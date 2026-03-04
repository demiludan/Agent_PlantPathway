from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from .models import CreateMessageRequest, RunStatus, StageName, StageStatus
from .state import InMemoryStore


class RunnerService:
    """Runs the existing CLI pipeline via subprocess and streams logs to the store."""

    def __init__(self, store: InMemoryStore, repo_root: Path | str | None = None, python_path: str | None = None):
        self.store = store
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
        self.python_path = python_path or sys.executable

    async def start_run(self, session_id: str, payload: CreateMessageRequest) -> str:
        run = await self.store.create_run(session_id, payload)
        asyncio.create_task(self._execute(run.id, payload))
        return run.id

    async def _execute(self, run_id: str, payload: CreateMessageRequest) -> None:
        await self.store.update_stage(run_id, StageName.queue, StageStatus.completed)
        await self.store.update_run(run_id, status=RunStatus.running)

        cmd = [self.python_path, "-u", "main.py", "--prompt", payload.prompt]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        current_stage: Optional[StageName] = None
        run_dir: Optional[str] = None
        report_path: Optional[str] = None

        async def set_stage(new_stage: StageName) -> None:
            nonlocal current_stage
            if current_stage and current_stage != new_stage:
                await self.store.update_stage(run_id, current_stage, StageStatus.completed)
            current_stage = new_stage
            await self.store.update_stage(run_id, new_stage, StageStatus.running)

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode(errors="ignore").rstrip("\n")
            stage = self._detect_stage(line)
            if stage:
                await set_stage(stage)
            if "Run directory:" in line:
                run_dir = line.split("Run directory:", 1)[1].strip()
            if "Report:" in line:
                report_path = line.split("Report:", 1)[1].strip()
            await self.store.append_log(run_id, line, stage=stage)

        await proc.wait()

        if current_stage:
            await self.store.update_stage(run_id, current_stage, StageStatus.completed)

        if proc.returncode == 0:
            await self.store.update_stage(run_id, StageName.report, StageStatus.completed)
            await self.store.update_run(
                run_id, status=RunStatus.completed,
                run_dir=run_dir, report_path=report_path, error=None,
            )
        else:
            await self.store.update_stage(run_id, StageName.report, StageStatus.error)
            await self.store.update_run(
                run_id, status=RunStatus.error,
                error=f"Process exited with code {proc.returncode}",
                run_dir=run_dir, report_path=report_path,
            )

    def _detect_stage(self, line: str) -> Optional[StageName]:
        lowered = line.lower()
        if "[pipeline] preprocessing" in lowered:
            return StageName.preprocess
        if "[pipeline] training" in lowered:
            return StageName.train
        if "[pipeline] inference" in lowered:
            return StageName.inference
        if "[pipeline] evaluating" in lowered:
            return StageName.evaluate
        return None
