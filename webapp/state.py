from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from .models import (
    CreateMessageRequest,
    CreateSessionRequest,
    LogEvent,
    Run,
    RunStatus,
    Session,
    Stage,
    StageName,
    StageStatus,
)


def _now() -> datetime:
    return datetime.utcnow()


class InMemoryStore:
    """Thread-safe asyncio store backing the temporary web session state."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._runs: Dict[str, Run] = {}
        self._logs: Dict[str, List[LogEvent]] = defaultdict(list)
        self._log_subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def create_session(self, payload: CreateSessionRequest) -> Session:
        async with self._lock:
            sid = str(uuid.uuid4())
            title = payload.title or "New Session"
            session = Session(id=sid, title=title, created_at=_now(), updated_at=_now())
            self._sessions[sid] = session
            return session

    async def list_sessions(self) -> List[Session]:
        async with self._lock:
            return list(self._sessions.values())

    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def create_run(self, session_id: str, payload: CreateMessageRequest) -> Run:
        async with self._lock:
            rid = str(uuid.uuid4())
            stages = [
                Stage(name=StageName.queue),
                Stage(name=StageName.preprocess),
                Stage(name=StageName.train),
                Stage(name=StageName.inference),
                Stage(name=StageName.evaluate),
                Stage(name=StageName.report),
            ]
            run = Run(
                id=rid, session_id=session_id, prompt=payload.prompt,
                status=RunStatus.queued, stages=stages,
                created_at=_now(), updated_at=_now(),
            )
            self._runs[rid] = run
            return run

    async def get_run(self, run_id: str) -> Optional[Run]:
        async with self._lock:
            return self._runs.get(run_id)

    async def list_runs_for_session(self, session_id: str) -> List[Run]:
        async with self._lock:
            return [r for r in self._runs.values() if r.session_id == session_id]

    async def update_run(self, run_id: str, **fields) -> Optional[Run]:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            data = run.dict()
            data.update(fields)
            data["updated_at"] = _now()
            self._runs[run_id] = Run(**data)
            return self._runs[run_id]

    async def update_stage(self, run_id: str, stage: StageName, status: StageStatus) -> Optional[Run]:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            updated_stages: List[Stage] = []
            now = _now()
            for s in run.stages:
                if s.name == stage:
                    start = s.started_at or now if status == StageStatus.running else s.started_at
                    finish = now if status in {StageStatus.completed, StageStatus.error, StageStatus.skipped} else s.finished_at
                    updated_stages.append(Stage(name=s.name, status=status, started_at=start, finished_at=finish))
                else:
                    updated_stages.append(s)
            run_data = run.dict()
            run_data["stages"] = [st.dict() for st in updated_stages]
            run_data["updated_at"] = now
            self._runs[run_id] = Run(**run_data)
            return self._runs[run_id]

    async def append_log(self, run_id: str, message: str, stage: Optional[StageName] = None) -> None:
        event = LogEvent(run_id=run_id, timestamp=_now(), message=message, stage=stage)
        async with self._lock:
            self._logs[run_id].append(event)
            for q in self._log_subscribers.get(run_id, []):
                q.put_nowait(event)

    async def get_logs(self, run_id: str, offset: int = 0, limit: int = 200) -> List[LogEvent]:
        async with self._lock:
            return self._logs.get(run_id, [])[offset : offset + limit]

    async def subscribe_logs(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._log_subscribers[run_id].append(queue)
        return queue

    async def unsubscribe_logs(self, run_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            subscribers = self._log_subscribers.get(run_id)
            if subscribers and queue in subscribers:
                subscribers.remove(queue)
