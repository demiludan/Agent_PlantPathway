from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


class PipelineRunner:
    def __init__(
        self,
        repo_root: Path,
        python_path: str = sys.executable,
        logs_dir: Optional[Path] = None,
        dry_run: bool = False,
    ):
        self.repo_root = Path(repo_root)
        self.python_path = python_path
        self.logs_dir = logs_dir
        self.dry_run = dry_run
        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: Iterable[str], log_name: str) -> subprocess.CompletedProcess:
        cmd_list = list(cmd)
        if self.dry_run:
            print(f"[dry-run] {' '.join(cmd_list)}")
            return subprocess.CompletedProcess(cmd_list, 0, b"", b"")
        log_file = None
        if self.logs_dir:
            log_path = self.logs_dir / f"{log_name}.log"
            log_file = open(log_path, "w")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd_list,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        output_lines = []
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
            if log_file:
                log_file.write(line)
                log_file.flush()

        proc.wait()
        if log_file:
            log_file.close()

        output = "".join(output_lines)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({proc.returncode}): {' '.join(cmd_list)}\n{output}"
            )

        return subprocess.CompletedProcess(cmd_list, proc.returncode, output, "")

    def run_preprocessing(self, experiment: str, config_path: Path) -> None:
        print(f"[pipeline] Preprocessing {experiment}...")
        cmd = [
            self.python_path, "-u", "data_preprocessing.py",
            "--experiment", experiment, "--config", str(config_path),
        ]
        self._run(cmd, log_name=f"preprocess_{experiment}")

    def run_train(self, experiment: str, config_path: Path) -> None:
        print(f"[pipeline] Training {experiment}...")
        cmd = [
            self.python_path, "-u", "train.py",
            "--experiment", experiment, "--config", str(config_path),
        ]
        self._run(cmd, log_name=f"train_{experiment}")

    def run_inference(self, experiment: str, config_path: Path, checkpoint: str, split: str) -> None:
        print(f"[pipeline] Inference {experiment} split={split}...")
        cmd = [
            self.python_path, "-u", "inference.py",
            "--experiment", experiment, "--split", split,
            "--checkpoint", checkpoint, "--config", str(config_path),
        ]
        self._run(cmd, log_name=f"inference_{experiment}_{split}")

    def run_evaluation(self, experiment: str, config_path: Path, checkpoint: str, split: str) -> None:
        print(f"[pipeline] Evaluating {experiment} split={split}...")
        cmd = [
            self.python_path, "-u", "evaluate.py",
            "--experiment", experiment, "--split", split,
            "--config", str(config_path), "--checkpoint", checkpoint,
        ]
        self._run(cmd, log_name=f"evaluate_{experiment}_{split}")

    def run_full_pipeline(
        self,
        data_experiment: str,
        model_experiment: str,
        config_path: Path,
        checkpoint: str,
        splits: Iterable[str],
        skip_preprocess: bool = False,
        skip_train: bool = False,
        skip_inference: bool = False,
        skip_evaluate: bool = False,
    ) -> Dict[str, str]:
        outputs = {}
        if not skip_preprocess:
            self.run_preprocessing(data_experiment, config_path)
            outputs["preprocess"] = f"{data_experiment}"
        if not skip_train:
            self.run_train(model_experiment, config_path)
            outputs["train"] = f"{model_experiment}"
        if not skip_inference:
            for split in splits:
                self.run_inference(model_experiment, config_path, checkpoint, split)
            outputs["inference"] = f"{model_experiment}:{','.join(splits)}"
        if not skip_evaluate:
            for split in splits:
                self.run_evaluation(model_experiment, config_path, checkpoint, split)
            outputs["evaluate"] = f"{model_experiment}:{','.join(splits)}"
        return outputs
