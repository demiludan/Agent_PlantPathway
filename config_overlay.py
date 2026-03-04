from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pyyaml is required. Install via `pip install pyyaml`.") from exc


class ConfigOverlay:
    def __init__(self, base_config_path: Path):
        self.base_config_path = Path(base_config_path)
        self._base = self._load()

    def _load(self) -> Dict:
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.base_config_path}")
        return yaml.safe_load(self.base_config_path.read_text())

    def build_overlay(
        self,
        data_experiment: str,
        model_experiment: str,
    ) -> Dict:
        cfg = copy.deepcopy(self._base)
        for name in (data_experiment, model_experiment):
            if name not in cfg:
                raise KeyError(f"Experiment '{name}' not found in config {self.base_config_path}")
        return cfg

    def save(self, cfg: Dict, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        return path
