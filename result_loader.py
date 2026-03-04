from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _collect_figures(base: Path, split: str) -> List[Path]:
    figures_dir = base / "figures" / split
    if not figures_dir.exists():
        return []
    return sorted(
        [p for p in figures_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".pdf", ".jpg"}]
    )


def load_evaluation_artifacts(
    experiment: str,
    checkpoint: str,
    splits: List[str],
    experiments_dir: Path = Path("experiments"),
) -> Dict[str, Dict]:
    """Load metrics and figure paths produced by evaluate.py."""
    checkpoint_dir = Path(checkpoint).stem
    output: Dict[str, Dict] = {}
    for split in splits:
        evaluate_dir = experiments_dir / experiment / "evaluate" / checkpoint_dir
        metrics_path = evaluate_dir / "metrics" / f"{experiment}_{split}_metrics.json"
        summary_path = evaluate_dir / "reports" / f"{experiment}_{split}_summary.txt"
        batch_results_path = evaluate_dir / f"{split}_batch_results.json"
        figures = _collect_figures(evaluate_dir, split)
        output[split] = {
            "metrics_path": metrics_path,
            "metrics": _load_json(metrics_path) or {},
            "batch_results_path": batch_results_path,
            "batch_results": _load_json(batch_results_path) or {},
            "figures_dir": evaluate_dir / "figures" / split,
            "figures": figures,
            "reports_dir": evaluate_dir / "reports",
            "summary_path": summary_path,
            "evaluate_dir": evaluate_dir,
        }
    return output
