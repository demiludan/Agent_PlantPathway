from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crew.pathway_crew import PathwayCrew
from agents.parsers import VALID_EXPERIMENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C3 vs C4 photosynthetic pathway classification orchestrator"
    )
    parser.add_argument("--prompt", type=str, help="Natural-language prompt (if omitted, prompts stdin)")
    parser.add_argument("--experiment", type=str, choices=VALID_EXPERIMENTS, help="Override: experiment name")
    parser.add_argument("--base-config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = args.prompt or input(
        "Enter request (e.g., 'Classify C3 vs C4 using CO2S curves'): "
    ).strip()

    crew = PathwayCrew(
        base_config=args.base_config,
        output_root=args.output_root,
        checkpoint=args.checkpoint,
        splits=args.splits,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train,
        skip_inference=args.skip_inference,
        skip_evaluate=args.skip_evaluate,
        dry_run=args.dry_run,
    )
    result = crew.run(prompt=prompt, experiment=args.experiment)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
