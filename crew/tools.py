from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from crewai.tools import tool

from agents.parsers import ParsedRequest
from agents.report_writer import final_report_writer
from config_overlay import ConfigOverlay
from pipeline_runner import PipelineRunner
from result_loader import load_evaluation_artifacts


def build_output_dir(base: Path, experiment: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / experiment / "assistant_runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass
class PipelineSettings:
    base_config: Path
    output_root: Path
    dry_run: bool = False
    repo_root: Path = Path(__file__).resolve().parent.parent


def build_run_pipeline_tool(settings: PipelineSettings):
    """Create a crewAI tool bound to the provided pipeline settings."""

    default_splits = ["train", "test"]

    @tool("run_classification_pipeline")
    def run_pipeline_tool(
        parsed: str,
        checkpoint: str = "best_model.pth",
        splits: Optional[List[str]] = None,
        skip_preprocess: bool = False,
        skip_train: bool = False,
        skip_inference: bool = False,
        skip_evaluate: bool = False,
    ) -> str:
        """Generate a config overlay, run preprocess->train->inference->evaluate, and return artifacts."""
        parsed_req = ParsedRequest(**parsed)
        if not parsed_req.experiment:
            return {"error": "No valid experiment provided."}

        data_experiment = parsed_req.data_experiment
        model_experiment = parsed_req.model_experiment
        if not data_experiment or not model_experiment:
            return {"error": "Unable to resolve experiment names."}

        run_dir = build_output_dir(settings.output_root, model_experiment)

        overlay = ConfigOverlay(settings.base_config)
        overlay_cfg = overlay.build_overlay(
            data_experiment=data_experiment,
            model_experiment=model_experiment,
        )
        generated_config = overlay.save(overlay_cfg, run_dir / "config.generated.yaml")

        runner = PipelineRunner(
            repo_root=settings.repo_root,
            logs_dir=run_dir / "logs",
            dry_run=settings.dry_run,
        )
        runner.run_full_pipeline(
            data_experiment=data_experiment,
            model_experiment=model_experiment,
            config_path=generated_config,
            checkpoint=checkpoint,
            splits=splits or default_splits,
            skip_preprocess=skip_preprocess,
            skip_train=skip_train,
            skip_inference=skip_inference,
            skip_evaluate=skip_evaluate,
        )

        artifacts = load_evaluation_artifacts(
            experiment=model_experiment,
            checkpoint=checkpoint,
            splits=splits or default_splits,
        )
        report_path = final_report_writer(
            experiment=model_experiment,
            artifacts=artifacts,
            output_dir=run_dir,
            config_path=generated_config,
        )

        return {
            "run_dir": str(run_dir),
            "config": str(generated_config),
            "report": str(report_path),
            "artifacts": artifacts,
        }

    return run_pipeline_tool
