from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crewai import Crew, Process

from agents.parsers import ParsedRequest, VALID_EXPERIMENTS
from crew.config_loader import AgentTaskRegistry
from crew.tools import PipelineSettings, build_run_pipeline_tool
from result_loader import load_evaluation_artifacts


class PathwayCrew:
    """Crew entrypoint that relies on LLM agents for experiment/config parsing."""

    _stage_flags = ["skip_preprocess", "skip_train", "skip_inference", "skip_evaluate"]

    def __init__(
        self,
        base_config: Path,
        output_root: Path,
        checkpoint: str,
        splits: List[str],
        skip_preprocess: bool = False,
        skip_train: bool = False,
        skip_inference: bool = False,
        skip_evaluate: bool = False,
        dry_run: bool = False,
    ):
        self.checkpoint = checkpoint
        self.splits = splits
        self.skip_preprocess = skip_preprocess
        self.skip_train = skip_train
        self.skip_inference = skip_inference
        self.skip_evaluate = skip_evaluate
        self.dry_run = dry_run

        settings = PipelineSettings(
            base_config=base_config,
            output_root=output_root,
            dry_run=dry_run,
        )
        self.pipeline_tool: Any = build_run_pipeline_tool(settings)

        config_root = Path(__file__).resolve().parent.parent / "config"
        self.registry = AgentTaskRegistry(config_root=config_root)

    def run(self, prompt: str, experiment: Optional[str] = None) -> str:
        parsed_req, invalid_response = self._parse_prompt_with_agents(prompt, experiment)
        if invalid_response:
            return invalid_response

        resolved_skips, sources, reasons, notes = self._resolve_skip_flags(prompt, parsed_req)
        self._log_skip_resolution(resolved_skips, sources, reasons, notes)

        tool_result = self.pipeline_tool.run(
            parsed=asdict(parsed_req),
            checkpoint=self.checkpoint,
            splits=self.splits,
            skip_preprocess=resolved_skips["skip_preprocess"],
            skip_train=resolved_skips["skip_train"],
            skip_inference=resolved_skips["skip_inference"],
            skip_evaluate=resolved_skips["skip_evaluate"],
        )
        if "error" in tool_result:
            return f"Pipeline failed: {tool_result['error']}"

        return self._format_summary(parsed_req, tool_result)

    def _parse_prompt_with_agents(
        self, prompt: str, experiment_override: Optional[str]
    ) -> Tuple[Optional[ParsedRequest], Optional[str]]:
        """Run LLM agents to extract experiment and config from prompt."""
        exp_agent = self.registry.create_agent("experiment_parser_agent")
        cfg_agent = self.registry.create_agent("config_parser_agent")

        parse_exp_task = self.registry.create_task(
            "experiment_parser",
            agent=exp_agent,
            prompt=prompt,
            allowed_experiments=", ".join(VALID_EXPERIMENTS),
            experiment_override=experiment_override or "",
        )
        parse_cfg_task = self.registry.create_task(
            "config_parser",
            agent=cfg_agent,
            prompt=prompt,
        )

        parse_crew = Crew(
            agents=[exp_agent, cfg_agent],
            tasks=[parse_exp_task, parse_cfg_task],
            process=Process.sequential,
            verbose=True,
        )
        parse_crew.kickoff()

        try:
            exp_payload = json.loads(parse_exp_task.output.raw)
            cfg_payload = json.loads(parse_cfg_task.output.raw)
        except Exception as exc:
            return None, self._invalid_response(f"Parser failed to return JSON ({exc}).")

        requested_exp = experiment_override or exp_payload.get("experiment")
        exp_errors = exp_payload.get("errors") or []
        if not requested_exp or requested_exp not in VALID_EXPERIMENTS:
            reason = exp_errors[0] if exp_errors else "Missing or invalid experiment."
            return None, self._invalid_response(reason)

        parsed_req = ParsedRequest(
            experiment=requested_exp,
            x_variable=cfg_payload.get("x_variable", "CO2S"),
            errors=[],
        )
        return parsed_req, None

    def _invalid_response(self, reason: str) -> str:
        agent = self.registry.create_agent("invalid_input_agent")
        task = self.registry.create_task(
            "invalid_input",
            agent=agent,
            reason=reason,
            allowed_experiments=", ".join(VALID_EXPERIMENTS),
            example_prompt="Classify C3 vs C4 using CO2S curves",
        )
        invalid_crew = Crew(
            agents=[agent], tasks=[task],
            process=Process.sequential, verbose=True,
        )
        invalid_crew.kickoff()
        return task.output.raw

    def _parse_skip_overrides(self, prompt: str) -> Dict[str, Any]:
        agent = self.registry.create_agent("command_parser_agent")
        task = self.registry.create_task("command_parser", agent=agent, prompt=prompt)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        crew.kickoff()
        try:
            return json.loads(task.output.raw)
        except Exception:
            return {}

    def _artifacts_available(self, parsed_req: ParsedRequest) -> bool:
        if not parsed_req or not parsed_req.model_experiment:
            return False
        artifacts = load_evaluation_artifacts(
            experiment=parsed_req.model_experiment,
            checkpoint=self.checkpoint,
            splits=self.splits,
        )
        for split_payload in artifacts.values():
            metrics_path = split_payload.get("metrics_path")
            if not metrics_path or not metrics_path.exists():
                return False
        return True

    def _resolve_skip_flags(
        self, prompt: str, parsed_req: ParsedRequest
    ) -> Tuple[Dict[str, bool], Dict[str, str], List[str], List[str]]:
        cli_flags = {flag: getattr(self, flag) for flag in self._stage_flags}
        payload = self._parse_skip_overrides(prompt)

        parsed_flags: Dict[str, bool] = {}
        reasons: List[str] = []
        for flag in self._stage_flags:
            val = payload.get(flag)
            if isinstance(val, bool):
                parsed_flags[flag] = val
        report_only = payload.get("report_only")
        if isinstance(report_only, bool):
            parsed_flags["_report_only"] = report_only
        reasons_payload = payload.get("reasons", [])
        if isinstance(reasons_payload, list):
            reasons = [str(r) for r in reasons_payload]

        resolved = dict(cli_flags)
        sources = {flag: "cli" for flag in self._stage_flags}
        for flag, val in parsed_flags.items():
            if flag in self._stage_flags:
                resolved[flag] = val
                sources[flag] = "prompt"

        notes: List[str] = []
        if parsed_flags.get("_report_only") is True:
            if self._artifacts_available(parsed_req):
                for flag in self._stage_flags:
                    resolved[flag] = True
                    sources[flag] = "prompt(report_only)"
                notes.append("Report-only honored: required artifacts found.")
            else:
                for flag in self._stage_flags:
                    resolved[flag] = False
                    sources[flag] = "fallback_missing_artifacts"
                notes.append("Report-only requested but artifacts missing; running all steps.")

        return resolved, sources, reasons, notes

    def _log_skip_resolution(self, resolved, sources, reasons, notes):
        print("[skip] resolved flags:")
        for flag in self._stage_flags:
            print(f"  - {flag}: {resolved[flag]} (source={sources.get(flag, 'unknown')})")
        if reasons:
            print("[skip] reasons:")
            for reason in reasons:
                print(f"  - {reason}")
        if notes:
            print("[skip] notes:")
            for note in notes:
                print(f"  - {note}")

    def _format_summary(self, parsed_req: ParsedRequest, tool_result: dict) -> str:
        lines = [
            f"Pipeline run completed for {parsed_req.model_experiment}.",
            f"- Run directory: {tool_result['run_dir']}",
            f"- Generated config: {tool_result['config']}",
            f"- Report: {tool_result['report']}",
        ]
        return "\n".join(lines)
