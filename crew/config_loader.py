from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from crewai import Agent, Task


class AgentTaskRegistry:
    """Loads agent/task definitions from YAML and instantiates crewAI objects."""

    def __init__(self, config_root: Path):
        config_root = Path(config_root)
        agents_path = config_root / "agents.yaml"
        tasks_path = config_root / "tasks.yaml"
        if not agents_path.exists() or not tasks_path.exists():
            raise FileNotFoundError(f"Missing agents/tasks config under {config_root}")
        self._agents: Dict[str, Dict[str, Any]] = yaml.safe_load(agents_path.read_text())
        self._tasks: Dict[str, Dict[str, Any]] = yaml.safe_load(tasks_path.read_text())

    def create_agent(self, key: str) -> Agent:
        data = self._agents[key]
        return Agent(
            role=data["role"],
            goal=data["goal"],
            backstory=data["backstory"],
            allow_delegation=data.get("allow_delegation", False),
            verbose=data.get("verbose", False),
        )

    def create_task(self, key: str, agent: Agent, **format_kwargs: Any) -> Task:
        cfg = self._tasks[key]
        description = cfg["description"].format(**format_kwargs)
        expected_output = cfg.get("expected_output", "")
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
