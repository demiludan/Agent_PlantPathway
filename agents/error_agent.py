from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorSummary:
    command: str
    exit_code: Optional[int]
    stderr: str

    def to_markdown(self) -> str:
        lines = [
            f"- command: `{self.command}`",
            f"- exit_code: {self.exit_code if self.exit_code is not None else 'unknown'}",
            f"- stderr: {self.stderr.strip() or 'n/a'}",
        ]
        return "\n".join(lines)


def summarize_failure(command: str, exit_code: Optional[int], stderr: str) -> ErrorSummary:
    return ErrorSummary(command=command, exit_code=exit_code, stderr=stderr or "")
