from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentTask:
    """A single agent task: seed workspace + instruction + verification command."""

    task_id: str
    workspace_files: dict[str, str]
    user_instruction: str
    judge_command: list[str]
    timeout_seconds: int = 1200
    max_turns: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Final result of running an AgentTask through the agent loop."""

    task_id: str
    model_id: str
    success: bool
    turns: int
    final_text: str
    judge_stdout: str
    judge_exit_code: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    trace: list[dict[str, Any]]


class AgentBenchmark(ABC):
    """An agent benchmark is a named collection of AgentTasks."""

    name: str = ""
    track: str = "agent"
    version: str = "v1"

    @abstractmethod
    def load_tasks(self, *, subset: str | None = None) -> Iterable[AgentTask]: ...
