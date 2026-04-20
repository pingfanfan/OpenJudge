from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from prism.judges.base import Judge


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    task_id: str
    version: str
    messages: list[dict[str, Any]]
    expected: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """A benchmark is a named, versioned, loader+judge pair."""

    name: str = ""
    track: str = "limit"
    version: str = "v1"

    @abstractmethod
    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]: ...

    @abstractmethod
    def make_judge(self, prompt: PromptSpec) -> Judge: ...
