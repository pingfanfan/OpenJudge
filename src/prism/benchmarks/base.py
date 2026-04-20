from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from prism.judges.base import Judge

if TYPE_CHECKING:
    from prism.adapters.base import Adapter


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

    # True if this benchmark constructs an LLMJudge. Callers must provide
    # llm_judge_adapter when calling make_judge on such benchmarks.
    needs_llm_judge: bool = False

    # Per-subset item caps. Subclasses override to specialize.
    subset_caps: dict[str, int | None] = {
        "quick": 100,
        "standard": 500,
        "full": None,
    }

    @abstractmethod
    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]: ...

    @abstractmethod
    def make_judge(
        self,
        prompt: PromptSpec,
        *,
        llm_judge_adapter: Adapter | None = None,
    ) -> Judge: ...

    def _cap_for(self, subset: str | None) -> int | None:
        if subset is None:
            return self.subset_caps.get("quick")
        return self.subset_caps.get(subset, self.subset_caps.get("quick"))


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, type[Benchmark]] = {}

    def register(self, cls: type[Benchmark]) -> None:
        if not cls.name:
            raise ValueError("Benchmark class must set a non-empty `name`")
        if cls.name in self._by_name:
            raise ValueError(f"Benchmark {cls.name!r} already registered")
        self._by_name[cls.name] = cls

    def get(self, name: str) -> Benchmark:
        if name not in self._by_name:
            raise KeyError(f"unknown benchmark: {name!r}")
        return self._by_name[name]()

    def get_class(self, name: str) -> type[Benchmark]:
        if name not in self._by_name:
            raise KeyError(f"unknown benchmark: {name!r}")
        return self._by_name[name]

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())
