from __future__ import annotations

from prism.agent.task import AgentBenchmark


class AgentBenchmarkRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, type[AgentBenchmark]] = {}

    def register(self, cls: type[AgentBenchmark]) -> None:
        if not cls.name:
            raise ValueError("AgentBenchmark class must set a non-empty `name`")
        if cls.name in self._by_name:
            raise ValueError(f"AgentBenchmark {cls.name!r} already registered")
        self._by_name[cls.name] = cls

    def get(self, name: str) -> AgentBenchmark:
        if name not in self._by_name:
            raise KeyError(f"unknown agent benchmark: {name!r}")
        return self._by_name[name]()

    def get_class(self, name: str) -> type[AgentBenchmark]:
        if name not in self._by_name:
            raise KeyError(f"unknown agent benchmark: {name!r}")
        return self._by_name[name]

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())


def agent_registry() -> AgentBenchmarkRegistry:
    """Build a fresh registry with the shipped agent benchmarks."""
    from prism.benchmarks.toy_agent.benchmark import ToyAgentBenchmark

    reg = AgentBenchmarkRegistry()
    reg.register(ToyAgentBenchmark)
    return reg
