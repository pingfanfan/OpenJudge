import pytest

from prism.agent.registry import AgentBenchmarkRegistry
from prism.agent.task import AgentBenchmark, AgentTask


class _FakeAgent(AgentBenchmark):
    name = "fake_agent"
    version = "v1"

    def load_tasks(self, *, subset=None):
        return iter([
            AgentTask(task_id="x", workspace_files={}, user_instruction="", judge_command=[]),
        ])


def test_register_and_get():
    reg = AgentBenchmarkRegistry()
    reg.register(_FakeAgent)
    bm = reg.get("fake_agent")
    assert isinstance(bm, _FakeAgent)


def test_duplicate_raises():
    reg = AgentBenchmarkRegistry()
    reg.register(_FakeAgent)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_FakeAgent)


def test_missing_raises():
    reg = AgentBenchmarkRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_list_names():
    reg = AgentBenchmarkRegistry()
    reg.register(_FakeAgent)
    assert reg.names() == ["fake_agent"]


def test_get_class():
    reg = AgentBenchmarkRegistry()
    reg.register(_FakeAgent)
    cls = reg.get_class("fake_agent")
    assert cls is _FakeAgent
