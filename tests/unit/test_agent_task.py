import pytest

from prism.agent.task import AgentBenchmark, AgentResult, AgentTask


def test_agent_task_minimal():
    t = AgentTask(
        task_id="t1",
        workspace_files={"a.py": "print(1)"},
        user_instruction="Run a.py",
        judge_command=["python", "a.py"],
    )
    assert t.task_id == "t1"
    assert t.workspace_files == {"a.py": "print(1)"}
    assert t.timeout_seconds == 1200
    assert t.max_turns == 30


def test_agent_task_frozen():
    t = AgentTask(
        task_id="t1", workspace_files={}, user_instruction="x", judge_command=[],
    )
    with pytest.raises(AttributeError):
        t.task_id = "t2"  # type: ignore


def test_agent_result_minimal():
    r = AgentResult(
        task_id="t1", model_id="m1",
        success=True, turns=3, final_text="done",
        judge_stdout="1 passed", judge_exit_code=0,
        tokens_in=100, tokens_out=50, latency_ms=1500.0, cost_usd=0.002,
        trace=[{"turn": 0, "type": "tool_call"}],
    )
    assert r.success is True
    assert r.turns == 3


def test_agent_benchmark_is_abstract():
    with pytest.raises(TypeError):
        AgentBenchmark()  # type: ignore[abstract]
