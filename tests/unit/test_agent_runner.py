import json
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.agent import AgentRunner
from prism.service import RunService


class _SolvingAdapter(Adapter):
    """Scripted adapter that solves both toy tasks by issuing write_file then empty final."""

    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        last_user = next(
            (m for m in reversed(request.messages) if m.get("role") == "user"),
            {},
        )
        instr = last_user.get("content", "")
        has_tool_call_done = any(
            m.get("role") == "assistant" and m.get("tool_calls")
            for m in request.messages
        )
        if has_tool_call_done:
            return _final_response("Done.")

        if "add" in instr:
            return _tool_call_response(
                "write_file",
                {"path": "solution.py", "content": "def add(a, b):\n    return a + b\n"},
            )
        if "reverse" in instr:
            return _tool_call_response(
                "write_file",
                {"path": "solution.py", "content": "def reverse(s):\n    return s[::-1]\n"},
            )
        return _final_response("I don't know what to do.")


def _tool_call_response(tool: str, args: dict) -> AdapterResponse:
    raw = {"choices": [{"message": {
        "content": "",
        "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": tool, "arguments": json.dumps(args)},
        }],
    }, "finish_reason": "tool_calls"}]}
    return AdapterResponse(
        text="", reasoning_text=None,
        tokens_in=50, tokens_out=20, latency_ms=10.0, cost_usd=0.0, raw=raw,
        finish_reason="tool_calls",
    )


def _final_response(text: str) -> AdapterResponse:
    raw = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    return AdapterResponse(
        text=text, reasoning_text=None,
        tokens_in=10, tokens_out=5, latency_ms=1.0, cost_usd=0.0, raw=raw,
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_agent_runner_executes_toy_benchmark(tmp_path: Path):
    from prism.benchmarks.toy_agent.benchmark import ToyAgentBenchmark
    bm = ToyAgentBenchmark()
    profile = ModelProfile(
        id="solve", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    adapter = _SolvingAdapter(profile)

    svc = RunService(
        db_path=tmp_path / "p.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = AgentRunner(service=svc)
    summary = await runner.run(benchmark=bm, profile=profile, adapter=adapter)

    assert summary["task_count"] == 2
    assert summary["success_rate"] == pytest.approx(1.0)
    trace_files = sorted((tmp_path / "a").rglob("agent/*.json"))
    assert len(trace_files) == 2
    t = json.loads(trace_files[0].read_text())
    assert "task_id" in t
    assert "trace" in t
    assert "judge_stdout" in t
