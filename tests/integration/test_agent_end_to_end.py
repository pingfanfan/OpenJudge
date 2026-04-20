"""End-to-end agent integration test.

A scripted adapter plays the role of "a model that correctly writes the
solution file"; the actual tool execution, workspace I/O, and pytest judge
all run for real.
"""
import json
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.toy_agent.benchmark import ToyAgentBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.agent import AgentRunner
from prism.service import RunService


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


class _SolvingAdapter(Adapter):
    """Returns a write_file call for the first turn on each task, then 'Done.'."""
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        last_user = next(
            (m for m in reversed(request.messages) if m.get("role") == "user"),
            {},
        )
        instr = last_user.get("content", "")
        done = any(
            m.get("role") == "assistant" and m.get("tool_calls")
            for m in request.messages
        )
        if done:
            return _final_response("Done.")
        if "add" in instr:
            return _tool_call_response("write_file", {
                "path": "solution.py",
                "content": "def add(a, b):\n    return a + b\n",
            })
        if "reverse" in instr:
            return _tool_call_response("write_file", {
                "path": "solution.py",
                "content": "def reverse(s):\n    return s[::-1]\n",
            })
        return _final_response("I don't know.")


class _BrokenAdapter(Adapter):
    """Never writes a solution; just returns a final 'I give up' immediately."""
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        return _final_response("I give up.")


@pytest.mark.asyncio
async def test_solving_adapter_gets_100_percent(tmp_path: Path):
    bm = ToyAgentBenchmark()
    profile = ModelProfile(
        id="solve", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    svc = RunService(
        db_path=tmp_path / "p.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = AgentRunner(service=svc)
    summary = await runner.run(
        benchmark=bm, profile=profile, adapter=_SolvingAdapter(profile),
    )
    assert summary["task_count"] == 2
    assert summary["success_rate"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_broken_adapter_gets_0_percent(tmp_path: Path):
    bm = ToyAgentBenchmark()
    profile = ModelProfile(
        id="broken", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    svc = RunService(
        db_path=tmp_path / "p.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = AgentRunner(service=svc)
    summary = await runner.run(
        benchmark=bm, profile=profile, adapter=_BrokenAdapter(profile),
    )
    assert summary["task_count"] == 2
    assert summary["success_rate"] == pytest.approx(0.0)
