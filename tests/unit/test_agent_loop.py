from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.agent.loop import run_agent_loop


class _ScriptedAdapter(Adapter):
    def __init__(self, profile, responses: list[AdapterResponse]) -> None:
        super().__init__(profile)
        self._responses = responses
        self._idx = 0

    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        r = self._responses[self._idx]
        self._idx += 1
        return r


def _resp_with_tool_call(tool_name: str, args: dict, call_id: str = "c1") -> AdapterResponse:
    import json
    raw = {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(args)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    return AdapterResponse(
        text="", reasoning_text=None,
        tokens_in=10, tokens_out=5, latency_ms=1.0, cost_usd=0.0, raw=raw,
        finish_reason="tool_calls",
    )


def _resp_final(text: str) -> AdapterResponse:
    raw = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    return AdapterResponse(
        text=text, reasoning_text=None,
        tokens_in=10, tokens_out=5, latency_ms=1.0, cost_usd=0.0, raw=raw,
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_agent_loop_write_then_finish(tmp_path: Path):
    from prism.config.model_profile import ModelProfile
    profile = ModelProfile(id="m1", provider="openai", model="x")
    adapter = _ScriptedAdapter(profile, [
        _resp_with_tool_call("write_file", {"path": "out.txt", "content": "done"}, call_id="c1"),
        _resp_final("All done."),
    ])
    result = await run_agent_loop(
        adapter=adapter,
        workspace=tmp_path,
        user_instruction="Write out.txt with 'done'.",
        max_turns=5,
    )
    assert (tmp_path / "out.txt").read_text() == "done"
    assert result.turns == 2
    assert result.final_text == "All done."
    assert result.tokens_in == 20
    assert len(result.trace) >= 3
    types = [t["type"] for t in result.trace]
    assert "tool_call" in types
    assert "final" in types


@pytest.mark.asyncio
async def test_agent_loop_respects_max_turns(tmp_path: Path):
    from prism.config.model_profile import ModelProfile
    profile = ModelProfile(id="m1", provider="openai", model="x")
    looping = [_resp_with_tool_call("bash", {"command": "true"}, call_id=f"c{i}") for i in range(100)]
    adapter = _ScriptedAdapter(profile, looping)

    result = await run_agent_loop(
        adapter=adapter,
        workspace=tmp_path,
        user_instruction="Keep trying.",
        max_turns=3,
    )
    assert result.turns == 3
    assert any(
        t.get("type") == "truncated" or "max_turns" in str(t)
        for t in result.trace
    )
