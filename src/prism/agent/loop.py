"""Inline agent loop — multi-turn tool-calling against any Adapter.

Parses tool calls from OpenAI-format Adapter responses (via
`raw.choices[0].message.tool_calls`), executes them against a workspace,
and feeds the tool results back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism.adapters.base import Adapter, AdapterRequest
from prism.agent.tools import AGENT_TOOL_SCHEMAS, execute_tool

_SYSTEM_PROMPT = (
    "You are a coding agent with access to tools that can read and write files, "
    "and run bash commands. Complete the user's task by calling tools, then "
    "respond with a final message once you are done. Do not call tools after "
    "you have completed the task."
)


@dataclass
class _LoopResult:
    """Internal result of the agent loop, consumed by AgentRunner."""

    turns: int
    final_text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    trace: list[dict[str, Any]]


def _extract_tool_calls(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from an OpenAI-format response.raw."""
    try:
        msg = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return []
    calls = msg.get("tool_calls") or []
    return [c for c in calls if isinstance(c, dict)]


async def run_agent_loop(
    *,
    adapter: Adapter,
    workspace: Path,
    user_instruction: str,
    max_turns: int = 30,
) -> _LoopResult:
    """Run an inline multi-turn agent loop against `workspace`."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction},
    ]
    trace: list[dict[str, Any]] = [
        {"type": "user_instruction", "text": user_instruction},
    ]

    tokens_in = 0
    tokens_out = 0
    latency_ms = 0.0
    cost_usd = 0.0
    final_text = ""

    for turn in range(max_turns):
        resp = await adapter.complete(AdapterRequest(
            messages=messages,
            tools=AGENT_TOOL_SCHEMAS,
            max_output_tokens=4096,
        ))
        tokens_in += resp.tokens_in
        tokens_out += resp.tokens_out
        latency_ms += resp.latency_ms
        cost_usd += resp.cost_usd

        tool_calls = _extract_tool_calls(resp.raw)

        if not tool_calls:
            final_text = resp.text
            trace.append({"turn": turn, "type": "final", "text": resp.text})
            return _LoopResult(
                turns=turn + 1, final_text=final_text,
                tokens_in=tokens_in, tokens_out=tokens_out,
                latency_ms=latency_ms, cost_usd=cost_usd,
                trace=trace,
            )

        messages.append({
            "role": "assistant",
            "content": resp.text or None,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args, workspace=workspace)
            trace.append({
                "turn": turn, "type": "tool_call",
                "tool": name, "args": args, "result": result[:2000],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })

    trace.append({"type": "truncated", "reason": f"hit max_turns={max_turns}"})
    return _LoopResult(
        turns=max_turns, final_text=final_text or "(agent did not finish)",
        tokens_in=tokens_in, tokens_out=tokens_out,
        latency_ms=latency_ms, cost_usd=cost_usd,
        trace=trace,
    )
