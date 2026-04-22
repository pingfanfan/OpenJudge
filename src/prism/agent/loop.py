"""Inline agent loop — multi-turn tool-calling against any Adapter.

Two modes:
  * Native (default): pass `tools=AGENT_TOOL_SCHEMAS`, parse from
    `raw.choices[0].message.tool_calls`.
  * Prompted (`prompted_tool_use=True`): omit `tools`, describe the tool
    protocol in the system prompt, parse <tool_use>{json}</tool_use> blocks
    from response text. For endpoints that reject OpenAI-format custom tool
    schemas.
"""
from __future__ import annotations

import json
import re
import uuid
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

_PROMPTED_SYSTEM_PROMPT = """You are a coding agent. Complete the user's task by calling tools.

Tools available:
  read_file(path: str) -> str           — read a file inside the workspace
  write_file(path: str, content: str)   — create or overwrite a file
  bash(command: str) -> str             — run a shell command in the workspace

To call a tool, emit exactly one or more XML blocks in this format:

<tool_use>
{"name": "bash", "arguments": {"command": "ls -la"}}
</tool_use>

The arguments object MUST be valid JSON. After each <tool_use> block I will
execute the call and feed the result back to you in the next turn. You can
emit multiple <tool_use> blocks in a single turn.

When the task is finished, respond with a plain-text final message and do NOT
emit any more <tool_use> blocks."""

_TOOL_USE_RE = re.compile(r"<tool_use>\s*(\{.*?\})\s*</tool_use>", re.DOTALL)


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


def _extract_prompted_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse <tool_use>{...}</tool_use> blocks from model response text.

    Returns OpenAI-format tool_call dicts so the downstream executor path is
    shared between native and prompted modes.
    """
    calls: list[dict[str, Any]] = []
    for match in _TOOL_USE_RE.finditer(text or ""):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = parsed.get("name") or parsed.get("tool") or ""
        arguments = parsed.get("arguments") or parsed.get("args") or {}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    return calls


async def run_agent_loop(
    *,
    adapter: Adapter,
    workspace: Path,
    user_instruction: str,
    max_turns: int = 30,
    prompted_tool_use: bool = False,
) -> _LoopResult:
    """Run an inline multi-turn agent loop against `workspace`."""
    system_prompt = _PROMPTED_SYSTEM_PROMPT if prompted_tool_use else _SYSTEM_PROMPT
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
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
        request_kwargs: dict[str, Any] = {
            "messages": messages,
            "max_output_tokens": 4096,
        }
        if not prompted_tool_use:
            request_kwargs["tools"] = AGENT_TOOL_SCHEMAS
        resp = await adapter.complete(AdapterRequest(**request_kwargs))
        tokens_in += resp.tokens_in
        tokens_out += resp.tokens_out
        latency_ms += resp.latency_ms
        cost_usd += resp.cost_usd

        if prompted_tool_use:
            tool_calls = _extract_prompted_tool_calls(resp.text)
        else:
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

        if prompted_tool_use:
            messages.append({"role": "assistant", "content": resp.text})
        else:
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
            if prompted_tool_use:
                messages.append({
                    "role": "user",
                    "content": f"<tool_result name=\"{name}\">\n{result}\n</tool_result>",
                })
            else:
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
