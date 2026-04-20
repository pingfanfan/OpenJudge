# Prism P3a — Agent Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑通 Prism 的第二条赛道（Agent）的端到端 pipeline：从"用户指令 + 工作目录" → 多轮 tool-calling 循环 → 执行硬判（pytest/build）→ 记录完整 trace → 持久化得分。P3a 是走通形状的最小实现，用**内联 agent 循环**（LiteLLMAdapter + OpenAI tool-calling 规范）证明接口合理，为后续 P3b 替换为真实 Claude Code CLI 留好插拔点。

**Architecture:** 新增 `prism.agent` 子包承载 AgentTask / 工具 / agent 循环 / workspace 上下文；`prism.runners.agent.AgentRunner` 负责每个 task 的生命周期（建 tmpdir → 解 workspace → 跑循环 → 硬判 → 持久化）。CLI 的 `prism run` 按 `--track` 分发到 `LimitRunner` 或 `AgentRunner`。评分复用现有 `Score` 表（score=1 if 硬判过 else 0）；多轮 trace 以 JSON 存到 `artifacts/<run_id>/agent/<task_id>.json`。

**Tech Stack:** 基于 P1–P2f，无新依赖（工具执行用 `subprocess` + `shutil`；tool-calling 走 LiteLLM 原生 `tools=[...]` 参数）。

---

## 参考文档

- 设计文档：§3.1 Runner 模块、§5 Agent 赛道（§5.3 执行流程是 P3a 的直接目标）
- P2f plan：已完成的 Limit 赛道收尾

---

## 范围边界

**In scope (P3a):**
- `prism.agent` 子包：`AgentTask` dataclass、`AgentBenchmark` ABC、`AgentResult` dataclass、三种工具（`read_file` / `write_file` / `bash`）、`run_agent_loop` 内联循环、workspace 上下文管理器、`run_hard_judge` 硬判函数
- `prism.runners.agent.AgentRunner`：工作目录 × 模型 → `AgentResult`
- `prism.benchmarks.toy_agent`：一个"实现 add 函数"玩具任务 benchmark（2 个 task），用于端到端验证
- `agent_registry()` 工厂（独立于 `default_registry`）
- CLI：`prism run --track agent --benchmark toy_agent --model <yaml>` 分发到 `AgentRunner`
- 持久化：Response.text=final agent 消息；Score=硬判结果；完整 trace 进 artifact JSON
- 端到端集成测试：fake adapter 模拟一个会调 `write_file` + `bash` 的 agent，验证 workspace 修改、硬判通过、trace 完整

**Out of scope（后续 plan）:**
- 真实 Claude Code CLI 集成（subprocess + Router 代理）— **P3b**
- SWE-Bench / Terminal-Bench / Tau-Bench / Aider / CORE-Bench 学术 benchmark — **P3c**
- Prism Real Tasks（30 个真实工程任务）+ 软判 LLM judge — **P3d**
- 工具调用深度曲线 / 大 repo 压力 / 错误恢复测试 — **P3e**
- 资源限制（CPU / 内存 / 网络隔离）— 未规划（沙箱化留给独立 plan）

---

## 关键设计决策

### 1. Agent vs Limit 的 ABC 关系
不复用 `Benchmark` ABC。原因：`load_prompts` 返回单轮 `PromptSpec`，`load_tasks` 返回多轮 `AgentTask`；`make_judge(prompt)` 对 Agent 没意义（硬判是 `judge_command`）。两者是**兄弟 ABC**，都注册到各自的 registry。

### 2. 工具格式
走 **OpenAI tool-calling 规范**（LiteLLM 原生支持，各厂商自动翻译）：
```python
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "...",
    "parameters": {"type": "object", "properties": {...}, "required": [...]}
  }
}
```
Adapter 层无需改动 —— P1 的 `AdapterRequest.tools` 字段已是 `list[dict[str, Any]]`。

### 3. Workspace 表示
P3a 用 `dict[str, str]`（文件路径 → 内容字符串）。真实 repo 的 tarball 留 P3c。原因：P3a 的玩具任务 3-5 个小文件就够，dict 更易构造 / 序列化 / 测试。

### 4. 持久化映射
- `Response.text` = 最终一轮 assistant 消息的 content
- `Response.tokens_in/out/cost/latency` = 跨所有轮次累计
- `Score.score` = 1.0 if 硬判 exit=0 else 0.0，`Score.judge` = "agent_hard"
- Artifact `agent/<task_id>-seed<N>.json` = 完整 trace（所有轮次 messages + tool calls + tool results + final diff + judge output）

### 5. 限流与并发
复用 `OrchestratorRunner` 不合适（它是单次 complete 设计）。AgentRunner 自己管理并发：`asyncio.Semaphore` + 每 model 限流。为简化 P3a，串行跑（`max_concurrency=1`），优化留后续。

---

## 文件结构（P3a 完成后新增 / 修改）

```
src/prism/
├── agent/                              # NEW 子包
│   ├── __init__.py
│   ├── task.py                          # AgentTask + AgentBenchmark ABC + AgentResult
│   ├── tools.py                         # tool schemas + execute_tool
│   ├── workspace.py                     # workspace_context + apply_files
│   ├── loop.py                          # run_agent_loop (inline multi-turn)
│   ├── judge.py                         # run_hard_judge (subprocess)
│   └── registry.py                      # agent_registry() factory
├── runners/
│   └── agent.py                         # NEW — AgentRunner
├── benchmarks/
│   └── toy_agent/                       # NEW — proof-of-concept benchmark
│       ├── __init__.py
│       └── benchmark.py
├── cli.py                               # MODIFY — dispatch on --track

tests/
├── unit/
│   ├── test_agent_task.py               # NEW
│   ├── test_agent_tools.py              # NEW
│   ├── test_agent_workspace.py          # NEW
│   ├── test_agent_loop.py               # NEW
│   ├── test_agent_judge.py              # NEW
│   ├── test_agent_runner.py             # NEW
│   ├── test_agent_registry.py           # NEW
│   └── test_toy_agent_benchmark.py      # NEW
└── integration/
    └── test_agent_end_to_end.py         # NEW
```

---

## Task 1: AgentTask + AgentBenchmark ABC + AgentResult

**Files:**
- Create: `src/prism/agent/__init__.py`
- Create: `src/prism/agent/task.py`
- Test: `tests/unit/test_agent_task.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_agent_task.py`:
```python
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
    assert t.timeout_seconds == 1200  # default
    assert t.max_turns == 30  # default


def test_agent_task_frozen():
    t = AgentTask(
        task_id="t1", workspace_files={}, user_instruction="x", judge_command=[],
    )
    with pytest.raises(Exception):  # dataclass FrozenInstanceError
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
```

- [ ] **Step 2: Fail**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/unit/test_agent_task.py -v
```

- [ ] **Step 3: Implement**

Create `src/prism/agent/__init__.py`:
```python
from prism.agent.task import AgentBenchmark, AgentResult, AgentTask

__all__ = ["AgentBenchmark", "AgentResult", "AgentTask"]
```

Create `src/prism/agent/task.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentTask:
    """A single agent task: seed workspace + instruction + verification command."""

    task_id: str
    workspace_files: dict[str, str]
    user_instruction: str
    judge_command: list[str]
    timeout_seconds: int = 1200
    max_turns: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Final result of running an AgentTask through the agent loop."""

    task_id: str
    model_id: str
    success: bool            # judge_exit_code == 0
    turns: int               # number of agent-model exchanges used
    final_text: str          # last assistant message text
    judge_stdout: str        # combined stdout+stderr of judge_command
    judge_exit_code: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    trace: list[dict[str, Any]]


class AgentBenchmark(ABC):
    """An agent benchmark is a named collection of AgentTasks."""

    name: str = ""
    track: str = "agent"
    version: str = "v1"

    @abstractmethod
    def load_tasks(self, *, subset: str | None = None) -> Iterable[AgentTask]: ...
```

- [ ] **Step 4: Pass — 4 tests**

- [ ] **Step 5: Commit**

```bash
cd /Users/pingfan/projects/prism
git add src/prism/agent tests/unit/test_agent_task.py
git commit -m "feat(agent): add AgentTask, AgentResult, AgentBenchmark ABC"
```

---

## Task 2: Agent tools — schemas + executor

**Files:**
- Create: `src/prism/agent/tools.py`
- Test: `tests/unit/test_agent_tools.py`

Three tools: `read_file`, `write_file`, `bash`. All constrained to operate inside a given workspace root — no path escapes.

- [ ] **Step 1: Failing test**

Create `tests/unit/test_agent_tools.py`:
```python
from pathlib import Path

import pytest

from prism.agent.tools import AGENT_TOOL_SCHEMAS, execute_tool


def test_tool_schemas_shape():
    assert len(AGENT_TOOL_SCHEMAS) == 3
    names = {t["function"]["name"] for t in AGENT_TOOL_SCHEMAS}
    assert names == {"read_file", "write_file", "bash"}


def test_read_file(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("world")
    out = execute_tool("read_file", {"path": "hello.txt"}, workspace=tmp_path)
    assert out == "world"


def test_read_file_missing(tmp_path: Path):
    out = execute_tool("read_file", {"path": "nope.txt"}, workspace=tmp_path)
    assert "Error" in out or "not found" in out.lower()


def test_read_file_path_escape_rejected(tmp_path: Path):
    (tmp_path.parent / "outside.txt").write_text("secret")
    out = execute_tool("read_file", {"path": "../outside.txt"}, workspace=tmp_path)
    assert "Error" in out or "outside" in out.lower()


def test_write_file(tmp_path: Path):
    out = execute_tool(
        "write_file",
        {"path": "new.txt", "content": "hello"},
        workspace=tmp_path,
    )
    assert "success" in out.lower() or "wrote" in out.lower()
    assert (tmp_path / "new.txt").read_text() == "hello"


def test_write_file_creates_parent_dirs(tmp_path: Path):
    execute_tool(
        "write_file",
        {"path": "sub/deep/x.txt", "content": "ok"},
        workspace=tmp_path,
    )
    assert (tmp_path / "sub" / "deep" / "x.txt").read_text() == "ok"


def test_bash_simple(tmp_path: Path):
    (tmp_path / "greet.txt").write_text("hi")
    out = execute_tool("bash", {"command": "cat greet.txt"}, workspace=tmp_path)
    assert "hi" in out


def test_bash_nonzero_exit(tmp_path: Path):
    out = execute_tool("bash", {"command": "exit 7"}, workspace=tmp_path)
    # Should report exit code
    assert "7" in out


def test_bash_timeout(tmp_path: Path):
    out = execute_tool(
        "bash",
        {"command": "sleep 10"},
        workspace=tmp_path,
        bash_timeout_sec=1,
    )
    assert "timed out" in out.lower() or "timeout" in out.lower()


def test_unknown_tool(tmp_path: Path):
    out = execute_tool("nonexistent", {}, workspace=tmp_path)
    assert "unknown" in out.lower() or "error" in out.lower()
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/agent/tools.py`:
```python
"""Agent tools: read_file, write_file, bash.

All tools are sandboxed to a given `workspace` Path — paths that resolve
outside the workspace are rejected.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace. Returns the file contents as a string, or an error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the workspace root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a file in the workspace. Parent directories are created as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path."},
                    "content": {"type": "string", "description": "File contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the workspace directory. Returns combined stdout+stderr and the exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                },
                "required": ["command"],
            },
        },
    },
]


def _resolve_inside(workspace: Path, relative: str) -> Path | None:
    """Resolve a relative path inside workspace. Returns None if it escapes."""
    workspace = workspace.resolve()
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    return candidate


def execute_tool(
    name: str,
    args: dict[str, Any],
    *,
    workspace: Path,
    bash_timeout_sec: float = 60.0,
) -> str:
    """Dispatch to the named tool. Returns a string result (success or error)."""
    if name == "read_file":
        relative = args.get("path", "")
        path = _resolve_inside(workspace, relative)
        if path is None:
            return f"Error: path {relative!r} escapes the workspace."
        if not path.exists():
            return f"Error: file not found: {relative}"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading {relative}: {type(e).__name__}: {e}"

    if name == "write_file":
        relative = args.get("path", "")
        content = args.get("content", "")
        path = _resolve_inside(workspace, relative)
        if path is None:
            return f"Error: path {relative!r} escapes the workspace."
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {relative}."

    if name == "bash":
        command = args.get("command", "")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=bash_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {bash_timeout_sec}s"
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            return f"Exit code {proc.returncode}. Output:\n{output}"
        return output or "(no output)"

    return f"Error: unknown tool {name!r}"
```

- [ ] **Step 4: Pass — 10 tests**

- [ ] **Step 5: Commit**

```bash
git add src/prism/agent/tools.py tests/unit/test_agent_tools.py
git commit -m "feat(agent): three sandboxed tools — read_file, write_file, bash"
```

---

## Task 3: Workspace context manager

**Files:**
- Create: `src/prism/agent/workspace.py`
- Test: `tests/unit/test_agent_workspace.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_agent_workspace.py`:
```python
from prism.agent.workspace import workspace_context


def test_workspace_creates_files():
    files = {"a.py": "print(1)", "sub/b.txt": "hello"}
    with workspace_context(files) as ws:
        assert (ws / "a.py").read_text() == "print(1)"
        assert (ws / "sub" / "b.txt").read_text() == "hello"
        assert ws.exists()
    # After exit, directory is cleaned up
    assert not ws.exists()


def test_workspace_empty_files():
    with workspace_context({}) as ws:
        assert ws.exists()
        assert list(ws.iterdir()) == []


def test_workspace_nested_dirs_created():
    files = {"a/b/c/d.txt": "deep"}
    with workspace_context(files) as ws:
        assert (ws / "a" / "b" / "c" / "d.txt").read_text() == "deep"
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/agent/workspace.py`:
```python
"""Workspace lifecycle management for agent tasks."""
from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def workspace_context(files: dict[str, str]) -> Iterator[Path]:
    """Create a tempdir, populate with files, yield the Path, cleanup on exit."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        yield root
```

- [ ] **Step 4: Pass — 3 tests**

- [ ] **Step 5: Commit**

```bash
git add src/prism/agent/workspace.py tests/unit/test_agent_workspace.py
git commit -m "feat(agent): workspace_context manager for seeded tmpdir + cleanup"
```

---

## Task 4: Inline agent loop (multi-turn tool calling)

**Files:**
- Create: `src/prism/agent/loop.py`
- Test: `tests/unit/test_agent_loop.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_agent_loop.py`:
```python
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.agent.loop import run_agent_loop


class _ScriptedAdapter(Adapter):
    """Replay a fixed list of responses in sequence."""

    def __init__(self, profile, responses: list[AdapterResponse]) -> None:
        super().__init__(profile)
        self._responses = responses
        self._idx = 0

    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        r = self._responses[self._idx]
        self._idx += 1
        return r


def _resp_with_tool_call(tool_name: str, args: dict, call_id: str = "c1") -> AdapterResponse:
    """Build an AdapterResponse that contains a tool call in its `raw`."""
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
    """Scripted: turn 0 writes a file, turn 1 returns final text."""
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
    assert result.turns == 2  # 1 tool-call round + 1 final round
    assert result.final_text == "All done."
    assert result.tokens_in == 20  # 2 × 10
    assert len(result.trace) >= 3  # user msg + tool call + tool result + final
    # Trace entries should have `type` field
    types = [t["type"] for t in result.trace]
    assert "tool_call" in types
    assert "final" in types


@pytest.mark.asyncio
async def test_agent_loop_respects_max_turns(tmp_path: Path):
    """If adapter keeps calling tools forever, loop should terminate at max_turns."""
    from prism.config.model_profile import ModelProfile
    profile = ModelProfile(id="m1", provider="openai", model="x")
    # Infinite loop of bash commands
    looping = [_resp_with_tool_call("bash", {"command": "true"}, call_id=f"c{i}") for i in range(100)]
    adapter = _ScriptedAdapter(profile, looping)

    result = await run_agent_loop(
        adapter=adapter,
        workspace=tmp_path,
        user_instruction="Keep trying.",
        max_turns=3,
    )
    assert result.turns == 3
    # Last entry in trace notes the truncation
    assert any(
        t.get("type") == "truncated" or "max_turns" in str(t)
        for t in result.trace
    )
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/agent/loop.py`:
```python
"""Inline agent loop — multi-turn tool-calling against any Adapter.

This is P3a's proof-of-concept runner. It parses tool calls from
OpenAI-format Adapter responses (via `raw.choices[0].message.tool_calls`),
executes them against a workspace, and feeds the tool results back.

Later replaced by P3b's real-Claude-Code subprocess runner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prism.adapters.base import Adapter, AdapterRequest
from prism.agent.tools import AGENT_TOOL_SCHEMAS, execute_tool

_SYSTEM_PROMPT = """You are a coding agent with access to tools that can read and write files, and run bash commands. Complete the user's task by calling tools, then respond with a final message once you are done. Do not call tools after you have completed the task."""


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

        # Append assistant message with tool calls to conversation.
        messages.append({
            "role": "assistant",
            "content": resp.text or None,
            "tool_calls": tool_calls,
        })

        # Execute each tool and append tool_result messages.
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

    # Exhausted max_turns without a final message.
    trace.append({"type": "truncated", "reason": f"hit max_turns={max_turns}"})
    return _LoopResult(
        turns=max_turns, final_text=final_text or "(agent did not finish)",
        tokens_in=tokens_in, tokens_out=tokens_out,
        latency_ms=latency_ms, cost_usd=cost_usd,
        trace=trace,
    )
```

- [ ] **Step 4: Pass — 2 tests**

- [ ] **Step 5: Commit**

```bash
git add src/prism/agent/loop.py tests/unit/test_agent_loop.py
git commit -m "feat(agent): inline run_agent_loop with tool-calling multi-turn"
```

---

## Task 5: Hard judgment runner

**Files:**
- Create: `src/prism/agent/judge.py`
- Test: `tests/unit/test_agent_judge.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_agent_judge.py`:
```python
from pathlib import Path

from prism.agent.judge import run_hard_judge


def test_judge_succeeds_when_command_exits_0(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hi")
    result = run_hard_judge(
        command=["cat", "hello.txt"],
        workspace=tmp_path,
        timeout_sec=10,
    )
    assert result.exit_code == 0
    assert "hi" in result.stdout
    assert result.success is True


def test_judge_fails_on_nonzero_exit(tmp_path: Path):
    result = run_hard_judge(
        command=["bash", "-c", "exit 3"],
        workspace=tmp_path,
        timeout_sec=10,
    )
    assert result.exit_code == 3
    assert result.success is False


def test_judge_timeout(tmp_path: Path):
    result = run_hard_judge(
        command=["sleep", "10"],
        workspace=tmp_path,
        timeout_sec=1,
    )
    assert result.success is False
    assert "timed out" in result.stdout.lower() or result.exit_code != 0


def test_judge_runs_in_workspace(tmp_path: Path):
    """pwd must report the workspace path."""
    result = run_hard_judge(command=["pwd"], workspace=tmp_path, timeout_sec=5)
    assert str(tmp_path) in result.stdout
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/agent/judge.py`:
```python
"""Hard judgment for agent tasks — run a command in the workspace and return pass/fail."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JudgeOutcome:
    exit_code: int
    stdout: str  # combined stdout+stderr
    success: bool  # exit_code == 0


def run_hard_judge(
    *,
    command: list[str],
    workspace: Path,
    timeout_sec: float = 300.0,
) -> JudgeOutcome:
    """Run `command` in `workspace`. Return JudgeOutcome with exit code + combined output."""
    try:
        proc = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        return JudgeOutcome(
            exit_code=124,
            stdout=f"timed out after {timeout_sec}s: {e.cmd}",
            success=False,
        )
    output = (proc.stdout + proc.stderr).strip()
    return JudgeOutcome(
        exit_code=proc.returncode,
        stdout=output,
        success=proc.returncode == 0,
    )
```

- [ ] **Step 4: Pass — 4 tests**

- [ ] **Step 5: Commit**

```bash
git add src/prism/agent/judge.py tests/unit/test_agent_judge.py
git commit -m "feat(agent): run_hard_judge subprocess runner with timeout"
```

---

## Task 6: Agent registry

**Files:**
- Create: `src/prism/agent/registry.py`
- Test: `tests/unit/test_agent_registry.py`

Mirrors `BenchmarkRegistry` but for `AgentBenchmark`. Separate to avoid mixing tracks.

- [ ] **Step 1: Failing test**

Create `tests/unit/test_agent_registry.py`:
```python
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
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/agent/registry.py`:
```python
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
```

Update `src/prism/agent/__init__.py` to export:
```python
from prism.agent.registry import AgentBenchmarkRegistry, agent_registry
from prism.agent.task import AgentBenchmark, AgentResult, AgentTask

__all__ = [
    "AgentBenchmark",
    "AgentBenchmarkRegistry",
    "AgentResult",
    "AgentTask",
    "agent_registry",
]
```

Note: `agent_registry()` imports `ToyAgentBenchmark` which we'll create in Task 7. This is fine — Task 7 tests alone run `agent_registry()` and will fail if ordering is wrong; the test here only directly tests `AgentBenchmarkRegistry`.

- [ ] **Step 4: Pass — 5 tests**

Note: the `agent_registry()` factory won't be callable until Task 7 lands the ToyAgentBenchmark. Only test the `AgentBenchmarkRegistry` class directly at this step.

- [ ] **Step 5: Commit**

```bash
git add src/prism/agent/registry.py src/prism/agent/__init__.py tests/unit/test_agent_registry.py
git commit -m "feat(agent): AgentBenchmarkRegistry + agent_registry() factory"
```

---

## Task 7: ToyAgentBenchmark — proof-of-concept benchmark

**Files:**
- Create: `src/prism/benchmarks/toy_agent/__init__.py`
- Create: `src/prism/benchmarks/toy_agent/benchmark.py`
- Test: `tests/unit/test_toy_agent_benchmark.py`

Two toy tasks, both of shape "implement a small function, then run pytest". Proves the full loop end-to-end.

- [ ] **Step 1: Failing test**

Create `tests/unit/test_toy_agent_benchmark.py`:
```python
from prism.benchmarks.toy_agent.benchmark import ToyAgentBenchmark


def test_load_tasks_produces_two_tasks():
    bm = ToyAgentBenchmark()
    tasks = list(bm.load_tasks(subset="full"))
    assert len(tasks) == 2
    ids = [t.task_id for t in tasks]
    assert "toy-add" in ids
    assert "toy-reverse" in ids


def test_tasks_have_workspace_and_judge_command():
    bm = ToyAgentBenchmark()
    task = next(iter(bm.load_tasks(subset="full")))
    assert task.workspace_files  # non-empty
    assert task.judge_command  # non-empty list
    assert "solution.py" in task.workspace_files


def test_add_task_judge_fails_on_unedited_solution():
    """The seed solution.py has a TODO stub; running pytest should fail before agent edits it.

    This test confirms the seed is broken (so the judge is actually measuring something).
    """
    import subprocess
    import tempfile
    from pathlib import Path

    bm = ToyAgentBenchmark()
    task = next(t for t in bm.load_tasks(subset="full") if t.task_id == "toy-add")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, content in task.workspace_files.items():
            (root / name).write_text(content)
        result = subprocess.run(
            task.judge_command, cwd=str(root),
            capture_output=True, text=True, timeout=30,
        )
    assert result.returncode != 0
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/benchmarks/toy_agent/__init__.py` (empty).

Create `src/prism/benchmarks/toy_agent/benchmark.py`:
```python
"""Toy agent benchmark — two tiny "implement this function" tasks.

Purpose: prove the P3a agent pipeline end-to-end without requiring a real
benchmark dataset. Academic benchmarks land in P3c.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable

from prism.agent.task import AgentBenchmark, AgentTask


_ADD_TASK = AgentTask(
    task_id="toy-add",
    workspace_files={
        "solution.py": "def add(a, b):\n    # TODO: implement\n    pass\n",
        "test_solution.py": (
            "from solution import add\n"
            "\n"
            "def test_add_positive():\n"
            "    assert add(2, 3) == 5\n"
            "\n"
            "def test_add_negative():\n"
            "    assert add(-1, 1) == 0\n"
        ),
    },
    user_instruction=(
        "Implement the `add(a, b)` function in solution.py so that the tests "
        "in test_solution.py all pass. Run the tests with pytest to verify."
    ),
    judge_command=[sys.executable, "-m", "pytest", "-q", "test_solution.py"],
    timeout_seconds=60,
    max_turns=10,
)

_REVERSE_TASK = AgentTask(
    task_id="toy-reverse",
    workspace_files={
        "solution.py": "def reverse(s):\n    # TODO\n    pass\n",
        "test_solution.py": (
            "from solution import reverse\n"
            "\n"
            "def test_reverse_hello():\n"
            "    assert reverse('hello') == 'olleh'\n"
            "\n"
            "def test_reverse_empty():\n"
            "    assert reverse('') == ''\n"
        ),
    },
    user_instruction=(
        "Implement the `reverse(s)` function in solution.py so that "
        "test_solution.py passes under pytest."
    ),
    judge_command=[sys.executable, "-m", "pytest", "-q", "test_solution.py"],
    timeout_seconds=60,
    max_turns=10,
)


class ToyAgentBenchmark(AgentBenchmark):
    name = "toy_agent"
    track = "agent"
    version = "v1"

    def load_tasks(self, *, subset: str | None = None) -> Iterable[AgentTask]:
        yield _ADD_TASK
        yield _REVERSE_TASK
```

- [ ] **Step 4: Pass — 3 tests**

- [ ] **Step 5: Commit**

```bash
git add src/prism/benchmarks/toy_agent tests/unit/test_toy_agent_benchmark.py
git commit -m "feat(benchmarks): ToyAgentBenchmark (2 add/reverse tasks for P3a POC)"
```

---

## Task 8: AgentRunner — orchestrate one benchmark end-to-end

**Files:**
- Create: `src/prism/runners/agent.py`
- Test: `tests/unit/test_agent_runner.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_agent_runner.py`:
```python
import json
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.agent import AgentRunner
from prism.service import RunService


class _SolvingAdapter(Adapter):
    """Scripted adapter that solves both toy tasks by issuing write_file then empty final."""

    def __init__(self, profile) -> None:
        super().__init__(profile)
        self._calls_by_task: dict[str, list[AdapterResponse]] = {}

    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        # Detect which task by looking at the last user message in request.
        last_user = next(
            (m for m in reversed(request.messages) if m.get("role") == "user"),
            {},
        )
        instr = last_user.get("content", "")

        # Detect if we've already sent the solve for this conversation.
        # Simple heuristic: if a prior assistant message has a tool_calls entry, we're done → return final text.
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

    # Both tasks should succeed
    assert summary["task_count"] == 2
    assert summary["success_rate"] == pytest.approx(1.0)
    # Artifacts: each task has a JSON trace
    trace_files = sorted((tmp_path / "a").rglob("agent/*.json"))
    assert len(trace_files) == 2
    t = json.loads(trace_files[0].read_text())
    assert "task_id" in t
    assert "trace" in t
    assert "judge_stdout" in t
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/runners/agent.py`:
```python
"""AgentRunner — orchestrate an AgentBenchmark against one (profile, adapter)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from prism.adapters.base import Adapter
from prism.agent.judge import run_hard_judge
from prism.agent.loop import run_agent_loop
from prism.agent.task import AgentBenchmark, AgentResult, AgentTask
from prism.agent.workspace import workspace_context
from prism.config.model_profile import ModelProfile
from prism.service import RunService
from prism.storage.schema import Prompt, Response, Score, Task


class AgentRunner:
    """Runs an AgentBenchmark against one (profile, adapter).

    For each AgentTask:
      1. Create a tmp workspace with seeded files
      2. Run inline agent loop (multi-turn tool calling)
      3. Run hard judge (e.g. pytest) in the workspace
      4. Persist Response + Score rows and a trace JSON artifact
    """

    def __init__(self, *, service: RunService) -> None:
        self.service = service

    async def run(
        self,
        *,
        benchmark: AgentBenchmark,
        profile: ModelProfile,
        adapter: Adapter,
        subset: str | None = "quick",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if run_id is None:
            run_id = await self.service.create_run(suite=f"{benchmark.name}-agent")

        await self.service.register_model(profile)
        await self.service.register_task(
            task_id=benchmark.name,
            benchmark=benchmark.name,
            track=benchmark.track,
        )

        tasks = list(benchmark.load_tasks(subset=subset))
        results: list[AgentResult] = []

        for task in tasks:
            result = await self._run_one(run_id=run_id, benchmark=benchmark,
                                          task=task, profile=profile, adapter=adapter)
            results.append(result)

        await self._mark_run_done(run_id)
        return self._summarize(results, run_id=run_id)

    async def _run_one(
        self, *, run_id: str, benchmark: AgentBenchmark, task: AgentTask,
        profile: ModelProfile, adapter: Adapter,
    ) -> AgentResult:
        # Register a synthetic "prompt" for this task so the Response FK is satisfied.
        prompt_id = f"{benchmark.name}-{task.task_id}"
        await self.service.register_prompt(
            prompt_id=prompt_id,
            task_id=benchmark.name,
            version=benchmark.version,
            text=task.user_instruction,
        )

        with workspace_context(task.workspace_files) as workspace:
            loop_result = await run_agent_loop(
                adapter=adapter,
                workspace=workspace,
                user_instruction=task.user_instruction,
                max_turns=task.max_turns,
            )
            judge_outcome = run_hard_judge(
                command=task.judge_command,
                workspace=workspace,
                timeout_sec=task.timeout_seconds,
            )

        result = AgentResult(
            task_id=task.task_id,
            model_id=profile.id,
            success=judge_outcome.success,
            turns=loop_result.turns,
            final_text=loop_result.final_text,
            judge_stdout=judge_outcome.stdout,
            judge_exit_code=judge_outcome.exit_code,
            tokens_in=loop_result.tokens_in,
            tokens_out=loop_result.tokens_out,
            latency_ms=loop_result.latency_ms,
            cost_usd=loop_result.cost_usd,
            trace=loop_result.trace,
        )

        await self._persist(run_id=run_id, prompt_id=prompt_id, result=result)
        return result

    async def _persist(
        self, *, run_id: str, prompt_id: str, result: AgentResult
    ) -> None:
        async with self.service.db.session() as s:
            row = Response(
                run_id=run_id,
                model_id=result.model_id,
                prompt_id=prompt_id,
                seed=0,
                text=result.final_text,
                reasoning_text=None,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                finish_reason="done" if result.success else "failed",
            )
            s.add(row)
            await s.flush()
            s.add(Score(
                response_id=row.id,
                judge="agent_hard",
                score=1.0 if result.success else 0.0,
                confidence=1.0,
                reasoning=result.judge_stdout[:500],
            ))
            await s.commit()

        self.service.artifacts.put(
            run_id,
            f"agent/{result.task_id}.json",
            asdict(result),
        )

    async def _mark_run_done(self, run_id: str) -> None:
        from prism.storage.schema import Run
        async with self.service.db.session() as s:
            run = await s.get(Run, run_id)
            if run is not None:
                run.status = "done"
                await s.commit()

    @staticmethod
    def _summarize(results: list[AgentResult], *, run_id: str) -> dict[str, Any]:
        if not results:
            return {"run_id": run_id, "task_count": 0, "success_rate": 0.0, "total_cost_usd": 0.0}
        success = sum(1 for r in results if r.success)
        cost = sum(r.cost_usd for r in results)
        return {
            "run_id": run_id,
            "task_count": len(results),
            "success_rate": success / len(results),
            "total_cost_usd": cost,
        }
```

- [ ] **Step 4: Pass — 1 test + full suite**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/unit/test_agent_runner.py -v
uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add src/prism/runners/agent.py tests/unit/test_agent_runner.py
git commit -m "feat(runners): AgentRunner orchestrating workspace + loop + judge + persist"
```

---

## Task 9: CLI — dispatch on `--track`

**Files:**
- Modify: `src/prism/cli.py`
- Modify: `tests/e2e/test_cli_run.py`

- [ ] **Step 1: Failing test**

Append to `tests/e2e/test_cli_run.py`:
```python
def test_run_agent_track_help():
    """--track agent should not error and should list toy_agent in benchmarks."""
    result = runner.invoke(app, ["run", "--help"])
    # --track flag already exists from P2a; no new flag needed.
    assert result.exit_code == 0


def test_run_agent_track_invokes_agent_runner(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text("id: test\nprovider: openai\nmodel: x\n")

    from unittest.mock import AsyncMock, patch
    fake_result = {"run_id": "run-x", "task_count": 2, "success_rate": 1.0, "total_cost_usd": 0.0}

    with patch("prism.cli.AgentRunner") as MockRunner, \
         patch("prism.cli.LiteLLMAdapter") as _MockAdapter:
        instance = MockRunner.return_value
        instance.run = AsyncMock(return_value=fake_result)

        result = runner.invoke(app, [
            "run", "--track", "agent", "--benchmark", "toy_agent",
            "--model", str(model_cfg),
            "--work-dir", str(tmp_path),
        ])

    assert result.exit_code == 0, result.stdout
    assert "success_rate" in result.stdout
    assert "1.0" in result.stdout


def test_run_agent_unknown_benchmark(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text("id: test\nprovider: openai\nmodel: x\n")
    result = runner.invoke(app, [
        "run", "--track", "agent", "--benchmark", "nonexistent",
        "--model", str(model_cfg),
        "--work-dir", str(tmp_path),
    ])
    assert result.exit_code != 0
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Modify `src/prism/cli.py`**

Add imports alongside existing:
```python
from prism.agent import agent_registry
from prism.runners.agent import AgentRunner
```

In `run_cmd`, locate the existing `if track != "limit":` guard and replace it with track-based dispatch. The clean refactor:

Find this block:
```python
    if track != "limit":
        console.print(f"[red]Only --track limit is implemented in P2a[/red]")
        raise typer.Exit(code=2)

    try:
        bm_cls = default_registry().get_class(benchmark)
    except KeyError:
        ...
```

Replace with:
```python
    if track not in ("limit", "agent"):
        console.print(f"[red]Unknown track: {track!r}. Supported: limit | agent[/red]")
        raise typer.Exit(code=2)

    if track == "agent":
        try:
            bm_cls = agent_registry().get_class(benchmark)
        except KeyError:
            console.print(
                f"[red]Unknown agent benchmark: {benchmark!r}. "
                f"Known: {agent_registry().names()}[/red]"
            )
            raise typer.Exit(code=2) from None
        bm = bm_cls()
        profile = load_model_profile(model)
        adapter = LiteLLMAdapter(profile)

        work_dir.mkdir(parents=True, exist_ok=True)
        svc = RunService(
            db_path=work_dir / "prism.db",
            artifacts_root=work_dir / "artifacts",
            checkpoint_path=work_dir / "checkpoint.db",
        )

        async def _run_agent() -> dict:
            await svc.init()
            agent_runner = AgentRunner(service=svc)
            return await agent_runner.run(
                benchmark=bm,
                profile=profile,
                adapter=adapter,
                subset=subset,
            )

        result = asyncio.run(_run_agent())
        console.print(json.dumps(result, indent=2))
        return

    # Existing limit-track path below, unchanged.
    try:
        bm_cls = default_registry().get_class(benchmark)
    except KeyError:
        ...
```

The existing Limit-track code below runs as-is.

- [ ] **Step 4: Pass — 3 tests + full suite**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/e2e/test_cli_run.py -v
uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add src/prism/cli.py tests/e2e/test_cli_run.py
git commit -m "feat(cli): dispatch `prism run --track agent` to AgentRunner"
```

---

## Task 10: End-to-end integration test (real subprocess)

**Files:**
- Test: `tests/integration/test_agent_end_to_end.py`

This test uses the real `_SolvingAdapter` → real tool execution → real pytest judge. No mocks on tools/judge — just the scripted adapter standing in for a model.

- [ ] **Step 1: Create test**

```python
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
    """Adapter that doesn't edit the workspace should fail hard judgment."""
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
```

- [ ] **Step 2: Run + full suite**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/integration/test_agent_end_to_end.py -v
uv run pytest
```

Expected: 2 new tests pass. The solving adapter should get 2/2 tasks through pytest; the broken adapter should get 0/2.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_agent_end_to_end.py
git commit -m "test(integration): agent end-to-end (solving + broken adapter scenarios)"
```

---

## Task 11: README + spec + final verification + tag

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md` (in-repo)

- [ ] **Step 1: Update README**

Add a new section below the Limit-track examples but before `## Publishing a leaderboard`:

```markdown
## Agent track (P3a walking skeleton)

Prism's second track runs multi-turn tool-calling agents. P3a ships with a toy
benchmark (`toy_agent`) that verifies the pipeline end-to-end without needing
real Claude Code integration:

```bash
uv run prism run --track agent --benchmark toy_agent \
    --model configs/models/gpt-5-high.example.yaml
```

The runner spins up a tempdir per task, lets the model call `read_file` /
`write_file` / `bash` tools, runs the task's `judge_command` (e.g., `pytest`),
and records a full trace in `artifacts/<run_id>/agent/<task_id>.json`.

P3b will replace the inline agent loop with a real Claude Code subprocess;
P3c adds SWE-Bench / Terminal-Bench / Tau-Bench / Aider / CORE-Bench;
P3d adds 30 Prism Real Tasks with LLM-judged code quality.
```

Update the roadmap line to:
```
P3b (Claude Code CLI), P3c (academic benchmarks), P3d (Prism Real Tasks), P4 (Meta-Ability), P5 (Web UI) are planned.
```

- [ ] **Step 2: Update spec status**

In the in-repo spec, change status to:
```
- **状态**：P1 + P2(a-f) + P3a 完成（Limit 赛道完整 + Agent pipeline walking skeleton）；P3b Claude Code 集成待启动
```

- [ ] **Step 3: Final verification**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /Users/pingfan/projects/prism
make all
```

Fix any lint/mypy issues minimally.

- [ ] **Step 4: Smoke tests**

```bash
uv run prism doctor
uv run prism run --track agent --help
```

- [ ] **Step 5: Commit + tag**

```bash
git status --porcelain
# README + spec modified; plan doc untracked; possibly lint-fixed files.

git add README.md docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md
git commit -m "docs: P3a — agent pipeline walking skeleton"

git add -A
git diff --cached --stat
git commit -m "chore(p3a): commit P3a plan doc + lint cleanup" || echo "nothing further"

git tag -a p3a-agent-skeleton -m "P3a: Agent walking skeleton — AgentTask/Tools/Loop/Judge/Runner + toy_agent"
git tag
git log --oneline --decorate -n 15
```

- [ ] **Step 6: Stats**

```bash
echo "=== P3a Stats ==="
uv run pytest --collect-only -q 2>&1 | tail -3
echo "Commits since p2f:"
git rev-list --count p2f-leaderboard..HEAD
echo "--- clean tree check ---"
git status --porcelain
```

## Report (final)

- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- `make all` output (lint fixes applied)
- Test count (~240+ expected)
- Tag `p3a-agent-skeleton` SHA
- Commits since `p2f-leaderboard`
- Confirm clean tree
- Concerns

---

## Self-Review Checklist

- [ ] `AgentTask` is frozen; `AgentResult` is a regular dataclass
- [ ] Three tools all reject paths that escape `workspace`
- [ ] `bash` tool times out cleanly and reports it
- [ ] `workspace_context` creates nested dirs and cleans up
- [ ] `run_agent_loop` respects `max_turns` (no infinite loop)
- [ ] `run_hard_judge` returns `JudgeOutcome` with `.success` correctly computed
- [ ] `AgentRunner` persists: 1 Response row + 1 Score row + 1 artifact JSON per task
- [ ] ToyAgent's seed solution.py fails the judge command (non-trivial tests)
- [ ] CLI `--track agent` dispatches to `AgentRunner`, errors on unknown track / benchmark
- [ ] Integration test with a "broken" adapter yields 0% success (proves judge actually measures)
- [ ] `make all` green; tag `p3a-agent-skeleton` on HEAD

---

## P3a Success Criteria

- `prism run --track agent --benchmark toy_agent --model <yaml>` completes and prints `{"task_count": 2, "success_rate": <0..1>, ...}`
- A real model (given API keys) that can write Python code should score 1.0 on `toy_agent`
- Agent trace artifacts contain every tool call with its args + first 2000 chars of result
- `AgentTask` / `AgentRunner` / `AgentResult` interfaces are stable — P3b can replace `run_agent_loop` with a Claude Code subprocess call without other code changes
- All P1–P2f tests still pass; P3a adds ~30 new tests with no flakes
