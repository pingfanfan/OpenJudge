# Prism P2a — Limit Runner Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 P1 Foundation 堆成一个可运行的 benchmark — 端到端打通 "`prism run --track limit --benchmark mmlu_pro --model X` → 得到分数"，验证 benchmark 插件抽象通用到 MCQ / 数值 / 代码执行三种 judge 类型。

**Architecture:** 引入 `Benchmark` ABC 抽象（每个 benchmark 一个独立模块，实现 `load_prompts()` + `make_judge()`）、`LimitRunner` 把 benchmark → prompts → adapter → judge → score 串起来、`RunService.execute()` 扩展接受 judges 参数并持久化 Score 行。数据源统一走 HuggingFace `datasets` 库（首次运行缓存到 `~/.cache/huggingface/`）。新增 `PytestJudge` 在临时目录用 subprocess 跑模型生成的代码。

**Tech Stack:** 基于 P1（uv, Pydantic v2, SQLAlchemy, asyncio, Typer, pytest）+ 新增 `datasets>=3.0`（HuggingFace）、subprocess 代码执行。

---

## 参考文档

- 设计文档：`docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md`（§4 Limit 赛道、§8 Judge 层、§3.1 关键模块）
- P1 plan：`docs/superpowers/plans/2026-04-20-prism-p1-foundation.md`
- P1 代码：Git tag `p1-foundation`，相关模块 `src/prism/{config,adapters,storage,orchestrator,judges,service,cli}`

---

## 范围边界

**In scope (P2a):**
- Benchmark 插件抽象（`Benchmark` ABC + `PromptSpec` + `BenchmarkRegistry`）
- 3 个初始 benchmark：MMLU-Pro subset / AIME 2024+2025 / HumanEval+ subset
- `PytestJudge`（代码执行评分）
- `LimitRunner`（benchmark → prompts → adapter → judge → score）
- `RunService.execute()` 扩展接受 judges 参数 + Score 行持久化
- `prism run --track limit --benchmark <name> --model <config.yaml>` CLI
- 一个配置 YAML 示例 + 一个 suite YAML 示例
- 端到端集成测试（用 FakeAdapter + 固定题目）

**Out of scope (留给 P2b / P2c):**
- 另外 7 个维度的 benchmark loader
- Context Length Staircase / Reasoning Effort Sweep / Contamination Probe
- Leaderboard HTML 生成
- Multiple-model 并行 + 并发优化
- Self-verification loop（P4）

---

## 文件结构（P2a 完成后新增 / 修改）

```
src/prism/
├── benchmarks/                       # NEW 包
│   ├── __init__.py
│   ├── base.py                       # Benchmark ABC, PromptSpec, BenchmarkRegistry
│   ├── dataset_cache.py              # HF datasets 加载包装 + mock hook
│   ├── mmlu_pro/
│   │   ├── __init__.py
│   │   └── benchmark.py              # MMLUProBenchmark
│   ├── aime/
│   │   ├── __init__.py
│   │   └── benchmark.py              # AIMEBenchmark
│   └── humaneval/
│       ├── __init__.py
│       └── benchmark.py              # HumanEvalBenchmark
├── runners/                          # NEW 包
│   ├── __init__.py
│   └── limit.py                      # LimitRunner
├── judges/
│   └── code_exec.py                  # NEW — PytestJudge
├── service.py                        # MODIFY — execute() 加 judges 参数 + Score 持久化
└── cli.py                            # MODIFY — 加 `run` 命令

configs/                              # NEW
├── models/
│   └── gpt-5-high.example.yaml
└── suites/
    └── limit-quick.example.yaml

tests/
├── unit/
│   ├── test_benchmark_base.py         # NEW
│   ├── test_benchmark_registry.py     # NEW
│   ├── test_dataset_cache.py          # NEW
│   ├── test_mmlu_pro_benchmark.py     # NEW
│   ├── test_aime_benchmark.py         # NEW
│   ├── test_humaneval_benchmark.py    # NEW
│   ├── test_pytest_judge.py           # NEW
│   └── test_limit_runner.py           # NEW
├── integration/
│   └── test_limit_run_end_to_end.py   # NEW
└── fixtures/
    ├── mmlu_pro_sample.jsonl          # NEW — 小样本，供 mock 加载
    ├── aime_sample.jsonl
    └── humaneval_sample.jsonl
```

---

## Task 1：添加 HuggingFace datasets 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1：添加 datasets 到 dependencies**

In `pyproject.toml`, under `[project] dependencies`, append:
```
    "datasets>=3.0",
```

Keep existing deps unchanged; final list should include both `litellm>=1.50.0` and the new `datasets>=3.0`.

- [ ] **Step 2：`uv sync` 确保依赖可解**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /Users/pingfan/projects/prism
uv sync --extra dev
```
Expected: no conflict; `datasets` and its deps (pyarrow, dill, etc) get locked.

- [ ] **Step 3：确认 `make all` 仍然绿**

```bash
make all
```
Expected: 71 passed, ruff + mypy clean.

- [ ] **Step 4：Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add huggingface datasets for benchmark loading"
```

---

## Task 2：Benchmark ABC + PromptSpec 数据类型

**Files:**
- Create: `src/prism/benchmarks/__init__.py`
- Create: `src/prism/benchmarks/base.py`
- Test: `tests/unit/test_benchmark_base.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_benchmark_base.py`:
```python
import pytest

from prism.benchmarks.base import Benchmark, PromptSpec


def test_prompt_spec_minimal():
    ps = PromptSpec(
        prompt_id="mmlu_pro-0001",
        task_id="mmlu_pro",
        version="v1",
        messages=[{"role": "user", "content": "Q: 2+2?"}],
        expected="4",
    )
    assert ps.prompt_id == "mmlu_pro-0001"
    assert ps.messages[0]["role"] == "user"
    assert ps.metadata == {}


def test_prompt_spec_with_metadata():
    ps = PromptSpec(
        prompt_id="x",
        task_id="x",
        version="v1",
        messages=[],
        expected=None,
        metadata={"choices": ["A", "B", "C", "D"]},
    )
    assert ps.metadata["choices"] == ["A", "B", "C", "D"]


def test_benchmark_is_abstract():
    with pytest.raises(TypeError):
        Benchmark()  # type: ignore[abstract]
```

- [ ] **Step 2：运行失败**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/unit/test_benchmark_base.py -v
```
Expected: ImportError.

- [ ] **Step 3：实现**

Create `src/prism/benchmarks/__init__.py` (empty).

Create `src/prism/benchmarks/base.py`:
```python
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from prism.judges.base import Judge


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

    @abstractmethod
    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]: ...

    @abstractmethod
    def make_judge(self, prompt: PromptSpec) -> Judge: ...
```

- [ ] **Step 4：测试通过**

```bash
uv run pytest tests/unit/test_benchmark_base.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/benchmarks tests/unit/test_benchmark_base.py
git commit -m "feat(benchmarks): add Benchmark ABC and PromptSpec"
```

---

## Task 3：Benchmark Registry

**Files:**
- Modify: `src/prism/benchmarks/base.py`
- Test: `tests/unit/test_benchmark_registry.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_benchmark_registry.py`:
```python
import pytest

from prism.benchmarks.base import Benchmark, BenchmarkRegistry, PromptSpec
from prism.judges.base import Judge, JudgeResult
from prism.judges.rules import ExactMatchJudge


class _FakeBenchmark(Benchmark):
    name = "fake"
    version = "v1"

    def load_prompts(self, *, subset=None):
        return iter([])

    def make_judge(self, prompt: PromptSpec) -> Judge:
        return ExactMatchJudge()


def test_register_and_lookup():
    reg = BenchmarkRegistry()
    reg.register(_FakeBenchmark)
    bm = reg.get("fake")
    assert isinstance(bm, _FakeBenchmark)


def test_lookup_missing_raises():
    reg = BenchmarkRegistry()
    with pytest.raises(KeyError, match="unknown"):
        reg.get("unknown")


def test_duplicate_register_raises():
    reg = BenchmarkRegistry()
    reg.register(_FakeBenchmark)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_FakeBenchmark)


def test_list_names():
    reg = BenchmarkRegistry()
    reg.register(_FakeBenchmark)
    assert list(reg.names()) == ["fake"]
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Modify `src/prism/benchmarks/base.py` — add at the end:
```python
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

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())
```

- [ ] **Step 4：Pass** — 4 tests.

- [ ] **Step 5：Commit**

```bash
git add src/prism/benchmarks/base.py tests/unit/test_benchmark_registry.py
git commit -m "feat(benchmarks): add BenchmarkRegistry"
```

---

## Task 4：Dataset cache wrapper（HF datasets 隔离 + mock hook）

**Files:**
- Create: `src/prism/benchmarks/dataset_cache.py`
- Test: `tests/unit/test_dataset_cache.py`
- Create: `tests/fixtures/mmlu_pro_sample.jsonl`（几行示例用于 mock）

- [ ] **Step 1：写失败测试**

Create `tests/fixtures/mmlu_pro_sample.jsonl`:
```json
{"question_id": "q1", "question": "What is 2+2?", "options": ["3", "4", "5", "6"], "answer": "B", "category": "math"}
{"question_id": "q2", "question": "Capital of France?", "options": ["Berlin", "London", "Paris", "Madrid"], "answer": "C", "category": "geography"}
```

Create `tests/unit/test_dataset_cache.py`:
```python
from pathlib import Path
from unittest.mock import patch

import pytest

from prism.benchmarks.dataset_cache import load_dataset_cached


def test_load_dataset_cached_with_local_jsonl_fixture(tmp_path: Path):
    # We pass a local jsonl file directly — the cache wrapper supports file:// URIs
    # or explicit file paths, for tests that don't want to hit HuggingFace.
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    rows = list(load_dataset_cached(source=str(fixture), format="jsonl"))
    assert len(rows) == 2
    assert rows[0]["question"] == "What is 2+2?"
    assert rows[1]["answer"] == "C"


def test_load_dataset_cached_falls_back_to_hf(tmp_path: Path):
    """When format is 'hf', we delegate to the datasets library."""
    with patch("prism.benchmarks.dataset_cache.datasets") as mock_datasets:
        mock_datasets.load_dataset.return_value = [
            {"question_id": "q1"}, {"question_id": "q2"}
        ]
        rows = list(load_dataset_cached(source="TIGER-Lab/MMLU-Pro", format="hf", split="test"))
        assert len(rows) == 2
        mock_datasets.load_dataset.assert_called_once_with("TIGER-Lab/MMLU-Pro", split="test")


def test_load_dataset_cached_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown format"):
        list(load_dataset_cached(source="x", format="xml"))
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Create `src/prism/benchmarks/dataset_cache.py`:
```python
"""Dataset loading wrapper with a mock-friendly indirection over HuggingFace datasets.

Two supported formats:
- `jsonl` — read rows from a local file (or `file://` URI). Used by tests.
- `hf`    — delegate to `datasets.load_dataset`. Used in real runs.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import datasets  # noqa: F401 — imported at module level so tests can patch it


def load_dataset_cached(
    *, source: str, format: str = "hf", split: str = "test", **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Yield rows from a dataset. Thin wrapper to keep HF dependency test-patchable."""
    if format == "jsonl":
        path = Path(source.removeprefix("file://"))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if format == "hf":
        ds = datasets.load_dataset(source, split=split, **kwargs)
        yield from ds
        return

    raise ValueError(f"unknown format: {format!r}")
```

- [ ] **Step 4：Pass** — 3 tests.

- [ ] **Step 5：Commit**

```bash
git add src/prism/benchmarks/dataset_cache.py tests/unit/test_dataset_cache.py tests/fixtures/mmlu_pro_sample.jsonl
git commit -m "feat(benchmarks): add dataset cache wrapper with jsonl + hf sources"
```

---

## Task 5：MMLU-Pro benchmark

**Files:**
- Create: `src/prism/benchmarks/mmlu_pro/__init__.py`
- Create: `src/prism/benchmarks/mmlu_pro/benchmark.py`
- Test: `tests/unit/test_mmlu_pro_benchmark.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_mmlu_pro_benchmark.py`:
```python
from pathlib import Path

from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark
from prism.judges.rules import RegexJudge


def test_load_prompts_from_local_fixture():
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl", subset_size=None)
    prompts = list(bm.load_prompts())
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "mmlu_pro"
    assert first.expected == "B"
    assert "What is 2+2?" in first.messages[0]["content"]
    # Choices included in prompt, lettered A/B/C/D
    assert "A. 3" in first.messages[0]["content"]
    assert "D. 6" in first.messages[0]["content"]
    # prompt_id is stable across loads (derived from question_id)
    assert first.prompt_id == "mmlu_pro-q1"


def test_make_judge_is_regex():
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts()))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)


def test_subset_size_limits_output():
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl", subset_size=1)
    prompts = list(bm.load_prompts())
    assert len(prompts) == 1
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Create `src/prism/benchmarks/mmlu_pro/__init__.py` (empty).

Create `src/prism/benchmarks/mmlu_pro/benchmark.py`:
```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.rules import RegexJudge

_PROMPT_TEMPLATE = """Answer the following multiple-choice question. Respond with ONLY the letter (A/B/C/D) on the last line, prefixed by "Answer:".

Question: {question}

Choices:
{choices_block}

Think step by step, then give your final answer as "Answer: X"."""

_JUDGE_PATTERN = r"Answer:\s*([A-D])\b"


class MMLUProBenchmark(Benchmark):
    name = "mmlu_pro"
    track = "limit"
    version = "v1"

    def __init__(
        self,
        *,
        source: str = "TIGER-Lab/MMLU-Pro",
        source_format: str = "hf",
        split: str = "test",
        subset_size: int | None = 200,
    ) -> None:
        self.source = source
        self.source_format = source_format
        self.split = split
        self.subset_size = subset_size

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        size = self.subset_size if subset != "full" else None
        yielded = 0
        for row in load_dataset_cached(
            source=self.source, format=self.source_format, split=self.split
        ):
            if size is not None and yielded >= size:
                break
            yield self._row_to_prompt(row)
            yielded += 1

    def make_judge(self, prompt: PromptSpec) -> Judge:
        return RegexJudge(pattern=_JUDGE_PATTERN)

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        qid = str(row.get("question_id") or row.get("id") or row["question"][:32])
        options = row["options"]
        choices_block = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))
        content = _PROMPT_TEMPLATE.format(question=row["question"], choices_block=choices_block)
        return PromptSpec(
            prompt_id=f"mmlu_pro-{qid}",
            task_id="mmlu_pro",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=row["answer"],
            metadata={"category": row.get("category", "unknown")},
        )
```

- [ ] **Step 4：Pass** — 3 tests.

- [ ] **Step 5：Commit**

```bash
git add src/prism/benchmarks/mmlu_pro tests/unit/test_mmlu_pro_benchmark.py
git commit -m "feat(benchmarks): add MMLU-Pro benchmark (regex-based MCQ judge)"
```

---

## Task 6：AIME benchmark

**Files:**
- Create: `src/prism/benchmarks/aime/__init__.py`
- Create: `src/prism/benchmarks/aime/benchmark.py`
- Create: `tests/fixtures/aime_sample.jsonl`
- Test: `tests/unit/test_aime_benchmark.py`

- [ ] **Step 1：写 fixture + 失败测试**

Create `tests/fixtures/aime_sample.jsonl`:
```json
{"id": "aime-2024-1", "year": 2024, "problem_number": 1, "problem": "Find the smallest positive integer n such that ...", "answer": "42"}
{"id": "aime-2024-2", "year": 2024, "problem_number": 2, "problem": "Let ABC be a triangle with ...", "answer": "7"}
```

Create `tests/unit/test_aime_benchmark.py`:
```python
from pathlib import Path

import pytest

from prism.benchmarks.aime.benchmark import AIMEBenchmark
from prism.judges.rules import NumericJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "aime_sample.jsonl"
    bm = AIMEBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts())
    assert len(prompts) == 2
    assert prompts[0].task_id == "aime"
    assert prompts[0].expected == "42"
    assert "smallest positive integer" in prompts[0].messages[0]["content"]
    assert prompts[0].prompt_id == "aime-aime-2024-1"


def test_judge_is_numeric():
    fixture = Path(__file__).parent.parent / "fixtures" / "aime_sample.jsonl"
    bm = AIMEBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts()))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, NumericJudge)


@pytest.mark.asyncio
async def test_judge_accepts_trailing_answer():
    fixture = Path(__file__).parent.parent / "fixtures" / "aime_sample.jsonl"
    bm = AIMEBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts()))
    judge = bm.make_judge(prompt)
    # The numeric judge extracts the LAST number in the output, so trailing reasoning works.
    output = "After computation we find n = 41, but actually rechecking gives 42."
    result = await judge.judge(output=output, expected=prompt.expected)
    assert result.score == 1.0
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Create `src/prism/benchmarks/aime/__init__.py` (empty).

Create `src/prism/benchmarks/aime/benchmark.py`:
```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.rules import NumericJudge

_PROMPT_TEMPLATE = """You are solving an AIME (American Invitational Mathematics Examination) problem. The answer is always a non-negative integer between 0 and 999.

Problem:
{problem}

Work through the problem step by step. Give your final answer on the last line as a single integer (no units, no words, just the number)."""


class AIMEBenchmark(Benchmark):
    name = "aime"
    track = "limit"
    version = "v1"

    def __init__(
        self,
        *,
        source: str = "Maxwell-Jia/AIME_2024",
        source_format: str = "hf",
        split: str = "train",
    ) -> None:
        self.source = source
        self.source_format = source_format
        self.split = split

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        for row in load_dataset_cached(
            source=self.source, format=self.source_format, split=self.split
        ):
            yield self._row_to_prompt(row)

    def make_judge(self, prompt: PromptSpec) -> Judge:
        return NumericJudge()

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        rid = str(row.get("id") or f"{row.get('year', 'x')}-{row.get('problem_number', 'x')}")
        content = _PROMPT_TEMPLATE.format(problem=row["problem"])
        return PromptSpec(
            prompt_id=f"aime-{rid}",
            task_id="aime",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=str(row["answer"]),
            metadata={"year": row.get("year"), "problem_number": row.get("problem_number")},
        )
```

- [ ] **Step 4：Pass** — 3 tests.

- [ ] **Step 5：Commit**

```bash
git add src/prism/benchmarks/aime tests/fixtures/aime_sample.jsonl tests/unit/test_aime_benchmark.py
git commit -m "feat(benchmarks): add AIME benchmark (numeric judge)"
```

---

## Task 7：PytestJudge（代码执行 judge）

**Files:**
- Create: `src/prism/judges/code_exec.py`
- Test: `tests/unit/test_pytest_judge.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_pytest_judge.py`:
```python
import pytest

from prism.judges.code_exec import PytestJudge


@pytest.mark.asyncio
async def test_passing_code_scores_1():
    code = """
def add(a, b):
    return a + b
"""
    test_code = """
def test_add():
    from solution import add
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
"""
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code, expected="")
    assert r.score == 1.0
    assert r.confidence == 1.0


@pytest.mark.asyncio
async def test_failing_code_scores_0():
    code = """
def add(a, b):
    return a - b   # bug
"""
    test_code = """
def test_add():
    from solution import add
    assert add(2, 3) == 5
"""
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code, expected="")
    assert r.score == 0.0
    assert r.reasoning is not None
    assert "AssertionError" in r.reasoning or "failed" in r.reasoning.lower()


@pytest.mark.asyncio
async def test_code_with_syntax_error_scores_0():
    code = "def add(a, b:"
    test_code = "def test_x():\n    from solution import add"
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code, expected="")
    assert r.score == 0.0


@pytest.mark.asyncio
async def test_extracts_code_block_if_output_wraps_in_fence():
    code_in_fence = "```python\ndef add(a, b):\n    return a + b\n```"
    test_code = """
def test_add():
    from solution import add
    assert add(1, 2) == 3
"""
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code_in_fence, expected="")
    assert r.score == 1.0
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Create `src/prism/judges/code_exec.py`:
```python
"""PytestJudge — scores a model-generated Python solution by running pytest against a fixed test file in a subprocess.

Behavior:
- Extracts code from ```python ... ``` fence if present (else treats whole output as code).
- Writes code to `solution.py` + the test_code to `test_solution.py` in a tempdir.
- Runs `python -m pytest -q test_solution.py` with timeout.
- Score = 1.0 if exit code 0, else 0.0. Reasoning captures the last ~500 chars of stderr/stdout for debugging.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from prism.judges.base import Judge, JudgeResult

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)


class PytestJudge(Judge):
    name = "pytest"

    def __init__(self, *, test_code: str, timeout_sec: float = 30.0) -> None:
        self.test_code = test_code
        self.timeout_sec = timeout_sec

    async def judge(self, *, output: str, expected: str) -> JudgeResult:
        code = self._extract_code(output)
        return await asyncio.to_thread(self._run_pytest_sync, code)

    @staticmethod
    def _extract_code(output: str) -> str:
        m = _CODE_FENCE_RE.search(output)
        return m.group(1) if m else output

    def _run_pytest_sync(self, code: str) -> JudgeResult:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "solution.py").write_text(code, encoding="utf-8")
            (tmp_path / "test_solution.py").write_text(self.test_code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "--no-header", "test_solution.py"],
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return JudgeResult(
                    score=0.0, confidence=1.0, reasoning=f"pytest timed out after {self.timeout_sec}s"
                )

        output_text = (proc.stdout + "\n" + proc.stderr).strip()[-800:]
        if proc.returncode == 0:
            return JudgeResult(score=1.0, confidence=1.0, reasoning="all tests passed")
        return JudgeResult(score=0.0, confidence=1.0, reasoning=output_text or "tests failed")
```

- [ ] **Step 4：Pass** — 4 tests.

```bash
uv run pytest tests/unit/test_pytest_judge.py -v
```
(This test writes code + runs pytest in a subprocess — first run may take ~1 second.)

- [ ] **Step 5：Commit**

```bash
git add src/prism/judges/code_exec.py tests/unit/test_pytest_judge.py
git commit -m "feat(judges): add PytestJudge for subprocess-based code execution scoring"
```

---

## Task 8：HumanEval+ benchmark

**Files:**
- Create: `src/prism/benchmarks/humaneval/__init__.py`
- Create: `src/prism/benchmarks/humaneval/benchmark.py`
- Create: `tests/fixtures/humaneval_sample.jsonl`
- Test: `tests/unit/test_humaneval_benchmark.py`

- [ ] **Step 1：Fixture + test**

Create `tests/fixtures/humaneval_sample.jsonl`:
```json
{"task_id": "HumanEval/0", "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n", "canonical_solution": "    return a + b\n", "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n\ncheck(add)\n", "entry_point": "add"}
```

Create `tests/unit/test_humaneval_benchmark.py`:
```python
from pathlib import Path

from prism.benchmarks.humaneval.benchmark import HumanEvalBenchmark
from prism.judges.code_exec import PytestJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "humaneval_sample.jsonl"
    bm = HumanEvalBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts())
    assert len(prompts) == 1
    p = prompts[0]
    assert p.task_id == "humaneval"
    assert p.prompt_id == "humaneval-HumanEval/0"
    assert "def add(a, b)" in p.messages[0]["content"]
    assert p.metadata["entry_point"] == "add"
    assert "def check" in p.metadata["test_code"]


def test_judge_is_pytest():
    fixture = Path(__file__).parent.parent / "fixtures" / "humaneval_sample.jsonl"
    bm = HumanEvalBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts()))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, PytestJudge)
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Create `src/prism/benchmarks/humaneval/__init__.py` (empty).

Create `src/prism/benchmarks/humaneval/benchmark.py`:
```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.code_exec import PytestJudge

_PROMPT_TEMPLATE = """Complete the following Python function. Return ONLY the complete function implementation in a ```python code block. Do not include any explanation text.

{prompt}"""

# Wrap HumanEval's check(<entry_point>) trailing call in a pytest test for the PytestJudge harness.
_TEST_WRAPPER_TEMPLATE = """
from solution import {entry_point}

{raw_test}

def test_humaneval():
    # The raw test ends with `check({entry_point})` — it already asserts when imported.
    # We run it again here to trigger pytest discovery.
    check({entry_point})
"""


class HumanEvalBenchmark(Benchmark):
    name = "humaneval"
    track = "limit"
    version = "v1"

    def __init__(
        self,
        *,
        source: str = "evalplus/humanevalplus",
        source_format: str = "hf",
        split: str = "test",
    ) -> None:
        self.source = source
        self.source_format = source_format
        self.split = split

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        for row in load_dataset_cached(
            source=self.source, format=self.source_format, split=self.split
        ):
            yield self._row_to_prompt(row)

    def make_judge(self, prompt: PromptSpec) -> Judge:
        return PytestJudge(
            test_code=prompt.metadata["test_code"],
            timeout_sec=30.0,
        )

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        task_id = row["task_id"]
        entry_point = row["entry_point"]
        raw_test = row["test"]
        test_code = _TEST_WRAPPER_TEMPLATE.format(entry_point=entry_point, raw_test=raw_test)
        content = _PROMPT_TEMPLATE.format(prompt=row["prompt"])
        return PromptSpec(
            prompt_id=f"humaneval-{task_id}",
            task_id="humaneval",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=row.get("canonical_solution"),
            metadata={
                "entry_point": entry_point,
                "test_code": test_code,
                "raw_test": raw_test,
            },
        )
```

- [ ] **Step 4：Pass** — 2 tests.

- [ ] **Step 5：Commit**

```bash
git add src/prism/benchmarks/humaneval tests/fixtures/humaneval_sample.jsonl tests/unit/test_humaneval_benchmark.py
git commit -m "feat(benchmarks): add HumanEval+ benchmark (pytest judge)"
```

---

## Task 9：扩展 Score 表的存储 + RunService.execute 接受 judges 参数

**Files:**
- Modify: `src/prism/service.py`
- Test: `tests/integration/test_limit_run_end_to_end.py` (created in later task; verify stub now)

- [ ] **Step 1：先写扩展后的失败测试（service 层单元测试）**

Create `tests/unit/test_service_judge_wiring.py`:
```python
from pathlib import Path
from typing import Any

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import ModelProfile, RateLimit
from prism.judges.base import Judge, JudgeResult
from prism.judges.rules import ExactMatchJudge
from prism.service import RunService
from prism.storage.schema import Score
from sqlalchemy import select


class EchoAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(
            text="hello",
            reasoning_text=None,
            tokens_in=1,
            tokens_out=1,
            latency_ms=1.0,
            cost_usd=0.0,
            raw={},
        )


@pytest.mark.asyncio
async def test_execute_persists_scores_when_judges_provided(tmp_path: Path):
    profile = ModelProfile(
        id="echo",
        provider="openai",
        model="echo",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    adapter = EchoAdapter(profile)

    svc = RunService(
        db_path=tmp_path / "prism.db",
        artifacts_root=tmp_path / "artifacts",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()

    run_id = await svc.create_run(suite="smoke")
    await svc.register_model(profile)
    await svc.register_task(task_id="t1", benchmark="smoke", track="limit")
    await svc.register_prompt(
        prompt_id="p1", task_id="t1", version="v1", text="say hello"
    )

    await svc.execute(
        run_id=run_id,
        profiles={profile.id: profile},
        adapters={profile.id: adapter},
        prompts={"p1": [{"role": "user", "content": "say hello"}]},
        judges={"p1": ExactMatchJudge()},
        expected={"p1": "hello"},
        seeds=[0],
    )

    async with svc.db.session() as s:
        scores = list((await s.execute(select(Score))).scalars())

    assert len(scores) == 1
    assert scores[0].score == 1.0
    assert scores[0].judge == "exact_match"


@pytest.mark.asyncio
async def test_execute_without_judges_does_not_persist_scores(tmp_path: Path):
    """Back-compat: judges is optional; omitting it keeps old behavior."""
    profile = ModelProfile(
        id="echo", provider="openai", model="echo",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    adapter = EchoAdapter(profile)
    svc = RunService(
        db_path=tmp_path / "prism.db",
        artifacts_root=tmp_path / "artifacts",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    run_id = await svc.create_run(suite="s")
    await svc.register_model(profile)
    await svc.register_task(task_id="t1", benchmark="s", track="limit")
    await svc.register_prompt(prompt_id="p1", task_id="t1", version="v1", text="x")

    await svc.execute(
        run_id=run_id,
        profiles={profile.id: profile},
        adapters={profile.id: adapter},
        prompts={"p1": [{"role": "user", "content": "x"}]},
        seeds=[0],
    )

    async with svc.db.session() as s:
        count = (await s.execute(select(Score))).all()
    assert len(count) == 0
```

- [ ] **Step 2：fail** — execute doesn't accept `judges` yet.

- [ ] **Step 3：Implement — extend `RunService.execute` signature**

Modify `src/prism/service.py`:

First, add the import near the top:
```python
from prism.judges.base import Judge
from prism.storage.schema import Score
```
(These may not exist yet — add them.)

Then, modify `execute` signature and body. Replace the existing `execute` method with:
```python
    async def execute(
        self,
        *,
        run_id: str,
        profiles: dict[str, ModelProfile],
        adapters: dict[str, Adapter],
        prompts: dict[str, list[dict[str, Any]]],
        seeds: list[int],
        judges: dict[str, Judge] | None = None,
        expected: dict[str, str | None] | None = None,
        max_concurrency: int = 8,
    ) -> None:
        runner = OrchestratorRunner(
            adapters=adapters,
            profiles=profiles,
            checkpoint_path=self.checkpoint_path,
        )
        await runner.init()

        cells = list(expand_matrix(
            models=list(profiles.values()),
            prompt_ids=list(prompts.keys()),
            seeds=seeds,
        ))

        async def _persist(cell: Cell, resp: AdapterResponse) -> None:
            async with self.db.session() as s:
                row = Response(
                    run_id=run_id,
                    model_id=cell.model_id,
                    prompt_id=cell.prompt_id,
                    seed=cell.seed,
                    text=resp.text,
                    reasoning_text=resp.reasoning_text,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    latency_ms=resp.latency_ms,
                    cost_usd=resp.cost_usd,
                    finish_reason=resp.finish_reason,
                )
                s.add(row)
                await s.flush()  # obtain row.id
                if judges is not None and cell.prompt_id in judges:
                    judge = judges[cell.prompt_id]
                    exp = (expected or {}).get(cell.prompt_id) or ""
                    try:
                        jr = await judge.judge(output=resp.text, expected=exp)
                    except Exception as e:  # judge failure is recorded, not fatal
                        s.add(Score(
                            response_id=row.id, judge=judge.name,
                            score=0.0, confidence=0.0,
                            reasoning=f"judge raised {type(e).__name__}: {e}",
                        ))
                    else:
                        s.add(Score(
                            response_id=row.id, judge=judge.name,
                            score=jr.score, confidence=jr.confidence,
                            reasoning=jr.reasoning,
                        ))
                await s.commit()
            self.artifacts.put(
                run_id,
                f"responses/{cell.model_id}/{cell.prompt_id}-seed{cell.seed}.json",
                resp.model_dump(),
            )

        await runner.run(
            run_id=run_id, cells=cells, prompts=prompts,
            on_done=_persist, max_concurrency=max_concurrency,
        )

        async with self.db.session() as s:
            run = await s.get(Run, run_id)
            if run is not None:
                run.status = "done"
                await s.commit()
```

Note: the key new behaviors are (1) `flush()` to get `row.id` before constructing `Score`, (2) judge exception protection — a failing judge doesn't crash the whole run; it's recorded with score=0 and confidence=0.

- [ ] **Step 4：Run unit tests + full suite**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/unit/test_service_judge_wiring.py -v
uv run pytest
```
Expected: both new tests pass. Pre-existing `test_full_run_lifecycle` (integration, Task 17) should still pass since `judges` is optional.

- [ ] **Step 5：Commit**

```bash
git add src/prism/service.py tests/unit/test_service_judge_wiring.py
git commit -m "feat(service): execute() accepts judges+expected, persists Score rows"
```

---

## Task 10：LimitRunner

**Files:**
- Create: `src/prism/runners/__init__.py`
- Create: `src/prism/runners/limit.py`
- Test: `tests/unit/test_limit_runner.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_limit_runner.py`:
```python
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.limit import LimitRunner
from prism.service import RunService


class CorrectAdapter(Adapter):
    """Always answers 'Answer: B' — matches MMLU-Pro fixture where q1's expected is B."""
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(
            text="The reasoning leads to B.\n\nAnswer: B",
            reasoning_text=None,
            tokens_in=5, tokens_out=5, latency_ms=1.0, cost_usd=0.0, raw={},
        )


@pytest.mark.asyncio
async def test_limit_runner_executes_benchmark(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl", subset_size=None)

    profile = ModelProfile(
        id="m1", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    svc = RunService(
        db_path=tmp_path / "p.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = LimitRunner(service=svc)
    result = await runner.run(
        benchmark=bm,
        profile=profile,
        adapter=CorrectAdapter(profile),
        seeds=[0],
        subset=None,
    )

    assert result["prompt_count"] == 2
    # q1 expected B (correct), q2 expected C (model always says B → wrong)
    assert result["pass_at_1"] == pytest.approx(0.5)
    assert result["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_limit_runner_no_seeds_defaults_to_single_seed(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl", subset_size=1)
    profile = ModelProfile(
        id="m1", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    svc = RunService(
        db_path=tmp_path / "p.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = LimitRunner(service=svc)
    result = await runner.run(
        benchmark=bm, profile=profile, adapter=CorrectAdapter(profile),
    )
    assert result["prompt_count"] == 1
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Create `src/prism/runners/__init__.py` (empty).

Create `src/prism/runners/limit.py`:
```python
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from prism.adapters.base import Adapter
from prism.benchmarks.base import Benchmark
from prism.config.model_profile import ModelProfile
from prism.service import RunService
from prism.storage.schema import Response, Score


class LimitRunner:
    """Runs a single Benchmark against a single (profile, adapter) pair via RunService.

    Produces an aggregate result summary: prompt_count, pass_at_1, total_cost_usd.
    """

    def __init__(self, *, service: RunService) -> None:
        self.service = service

    async def run(
        self,
        *,
        benchmark: Benchmark,
        profile: ModelProfile,
        adapter: Adapter,
        seeds: list[int] | None = None,
        subset: str | None = "quick",
        run_id: str | None = None,
        max_concurrency: int = 8,
    ) -> dict[str, Any]:
        seeds = seeds if seeds is not None else [0]

        # Ensure a run exists.
        if run_id is None:
            run_id = await self.service.create_run(suite=f"{benchmark.name}-{subset or 'default'}")

        await self.service.register_model(profile)
        await self.service.register_task(
            task_id=benchmark.name, benchmark=benchmark.name, track=benchmark.track,
        )

        prompts_to_send: dict[str, list[dict[str, Any]]] = {}
        expected_map: dict[str, str | None] = {}
        judges: dict[str, Any] = {}

        for spec in benchmark.load_prompts(subset=subset):
            await self.service.register_prompt(
                prompt_id=spec.prompt_id,
                task_id=benchmark.name,
                version=spec.version,
                text=spec.messages[-1]["content"],
                system=None,
            )
            prompts_to_send[spec.prompt_id] = spec.messages
            expected_map[spec.prompt_id] = spec.expected
            judges[spec.prompt_id] = benchmark.make_judge(spec)

        await self.service.execute(
            run_id=run_id,
            profiles={profile.id: profile},
            adapters={profile.id: adapter},
            prompts=prompts_to_send,
            seeds=seeds,
            judges=judges,
            expected=expected_map,
            max_concurrency=max_concurrency,
        )

        return await self._summarize(run_id=run_id)

    async def _summarize(self, *, run_id: str) -> dict[str, Any]:
        async with self.service.db.session() as s:
            # All scores joined with responses (run_id filter on response)
            rows = list((await s.execute(
                select(Score.score, Response.cost_usd, Response.prompt_id)
                .join(Response, Score.response_id == Response.id)
                .where(Response.run_id == run_id)
            )).all())
        if not rows:
            return {"run_id": run_id, "prompt_count": 0, "pass_at_1": 0.0, "total_cost_usd": 0.0}

        # pass_at_1: per unique prompt_id, was the first-seed score == 1.0?
        # Simpler: mean score across all scored responses — correct when seeds=[0].
        scores = [r[0] for r in rows]
        costs = [r[1] for r in rows]
        unique_prompts = {r[2] for r in rows}
        return {
            "run_id": run_id,
            "prompt_count": len(unique_prompts),
            "pass_at_1": sum(scores) / len(scores),
            "total_cost_usd": float(sum(costs)),
        }
```

- [ ] **Step 4：Pass** — 2 tests.

- [ ] **Step 5：Commit**

```bash
git add src/prism/runners tests/unit/test_limit_runner.py
git commit -m "feat(runners): add LimitRunner that wires benchmark→adapter→judge→score"
```

---

## Task 11：注册 3 个 benchmark 到全局 registry

**Files:**
- Modify: `src/prism/benchmarks/__init__.py`
- Test: `tests/unit/test_global_registry.py`

- [ ] **Step 1：写失败测试**

Create `tests/unit/test_global_registry.py`:
```python
from prism.benchmarks import default_registry


def test_default_registry_has_three_benchmarks():
    names = default_registry().names()
    assert set(names) == {"mmlu_pro", "aime", "humaneval"}


def test_default_registry_returns_fresh_instance_each_time():
    r1 = default_registry()
    r2 = default_registry()
    # default_registry() constructs fresh registry each call (no shared mutable state).
    r1_bm = r1.get("mmlu_pro")
    r2_bm = r2.get("mmlu_pro")
    assert r1_bm is not r2_bm
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Modify `src/prism/benchmarks/__init__.py`:
```python
from prism.benchmarks.base import Benchmark, BenchmarkRegistry, PromptSpec


def default_registry() -> BenchmarkRegistry:
    """Build a fresh registry pre-populated with P2a benchmarks."""
    from prism.benchmarks.aime.benchmark import AIMEBenchmark
    from prism.benchmarks.humaneval.benchmark import HumanEvalBenchmark
    from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark

    reg = BenchmarkRegistry()
    reg.register(MMLUProBenchmark)
    reg.register(AIMEBenchmark)
    reg.register(HumanEvalBenchmark)
    return reg


__all__ = ["Benchmark", "BenchmarkRegistry", "PromptSpec", "default_registry"]
```

- [ ] **Step 4：Pass** — 2 tests.

- [ ] **Step 5：Commit**

```bash
git add src/prism/benchmarks/__init__.py tests/unit/test_global_registry.py
git commit -m "feat(benchmarks): default_registry() factory with 3 registered benchmarks"
```

---

## Task 12：CLI `prism run --track limit`

**Files:**
- Modify: `src/prism/cli.py`
- Test: `tests/e2e/test_cli_run.py`

- [ ] **Step 1：写失败测试**

Create `tests/e2e/test_cli_run.py`:
```python
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_run_command_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--benchmark" in result.stdout
    assert "--model" in result.stdout


def test_run_command_rejects_unknown_benchmark(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text(
        "id: test\nprovider: openai\nmodel: gpt-4o\n"
    )
    result = runner.invoke(app, [
        "run", "--track", "limit", "--benchmark", "nonexistent",
        "--model", str(model_cfg),
    ])
    assert result.exit_code != 0
    assert "nonexistent" in (result.stdout + (result.stderr or ""))


def test_run_command_invokes_limit_runner(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text(
        "id: test\nprovider: openai\nmodel: gpt-4o\n"
    )
    fake_result = {"run_id": "run-x", "prompt_count": 2, "pass_at_1": 0.5, "total_cost_usd": 0.0}

    with patch("prism.cli.LimitRunner") as MockRunner:
        instance = MockRunner.return_value
        instance.run = AsyncMock(return_value=fake_result)

        result = runner.invoke(app, [
            "run", "--track", "limit", "--benchmark", "mmlu_pro",
            "--model", str(model_cfg),
            "--work-dir", str(tmp_path),
            "--benchmark-source", str(Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"),
            "--benchmark-format", "jsonl",
        ])

    assert result.exit_code == 0
    assert "prompt_count" in result.stdout
    assert "0.5" in result.stdout  # pass_at_1
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement — extend `src/prism/cli.py`**

Add these imports at the top of `src/prism/cli.py` (below the existing imports):
```python
import asyncio
import json

from prism.adapters.litellm_adapter import LiteLLMAdapter
from prism.benchmarks import default_registry
from prism.config.loader import load_model_profile
from prism.runners.limit import LimitRunner
from prism.service import RunService
```

Then add the new command (append at end of file, before `if __name__ == "__main__"`):
```python
@app.command(name="run")
def run_cmd(
    track: str = typer.Option(..., "--track", help="Track: limit|agent|taste (P2a supports limit)"),
    benchmark: str = typer.Option(..., "--benchmark", help="Benchmark name (see registry)"),
    model: Path = typer.Option(..., "--model", exists=True, help="Path to model profile YAML"),
    work_dir: Path = typer.Option(
        Path.cwd() / ".prism_runs",
        "--work-dir",
        help="Directory for SQLite DB + artifacts + checkpoint",
    ),
    subset: str | None = typer.Option("quick", "--subset", help="Benchmark subset (quick|standard|full)"),
    seeds: str = typer.Option("0", "--seeds", help="Comma-separated integer seeds"),
    max_concurrency: int = typer.Option(8, "--max-concurrency"),
    benchmark_source: str | None = typer.Option(
        None, "--benchmark-source", help="Override benchmark source (e.g., local jsonl path for testing)"
    ),
    benchmark_format: str | None = typer.Option(
        None, "--benchmark-format", help="jsonl|hf"
    ),
) -> None:
    """Run a benchmark against a model, producing scored results."""
    if track != "limit":
        console.print(f"[red]Only --track limit is implemented in P2a[/red]")
        raise typer.Exit(code=2)

    try:
        bm_cls = default_registry()._by_name[benchmark]
    except KeyError:
        console.print(f"[red]Unknown benchmark: {benchmark!r}. Known: {default_registry().names()}[/red]")
        raise typer.Exit(code=2)

    # Instantiate benchmark (allow override for source/format in tests)
    kwargs: dict = {}
    if benchmark_source is not None:
        kwargs["source"] = benchmark_source
    if benchmark_format is not None:
        kwargs["source_format"] = benchmark_format
    bm = bm_cls(**kwargs)

    profile = load_model_profile(model)
    adapter = LiteLLMAdapter(profile)

    work_dir.mkdir(parents=True, exist_ok=True)
    svc = RunService(
        db_path=work_dir / "prism.db",
        artifacts_root=work_dir / "artifacts",
        checkpoint_path=work_dir / "checkpoint.db",
    )

    seeds_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    async def _run() -> dict:
        await svc.init()
        limit = LimitRunner(service=svc)
        return await limit.run(
            benchmark=bm,
            profile=profile,
            adapter=adapter,
            seeds=seeds_list,
            subset=subset,
            max_concurrency=max_concurrency,
        )

    result = asyncio.run(_run())
    console.print(json.dumps(result, indent=2))
```

Note: accessing `default_registry()._by_name[benchmark]` goes through a private attribute — that's OK because the CLI uses the benchmark *class* to forward kwargs. If the reviewer flags this, add a `get_class(name)` method to `BenchmarkRegistry` later. For P2a acceptable.

- [ ] **Step 4：Pass**

```bash
uv run pytest tests/e2e/test_cli_run.py -v
```
Expected: 3 passed.

- [ ] **Step 5：Commit**

```bash
git add src/prism/cli.py tests/e2e/test_cli_run.py
git commit -m "feat(cli): add `prism run --track limit --benchmark X --model Y` command"
```

---

## Task 13：BenchmarkRegistry.get_class 方法 + 重构 CLI 不用私有属性

**Files:**
- Modify: `src/prism/benchmarks/base.py`
- Modify: `src/prism/cli.py`
- Test: `tests/unit/test_benchmark_registry.py` (extend)

- [ ] **Step 1：写失败测试（extend existing file）**

Append to `tests/unit/test_benchmark_registry.py`:
```python
def test_get_class_returns_class_not_instance():
    reg = BenchmarkRegistry()
    reg.register(_FakeBenchmark)
    cls = reg.get_class("fake")
    assert cls is _FakeBenchmark


def test_get_class_missing_raises():
    reg = BenchmarkRegistry()
    with pytest.raises(KeyError):
        reg.get_class("unknown")
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Implement**

Add to `BenchmarkRegistry` in `src/prism/benchmarks/base.py`:
```python
    def get_class(self, name: str) -> type[Benchmark]:
        if name not in self._by_name:
            raise KeyError(f"unknown benchmark: {name!r}")
        return self._by_name[name]
```

Then in `src/prism/cli.py`, change:
```python
    try:
        bm_cls = default_registry()._by_name[benchmark]
    except KeyError:
```
to:
```python
    try:
        bm_cls = default_registry().get_class(benchmark)
    except KeyError:
```

- [ ] **Step 4：Pass** — both new tests + existing CLI test.

- [ ] **Step 5：Commit**

```bash
git add src/prism/benchmarks/base.py src/prism/cli.py tests/unit/test_benchmark_registry.py
git commit -m "refactor(benchmarks): expose get_class() instead of CLI accessing _by_name"
```

---

## Task 14：example YAML 配置文件

**Files:**
- Create: `configs/models/gpt-5-high.example.yaml`
- Create: `configs/models/claude-opus-4-7-max.example.yaml`
- Create: `configs/suites/limit-quick.example.yaml`

- [ ] **Step 1：Create configs**

Create `configs/models/gpt-5-high.example.yaml`:
```yaml
id: gpt-5@high
display_name: "GPT-5 (reasoning_effort=high)"
provider: openai
model: gpt-5
reasoning_effort: high
rate_limit:
  rpm: 100
  tpm: 500000
cost:
  input_per_mtok: 10.0
  output_per_mtok: 40.0
```

Create `configs/models/claude-opus-4-7-max.example.yaml`:
```yaml
id: claude-opus-4-7@max
display_name: "Claude Opus 4.7 (thinking=max)"
provider: anthropic
model: claude-opus-4-7
thinking:
  enabled: true
  effort: max
rate_limit:
  rpm: 50
  tpm: 400000
cost:
  input_per_mtok: 15.0
  output_per_mtok: 75.0
```

Create `configs/suites/limit-quick.example.yaml`:
```yaml
# P2a quick suite (example). The CLI does not yet read suite files directly — this is a
# human-readable record of what a "quick" run looks like. CLI execution still takes --benchmark
# and --model individually. Suite-file driving lands in P2c.
name: limit-quick
track: limit
benchmarks:
  - name: mmlu_pro
    subset: quick
  - name: aime
    subset: full
  - name: humaneval
    subset: quick
sampling:
  n_seeds: 1
  seeds: [0]
```

- [ ] **Step 2：Sanity-check that the example YAML loads (smoke)**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /Users/pingfan/projects/prism
uv run python -c "from prism.config.loader import load_model_profile; print(load_model_profile('configs/models/gpt-5-high.example.yaml').id)"
```
Expected: `gpt-5@high`.

- [ ] **Step 3：Commit**

```bash
git add configs
git commit -m "docs(configs): add example model profiles and quick suite manifest"
```

---

## Task 15：端到端集成测试（使用 FakeAdapter 跑 MMLU-Pro fixture）

**Files:**
- Test: `tests/integration/test_limit_run_end_to_end.py`

- [ ] **Step 1：写端到端测试**

Create `tests/integration/test_limit_run_end_to_end.py`:
```python
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.limit import LimitRunner
from prism.service import RunService


class _CheatingAdapter(Adapter):
    """Always answers the first expected option: 'Answer: B'.

    The MMLU-Pro fixture has:
      q1 expected B  → correct
      q2 expected C  → wrong
    """
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(
            text="Reasoning... Answer: B",
            reasoning_text=None,
            tokens_in=10, tokens_out=3, latency_ms=1.0, cost_usd=0.0, raw={},
        )


@pytest.mark.asyncio
async def test_limit_runner_end_to_end_persists_all_layers(tmp_path: Path):
    """End-to-end: benchmark load → adapter call → judge → Score row → summary."""
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl", subset_size=None)

    profile = ModelProfile(
        id="fake", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    svc = RunService(
        db_path=tmp_path / "prism.db",
        artifacts_root=tmp_path / "artifacts",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()

    limit = LimitRunner(service=svc)
    result = await limit.run(
        benchmark=bm, profile=profile, adapter=_CheatingAdapter(profile),
    )

    assert result["prompt_count"] == 2
    assert result["pass_at_1"] == pytest.approx(0.5)

    # Verify DB has both responses and both scores
    from sqlalchemy import select
    from prism.storage.schema import Response, Score
    async with svc.db.session() as s:
        resps = list((await s.execute(select(Response))).scalars())
        scores = list((await s.execute(select(Score))).scalars())
    assert len(resps) == 2
    assert len(scores) == 2
    assert sum(sc.score for sc in scores) == 1.0  # 1 correct, 1 wrong

    # Verify artifacts were written
    artifact_files = sorted((tmp_path / "artifacts").rglob("*.json"))
    assert len(artifact_files) == 2
    assert all("responses" in str(p) for p in artifact_files)
```

- [ ] **Step 2：Run**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/integration/test_limit_run_end_to_end.py -v
```
Expected: 1 passed.

- [ ] **Step 3：Run full suite**

```bash
uv run pytest
```
Expected: all previously-passing tests + the new ones. Count should be at least 71 + the additions from P2a tasks so far.

- [ ] **Step 4：Commit**

```bash
git add tests/integration/test_limit_run_end_to_end.py
git commit -m "test(integration): end-to-end limit run persists responses + scores + artifacts"
```

---

## Task 16：更新 `prism doctor` 报告 benchmark registry + track 支持

**Files:**
- Modify: `src/prism/cli.py`
- Modify: `tests/e2e/test_cli.py`

- [ ] **Step 1：Extend test**

Modify `tests/e2e/test_cli.py` — update `test_doctor_reports_python`:
```python
def test_doctor_reports_python():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "python" in result.stdout.lower()
    assert "litellm" in result.stdout.lower()
    # P2a additions:
    assert "benchmarks" in result.stdout.lower()
    assert "mmlu_pro" in result.stdout.lower()
```

- [ ] **Step 2：fail**

- [ ] **Step 3：Modify doctor**

In `src/prism/cli.py`, inside the `doctor()` function, **before** `console.print(table)`, add:
```python
    # Benchmarks registered
    from prism.benchmarks import default_registry
    bench_names = default_registry().names()
    table.add_row("benchmarks", ", ".join(bench_names), "OK" if bench_names else "WARN")
```

- [ ] **Step 4：Pass**

```bash
uv run pytest tests/e2e/test_cli.py -v
uv run prism doctor
```
Expected: test passes; doctor's output now includes a `benchmarks` row listing `aime, humaneval, mmlu_pro`.

- [ ] **Step 5：Commit**

```bash
git add src/prism/cli.py tests/e2e/test_cli.py
git commit -m "feat(cli): doctor reports registered benchmarks"
```

---

## Task 17：更新 README + docs 状态

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md`

- [ ] **Step 1：Update README**

Replace the "## Architecture (P1)" section in `README.md` with:

`````markdown
## Architecture

**P1 Foundation (complete, tag `p1-foundation`):**

- `prism.adapters` — LiteLLM-based model adapter with thinking/reasoning_effort translation
- `prism.storage` — SQLite schema + async session + JSON artifact store
- `prism.orchestrator` — execution matrix, rate limiter, checkpoint, async runner
- `prism.judges` — Tier 1 rule judges + Tier 2 LLM judge + pytest judge
- `prism.service` — top-level orchestration service
- `prism.cli` — Typer CLI entry point

**P2a Limit Runner (complete):**

- `prism.benchmarks` — Benchmark ABC + PromptSpec + Registry
- `prism.benchmarks.mmlu_pro` / `aime` / `humaneval` — 3 initial benchmarks
- `prism.runners.limit` — LimitRunner: benchmark → adapter → judge → score

**Example:**

```bash
uv run prism run --track limit \
    --benchmark mmlu_pro \
    --model configs/models/gpt-5-high.example.yaml
```

P2b (more benchmarks), P2c (special views + leaderboard), P3 (Agent Runner), P4 (Meta-Ability), P5 (Web UI) are planned.
`````

- [ ] **Step 2：Update design spec status**

In `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md`, find the status line and update to:
```
- **状态**：P1 Foundation + P2a Limit Runner walking skeleton 完成；P2b benchmark 扩展待启动
```

- [ ] **Step 3：Commit**

```bash
git add README.md docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md
git commit -m "docs: mark P2a Limit Runner walking skeleton complete"
```

---

## Task 18：P2a 完工验证 + git tag

**Files:** 无新增。

- [ ] **Step 1：Run full suite**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /Users/pingfan/projects/prism
make all
```
Expected: all green. Test count increased by approximately 20+ from P1 baseline (71 → ~95).

- [ ] **Step 2：Smoke-test CLI end to end (no real API call)**

```bash
uv run prism doctor
uv run prism --help
uv run prism run --help
```
Expected: all render; `run` subcommand visible.

- [ ] **Step 3：Tag**

```bash
git tag -a p2a-limit-runner -m "P2a Limit Runner walking skeleton: Benchmark ABC, 3 benchmarks, LimitRunner, judge wiring, prism run CLI"
git tag
git log --oneline -n 25
```

- [ ] **Step 4：Final report — print stats**

```bash
echo "=== P2a Stats ==="
echo "Tests:"
uv run pytest --collect-only -q 2>&1 | tail -3
echo "Commits since p1-foundation:"
git rev-list --count p1-foundation..HEAD
echo "Python files added:"
git diff --stat p1-foundation..HEAD -- 'src/prism/*.py' | tail -1
```

## Report (Task 18 final)

- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- `make all` output
- Final test count
- Tag `p2a-limit-runner` on which commit
- Commit count since `p1-foundation`
- Any leftover concerns

---

## Self-Review checklist（completion 时自查）

- [ ] Each benchmark has loader + judge factory + prompt template + fixture
- [ ] All 3 benchmarks registered in `default_registry()`
- [ ] `RunService.execute(judges=..., expected=...)` accepts optional judges param; omitting it preserves P1 behavior
- [ ] `Score` rows persisted to DB; `LimitRunner._summarize` reads them for pass@1
- [ ] CLI `prism run --track limit` is invokable and prints JSON result
- [ ] Example configs loadable via `load_model_profile`
- [ ] End-to-end integration test persists responses + scores + artifacts
- [ ] `prism doctor` lists registered benchmarks
- [ ] `make all` green, tag `p2a-limit-runner` on latest commit
- [ ] P1's technical debts (C1 gather, C3 async judge, datetime.utcnow) are all resolved in P1 patch — P2a inherits a clean base

---

## P2a Success Criteria

- `prism run --track limit --benchmark mmlu_pro --model <yaml>` runs end-to-end, producing a JSON summary with `prompt_count`, `pass_at_1`, `total_cost_usd`
- Same works for `--benchmark aime` (numeric judge) and `--benchmark humaneval` (PytestJudge) 
- Score rows are persisted; a third-party query on SQLite can reconstruct results
- Adding a 4th benchmark requires only: new module under `benchmarks/<name>/`, registration in `default_registry()`, no changes to `LimitRunner` / `RunService` / `cli.py`
- Judge failure does not crash the run (exception-protected in `RunService.execute`)
- All P1 tests still pass; P2a adds ~20+ new tests with no flakes
