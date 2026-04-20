# Prism P2e — Long Context Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 添加 Prism 设计规范最后一个缺失的维度 **a（长上下文）**。落地 **NIAH**（Needle-in-a-Haystack 单针）与 **RULER MK-NIAH**（多针变体）两个 benchmark，支持 Context Length Staircase — 同一 task 在多个上下文长度（1K / 4K / 16K / 64K / 256K / 1M）上各跑一次。

**Architecture:** 每个长上下文 benchmark 的一个 YAML 行产出 N × M 个 PromptSpec — N 条上下文长度，M 条 needle 位置（深度 0%/25%/50%/75%/100%）。haystack 在 runtime 从打包的公共领域语料里组装到目标 token 数。评分用 RegexJudge（NIAH 有明确的 needle 字符串可提取）。

**Tech Stack:** 基于 P1–P2d，无新依赖（token 计数用 `len(text) / 4` 字符级近似；精确 tokenizer 留 P2f）。

---

## 参考文档

- 设计文档：§4.1 维度 a、§4.2 Context Length Staircase
- P2d plan：`docs/superpowers/plans/2026-04-20-prism-p2d-multimodal-expansion.md`

---

## 范围边界

**In scope (P2e):**
- **2 个新 benchmark**：NIAH（单针）、RULER MK-NIAH（多针 4 选 1）
- `src/prism/utils/haystack.py` — token 计数近似 + 语料组装到目标长度 + needle 插入
- `src/prism/utils/corpus/niah_corpus.txt` — 公共领域语料（Project Gutenberg 选段，~100KB）
- Benchmark 产出 N × M 个 PromptSpec（Staircase × 深度），metadata 带 `context_tokens` / `needle_depth`
- 一个集成测试：NIAH 短上下文、固定 needle 位置，fake adapter 返回 needle，验证完整管线
- 默认 registry 扩至 17、doctor 列 17

**Out of scope（后续 plan）:**
- 精确 tokenizer（tiktoken / 各厂商原生 tokenizer）— P2f
- RULER 其它 11 种子任务（multi-value / multi-query / variable tracking / CWE / FWE / QA）— 独立 plan
- Context Staircase 的专项可视化产出（能力-长度曲线）— P2f（leaderboard 一起做）
- 1M context 的实际运行稳定性测试 — 需要真实模型环境，不在 P2e 单元测试范围

---

## 文件结构

```
src/prism/
├── utils/
│   ├── haystack.py                     # NEW — token approx + haystack build
│   └── corpus/
│       ├── __init__.py                 # NEW
│       └── niah_corpus.txt              # NEW — public-domain source text
├── benchmarks/
│   ├── __init__.py                     # MODIFY — register NIAH, RULER
│   ├── niah/                           # NEW
│   │   ├── __init__.py
│   │   └── benchmark.py
│   └── ruler_mk/                       # NEW (RULER multi-key NIAH)
│       ├── __init__.py
│       └── benchmark.py

tests/
├── unit/
│   ├── test_haystack_utils.py          # NEW
│   ├── test_niah_benchmark.py          # NEW
│   ├── test_ruler_mk_benchmark.py      # NEW
│   ├── test_global_registry.py         # MODIFY — 17 benchmarks
│   └── test_cli.py                     # MODIFY — doctor lists 17
└── integration/
    └── test_niah_end_to_end.py         # NEW
```

---

## Task 1: Haystack utilities (token approx + haystack build)

**Files:**
- Create: `src/prism/utils/corpus/__init__.py` (empty)
- Create: `src/prism/utils/corpus/niah_corpus.txt`
- Create: `src/prism/utils/haystack.py`
- Test: `tests/unit/test_haystack_utils.py`

- [ ] **Step 1: Create corpus file**

Create `src/prism/utils/corpus/niah_corpus.txt` with a reasonable public-domain filler. A simple approach: use a short repeatable paragraph about general knowledge (since we'll wrap and repeat it to hit target token counts):

```
The origin of knowledge is a topic that philosophers have debated for centuries. Some argue
that all knowledge is derived from sensory experience, a position known as empiricism. Others
maintain that certain truths can be known through reason alone, without reference to experience,
a position known as rationalism. Immanuel Kant attempted to synthesize these two views in his
Critique of Pure Reason, arguing that while all knowledge begins with experience, not all of it
arises out of experience. He proposed that the mind possesses innate categories through which
sensory data must be structured before it can become intelligible.

The scientific method emerged as a practical refinement of these philosophical inquiries. By
emphasizing hypothesis, experiment, and reproducibility, natural philosophers of the 17th and
18th centuries built a framework for accumulating reliable knowledge about the physical world.
Galileo's astronomical observations, Newton's laws of motion, and Boyle's investigations of
gases were all early examples of this approach in action. Over time, the method was extended
to chemistry, biology, and eventually to the social sciences.

In the 20th century, the philosophy of science became more self-aware. Karl Popper argued that
scientific theories can never be definitively proven, only falsified by observation. Thomas
Kuhn described science as proceeding through paradigm shifts rather than smooth accumulation.
Imre Lakatos proposed research programmes as units of evaluation, combining hard-core theoretical
commitments with a protective belt of auxiliary hypotheses. Each of these thinkers complicated
the picture of how knowledge grows.

More recently, the rise of computational methods has transformed every empirical field. Large
datasets, statistical inference, machine learning, and simulation have become as central to
scientific practice as the laboratory bench once was. This shift has raised fresh questions
about what counts as an explanation, how to interpret predictive models whose internal workings
are opaque, and how to maintain intellectual humility when powerful tools are pointed at
complex systems.
```

This paragraph is ~400 tokens (~1600 chars). Repeating it will provide filler up to any target length.

- [ ] **Step 2: Failing test**

Create `tests/unit/test_haystack_utils.py`:
```python
from prism.utils.haystack import (
    approximate_token_count,
    build_haystack,
    load_corpus,
)


def test_approximate_token_count():
    """1 token ≈ 4 English chars."""
    assert approximate_token_count("") == 0
    assert 90 <= approximate_token_count("a" * 400) <= 110


def test_load_corpus_returns_nonempty_text():
    text = load_corpus()
    assert len(text) > 100  # corpus has some content
    assert "The" in text or "the" in text


def test_build_haystack_hits_target_length_within_tolerance():
    needle = "The special code is HONEYBADGER."
    target_tokens = 500
    hs = build_haystack(target_tokens=target_tokens, needle=needle, needle_depth=0.5)
    actual = approximate_token_count(hs)
    # Within ±10% of target
    assert 0.9 * target_tokens <= actual <= 1.1 * target_tokens
    # Needle is present
    assert needle in hs


def test_build_haystack_needle_at_expected_depth():
    needle = "The magic phrase is PURPLE_ELEPHANT_42."
    # Shorter target so we can precisely place the needle
    hs = build_haystack(target_tokens=400, needle=needle, needle_depth=0.25)
    # Needle index should be near 25% of the haystack
    idx = hs.find(needle)
    assert idx > 0, "needle should be inserted"
    relative_depth = idx / len(hs)
    assert 0.15 <= relative_depth <= 0.35, f"needle at {relative_depth:.2f}, expected ~0.25"


def test_build_haystack_rejects_bad_depth():
    import pytest
    with pytest.raises(ValueError, match="depth"):
        build_haystack(target_tokens=100, needle="x", needle_depth=1.5)
    with pytest.raises(ValueError, match="depth"):
        build_haystack(target_tokens=100, needle="x", needle_depth=-0.1)
```

- [ ] **Step 3: Fail**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/unit/test_haystack_utils.py -v
```

- [ ] **Step 4: Implement**

Create `src/prism/utils/corpus/__init__.py` (empty).

Create `src/prism/utils/haystack.py`:
```python
"""Utilities for Needle-in-a-Haystack style long-context benchmarks.

v0.1 uses a simple character-based approximation for token counting (1 token ≈ 4
English chars). Precise tokenization is deferred to a later plan — the
approximation is adequate for generating targets that are "close enough" to
well-known context sizes for benchmark purposes.
"""
from __future__ import annotations

from pathlib import Path

_CORPUS_FILE = Path(__file__).parent / "corpus" / "niah_corpus.txt"
_CHARS_PER_TOKEN = 4


def approximate_token_count(text: str) -> int:
    """Return approximate token count (chars // 4, English heuristic)."""
    return len(text) // _CHARS_PER_TOKEN


def load_corpus() -> str:
    """Return the bundled filler corpus used to build haystacks."""
    return _CORPUS_FILE.read_text(encoding="utf-8")


def build_haystack(
    *,
    target_tokens: int,
    needle: str,
    needle_depth: float,
) -> str:
    """Generate a haystack of approximately `target_tokens` tokens with `needle`
    inserted at relative position `needle_depth` ∈ [0.0, 1.0].

    depth=0.0 → needle near the start (after the first filler chunk)
    depth=0.5 → needle near the middle
    depth=1.0 → needle near the end
    """
    if not 0.0 <= needle_depth <= 1.0:
        raise ValueError(f"needle_depth must be in [0, 1], got {needle_depth}")

    corpus = load_corpus()
    target_chars = target_tokens * _CHARS_PER_TOKEN

    # Build filler by repeating the corpus until we have enough chars.
    filler_parts = []
    filler_len = 0
    while filler_len < target_chars:
        filler_parts.append(corpus)
        filler_len += len(corpus)
    filler = "".join(filler_parts)[:target_chars]

    # Compute split point for needle insertion.
    needle_pos = int(len(filler) * needle_depth)
    # Snap to nearest whitespace to avoid splitting a word.
    while needle_pos < len(filler) and filler[needle_pos] not in " \n":
        needle_pos += 1

    haystack = filler[:needle_pos] + " " + needle + " " + filler[needle_pos:]
    return haystack
```

- [ ] **Step 5: Pass — 5 tests**

- [ ] **Step 6: Commit**

```bash
cd /Users/pingfan/projects/prism
git add src/prism/utils/haystack.py src/prism/utils/corpus tests/unit/test_haystack_utils.py
git commit -m "feat(utils): haystack builder + token approximator + bundled corpus"
```

---

## Task 2: NIAH benchmark

**Files:**
- Create: `src/prism/benchmarks/niah/__init__.py` (empty)
- Create: `src/prism/benchmarks/niah/benchmark.py`
- Test: `tests/unit/test_niah_benchmark.py`

NIAH produces **no fixture file** — it generates prompts programmatically from constructor args (lengths + depths + a fixed needle template).

- [ ] **Step 1: Test**

Create `tests/unit/test_niah_benchmark.py`:
```python
from prism.benchmarks.niah.benchmark import NIAHBenchmark
from prism.judges.rules import RegexJudge


def test_default_lengths_and_depths_produce_prompt_matrix():
    bm = NIAHBenchmark(
        lengths=[1024, 4096],
        depths=[0.0, 0.5, 1.0],
    )
    prompts = list(bm.load_prompts(subset="full"))
    # 2 lengths × 3 depths = 6 prompts
    assert len(prompts) == 6
    # Each prompt has context_tokens and needle_depth in metadata
    for p in prompts:
        assert "context_tokens" in p.metadata
        assert "needle_depth" in p.metadata
        assert p.metadata["context_tokens"] in (1024, 4096)
        assert p.metadata["needle_depth"] in (0.0, 0.5, 1.0)


def test_prompt_contains_needle():
    bm = NIAHBenchmark(lengths=[512], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    # The haystack (first user message) must contain the needle phrase.
    content = prompt.messages[0]["content"]
    # NIAH needles have the form "The special passcode is <UUID-style>."
    assert "special passcode" in content.lower()
    # The expected answer is the code itself, which should be extractable from content
    # by the regex judge.
    assert prompt.expected is not None


def test_judge_is_regex():
    bm = NIAHBenchmark(lengths=[512], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)


def test_prompt_id_encodes_length_and_depth():
    bm = NIAHBenchmark(lengths=[1024], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    # prompt_id format: niah-len<N>-depth<XX>-<seed>
    assert "niah-" in prompt.prompt_id
    assert "len1024" in prompt.prompt_id
    assert "depth50" in prompt.prompt_id


def test_needle_is_deterministic_per_position():
    """Running load_prompts twice should produce the same needles."""
    bm1 = NIAHBenchmark(lengths=[1024], depths=[0.5], seed=7)
    bm2 = NIAHBenchmark(lengths=[1024], depths=[0.5], seed=7)
    p1 = next(iter(bm1.load_prompts(subset="full")))
    p2 = next(iter(bm2.load_prompts(subset="full")))
    assert p1.expected == p2.expected
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/benchmarks/niah/__init__.py` (empty).

Create `src/prism/benchmarks/niah/benchmark.py`:
```python
"""Needle-in-a-Haystack (single-needle) benchmark.

Each benchmark instance emits N×M prompts where
  N = len(lengths) is the number of context-length points on the staircase, and
  M = len(depths) is the number of needle-depth positions tested.

Score is extracted by regex from the model's response — the needle itself
is a unique ASCII string embedded in the haystack.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.judges.base import Judge
from prism.judges.rules import RegexJudge
from prism.utils.haystack import build_haystack

if TYPE_CHECKING:
    from prism.adapters.base import Adapter


_PROMPT_TEMPLATE = """Below is a long document. Read it carefully, then answer the question at the end.

<document>
{haystack}
</document>

Question: What is the special passcode mentioned in the document?

Respond with just the passcode on the last line, prefixed by "Answer:". For example: "Answer: ABC123"."""

_JUDGE_PATTERN = r"Answer:\s*([A-Z0-9_-]{6,})\b"

# Pool of distinctive codes — seeded RNG picks one per (length, depth) cell.
_NEEDLE_CODES = [
    "HONEYBADGER_9412",
    "PURPLE_ELEPHANT_42",
    "QUANTUM_LIZARD_7731",
    "SILVER_TURTLE_88",
    "NEON_DRAGON_5521",
    "BRONZE_PHOENIX_33",
    "CRYSTAL_WHALE_616",
    "OBSIDIAN_HAWK_909",
    "EMERALD_JAGUAR_1024",
    "CARBON_FALCON_2048",
    "PLATINUM_SHARK_4096",
    "TITANIUM_LYNX_8192",
]


class NIAHBenchmark(Benchmark):
    name = "niah"
    track = "limit"
    version = "v1"
    # Staircase defaults: a small quick tier, full = the classic six rungs.
    subset_caps = {"quick": 6, "standard": 12, "full": None}

    def __init__(
        self,
        *,
        lengths: list[int] | None = None,
        depths: list[float] | None = None,
        seed: int = 42,
    ) -> None:
        self.lengths = lengths if lengths is not None else [1024, 4096, 16384, 65536, 262144, 1048576]
        self.depths = depths if depths is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        self.seed = seed

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        rng = random.Random(self.seed)
        yielded = 0
        for length in self.lengths:
            for depth in self.depths:
                if cap is not None and yielded >= cap:
                    return
                needle_code = _NEEDLE_CODES[rng.randrange(len(_NEEDLE_CODES))]
                yield self._build_prompt(length=length, depth=depth, needle_code=needle_code)
                yielded += 1

    def make_judge(
        self,
        prompt: PromptSpec,
        *,
        llm_judge_adapter: Adapter | None = None,
    ) -> Judge:
        return RegexJudge(pattern=_JUDGE_PATTERN)

    @staticmethod
    def _build_prompt(*, length: int, depth: float, needle_code: str) -> PromptSpec:
        needle_phrase = f"The special passcode is {needle_code}."
        haystack = build_haystack(
            target_tokens=length, needle=needle_phrase, needle_depth=depth
        )
        content = _PROMPT_TEMPLATE.format(haystack=haystack)
        depth_pct = int(depth * 100)
        prompt_id = f"niah-len{length}-depth{depth_pct:02d}"
        return PromptSpec(
            prompt_id=prompt_id,
            task_id="niah",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=needle_code,
            metadata={"context_tokens": length, "needle_depth": depth},
        )
```

- [ ] **Step 4: Pass — 5 tests**

- [ ] **Step 5: Commit**

```bash
git add src/prism/benchmarks/niah tests/unit/test_niah_benchmark.py
git commit -m "feat(benchmarks): add NIAH benchmark (Context Length Staircase)"
```

---

## Task 3: RULER MK-NIAH benchmark

**Files:**
- Create: `src/prism/benchmarks/ruler_mk/__init__.py` (empty)
- Create: `src/prism/benchmarks/ruler_mk/benchmark.py`
- Test: `tests/unit/test_ruler_mk_benchmark.py`

MK-NIAH = Multi-Key NIAH: the haystack contains 3 needles with different keys; the question asks the model to retrieve the value for ONE specific key. Tests associative retrieval, not just existence.

- [ ] **Step 1: Test**

Create `tests/unit/test_ruler_mk_benchmark.py`:
```python
from prism.benchmarks.ruler_mk.benchmark import RulerMKBenchmark
from prism.judges.rules import RegexJudge


def test_emits_staircase_times_depths_prompts():
    bm = RulerMKBenchmark(lengths=[1024, 4096], depths=[0.25, 0.75])
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 4  # 2 × 2
    for p in prompts:
        assert p.task_id == "ruler_mk"
        assert "context_tokens" in p.metadata
        assert "queried_key" in p.metadata


def test_prompt_contains_3_keys_and_asks_for_one():
    bm = RulerMKBenchmark(lengths=[1024], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    content = prompt.messages[0]["content"]
    # The haystack must contain 3 distinct "key: value" pairs.
    # Content includes a Question asking for a specific key.
    assert "Question" in content
    queried = prompt.metadata["queried_key"]
    # The queried key must appear in the question.
    assert queried in content


def test_judge_is_regex():
    bm = RulerMKBenchmark(lengths=[1024], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)


def test_expected_value_matches_queried_key():
    """For each prompt, prompt.expected must be the value associated with the queried key
    (found in the haystack)."""
    bm = RulerMKBenchmark(lengths=[1024], depths=[0.5])
    prompt = next(iter(bm.load_prompts(subset="full")))
    content = prompt.messages[0]["content"]
    # The expected value must appear somewhere in the haystack (as the value for queried_key).
    assert prompt.expected is not None
    assert prompt.expected in content
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

Create `src/prism/benchmarks/ruler_mk/__init__.py` (empty).

Create `src/prism/benchmarks/ruler_mk/benchmark.py`:
```python
"""RULER Multi-Key NIAH benchmark.

The haystack embeds 3 distinct "key: value" records at spread-out positions,
and the question asks for the value of one specific key. This tests
associative retrieval across long contexts, not just needle existence.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.judges.base import Judge
from prism.judges.rules import RegexJudge
from prism.utils.haystack import build_haystack

if TYPE_CHECKING:
    from prism.adapters.base import Adapter


_PROMPT_TEMPLATE = """Below is a long document containing multiple key-value records. Read it, then answer the question.

<document>
{haystack}
</document>

Question: What is the value associated with the key "{queried_key}"?

Respond with just the value on the last line, prefixed by "Answer:". For example: "Answer: XYZ123"."""

_JUDGE_PATTERN = r"Answer:\s*([A-Z0-9_-]{4,})\b"

# Small pool of key/value strings (unique codes avoid regex false matches).
_KEYS = [
    "alpha_key", "beta_key", "gamma_key", "delta_key", "epsilon_key",
    "zeta_key", "eta_key", "theta_key",
]
_VALUES = [
    "REDFOX_01", "BLUEJAY_02", "GREENOWL_03", "YELLOWBAT_04",
    "BLACKWOLF_05", "WHITECROW_06", "ORANGEMOTH_07", "SILVERCAT_08",
    "GOLDBEAR_09", "COPPERDEER_10", "BRONZEEAGLE_11", "CARBONMOOSE_12",
]


class RulerMKBenchmark(Benchmark):
    name = "ruler_mk"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 6, "standard": 12, "full": None}

    def __init__(
        self,
        *,
        lengths: list[int] | None = None,
        depths: list[float] | None = None,
        seed: int = 42,
    ) -> None:
        self.lengths = lengths if lengths is not None else [1024, 4096, 16384, 65536, 262144, 1048576]
        self.depths = depths if depths is not None else [0.25, 0.5, 0.75]
        self.seed = seed

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        rng = random.Random(self.seed)
        yielded = 0
        for length in self.lengths:
            for depth in self.depths:
                if cap is not None and yielded >= cap:
                    return
                yield self._build_prompt(length=length, depth=depth, rng=rng)
                yielded += 1

    def make_judge(
        self,
        prompt: PromptSpec,
        *,
        llm_judge_adapter: Adapter | None = None,
    ) -> Judge:
        return RegexJudge(pattern=_JUDGE_PATTERN)

    def _build_prompt(
        self, *, length: int, depth: float, rng: random.Random
    ) -> PromptSpec:
        # Pick 3 distinct keys and values.
        keys = rng.sample(_KEYS, 3)
        values = rng.sample(_VALUES, 3)
        pairs = list(zip(keys, values))
        # The "main" needle is the queried key; the other 2 act as distractors
        # but they're embedded inside the filler by build_haystack via a compound needle.
        queried_key, queried_value = pairs[0]

        # Build a multi-record needle string; build_haystack inserts it as one block
        # at `depth`. Distractors live adjacent to the queried pair to test
        # associative retrieval, not just positional.
        needle_block = "\n".join(f"The {k}'s value is {v}." for k, v in pairs)
        haystack = build_haystack(
            target_tokens=length, needle=needle_block, needle_depth=depth
        )
        content = _PROMPT_TEMPLATE.format(haystack=haystack, queried_key=queried_key)
        depth_pct = int(depth * 100)
        prompt_id = f"ruler_mk-len{length}-depth{depth_pct:02d}-{queried_key}"
        return PromptSpec(
            prompt_id=prompt_id,
            task_id="ruler_mk",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=queried_value,
            metadata={
                "context_tokens": length,
                "needle_depth": depth,
                "queried_key": queried_key,
            },
        )
```

- [ ] **Step 4: Pass — 4 tests**

- [ ] **Step 5: Commit**

```bash
git add src/prism/benchmarks/ruler_mk tests/unit/test_ruler_mk_benchmark.py
git commit -m "feat(benchmarks): add RULER MK-NIAH benchmark (multi-key retrieval)"
```

---

## Task 4: Update default_registry + doctor + spec

**Files:**
- Modify: `src/prism/benchmarks/__init__.py`
- Modify: `tests/unit/test_global_registry.py`
- Modify: `tests/e2e/test_cli.py`
- Modify: `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md`

- [ ] **Step 1: Update registry test**

In `tests/unit/test_global_registry.py`, rename `test_default_registry_has_fifteen_benchmarks` to `test_default_registry_has_seventeen_benchmarks` and update:
```python
def test_default_registry_has_seventeen_benchmarks():
    names = default_registry().names()
    assert set(names) == {
        "mmlu_pro", "aime", "humaneval",
        "gpqa", "math500", "livecodebench",
        "ifeval", "ceval", "simpleqa", "truthfulqa",
        "harmbench", "xstest", "superclue",
        "mmmu", "mathvista",
        "niah", "ruler_mk",
    }
```

- [ ] **Step 2: Update doctor test**

In `tests/e2e/test_cli.py`, extend the doctor benchmark-name loop to 17:
```python
    for name in ("mmlu_pro", "aime", "humaneval", "gpqa", "math500",
                 "livecodebench", "ifeval", "ceval", "simpleqa", "truthfulqa",
                 "harmbench", "xstest", "superclue", "mmmu", "mathvista",
                 "niah", "ruler_mk"):
        assert name in result.stdout.lower(), f"missing benchmark {name} in doctor output"
```

- [ ] **Step 3: Fail**

- [ ] **Step 4: Update `src/prism/benchmarks/__init__.py`**

Add 2 imports alphabetically:
```python
    from prism.benchmarks.niah.benchmark import NIAHBenchmark
    from prism.benchmarks.ruler_mk.benchmark import RulerMKBenchmark
```

And 2 `reg.register(...)` calls after the existing 15:
```python
    reg.register(NIAHBenchmark)
    reg.register(RulerMKBenchmark)
```

- [ ] **Step 5: Update spec status**

In `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md` (in-repo path), change the status line to:
```
- **状态**：P1 + P2a + P2b + P2c + P2d + P2e 完成（17 benchmark 跨 10 维度；长上下文 Staircase 就绪）；P2f leaderboard + 专项视图待启动
```

- [ ] **Step 6: Pass + commit**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/unit/test_global_registry.py tests/e2e/test_cli.py -v
uv run pytest
git add src/prism/benchmarks/__init__.py tests/unit/test_global_registry.py tests/e2e/test_cli.py docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md
git commit -m "feat(benchmarks): register NIAH, RULER MK-NIAH (17 total); update doctor + spec"
```

---

## Task 5: Integration test — NIAH end-to-end

**Files:**
- Test: `tests/integration/test_niah_end_to_end.py`

- [ ] **Step 1: Test**

```python
"""Integration test: NIAH with fake adapter that finds and returns the needle."""
import re
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.niah.benchmark import NIAHBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.limit import LimitRunner
from prism.service import RunService


class _NeedleFindingAdapter(Adapter):
    """Extracts the needle phrase from the haystack and returns it as "Answer: <code>"."""
    _NEEDLE_RE = re.compile(r"The special passcode is ([A-Z0-9_-]+)\.")

    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        content = request.messages[-1]["content"]
        if not isinstance(content, str):
            content = ""
        m = self._NEEDLE_RE.search(content)
        text = f"Answer: {m.group(1)}" if m else "Answer: UNKNOWN"
        return AdapterResponse(
            text=text,
            reasoning_text=None,
            tokens_in=1000, tokens_out=5, latency_ms=1.0, cost_usd=0.0, raw={},
        )


@pytest.mark.asyncio
async def test_niah_pipeline_pass_at_1_is_1_when_adapter_finds_needle(tmp_path: Path):
    # Small lengths so the test is fast.
    bm = NIAHBenchmark(lengths=[512, 1024], depths=[0.0, 0.5, 1.0])

    profile = ModelProfile(
        id="needle-find", provider="openai", model="x",
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
        benchmark=bm, profile=profile, adapter=_NeedleFindingAdapter(profile),
        subset="full",
    )

    # 2 lengths × 3 depths = 6 prompts; all should pass since adapter always finds the needle.
    assert result["prompt_count"] == 6
    assert result["pass_at_1"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run + full suite**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/integration/test_niah_end_to_end.py -v
uv run pytest
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_niah_end_to_end.py
git commit -m "test(integration): NIAH end-to-end with needle-finding fake adapter"
```

---

## Task 6: Update README + final verification + tag

**Files:** Modify `README.md`; final `make all` + tag.

- [ ] **Step 1: Update README**

Update the benchmark list line in `## Architecture`:
```
- 17 benchmarks across 10 dimensions: `mmlu_pro`, `gpqa` (knowledge); `aime`, `math500` (math); `humaneval`, `livecodebench` (code); `ifeval` (instruction following); `ceval`, `superclue` (Chinese); `simpleqa`, `truthfulqa` (hallucination); `harmbench`, `xstest` (safety); `mmmu`, `mathvista` (multimodal); `niah`, `ruler_mk` (long context)
```

And update the roadmap:
```
P2f (leaderboard + special views), P3 (Agent Runner), P4 (Meta-Ability), P5 (Web UI) are planned.
```

- [ ] **Step 2: Final verification**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /Users/pingfan/projects/prism
make all
```
Expected: all green. Test count ~195+.

- [ ] **Step 3: Smoke test**

```bash
uv run prism doctor
```
Expected: 17 benchmarks listed.

- [ ] **Step 4: Tag**

```bash
git add README.md
git commit -m "docs: P2e — 17 benchmarks across 10 dimensions (add long context)"
git tag -a p2e-longcontext -m "P2e: NIAH + RULER MK-NIAH + haystack utilities + Context Length Staircase"
git log --oneline --decorate -n 15
```

- [ ] **Step 5: Stats**

```bash
echo "=== P2e Stats ==="
uv run pytest --collect-only -q 2>&1 | tail -3
echo "Commits since p2d:"
git rev-list --count p2d-multimodal..HEAD
echo "Benchmark count:"
uv run python -c "from prism.benchmarks import default_registry; print(len(default_registry().names()))"
```

## Report (final)

- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- `make all` output
- Test count (~195+ expected)
- Benchmark count (17 expected)
- Tag `p2e-longcontext` SHA
- Commits since `p2d-multimodal`
- Concerns

---

## Self-Review Checklist

- [ ] `approximate_token_count` gives 1 token ≈ 4 chars
- [ ] `build_haystack(target=N)` lands within ±10% of N tokens
- [ ] NIAH emits N × M prompts with `context_tokens` / `needle_depth` metadata
- [ ] RULER MK emits prompts whose question references one of 3 keys in the haystack
- [ ] Both benchmarks use `RegexJudge` to extract `Answer: <code>`
- [ ] Default registry lists 17 benchmarks; doctor test covers all 17
- [ ] Integration test passes for a fake needle-finding adapter (pass@1 == 1.0)
- [ ] Tag `p2e-longcontext` on HEAD
- [ ] All P1–P2d tests still pass

---

## P2e Success Criteria

- `prism run --track limit --benchmark niah --model <yaml>` runs end-to-end with Staircase prompts
- `prism run --track limit --benchmark ruler_mk --model <yaml>` runs end-to-end
- Metadata (`context_tokens`, `needle_depth`) is persisted in `PromptSpec.metadata` and should be readable from artifacts for later Staircase plotting (in P2f)
- 17 benchmarks listed in `prism doctor`
- Token approximation is documented as "char-based heuristic, precise tokenizer in P2f" — not hidden
- All P1–P2d tests still pass; P2e adds ~15 new tests with no flakes
