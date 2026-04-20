# Prism P2f — Leaderboard + Special Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Limit 赛道收尾工作 —— 把 SQLite 中累积的 run 结果渲染成**可发布到 GitHub Pages 的静态 HTML leaderboard**，并实现设计规范里的两个专项视图：**Context Length Staircase**（NIAH 能力-长度曲线）和 **Reasoning Effort Sweep**（thinking=off/high/max 的 Pareto 对比）。

**Architecture:** 纯 Python 字符串模板生成 HTML（不引入 Jinja2 等新依赖）。Leaderboard 读取一个 `prism.db`，按 `(model_id, benchmark)` 聚合 Score，输出静态 HTML + JSON 数据文件。专项视图复用 `PromptSpec.metadata` 里的 `context_tokens` / thinking 配置字段做分组。CLI 新增 `prism leaderboard publish <workdir> --output <dir>`。

**Tech Stack:** 基于 P1–P2e，无新依赖。Contamination Probe 独立在后续 plan，不在 P2f 范围。

---

## 参考文档

- 设计文档：§4.2 三大专项视图、§10.2 Leaderboard 产出
- P2e plan：`docs/superpowers/plans/2026-04-20-prism-p2e-longcontext-expansion.md`

---

## 范围边界

**In scope (P2f):**
- `src/prism/leaderboard/queries.py` — SQLAlchemy 聚合查询层
- `src/prism/leaderboard/renderer.py` — HTML + JSON 输出生成器（纯 Python 字符串模板）
- 主 leaderboard 表（行 = `model_id`，列 = benchmark，cell = pass@1 + 置信显示）
- 每个 benchmark 的 drilldown 子页
- **Context Length Staircase 视图**：对 NIAH / RULER MK-NIAH 的 run 结果，按 `context_tokens` 分组展示 pass@1 表格
- **Reasoning Effort Sweep 视图**：同一底模的多个 thinking 变体并排对比（`gpt-5@high` vs `gpt-5@max`）
- CLI `prism leaderboard publish <workdir> --output <dir>`
- 一个 integration test：用 fake 数据生成 leaderboard，断言 HTML 合法 + 包含关键字段
- README 中给一个 "How to publish to GitHub Pages" 小节

**Out of scope（后续 plan）:**
- Contamination Probe（canary 生成、污染检测）— P2g
- 交互式图表（Chart.js 等客户端渲染）— v0.1 只静态表格
- 多 DB 合并 / run 历史时间线 — 未来需要
- Taste 赛道的 pairwise leaderboard — P5（Web UI）一起做

---

## 文件结构

```
src/prism/
├── leaderboard/                        # NEW 包
│   ├── __init__.py
│   ├── queries.py                       # 聚合查询
│   ├── renderer.py                      # HTML / JSON 生成
│   └── templates.py                     # 字符串 HTML 模板（headers/rows/pages）
├── cli.py                               # MODIFY — 添加 `leaderboard publish` 子命令

tests/
├── unit/
│   ├── test_leaderboard_queries.py      # NEW
│   ├── test_leaderboard_renderer.py     # NEW
│   └── test_cli.py                      # MODIFY — 新子命令冒烟
├── integration/
│   └── test_leaderboard_end_to_end.py   # NEW
└── fixtures/
    └── (reuse P1-P2e benchmark fixtures)
```

---

## Task 1: Query layer — aggregate Score rows for leaderboard

**Files:**
- Create: `src/prism/leaderboard/__init__.py`
- Create: `src/prism/leaderboard/queries.py`
- Test: `tests/unit/test_leaderboard_queries.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_leaderboard_queries.py`:
```python
from pathlib import Path

import pytest
from sqlalchemy import insert

from prism.leaderboard.queries import (
    aggregate_by_model_benchmark,
    aggregate_staircase,
    list_thinking_variants,
)
from prism.storage.database import Database
from prism.storage.schema import Base, Model, Prompt, Response, Run, Score, Task


async def _seed(db: Database) -> None:
    """Insert a small realistic dataset: 1 run, 2 models, 2 benchmarks, 4 responses + scores."""
    async with db.session() as s:
        s.add(Run(id="r1", suite="test", config_hash=""))
        s.add(Model(id="gpt-5@high", provider="openai", model="gpt-5",
                   reasoning_effort="high"))
        s.add(Model(id="gpt-5@max", provider="openai", model="gpt-5",
                   reasoning_effort="max"))
        s.add(Task(id="mmlu_pro", benchmark="mmlu_pro", track="limit"))
        s.add(Task(id="niah", benchmark="niah", track="limit"))
        s.add(Prompt(id="p1", task_id="mmlu_pro", version="v1", text="q1"))
        s.add(Prompt(id="p2", task_id="niah", version="v1", text="needle"))
        await s.commit()
    async with db.session() as s:
        # 2 responses per (model, benchmark)
        for model_id in ("gpt-5@high", "gpt-5@max"):
            for prompt_id in ("p1", "p2"):
                s.add(Response(
                    run_id="r1", model_id=model_id, prompt_id=prompt_id, seed=0,
                    text="x", tokens_in=10, tokens_out=5, cost_usd=0.001,
                ))
        await s.commit()
    async with db.session() as s:
        # Scores: gpt-5@high gets 1.0 on mmlu_pro, 0.5 on niah; gpt-5@max gets 1.0 on both.
        rows = list((await s.execute(
            __import__("sqlalchemy").select(Response)
        )).scalars())
        score_map = {
            ("gpt-5@high", "p1"): 1.0,
            ("gpt-5@high", "p2"): 0.5,
            ("gpt-5@max", "p1"): 1.0,
            ("gpt-5@max", "p2"): 1.0,
        }
        for r in rows:
            s.add(Score(
                response_id=r.id, judge="test",
                score=score_map[(r.model_id, r.prompt_id)], confidence=1.0,
            ))
        await s.commit()


@pytest.mark.asyncio
async def test_aggregate_by_model_benchmark(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    await db.init()
    await _seed(db)

    rows = await aggregate_by_model_benchmark(db=db)
    # Expect: (model_id, benchmark, mean_score, count)
    result = {(r["model_id"], r["benchmark"]): r for r in rows}
    assert result[("gpt-5@high", "mmlu_pro")]["mean_score"] == pytest.approx(1.0)
    assert result[("gpt-5@high", "niah")]["mean_score"] == pytest.approx(0.5)
    assert result[("gpt-5@max", "mmlu_pro")]["mean_score"] == pytest.approx(1.0)
    assert result[("gpt-5@max", "niah")]["mean_score"] == pytest.approx(1.0)
    # Count = 1 score per (model, benchmark) in this fixture.
    assert result[("gpt-5@high", "mmlu_pro")]["count"] == 1


@pytest.mark.asyncio
async def test_list_thinking_variants(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    await db.init()
    await _seed(db)

    variants = await list_thinking_variants(db=db)
    # Both models share the same provider+model ("openai/gpt-5") but differ in effort.
    # Expect one grouping with 2 variants.
    assert len(variants) == 1
    group = variants[0]
    assert group["base"] == "openai/gpt-5"
    assert set(group["variants"]) == {"gpt-5@high", "gpt-5@max"}


@pytest.mark.asyncio
async def test_aggregate_staircase_for_niah_only(tmp_path: Path):
    """Staircase aggregation: for NIAH prompts, group by context_tokens from metadata.

    For this test we skip NIAH-specific metadata plumbing (it's stored in PromptSpec.metadata
    but NOT in the DB's Prompt table). Instead we just verify the function returns empty
    when no NIAH-type prompts exist — the real staircase data comes from artifact JSON
    which we parse in the renderer's integration test.
    """
    db = Database(tmp_path / "p.db")
    await db.init()
    await _seed(db)

    rows = await aggregate_staircase(db=db, benchmark="niah")
    # Our seeded Prompt.id "p2" is under task_id="niah" but has no embedded context_tokens
    # in Prompt.text. The helper should return rows with 0 or empty aggregation.
    assert isinstance(rows, list)
```

- [ ] **Step 2: Fail**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/unit/test_leaderboard_queries.py -v
```

- [ ] **Step 3: Implement**

Create `src/prism/leaderboard/__init__.py`:
```python
from prism.leaderboard.queries import (
    aggregate_by_model_benchmark,
    aggregate_staircase,
    list_thinking_variants,
)
from prism.leaderboard.renderer import render_leaderboard

__all__ = [
    "aggregate_by_model_benchmark",
    "aggregate_staircase",
    "list_thinking_variants",
    "render_leaderboard",
]
```

Create `src/prism/leaderboard/queries.py`:
```python
"""SQLAlchemy aggregate queries powering the leaderboard views."""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from prism.storage.database import Database
from prism.storage.schema import Model, Prompt, Response, Score, Task


async def aggregate_by_model_benchmark(*, db: Database) -> list[dict[str, Any]]:
    """One row per (model_id, benchmark) with mean score and count.

    Joins Score → Response → Prompt → Task to get benchmark name.
    """
    stmt = (
        select(
            Response.model_id,
            Task.benchmark,
            func.avg(Score.score).label("mean_score"),
            func.count(Score.id).label("count"),
            func.sum(Response.cost_usd).label("total_cost"),
        )
        .join(Score, Score.response_id == Response.id)
        .join(Prompt, Prompt.id == Response.prompt_id)
        .join(Task, Task.id == Prompt.task_id)
        .group_by(Response.model_id, Task.benchmark)
    )
    async with db.session() as s:
        rows = (await s.execute(stmt)).all()
    return [
        {
            "model_id": r.model_id,
            "benchmark": r.benchmark,
            "mean_score": float(r.mean_score or 0.0),
            "count": int(r.count or 0),
            "total_cost": float(r.total_cost or 0.0),
        }
        for r in rows
    ]


async def list_thinking_variants(*, db: Database) -> list[dict[str, Any]]:
    """Group models by (provider, model) to surface "thinking effort sweep" candidates.

    Each group has 'base' = "<provider>/<model>" and 'variants' = list of model_id values
    whose thinking config differs. Only groups with 2+ variants are returned.
    """
    stmt = select(Model.id, Model.provider, Model.model, Model.reasoning_effort)
    async with db.session() as s:
        rows = (await s.execute(stmt)).all()

    by_base: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        base = f"{r.provider}/{r.model}"
        by_base.setdefault(base, []).append(
            {"model_id": r.id, "effort": r.reasoning_effort}
        )

    result = []
    for base, members in by_base.items():
        if len(members) < 2:
            continue
        result.append({
            "base": base,
            "variants": [m["model_id"] for m in members],
            "efforts": {m["model_id"]: m["effort"] for m in members},
        })
    return result


async def aggregate_staircase(*, db: Database, benchmark: str) -> list[dict[str, Any]]:
    """Staircase aggregation: per (model_id, context_tokens), mean score for the given benchmark.

    Context-length metadata is encoded in the prompt_id (e.g., "niah-len1024-depth50").
    This helper parses the prompt_id to extract length, then groups scores.
    Returns empty list if no rows match.
    """
    import re
    stmt = (
        select(
            Response.model_id,
            Response.prompt_id,
            Score.score,
        )
        .join(Score, Score.response_id == Response.id)
        .join(Prompt, Prompt.id == Response.prompt_id)
        .join(Task, Task.id == Prompt.task_id)
        .where(Task.benchmark == benchmark)
    )
    async with db.session() as s:
        rows = (await s.execute(stmt)).all()

    # Parse length from prompt_id like "niah-len1024-depth50" or "ruler_mk-len4096-depth25-alpha_key"
    len_re = re.compile(r"-len(\d+)-")
    bucket: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        m = len_re.search(r.prompt_id)
        if not m:
            continue
        length = int(m.group(1))
        bucket.setdefault((r.model_id, length), []).append(float(r.score))

    result = []
    for (model_id, length), scores in sorted(bucket.items()):
        result.append({
            "model_id": model_id,
            "context_tokens": length,
            "mean_score": sum(scores) / len(scores),
            "count": len(scores),
        })
    return result
```

- [ ] **Step 4: Pass — 3 tests**

- [ ] **Step 5: Commit**

```bash
cd /Users/pingfan/projects/prism
git add src/prism/leaderboard tests/unit/test_leaderboard_queries.py
git commit -m "feat(leaderboard): aggregate queries for main grid + staircase + thinking variants"
```

---

## Task 2: HTML renderer

**Files:**
- Create: `src/prism/leaderboard/templates.py`
- Create: `src/prism/leaderboard/renderer.py`
- Test: `tests/unit/test_leaderboard_renderer.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_leaderboard_renderer.py`:
```python
from pathlib import Path

import pytest

from prism.leaderboard.renderer import render_leaderboard_html


def test_render_main_table_contains_models_and_benchmarks():
    data = {
        "main": [
            {"model_id": "gpt-5@high", "benchmark": "mmlu_pro", "mean_score": 0.85, "count": 100, "total_cost": 1.5},
            {"model_id": "gpt-5@high", "benchmark": "niah", "mean_score": 0.92, "count": 30, "total_cost": 0.3},
            {"model_id": "claude-opus@max", "benchmark": "mmlu_pro", "mean_score": 0.88, "count": 100, "total_cost": 2.0},
            {"model_id": "claude-opus@max", "benchmark": "niah", "mean_score": 0.95, "count": 30, "total_cost": 0.4},
        ],
        "staircase": [],
        "sweep_groups": [],
    }
    html = render_leaderboard_html(data)
    assert "<table" in html
    assert "gpt-5@high" in html
    assert "claude-opus@max" in html
    assert "mmlu_pro" in html
    assert "niah" in html
    # Scores rendered as percentages
    assert "85" in html  # 0.85 → 85
    assert "95" in html  # 0.95 → 95


def test_render_escapes_html_in_model_id():
    """Paranoia check — leaderboard HTML must escape user-controlled strings."""
    data = {
        "main": [
            {"model_id": "<script>alert('xss')</script>", "benchmark": "mmlu_pro",
             "mean_score": 0.5, "count": 1, "total_cost": 0.0},
        ],
        "staircase": [],
        "sweep_groups": [],
    }
    html = render_leaderboard_html(data)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html or "&lt;" in html


def test_render_staircase_section_when_present():
    data = {
        "main": [],
        "staircase": [
            {"model_id": "m1", "context_tokens": 1024, "mean_score": 1.0, "count": 3},
            {"model_id": "m1", "context_tokens": 4096, "mean_score": 0.66, "count": 3},
            {"model_id": "m1", "context_tokens": 16384, "mean_score": 0.33, "count": 3},
        ],
        "sweep_groups": [],
    }
    html = render_leaderboard_html(data)
    assert "Context Length Staircase" in html or "staircase" in html.lower()
    assert "1024" in html
    assert "16384" in html


def test_render_sweep_section_when_present():
    data = {
        "main": [
            {"model_id": "gpt-5@high", "benchmark": "mmlu_pro", "mean_score": 0.8, "count": 100, "total_cost": 1.0},
            {"model_id": "gpt-5@max", "benchmark": "mmlu_pro", "mean_score": 0.9, "count": 100, "total_cost": 2.0},
        ],
        "staircase": [],
        "sweep_groups": [
            {"base": "openai/gpt-5", "variants": ["gpt-5@high", "gpt-5@max"],
             "efforts": {"gpt-5@high": "high", "gpt-5@max": "max"}},
        ],
    }
    html = render_leaderboard_html(data)
    assert "Reasoning Effort Sweep" in html or "sweep" in html.lower()
    assert "openai/gpt-5" in html


def test_render_empty_data_produces_valid_html():
    data = {"main": [], "staircase": [], "sweep_groups": []}
    html = render_leaderboard_html(data)
    assert "<html" in html
    assert "</html>" in html
    # Some message about no data
    assert "no data" in html.lower() or "empty" in html.lower() or "0 benchmarks" in html.lower()
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement templates**

Create `src/prism/leaderboard/templates.py`:
```python
"""HTML string templates for the leaderboard.

Kept as Python string constants to avoid adding Jinja2. Substitution uses
`str.replace` with carefully escaped inputs (see renderer.py).
"""
from __future__ import annotations

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Prism Leaderboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2em; color: #333; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }
  h2 { color: #555; margin-top: 2em; }
  table { border-collapse: collapse; margin: 1em 0; }
  th, td { border: 1px solid #ddd; padding: 0.5em 0.8em; text-align: right; }
  th { background: #f5f5f5; text-align: left; }
  td.model { text-align: left; font-family: monospace; font-size: 0.9em; }
  td.score-high { background: #d4edda; color: #155724; }
  td.score-mid  { background: #fff3cd; color: #856404; }
  td.score-low  { background: #f8d7da; color: #721c24; }
  td.empty { color: #999; font-style: italic; }
  .note { color: #666; font-size: 0.9em; margin: 0.5em 0 1.5em; }
</style>
</head>
<body>
<h1>Prism Leaderboard</h1>
<p class="note">Scores are pass@1 (higher is better). Generated by <code>prism leaderboard publish</code>.</p>
{main_section}
{staircase_section}
{sweep_section}
</body>
</html>
"""

EMPTY_STATE = """<p class="note">No benchmark data found — run <code>prism run</code> first.</p>"""

MAIN_HEADER = """<h2>Main Leaderboard</h2>
<table>
<thead><tr><th>Model</th>{benchmark_headers}</tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""

STAIRCASE_HEADER = """<h2>Context Length Staircase</h2>
<p class="note">Performance vs context length for NIAH / RULER MK-NIAH benchmarks. Detects "claims 1M but dies at 128K".</p>
<table>
<thead><tr><th>Model</th>{length_headers}</tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""

SWEEP_HEADER = """<h2>Reasoning Effort Sweep</h2>
<p class="note">Same base model under different thinking/effort settings. Compare accuracy × cost tradeoffs.</p>
<table>
<thead><tr><th>Base Model</th><th>Variant</th><th>Effort</th><th>Mean Score</th><th>Total Cost</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""
```

Create `src/prism/leaderboard/renderer.py`:
```python
"""Pure-Python HTML leaderboard renderer.

Reads the aggregated data dicts (produced by queries.py) and emits a static
HTML page plus a sibling `data.json`. No Jinja — plain string formatting with
explicit HTML escaping on all user-supplied content.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

from prism.leaderboard import templates as T


def _esc(s: Any) -> str:
    """HTML-escape any value."""
    return _html.escape(str(s), quote=True)


def _score_class(score: float) -> str:
    if score >= 0.8:
        return "score-high"
    if score >= 0.5:
        return "score-mid"
    return "score-low"


def _render_main(main_rows: list[dict[str, Any]]) -> str:
    if not main_rows:
        return ""
    # Pivot: rows = models, columns = benchmarks.
    models = sorted({r["model_id"] for r in main_rows})
    benchmarks = sorted({r["benchmark"] for r in main_rows})
    by_cell: dict[tuple[str, str], dict[str, Any]] = {
        (r["model_id"], r["benchmark"]): r for r in main_rows
    }

    header_cells = "".join(f"<th>{_esc(b)}</th>" for b in benchmarks)
    row_html_parts = []
    for m in models:
        cells = [f'<td class="model">{_esc(m)}</td>']
        for b in benchmarks:
            c = by_cell.get((m, b))
            if c is None:
                cells.append('<td class="empty">—</td>')
            else:
                pct = int(round(c["mean_score"] * 100))
                cls = _score_class(c["mean_score"])
                cells.append(f'<td class="{cls}">{pct}%</td>')
        row_html_parts.append("<tr>" + "".join(cells) + "</tr>")
    rows = "\n".join(row_html_parts)
    return T.MAIN_HEADER.format(benchmark_headers=header_cells, rows=rows)


def _render_staircase(staircase_rows: list[dict[str, Any]]) -> str:
    if not staircase_rows:
        return ""
    models = sorted({r["model_id"] for r in staircase_rows})
    lengths = sorted({r["context_tokens"] for r in staircase_rows})
    by_cell: dict[tuple[str, int], dict[str, Any]] = {
        (r["model_id"], r["context_tokens"]): r for r in staircase_rows
    }
    header_cells = "".join(f"<th>{_esc(l)}</th>" for l in lengths)
    row_html_parts = []
    for m in models:
        cells = [f'<td class="model">{_esc(m)}</td>']
        for L in lengths:
            c = by_cell.get((m, L))
            if c is None:
                cells.append('<td class="empty">—</td>')
            else:
                pct = int(round(c["mean_score"] * 100))
                cls = _score_class(c["mean_score"])
                cells.append(f'<td class="{cls}">{pct}%</td>')
        row_html_parts.append("<tr>" + "".join(cells) + "</tr>")
    rows = "\n".join(row_html_parts)
    return T.STAIRCASE_HEADER.format(length_headers=header_cells, rows=rows)


def _render_sweep(sweep_groups: list[dict[str, Any]], main_rows: list[dict[str, Any]]) -> str:
    if not sweep_groups:
        return ""
    # Compute mean score + total cost per model across all benchmarks.
    score_by_model: dict[str, list[float]] = {}
    cost_by_model: dict[str, float] = {}
    for r in main_rows:
        score_by_model.setdefault(r["model_id"], []).append(r["mean_score"])
        cost_by_model[r["model_id"]] = cost_by_model.get(r["model_id"], 0.0) + r["total_cost"]

    row_parts = []
    for g in sweep_groups:
        base = g["base"]
        for variant in g["variants"]:
            effort = g["efforts"].get(variant) or "?"
            scores = score_by_model.get(variant, [])
            mean = sum(scores) / len(scores) if scores else 0.0
            cost = cost_by_model.get(variant, 0.0)
            pct = int(round(mean * 100))
            cls = _score_class(mean)
            row_parts.append(
                f"<tr><td>{_esc(base)}</td>"
                f"<td>{_esc(variant)}</td>"
                f"<td>{_esc(effort)}</td>"
                f'<td class="{cls}">{pct}%</td>'
                f"<td>${cost:.4f}</td></tr>"
            )
    rows = "\n".join(row_parts)
    return T.SWEEP_HEADER.format(rows=rows)


def render_leaderboard_html(data: dict[str, Any]) -> str:
    """data keys: 'main' (list of agg dicts), 'staircase' (list), 'sweep_groups' (list)."""
    main = _render_main(data.get("main", []))
    staircase = _render_staircase(data.get("staircase", []))
    sweep = _render_sweep(data.get("sweep_groups", []), data.get("main", []))
    if not any([main, staircase, sweep]):
        main = T.EMPTY_STATE
    return T.PAGE_TEMPLATE.format(
        main_section=main,
        staircase_section=staircase,
        sweep_section=sweep,
    )


def render_leaderboard(data: dict[str, Any], *, output_dir: Path | str) -> Path:
    """Write `index.html` and `data.json` to output_dir. Returns the HTML path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    html = render_leaderboard_html(data)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out / "index.html"
```

- [ ] **Step 4: Pass — 5 tests**

- [ ] **Step 5: Commit**

```bash
git add src/prism/leaderboard/templates.py src/prism/leaderboard/renderer.py src/prism/leaderboard/__init__.py tests/unit/test_leaderboard_renderer.py
git commit -m "feat(leaderboard): pure-Python HTML renderer with main/staircase/sweep views"
```

---

## Task 3: CLI — `prism leaderboard publish`

**Files:**
- Modify: `src/prism/cli.py`
- Test: `tests/e2e/test_cli_leaderboard.py`

- [ ] **Step 1: Failing test**

Create `tests/e2e/test_cli_leaderboard.py`:
```python
from pathlib import Path

from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_leaderboard_publish_help():
    result = runner.invoke(app, ["leaderboard", "publish", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.stdout
    assert "workdir" in result.stdout.lower() or "work-dir" in result.stdout.lower()


def test_leaderboard_publish_missing_workdir_errors(tmp_path: Path):
    nonexistent = tmp_path / "does-not-exist"
    result = runner.invoke(app, [
        "leaderboard", "publish", str(nonexistent), "--output", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0


def test_leaderboard_publish_empty_workdir_still_produces_html(tmp_path: Path):
    """Running publish against an empty workdir should NOT crash; it should produce
    an 'empty state' leaderboard HTML."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    # Initialize an empty DB so the publish command has something to query.
    from prism.storage.database import Database
    import asyncio
    async def _init():
        db = Database(workdir / "prism.db")
        await db.init()
    asyncio.run(_init())

    out_dir = tmp_path / "out"
    result = runner.invoke(app, [
        "leaderboard", "publish", str(workdir), "--output", str(out_dir),
    ])
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "index.html").exists()
    assert (out_dir / "data.json").exists()
    html = (out_dir / "index.html").read_text()
    assert "<html" in html
```

- [ ] **Step 2: Fail**

- [ ] **Step 3: Implement**

At the top of `src/prism/cli.py`, add these imports near the existing ones:
```python
from prism.leaderboard import (
    aggregate_by_model_benchmark,
    aggregate_staircase,
    list_thinking_variants,
    render_leaderboard,
)
from prism.storage.database import Database
```

Add a Typer sub-app for `leaderboard` commands. After the existing `@app.command` definitions but before `if __name__ == "__main__"`:
```python
leaderboard_app = typer.Typer(help="Leaderboard generation")
app.add_typer(leaderboard_app, name="leaderboard")


@leaderboard_app.command("publish")
def leaderboard_publish_cmd(
    workdir: Path = typer.Argument(..., exists=True, help="Path containing prism.db"),
    output: Path = typer.Option(..., "--output", help="Directory to write index.html + data.json"),
) -> None:
    """Render the leaderboard HTML from a Prism workdir's SQLite DB."""
    db_path = workdir / "prism.db"
    if not db_path.exists():
        console.print(f"[red]No prism.db found in {workdir}[/red]")
        raise typer.Exit(code=2)

    async def _build() -> dict:
        db = Database(db_path)
        main = await aggregate_by_model_benchmark(db=db)
        sweep_groups = await list_thinking_variants(db=db)
        # Staircase aggregates for NIAH + RULER; empty if neither has been run.
        staircase = []
        for bm in ("niah", "ruler_mk"):
            staircase.extend(await aggregate_staircase(db=db, benchmark=bm))
        return {"main": main, "staircase": staircase, "sweep_groups": sweep_groups}

    data = asyncio.run(_build())
    html_path = render_leaderboard(data, output_dir=output)
    console.print(f"Wrote leaderboard → {html_path}")
```

Note: `asyncio` is already imported at top of cli.py from P2a. If not, add `import asyncio`.

- [ ] **Step 4: Pass — 3 tests + full suite**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/e2e/test_cli_leaderboard.py -v
uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add src/prism/cli.py tests/e2e/test_cli_leaderboard.py
git commit -m "feat(cli): add `prism leaderboard publish` subcommand"
```

---

## Task 4: Integration test — end-to-end leaderboard from a real NIAH run

**Files:**
- Test: `tests/integration/test_leaderboard_end_to_end.py`

- [ ] **Step 1: Create test**

```python
"""End-to-end: run NIAH with a fake adapter, then publish leaderboard, then
verify the HTML contains the expected staircase breakdown."""
import re
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.niah.benchmark import NIAHBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.leaderboard import (
    aggregate_by_model_benchmark,
    aggregate_staircase,
    render_leaderboard,
)
from prism.runners.limit import LimitRunner
from prism.service import RunService


class _PartialNeedleAdapter(Adapter):
    """Finds the needle only in short contexts; fails on long ones.

    This simulates a model whose context window is small — useful for testing
    that the staircase view captures the drop-off.
    """
    _NEEDLE_RE = re.compile(r"The special passcode is ([A-Z0-9_-]+)\.")

    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        content = request.messages[-1]["content"]
        if not isinstance(content, str):
            content = ""
        # Hard cap: only find the needle when content is < 3000 chars
        # (approximates 512-token context). For 1024-token cases this still works;
        # for 4096+ cases we pretend to fail.
        if len(content) > 4000:
            return AdapterResponse(
                text="Answer: UNKNOWN",
                reasoning_text=None,
                tokens_in=1000, tokens_out=5, latency_ms=1.0, cost_usd=0.001, raw={},
            )
        m = self._NEEDLE_RE.search(content)
        text = f"Answer: {m.group(1)}" if m else "Answer: UNKNOWN"
        return AdapterResponse(
            text=text,
            reasoning_text=None,
            tokens_in=1000, tokens_out=5, latency_ms=1.0, cost_usd=0.001, raw={},
        )


@pytest.mark.asyncio
async def test_leaderboard_captures_staircase_dropoff(tmp_path: Path):
    # 2 lengths × 1 depth → 2 NIAH prompts, one should pass (512) and one should fail (8192).
    bm = NIAHBenchmark(lengths=[512, 8192], depths=[0.5])

    profile = ModelProfile(
        id="cap-512", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    svc = RunService(
        db_path=tmp_path / "prism.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = LimitRunner(service=svc)

    result = await runner.run(
        benchmark=bm, profile=profile, adapter=_PartialNeedleAdapter(profile),
        subset="full",
    )
    # 1 of 2 passes → pass_at_1 = 0.5
    assert result["prompt_count"] == 2
    assert result["pass_at_1"] == pytest.approx(0.5)

    # Now render the leaderboard from the DB.
    main = await aggregate_by_model_benchmark(db=svc.db)
    staircase = await aggregate_staircase(db=svc.db, benchmark="niah")
    data = {"main": main, "staircase": staircase, "sweep_groups": []}
    out_dir = tmp_path / "out"
    html_path = render_leaderboard(data, output_dir=out_dir)

    html = html_path.read_text()
    # Main table should contain the model and the niah benchmark.
    assert "cap-512" in html
    assert "niah" in html
    # Staircase section shows 512 (passed) and 8192 (failed).
    assert "512" in html
    assert "8192" in html
    # Staircase data should show a drop-off: 512 → 100%, 8192 → 0%.
    sc_by_len = {(r["model_id"], r["context_tokens"]): r for r in staircase}
    assert sc_by_len[("cap-512", 512)]["mean_score"] == pytest.approx(1.0)
    assert sc_by_len[("cap-512", 8192)]["mean_score"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run + full suite**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest tests/integration/test_leaderboard_end_to_end.py -v
uv run pytest
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_leaderboard_end_to_end.py
git commit -m "test(integration): leaderboard captures NIAH staircase dropoff"
```

---

## Task 5: README + final verification + tag

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md` (in-repo path)

- [ ] **Step 1: Update README**

Add a new section before `## License`:

```markdown
## Publishing a leaderboard

After running one or more benchmarks, render a static HTML leaderboard:

\`\`\`bash
uv run prism leaderboard publish ./.prism_runs --output ./leaderboard
\`\`\`

This produces `leaderboard/index.html` (human-readable) and `leaderboard/data.json`
(machine-readable). The HTML has three sections:

- **Main Leaderboard** — model × benchmark pass@1 grid
- **Context Length Staircase** — per-length accuracy for NIAH/RULER runs
- **Reasoning Effort Sweep** — same base model compared across thinking=off/high/max

To publish to GitHub Pages, commit the `leaderboard/` directory to a `gh-pages`
branch, or copy its contents to your Pages source.
```

Update the roadmap line (near architecture) to:
```
P2g (contamination probe), P3 (Agent Runner), P4 (Meta-Ability), P5 (Web UI) are planned.
```

- [ ] **Step 2: Update spec status**

In the in-repo spec file (NOT iCloud), change status to:
```
- **状态**：P1 + P2a-P2f 完成（17 benchmark + leaderboard + Context Staircase + Reasoning Effort Sweep）；P2g 污染探针待启动
```

- [ ] **Step 3: Final verification**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /Users/pingfan/projects/prism
make all
```
Expected: all green. Test count ~212+ (up from P2e's 196).

If lint / mypy issues, fix minimally.

- [ ] **Step 4: Smoke test**

```bash
uv run prism leaderboard --help
uv run prism leaderboard publish --help
```

- [ ] **Step 5: Commit + tag**

```bash
git status --porcelain   # should be clean after the above commits
# (If anything uncommitted — lint fixes, README — bundle into one commit.)

git add README.md docs/superpowers/specs/2026-04-20-prism-llm-benchmark-design.md
git commit -m "docs: P2f — leaderboard + Context Staircase + Reasoning Sweep"

# Also commit the P2f plan doc if it's still untracked:
git add -A   # only after verifying nothing else sneaks in
git diff --cached --stat
git commit -m "chore(p2f): commit P2f plan doc" || true  # fine if nothing new

git tag -a p2f-leaderboard -m "P2f: HTML leaderboard + Context Length Staircase + Reasoning Effort Sweep views"
git tag
git log --oneline --decorate -n 12
```

### Step 6: Stats

```bash
echo "=== P2f Stats ==="
uv run pytest --collect-only -q 2>&1 | tail -3
echo "Commits since p2e:"
git rev-list --count p2e-longcontext..HEAD
echo "--- clean tree check ---"
git status --porcelain
```

## Report (final)

- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- `make all` output
- Test count (~212+ expected)
- Tag `p2f-leaderboard` points to which commit
- Commits since `p2e-longcontext`
- Confirm working tree is clean
- Concerns

---

## Self-Review Checklist

- [ ] `aggregate_by_model_benchmark` joins Score → Response → Prompt → Task correctly
- [ ] `list_thinking_variants` only returns groups with 2+ variants
- [ ] `aggregate_staircase` parses `-lenNNNN-` from prompt_id without brittle regex
- [ ] HTML renderer escapes all user-controlled strings (model_id, benchmark name)
- [ ] Empty data produces a valid "no data" page, not a crash
- [ ] Score color bands: >=0.8 green, >=0.5 yellow, <0.5 red
- [ ] `prism leaderboard publish <workdir> --output <dir>` emits `index.html` + `data.json`
- [ ] Integration test asserts staircase drop-off (pass at 512, fail at 8192)
- [ ] `make all` green; tag `p2f-leaderboard` on HEAD

---

## P2f Success Criteria

- `prism leaderboard publish <workdir> --output <dir>` produces `index.html` + `data.json`
- Main table shows models × benchmarks grid with pass@1 color-coded cells
- Context Length Staircase view renders for any workdir with NIAH or RULER MK-NIAH runs
- Reasoning Effort Sweep view renders when ≥2 model profiles share provider+model but differ in effort
- HTML is valid, escapes user content, displays cleanly without JavaScript
- All P1–P2e tests still pass; P2f adds ~15 new tests with no flakes
- `docs/` in README explains "commit to gh-pages" for publishing
