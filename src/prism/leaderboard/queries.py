"""SQLAlchemy aggregate queries powering the leaderboard views."""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import func, select

from prism.storage.database import Database
from prism.storage.schema import Model, Prompt, Response, Score, Task


async def aggregate_by_model_benchmark(*, db: Database) -> list[dict[str, Any]]:
    """One row per (model_id, benchmark) with mean score and count."""
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
    """Group models by (provider, model); return only groups with 2+ variants."""
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
    """Per (model_id, context_tokens), mean score for a given staircase benchmark.

    Extracts context length from prompt_id pattern `-len<N>-`.
    """
    stmt = (
        select(Response.model_id, Response.prompt_id, Score.score)
        .join(Score, Score.response_id == Response.id)
        .join(Prompt, Prompt.id == Response.prompt_id)
        .join(Task, Task.id == Prompt.task_id)
        .where(Task.benchmark == benchmark)
    )
    async with db.session() as s:
        rows = (await s.execute(stmt)).all()

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
