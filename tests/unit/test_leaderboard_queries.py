from pathlib import Path

import pytest

from prism.leaderboard.queries import (
    aggregate_by_model_benchmark,
    aggregate_staircase,
    list_thinking_variants,
)
from prism.storage.database import Database
from prism.storage.schema import Model, Prompt, Response, Run, Score, Task


async def _seed(db: Database) -> None:
    async with db.session() as s:
        s.add(Run(id="r1", suite="test", config_hash=""))
        s.add(Model(id="gpt-5@high", provider="openai", model="gpt-5",
                   reasoning_effort="high"))
        s.add(Model(id="gpt-5@max", provider="openai", model="gpt-5",
                   reasoning_effort="max"))
        s.add(Task(id="mmlu_pro", benchmark="mmlu_pro", track="limit"))
        s.add(Task(id="niah", benchmark="niah", track="limit"))
        s.add(Prompt(id="p1", task_id="mmlu_pro", version="v1", text="q1"))
        s.add(Prompt(id="niah-len1024-depth50", task_id="niah", version="v1", text="needle"))
        s.add(Prompt(id="niah-len4096-depth50", task_id="niah", version="v1", text="needle"))
        await s.commit()
    async with db.session() as s:
        for model_id in ("gpt-5@high", "gpt-5@max"):
            for prompt_id in ("p1", "niah-len1024-depth50", "niah-len4096-depth50"):
                s.add(Response(
                    run_id="r1", model_id=model_id, prompt_id=prompt_id, seed=0,
                    text="x", tokens_in=10, tokens_out=5, cost_usd=0.001,
                ))
        await s.commit()
    async with db.session() as s:
        from sqlalchemy import select
        rows = list((await s.execute(select(Response))).scalars())
        score_map = {
            ("gpt-5@high", "p1"): 1.0,
            ("gpt-5@high", "niah-len1024-depth50"): 1.0,
            ("gpt-5@high", "niah-len4096-depth50"): 0.0,
            ("gpt-5@max", "p1"): 1.0,
            ("gpt-5@max", "niah-len1024-depth50"): 1.0,
            ("gpt-5@max", "niah-len4096-depth50"): 1.0,
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
    result = {(r["model_id"], r["benchmark"]): r for r in rows}
    assert result[("gpt-5@high", "mmlu_pro")]["mean_score"] == pytest.approx(1.0)
    assert result[("gpt-5@high", "niah")]["mean_score"] == pytest.approx(0.5)
    assert result[("gpt-5@max", "mmlu_pro")]["mean_score"] == pytest.approx(1.0)
    assert result[("gpt-5@max", "niah")]["mean_score"] == pytest.approx(1.0)
    assert result[("gpt-5@high", "mmlu_pro")]["count"] == 1


@pytest.mark.asyncio
async def test_list_thinking_variants(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    await db.init()
    await _seed(db)

    variants = await list_thinking_variants(db=db)
    assert len(variants) == 1
    group = variants[0]
    assert group["base"] == "openai/gpt-5"
    assert set(group["variants"]) == {"gpt-5@high", "gpt-5@max"}


@pytest.mark.asyncio
async def test_aggregate_staircase_for_niah(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    await db.init()
    await _seed(db)

    rows = await aggregate_staircase(db=db, benchmark="niah")
    # 2 models × 2 lengths = 4 rows
    by_cell = {(r["model_id"], r["context_tokens"]): r for r in rows}
    assert by_cell[("gpt-5@high", 1024)]["mean_score"] == pytest.approx(1.0)
    assert by_cell[("gpt-5@high", 4096)]["mean_score"] == pytest.approx(0.0)
    assert by_cell[("gpt-5@max", 1024)]["mean_score"] == pytest.approx(1.0)
    assert by_cell[("gpt-5@max", 4096)]["mean_score"] == pytest.approx(1.0)
