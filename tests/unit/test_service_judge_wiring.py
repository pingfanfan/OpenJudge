from pathlib import Path

import pytest
from sqlalchemy import select

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import ModelProfile, RateLimit
from prism.judges.base import Judge, JudgeResult
from prism.judges.rules import ExactMatchJudge
from prism.service import RunService
from prism.storage.schema import Score


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


class RaisingJudge(Judge):
    name = "raising_judge"

    async def judge(self, *, output: str, expected: str) -> JudgeResult:
        raise RuntimeError("intentional test failure")


@pytest.mark.asyncio
async def test_execute_records_score_when_judge_raises(tmp_path: Path):
    """Judge exceptions must be recorded as score=0, confidence=0 — not crash the run."""
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
        judges={"p1": RaisingJudge()},
        expected={"p1": "anything"},
        seeds=[0],
    )

    async with svc.db.session() as s:
        scores = list((await s.execute(select(Score))).scalars())

    assert len(scores) == 1
    assert scores[0].score == 0.0
    assert scores[0].confidence == 0.0
    assert scores[0].reasoning is not None
    assert "RuntimeError" in scores[0].reasoning
    assert "intentional test failure" in scores[0].reasoning
