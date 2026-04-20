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
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl")

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
        subset="full",
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
