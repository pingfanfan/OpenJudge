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
