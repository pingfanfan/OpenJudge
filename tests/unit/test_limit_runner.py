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
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl")

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
        subset="full",
    )

    assert result["prompt_count"] == 2
    # q1 expected B (correct), q2 expected C (model always says B → wrong)
    assert result["pass_at_1"] == pytest.approx(0.5)
    assert result["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_limit_runner_no_seeds_defaults_to_single_seed(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"

    class OneCap(MMLUProBenchmark):
        subset_caps = {"quick": 1, "standard": 1, "full": None}

    bm = OneCap(source=str(fixture), source_format="jsonl")
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
        subset="quick",
    )
    assert result["prompt_count"] == 1


@pytest.mark.asyncio
async def test_limit_runner_multiseed_averages_per_prompt(tmp_path: Path):
    """With 2 seeds × 2 prompts, pass_at_1 = mean over prompts of (mean over seeds).

    Adapter flips answer based on seed: seed 0 → B (correct for q1, wrong for q2),
    seed 1 → C (wrong for q1, correct for q2).
    Expected: q1 scores [1, 0] → mean 0.5; q2 scores [0, 1] → mean 0.5; pass_at_1 = 0.5.
    """
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl")

    profile = ModelProfile(
        id="m1", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )

    class FlipAdapter(Adapter):
        async def complete(self, request: AdapterRequest) -> AdapterResponse:
            letter = "B" if request.seed == 0 else "C"
            return AdapterResponse(
                text=f"Answer: {letter}",
                reasoning_text=None,
                tokens_in=1, tokens_out=1, latency_ms=1.0, cost_usd=0.0, raw={},
            )

    svc = RunService(
        db_path=tmp_path / "p.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = LimitRunner(service=svc)
    result = await runner.run(
        benchmark=bm, profile=profile, adapter=FlipAdapter(profile),
        seeds=[0, 1], subset="full",
    )
    # 2 prompts × 2 seeds = 4 scores; per-prompt means [0.5, 0.5]; pass_at_1 = 0.5
    assert result["prompt_count"] == 2
    assert result["pass_at_1"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_limit_runner_raises_when_benchmark_needs_judge_but_none_provided(tmp_path: Path):
    """If a benchmark.needs_llm_judge=True and no judge_adapter is passed, run raises."""
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"

    class _FakeLLMBenchmark(MMLUProBenchmark):
        needs_llm_judge = True

    bm = _FakeLLMBenchmark(source=str(fixture), source_format="jsonl")
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
    with pytest.raises(RuntimeError, match="requires.*judge"):
        await runner.run(
            benchmark=bm, profile=profile, adapter=CorrectAdapter(profile),
            subset="full",
        )


@pytest.mark.asyncio
async def test_limit_runner_passes_judge_adapter_to_make_judge(tmp_path: Path):
    """When judge_adapter is provided, it flows to benchmark.make_judge."""
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    received: list = []

    class _CapturingBenchmark(MMLUProBenchmark):
        needs_llm_judge = True

        def make_judge(self, prompt, *, llm_judge_adapter=None):
            received.append(llm_judge_adapter)
            return super().make_judge(prompt, llm_judge_adapter=llm_judge_adapter)

    bm = _CapturingBenchmark(source=str(fixture), source_format="jsonl")
    profile = ModelProfile(
        id="m1", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    judge_profile = ModelProfile(
        id="judge", provider="openai", model="judge",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    judge_adapter = CorrectAdapter(judge_profile)

    svc = RunService(
        db_path=tmp_path / "p.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = LimitRunner(service=svc)
    await runner.run(
        benchmark=bm, profile=profile, adapter=CorrectAdapter(profile),
        judge_adapter=judge_adapter, subset="full",
    )
    # Two prompts → make_judge called twice → both received the judge_adapter.
    assert len(received) == 2
    assert all(a is judge_adapter for a in received)
