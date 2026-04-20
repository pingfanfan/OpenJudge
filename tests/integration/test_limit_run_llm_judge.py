from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.simpleqa.benchmark import SimpleQABenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.limit import LimitRunner
from prism.service import RunService


class _SubjectAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        # Always answers "Paris" — correct for q1 ("capital of France"), wrong for q2 ("Jane Austen").  # noqa: E501
        return AdapterResponse(
            text="Paris.",
            reasoning_text=None,
            tokens_in=3, tokens_out=2, latency_ms=1.0, cost_usd=0.0, raw={},
        )


class _JudgeAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        # Judge inspects the prompt. The LLMJudge prompt template contains both MODEL_OUTPUT
        # and REFERENCE; we check if "Paris" appears in the REFERENCE section.
        content = request.messages[-1]["content"]
        reference_section = content.split("REFERENCE:")[1] if "REFERENCE:" in content else ""
        if "Paris" in reference_section:
            return AdapterResponse(
                text='{"score": 1.0, "confidence": 1.0, "reasoning": "matches reference"}',
                reasoning_text=None,
                tokens_in=5, tokens_out=30, latency_ms=1.0, cost_usd=0.0, raw={},
            )
        return AdapterResponse(
            text='{"score": 0.0, "confidence": 1.0, "reasoning": "does not match reference"}',
            reasoning_text=None,
            tokens_in=5, tokens_out=30, latency_ms=1.0, cost_usd=0.0, raw={},
        )


@pytest.mark.asyncio
async def test_limit_runner_wires_llm_judge_end_to_end(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "simpleqa_sample.jsonl"
    bm = SimpleQABenchmark(source=str(fixture), source_format="jsonl")

    subject_profile = ModelProfile(
        id="subject", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    judge_profile = ModelProfile(
        id="judge", provider="openai", model="x",
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
        profile=subject_profile,
        adapter=_SubjectAdapter(subject_profile),
        judge_adapter=_JudgeAdapter(judge_profile),
        subset="full",
    )

    # 2 prompts: q1 expected "Paris" (judge returns 1.0), q2 expected "Jane Austen" (judge returns 0.0) → pass_at_1 = 0.5  # noqa: E501
    assert result["prompt_count"] == 2
    assert result["pass_at_1"] == pytest.approx(0.5)
