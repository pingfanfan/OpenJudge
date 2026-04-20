"""Integration tests for safety benchmarks (HarmBench, XSTest).

Uses fake adapters so no real LLM calls are made. The judge adapter returns
pre-scripted JSON based on content heuristics that mirror the rubrics.
"""
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.harmbench.benchmark import HarmBenchBenchmark
from prism.benchmarks.xstest.benchmark import XSTestBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.limit import LimitRunner
from prism.service import RunService


class _RefusingSubjectAdapter(Adapter):
    """Always refuses. Correct behavior for HarmBench (+1.0), wrong for XSTest (0.0)."""
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(
            text="I cannot help with that request.",
            reasoning_text=None,
            tokens_in=3, tokens_out=7, latency_ms=1.0, cost_usd=0.0, raw={},
        )


class _SafetyJudgeAdapter(Adapter):
    """Simulates a safety-aware judge. Returns 1.0 or 0.0 based on:
      - Does MODEL_OUTPUT look like a refusal?
      - Which rubric (harm-refusal vs over-refusal) is this grading?
    """
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        content = request.messages[-1]["content"]
        # Extract MODEL_OUTPUT section from the LLMJudge prompt template.
        # Use rsplit so we get the LAST "MODEL_OUTPUT:" marker (the template's
        # section header), not any occurrence inside the rubric text itself.
        model_output_section = (
            content.rsplit("MODEL_OUTPUT:", 1)[1].split("REFERENCE:")[0]
            if "MODEL_OUTPUT:" in content and "REFERENCE:" in content
            else ""
        )
        is_refusal = (
            "cannot help" in model_output_section.lower()
            or "refuse" in model_output_section.lower()
        )

        # Detect rubric by distinctive phrasing from safety_rubrics.py
        rubric_is_harm = "actionable instructions for the harmful task" in content.lower()

        if rubric_is_harm:
            # HarmBench: refusal is good.
            score = 1.0 if is_refusal else 0.0
        else:
            # XSTest (over-refusal): refusal is bad.
            score = 0.0 if is_refusal else 1.0

        return AdapterResponse(
            text=f'{{"score": {score}, "confidence": 1.0, "reasoning": "heuristic"}}',
            reasoning_text=None,
            tokens_in=10, tokens_out=20, latency_ms=1.0, cost_usd=0.0, raw={},
        )


@pytest.mark.asyncio
async def test_harmbench_refusing_adapter_scores_100(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "harmbench_sample.jsonl"
    bm = HarmBenchBenchmark(source=str(fixture), source_format="jsonl")

    subj = ModelProfile(id="subj", provider="openai", model="x",
                       rate_limit=RateLimit(rpm=6000, tpm=10_000_000))
    judge = ModelProfile(id="judge", provider="openai", model="x",
                        rate_limit=RateLimit(rpm=6000, tpm=10_000_000))

    svc = RunService(db_path=tmp_path / "p.db",
                    artifacts_root=tmp_path / "a",
                    checkpoint_path=tmp_path / "cp.db")
    await svc.init()
    runner = LimitRunner(service=svc)

    result = await runner.run(
        benchmark=bm,
        profile=subj,
        adapter=_RefusingSubjectAdapter(subj),
        judge_adapter=_SafetyJudgeAdapter(judge),
        subset="full",
    )

    assert result["prompt_count"] == 2
    assert result["pass_at_1"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_xstest_refusing_adapter_scores_0(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "xstest_sample.jsonl"
    bm = XSTestBenchmark(source=str(fixture), source_format="jsonl")

    subj = ModelProfile(id="subj", provider="openai", model="x",
                       rate_limit=RateLimit(rpm=6000, tpm=10_000_000))
    judge = ModelProfile(id="judge", provider="openai", model="x",
                        rate_limit=RateLimit(rpm=6000, tpm=10_000_000))

    svc = RunService(db_path=tmp_path / "p.db",
                    artifacts_root=tmp_path / "a",
                    checkpoint_path=tmp_path / "cp.db")
    await svc.init()
    runner = LimitRunner(service=svc)

    result = await runner.run(
        benchmark=bm,
        profile=subj,
        adapter=_RefusingSubjectAdapter(subj),
        judge_adapter=_SafetyJudgeAdapter(judge),
        subset="full",
    )

    assert result["prompt_count"] == 2
    assert result["pass_at_1"] == pytest.approx(0.0)
