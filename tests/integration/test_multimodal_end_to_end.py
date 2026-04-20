"""Integration test verifying multimodal content flows adapter→judge→score end-to-end."""
from pathlib import Path

import pytest
from sqlalchemy import select

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.mmmu.benchmark import MMMUBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.limit import LimitRunner
from prism.service import RunService
from prism.storage.schema import Response


class _ColorPickingAdapter(Adapter):
    """Fake multimodal adapter that peeks at the image data URL and answers the color.

    Real multimodal models receive the image bytes and reason; this fake just
    verifies it received a list-content message with an image_url part, then
    always returns "Answer: A".
    """
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        parts = request.messages[-1]["content"]
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    _ = url  # confirms URL is present
                    return AdapterResponse(
                        text="Looking at the image... Answer: A",
                        reasoning_text=None,
                        tokens_in=5, tokens_out=5, latency_ms=1.0, cost_usd=0.0, raw={},
                    )
        return AdapterResponse(
            text="Answer: A",
            reasoning_text=None,
            tokens_in=5, tokens_out=5, latency_ms=1.0, cost_usd=0.0, raw={},
        )


@pytest.mark.asyncio
async def test_mmmu_multimodal_pipeline(tmp_path: Path):
    fixture = Path(__file__).parent.parent / "fixtures" / "mmmu_sample.jsonl"
    bm = MMMUBenchmark(
        source=str(fixture),
        source_format="jsonl",
        fixture_root=fixture.parent.parent,
    )

    profile = ModelProfile(
        id="mm", provider="openai", model="x",
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
        benchmark=bm, profile=profile, adapter=_ColorPickingAdapter(profile),
        subset="full",
    )

    # Fixture: q1 expected A (correct since adapter always says A); q2 expected C (wrong).
    assert result["prompt_count"] == 2
    assert result["pass_at_1"] == pytest.approx(0.5)

    async with svc.db.session() as s:
        responses = list((await s.execute(select(Response))).scalars())
    assert len(responses) == 2
    assert all("Answer: A" in r.text for r in responses)
