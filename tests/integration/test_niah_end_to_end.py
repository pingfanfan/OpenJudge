"""Integration test: NIAH with fake adapter that finds and returns the needle."""
import re
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.niah.benchmark import NIAHBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.runners.limit import LimitRunner
from prism.service import RunService


class _NeedleFindingAdapter(Adapter):
    """Extracts the needle phrase from the haystack and returns it as "Answer: <code>"."""
    _NEEDLE_RE = re.compile(r"The special passcode is ([A-Z0-9_-]+)\.")

    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        content = request.messages[-1]["content"]
        if not isinstance(content, str):
            content = ""
        m = self._NEEDLE_RE.search(content)
        text = f"Answer: {m.group(1)}" if m else "Answer: UNKNOWN"
        return AdapterResponse(
            text=text,
            reasoning_text=None,
            tokens_in=1000, tokens_out=5, latency_ms=1.0, cost_usd=0.0, raw={},
        )


@pytest.mark.asyncio
async def test_niah_pipeline_pass_at_1_is_1_when_adapter_finds_needle(tmp_path: Path):
    bm = NIAHBenchmark(lengths=[512, 1024], depths=[0.0, 0.5, 1.0])

    profile = ModelProfile(
        id="needle-find", provider="openai", model="x",
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
        benchmark=bm, profile=profile, adapter=_NeedleFindingAdapter(profile),
        subset="full",
    )

    assert result["prompt_count"] == 6
    assert result["pass_at_1"] == pytest.approx(1.0)
