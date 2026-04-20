"""End-to-end: run NIAH with a fake adapter that fails at long contexts, then
publish leaderboard, then verify the staircase shows the dropoff."""
import re
from pathlib import Path

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.benchmarks.niah.benchmark import NIAHBenchmark
from prism.config.model_profile import ModelProfile, RateLimit
from prism.leaderboard import (
    aggregate_by_model_benchmark,
    aggregate_staircase,
    render_leaderboard,
)
from prism.runners.limit import LimitRunner
from prism.service import RunService


class _PartialNeedleAdapter(Adapter):
    """Finds the needle only in short contexts; fails on long ones."""
    _NEEDLE_RE = re.compile(r"The special passcode is ([A-Z0-9_-]+)\.")

    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        content = request.messages[-1]["content"]
        if not isinstance(content, str):
            content = ""
        # Hard cap: pretend to fail when context > 4000 chars (~1000 tokens).
        if len(content) > 4000:
            return AdapterResponse(
                text="Answer: UNKNOWN",
                reasoning_text=None,
                tokens_in=1000, tokens_out=5, latency_ms=1.0, cost_usd=0.001, raw={},
            )
        m = self._NEEDLE_RE.search(content)
        text = f"Answer: {m.group(1)}" if m else "Answer: UNKNOWN"
        return AdapterResponse(
            text=text,
            reasoning_text=None,
            tokens_in=1000, tokens_out=5, latency_ms=1.0, cost_usd=0.001, raw={},
        )


@pytest.mark.asyncio
async def test_leaderboard_captures_staircase_dropoff(tmp_path: Path):
    bm = NIAHBenchmark(lengths=[512, 8192], depths=[0.5])

    profile = ModelProfile(
        id="cap-512", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    svc = RunService(
        db_path=tmp_path / "prism.db",
        artifacts_root=tmp_path / "a",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()
    runner = LimitRunner(service=svc)

    result = await runner.run(
        benchmark=bm, profile=profile, adapter=_PartialNeedleAdapter(profile),
        subset="full",
    )
    assert result["prompt_count"] == 2
    assert result["pass_at_1"] == pytest.approx(0.5)

    main = await aggregate_by_model_benchmark(db=svc.db)
    staircase = await aggregate_staircase(db=svc.db, benchmark="niah")
    data = {"main": main, "staircase": staircase, "sweep_groups": []}
    out_dir = tmp_path / "out"
    html_path = render_leaderboard(data, output_dir=out_dir)

    html = html_path.read_text()
    assert "cap-512" in html
    assert "niah" in html
    assert "512" in html
    assert "8192" in html

    sc_by_len = {(r["model_id"], r["context_tokens"]): r for r in staircase}
    assert sc_by_len[("cap-512", 512)]["mean_score"] == pytest.approx(1.0)
    assert sc_by_len[("cap-512", 8192)]["mean_score"] == pytest.approx(0.0)
