import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import ModelProfile, RateLimit
from prism.orchestrator.matrix import Cell
from prism.orchestrator.runner import OrchestratorRunner


class FakeAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        await asyncio.sleep(0.01)
        return AdapterResponse(
            text="ok",
            reasoning_text=None,
            tokens_in=10,
            tokens_out=5,
            latency_ms=10.0,
            cost_usd=0.0,
            raw={},
        )


@pytest.mark.asyncio
async def test_run_all_cells(tmp_path: Path):
    profile = ModelProfile(
        id="m1",
        provider="openai",
        model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    runner = OrchestratorRunner(
        adapters={"m1": FakeAdapter(profile)},
        profiles={"m1": profile},
        checkpoint_path=tmp_path / "cp.db",
    )
    await runner.init()

    cells = [Cell("m1", f"p{i}", 0) for i in range(5)]
    prompts = {f"p{i}": [{"role": "user", "content": str(i)}] for i in range(5)}

    results: list[tuple[Cell, AdapterResponse]] = []

    async def on_done(cell: Cell, resp: AdapterResponse) -> None:
        results.append((cell, resp))

    await runner.run(
        run_id="r",
        cells=cells,
        prompts=prompts,
        on_done=on_done,
        max_concurrency=3,
    )
    assert len(results) == 5
    assert all(r.text == "ok" for _, r in results)


@pytest.mark.asyncio
async def test_one_failing_cell_does_not_cancel_others(tmp_path: Path):
    profile = ModelProfile(
        id="m1", provider="openai", model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )

    call_count = {"n": 0}

    class FlakyAdapter(Adapter):
        async def complete(self, request: AdapterRequest) -> AdapterResponse:
            call_count["n"] += 1
            # Fail only on the cell whose content is "1"
            if request.messages[-1]["content"] == "1":
                raise RuntimeError("synthetic failure")
            return AdapterResponse(
                text="ok",
                reasoning_text=None,
                tokens_in=1,
                tokens_out=1,
                latency_ms=1.0,
                cost_usd=0.0,
                raw={},
            )

    runner = OrchestratorRunner(
        adapters={"m1": FlakyAdapter(profile)},
        profiles={"m1": profile},
        checkpoint_path=tmp_path / "cp.db",
    )
    await runner.init()

    cells = [Cell("m1", f"p{i}", 0) for i in range(5)]
    prompts = {f"p{i}": [{"role": "user", "content": str(i)}] for i in range(5)}

    succeeded: list[Cell] = []

    async def on_done(cell: Cell, _resp: AdapterResponse) -> None:
        succeeded.append(cell)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        await runner.run(
            run_id="r",
            cells=cells,
            prompts=prompts,
            on_done=on_done,
            max_concurrency=5,
        )

    # All 5 cells should have been attempted (one failed, four succeeded).
    assert call_count["n"] == 5
    # The 4 non-failing cells should have called on_done.
    assert len(succeeded) == 4
    # The failing cell should NOT be in succeeded.
    assert all(c.prompt_id != "p1" for c in succeeded)


@pytest.mark.asyncio
async def test_resume_skips_done(tmp_path: Path):
    profile = ModelProfile(
        id="m1",
        provider="openai",
        model="x",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
    )
    adapter = FakeAdapter(profile)
    adapter.complete = AsyncMock(wraps=adapter.complete)

    runner = OrchestratorRunner(
        adapters={"m1": adapter},
        profiles={"m1": profile},
        checkpoint_path=tmp_path / "cp.db",
    )
    await runner.init()

    cells = [Cell("m1", f"p{i}", 0) for i in range(3)]
    prompts = {f"p{i}": [{"role": "user", "content": str(i)}] for i in range(3)}

    await runner.run(run_id="r", cells=cells, prompts=prompts, on_done=None, max_concurrency=2)
    assert adapter.complete.await_count == 3

    # Second call should skip all (all done).
    await runner.run(run_id="r", cells=cells, prompts=prompts, on_done=None, max_concurrency=2)
    assert adapter.complete.await_count == 3
