from pathlib import Path
from typing import Any

import pytest

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import Cost, ModelProfile, RateLimit
from prism.service import RunService


class EchoAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        content = request.messages[-1]["content"]
        return AdapterResponse(
            text=content,  # Echo back
            reasoning_text=None,
            tokens_in=len(content) // 4 + 1,
            tokens_out=len(content) // 4 + 1,
            latency_ms=1.0,
            cost_usd=0.0,
            raw={},
        )


@pytest.mark.asyncio
async def test_full_run_lifecycle(tmp_path: Path):
    profile = ModelProfile(
        id="echo",
        provider="openai",  # pretend
        model="echo",
        rate_limit=RateLimit(rpm=6000, tpm=10_000_000),
        cost=Cost(),
    )
    adapter = EchoAdapter(profile)

    svc = RunService(
        db_path=tmp_path / "prism.db",
        artifacts_root=tmp_path / "artifacts",
        checkpoint_path=tmp_path / "cp.db",
    )
    await svc.init()

    prompts: dict[str, dict[str, Any]] = {
        "p1": {"version": "v1", "text": "What is 2+2?", "system": None, "task_id": "t1"},
        "p2": {"version": "v1", "text": "Say hello.", "system": None, "task_id": "t1"},
    }

    run_id = await svc.create_run(suite="smoke")
    await svc.register_model(profile)
    await svc.register_task(task_id="t1", benchmark="smoke", track="limit")
    for pid, meta in prompts.items():
        await svc.register_prompt(prompt_id=pid, **meta)

    await svc.execute(
        run_id=run_id,
        profiles={profile.id: profile},
        adapters={profile.id: adapter},
        prompts={pid: [{"role": "user", "content": m["text"]}] for pid, m in prompts.items()},
        seeds=[0],
        max_concurrency=2,
    )

    summary = await svc.summarize(run_id=run_id)
    assert summary["response_count"] == 2
    assert summary["total_cost_usd"] == 0.0
