from __future__ import annotations

from typing import Any

from sqlalchemy import select

from prism.adapters.base import Adapter
from prism.benchmarks.base import Benchmark
from prism.config.model_profile import ModelProfile
from prism.service import RunService
from prism.storage.schema import Response, Score


class LimitRunner:
    """Runs a single Benchmark against a single (profile, adapter) pair via RunService.

    Produces an aggregate result summary: prompt_count, pass_at_1, total_cost_usd.
    """

    def __init__(self, *, service: RunService) -> None:
        self.service = service

    async def run(
        self,
        *,
        benchmark: Benchmark,
        profile: ModelProfile,
        adapter: Adapter,
        seeds: list[int] | None = None,
        subset: str | None = "quick",
        run_id: str | None = None,
        max_concurrency: int = 8,
    ) -> dict[str, Any]:
        seeds = seeds if seeds is not None else [0]

        # Ensure a run exists.
        if run_id is None:
            run_id = await self.service.create_run(suite=f"{benchmark.name}-{subset or 'default'}")

        await self.service.register_model(profile)
        await self.service.register_task(
            task_id=benchmark.name, benchmark=benchmark.name, track=benchmark.track,
        )

        prompts_to_send: dict[str, list[dict[str, Any]]] = {}
        expected_map: dict[str, str | None] = {}
        judges: dict[str, Any] = {}

        for spec in benchmark.load_prompts(subset=subset):
            await self.service.register_prompt(
                prompt_id=spec.prompt_id,
                task_id=benchmark.name,
                version=spec.version,
                text=spec.messages[-1]["content"],
                system=None,
            )
            prompts_to_send[spec.prompt_id] = spec.messages
            expected_map[spec.prompt_id] = spec.expected
            judges[spec.prompt_id] = benchmark.make_judge(spec)

        await self.service.execute(
            run_id=run_id,
            profiles={profile.id: profile},
            adapters={profile.id: adapter},
            prompts=prompts_to_send,
            seeds=seeds,
            judges=judges,
            expected=expected_map,
            max_concurrency=max_concurrency,
        )

        return await self._summarize(run_id=run_id)

    async def _summarize(self, *, run_id: str) -> dict[str, Any]:
        async with self.service.db.session() as s:
            rows = list((await s.execute(
                select(Score.score, Response.cost_usd, Response.prompt_id)
                .join(Response, Score.response_id == Response.id)
                .where(Response.run_id == run_id)
            )).all())
        if not rows:
            return {"run_id": run_id, "prompt_count": 0, "pass_at_1": 0.0, "total_cost_usd": 0.0}

        scores = [r[0] for r in rows]
        costs = [r[1] for r in rows]
        unique_prompts = {r[2] for r in rows}
        return {
            "run_id": run_id,
            "prompt_count": len(unique_prompts),
            "pass_at_1": sum(scores) / len(scores),
            "total_cost_usd": float(sum(costs)),
        }
