from __future__ import annotations

from typing import Any

from sqlalchemy import select

from prism.adapters.base import Adapter
from prism.benchmarks.base import Benchmark
from prism.config.model_profile import ModelProfile
from prism.judges.base import Judge
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
        judge_adapter: Adapter | None = None,
        seeds: list[int] | None = None,
        subset: str | None = "quick",
        run_id: str | None = None,
        max_concurrency: int = 8,
    ) -> dict[str, Any]:
        if benchmark.needs_llm_judge and judge_adapter is None:
            raise RuntimeError(
                f"Benchmark {benchmark.name!r} requires an LLM judge model — "
                f"pass --judge-model <profile.yaml> (or judge_adapter=... in code)."
            )

        seeds = seeds if seeds is not None else [0]

        if run_id is None:
            run_id = await self.service.create_run(suite=f"{benchmark.name}-{subset or 'default'}")

        await self.service.register_model(profile)
        await self.service.register_task(
            task_id=benchmark.name, benchmark=benchmark.name, track=benchmark.track,
        )

        prompts_to_send: dict[str, list[dict[str, Any]]] = {}
        expected_map: dict[str, str | None] = {}
        judges: dict[str, Judge] = {}

        for spec in benchmark.load_prompts(subset=subset):
            await self.service.register_prompt(
                prompt_id=spec.prompt_id,
                task_id=benchmark.name,
                version=spec.version,
                text=spec.messages[-1]["content"] if isinstance(spec.messages[-1].get("content"), str) else "<multimodal>",
                system=None,
            )
            prompts_to_send[spec.prompt_id] = spec.messages
            expected_map[spec.prompt_id] = spec.expected
            judges[spec.prompt_id] = benchmark.make_judge(spec, llm_judge_adapter=judge_adapter)

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

        # Group scores by prompt_id, take mean over seeds per prompt, then mean over prompts.
        # This is the formally correct pass_at_1 under multi-seed.
        by_prompt: dict[str, list[float]] = {}
        for score, _cost, prompt_id in rows:
            by_prompt.setdefault(prompt_id, []).append(score)
        per_prompt_mean = [sum(v) / len(v) for v in by_prompt.values()]

        return {
            "run_id": run_id,
            "prompt_count": len(by_prompt),
            "pass_at_1": sum(per_prompt_mean) / len(per_prompt_mean),
            "total_cost_usd": float(sum(r[1] for r in rows)),
        }
