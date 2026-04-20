from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select

from prism.adapters.base import Adapter, AdapterResponse
from prism.config.model_profile import ModelProfile
from prism.orchestrator.matrix import Cell, expand_matrix
from prism.orchestrator.runner import OrchestratorRunner
from prism.storage.artifacts import ArtifactStore
from prism.storage.database import Database
from prism.storage.schema import Model, Prompt, Response, Run, Task


class RunService:
    def __init__(
        self,
        *,
        db_path: str | Path,
        artifacts_root: str | Path,
        checkpoint_path: str | Path,
    ) -> None:
        self.db = Database(db_path)
        self.artifacts = ArtifactStore(artifacts_root)
        self.checkpoint_path = Path(checkpoint_path)

    async def init(self) -> None:
        await self.db.init()

    async def create_run(self, *, suite: str, config_hash: str = "") -> str:
        run_id = f"run-{uuid4().hex[:12]}"
        async with self.db.session() as s:
            s.add(Run(id=run_id, suite=suite, config_hash=config_hash, status="running"))
            await s.commit()
        return run_id

    async def register_model(self, profile: ModelProfile) -> None:
        async with self.db.session() as s:
            if await s.get(Model, profile.id) is None:
                s.add(Model(
                    id=profile.id,
                    display_name=profile.display_name,
                    provider=profile.provider,
                    model=profile.model,
                    thinking_enabled=bool(profile.thinking and profile.thinking.enabled),
                    reasoning_effort=profile.reasoning_effort or (
                        profile.thinking.effort if profile.thinking else None
                    ),
                    cost_input_per_mtok=profile.cost.input_per_mtok,
                    cost_output_per_mtok=profile.cost.output_per_mtok,
                ))
                await s.commit()

    async def register_task(self, *, task_id: str, benchmark: str, track: str) -> None:
        async with self.db.session() as s:
            if await s.get(Task, task_id) is None:
                s.add(Task(id=task_id, benchmark=benchmark, track=track))
                await s.commit()

    async def register_prompt(
        self, *, prompt_id: str, task_id: str, version: str, text: str, system: str | None = None
    ) -> None:
        async with self.db.session() as s:
            if await s.get(Prompt, prompt_id) is None:
                s.add(Prompt(id=prompt_id, task_id=task_id, version=version, text=text, system=system))
                await s.commit()

    async def execute(
        self,
        *,
        run_id: str,
        profiles: dict[str, ModelProfile],
        adapters: dict[str, Adapter],
        prompts: dict[str, list[dict[str, Any]]],
        seeds: list[int],
        max_concurrency: int = 8,
    ) -> None:
        runner = OrchestratorRunner(
            adapters=adapters,
            profiles=profiles,
            checkpoint_path=self.checkpoint_path,
        )
        await runner.init()

        cells = list(expand_matrix(
            models=list(profiles.values()),
            prompt_ids=list(prompts.keys()),
            seeds=seeds,
        ))

        async def _persist(cell: Cell, resp: AdapterResponse) -> None:
            async with self.db.session() as s:
                s.add(Response(
                    run_id=run_id,
                    model_id=cell.model_id,
                    prompt_id=cell.prompt_id,
                    seed=cell.seed,
                    text=resp.text,
                    reasoning_text=resp.reasoning_text,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    latency_ms=resp.latency_ms,
                    cost_usd=resp.cost_usd,
                    finish_reason=resp.finish_reason,
                ))
                await s.commit()
            self.artifacts.put(
                run_id,
                f"responses/{cell.model_id}/{cell.prompt_id}-seed{cell.seed}.json",
                resp.model_dump(),
            )

        await runner.run(
            run_id=run_id, cells=cells, prompts=prompts,
            on_done=_persist, max_concurrency=max_concurrency,
        )

        async with self.db.session() as s:
            run = await s.get(Run, run_id)
            if run is not None:
                run.status = "done"
                await s.commit()

    async def summarize(self, *, run_id: str) -> dict[str, Any]:
        async with self.db.session() as s:
            count = (await s.execute(
                select(func.count()).select_from(Response).where(Response.run_id == run_id)
            )).scalar_one()
            total_cost = (await s.execute(
                select(func.coalesce(func.sum(Response.cost_usd), 0.0)).where(Response.run_id == run_id)
            )).scalar_one()
        return {"run_id": run_id, "response_count": count, "total_cost_usd": float(total_cost)}
