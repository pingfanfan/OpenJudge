"""AgentRunner — orchestrate an AgentBenchmark against one (profile, adapter)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from prism.adapters.base import Adapter
from prism.agent.judge import run_hard_judge
from prism.agent.loop import run_agent_loop
from prism.agent.task import AgentBenchmark, AgentResult, AgentTask
from prism.agent.workspace import workspace_context
from prism.config.model_profile import ModelProfile
from prism.service import RunService
from prism.storage.schema import Response, Run, Score


class AgentRunner:
    """Runs an AgentBenchmark against one (profile, adapter).

    For each AgentTask:
      1. Create a tmp workspace with seeded files
      2. Run inline agent loop (multi-turn tool calling)
      3. Run hard judge (e.g., pytest) in the workspace
      4. Persist Response + Score rows and a trace JSON artifact
    """

    def __init__(self, *, service: RunService) -> None:
        self.service = service

    async def run(
        self,
        *,
        benchmark: AgentBenchmark,
        profile: ModelProfile,
        adapter: Adapter,
        subset: str | None = "quick",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if run_id is None:
            run_id = await self.service.create_run(suite=f"{benchmark.name}-agent")

        await self.service.register_model(profile)
        await self.service.register_task(
            task_id=benchmark.name,
            benchmark=benchmark.name,
            track=benchmark.track,
        )

        tasks = list(benchmark.load_tasks(subset=subset))
        results: list[AgentResult] = []

        for task in tasks:
            result = await self._run_one(
                run_id=run_id, benchmark=benchmark,
                task=task, profile=profile, adapter=adapter,
            )
            results.append(result)

        await self._mark_run_done(run_id)
        return self._summarize(results, run_id=run_id)

    async def _run_one(
        self, *, run_id: str, benchmark: AgentBenchmark, task: AgentTask,
        profile: ModelProfile, adapter: Adapter,
    ) -> AgentResult:
        prompt_id = f"{benchmark.name}-{task.task_id}"
        await self.service.register_prompt(
            prompt_id=prompt_id,
            task_id=benchmark.name,
            version=benchmark.version,
            text=task.user_instruction,
        )

        with workspace_context(task.workspace_files) as workspace:
            loop_result = await run_agent_loop(
                adapter=adapter,
                workspace=workspace,
                user_instruction=task.user_instruction,
                max_turns=task.max_turns,
                prompted_tool_use=profile.prompted_tool_use,
            )
            judge_outcome = run_hard_judge(
                command=task.judge_command,
                workspace=workspace,
                timeout_sec=task.timeout_seconds,
            )

        result = AgentResult(
            task_id=task.task_id,
            model_id=profile.id,
            success=judge_outcome.success,
            turns=loop_result.turns,
            final_text=loop_result.final_text,
            judge_stdout=judge_outcome.stdout,
            judge_exit_code=judge_outcome.exit_code,
            tokens_in=loop_result.tokens_in,
            tokens_out=loop_result.tokens_out,
            latency_ms=loop_result.latency_ms,
            cost_usd=loop_result.cost_usd,
            trace=loop_result.trace,
        )

        await self._persist(run_id=run_id, prompt_id=prompt_id, result=result)
        return result

    async def _persist(
        self, *, run_id: str, prompt_id: str, result: AgentResult
    ) -> None:
        async with self.service.db.session() as s:
            row = Response(
                run_id=run_id,
                model_id=result.model_id,
                prompt_id=prompt_id,
                seed=0,
                text=result.final_text,
                reasoning_text=None,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                finish_reason="done" if result.success else "failed",
            )
            s.add(row)
            await s.flush()
            s.add(Score(
                response_id=row.id,
                judge="agent_hard",
                score=1.0 if result.success else 0.0,
                confidence=1.0,
                reasoning=result.judge_stdout[:500],
            ))
            await s.commit()

        self.service.artifacts.put(
            run_id,
            f"agent/{result.task_id}.json",
            asdict(result),
        )

    async def _mark_run_done(self, run_id: str) -> None:
        async with self.service.db.session() as s:
            run = await s.get(Run, run_id)
            if run is not None:
                run.status = "done"
                await s.commit()

    @staticmethod
    def _summarize(results: list[AgentResult], *, run_id: str) -> dict[str, Any]:
        if not results:
            return {"run_id": run_id, "task_count": 0, "success_rate": 0.0, "total_cost_usd": 0.0}
        success = sum(1 for r in results if r.success)
        cost = sum(r.cost_usd for r in results)
        return {
            "run_id": run_id,
            "task_count": len(results),
            "success_rate": success / len(results),
            "total_cost_usd": cost,
        }
