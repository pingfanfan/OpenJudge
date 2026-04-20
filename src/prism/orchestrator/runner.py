import asyncio
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.config.model_profile import ModelProfile
from prism.orchestrator.checkpoint import CheckpointStore
from prism.orchestrator.matrix import Cell
from prism.orchestrator.rate_limit import RateLimiter

OnDone = Callable[[Cell, AdapterResponse], Awaitable[None]] | None


class OrchestratorRunner:
    def __init__(
        self,
        *,
        adapters: dict[str, Adapter],
        profiles: dict[str, ModelProfile],
        checkpoint_path: str | Path,
    ) -> None:
        self.adapters = adapters
        self.profiles = profiles
        self.checkpoint = CheckpointStore(checkpoint_path)
        self._limiters: dict[str, RateLimiter] = {
            mid: RateLimiter(rpm=p.rate_limit.rpm, tpm=p.rate_limit.tpm)
            for mid, p in profiles.items()
        }

    async def init(self) -> None:
        await self.checkpoint.init()

    async def run(
        self,
        *,
        run_id: str,
        cells: Iterable[Cell],
        prompts: dict[str, list[dict[str, Any]]],
        on_done: OnDone,
        max_concurrency: int = 8,
    ) -> None:
        sem = asyncio.Semaphore(max_concurrency)
        cells_list = list(cells)

        async def _execute(cell: Cell) -> None:
            async with sem:
                profile = self.profiles[cell.model_id]
                adapter = self.adapters[cell.model_id]
                limiter = self._limiters[cell.model_id]
                messages = prompts[cell.prompt_id]

                # Approx token count: 4 chars per token; guard against multimodal list content
                approx_tokens = max(1, sum(
                    len(m.get("content", "")) if isinstance(m.get("content"), str) else 100
                    for m in messages
                ) // 4)
                await limiter.acquire(tokens=approx_tokens)

                await self.checkpoint.mark(run_id=run_id, cell=cell, status="running")
                try:
                    resp = await adapter.complete(AdapterRequest(
                        messages=messages,
                        max_output_tokens=4096,
                        seed=cell.seed,
                    ))
                    await self.checkpoint.mark(run_id=run_id, cell=cell, status="done")
                    if on_done is not None:
                        await on_done(cell, resp)
                except Exception:
                    await self.checkpoint.mark(run_id=run_id, cell=cell, status="failed")
                    raise

        pending: list[Cell] = []
        async for c in self.checkpoint.pending_cells(run_id, cells_list):
            pending.append(c)

        await asyncio.gather(*(_execute(c) for c in pending))
