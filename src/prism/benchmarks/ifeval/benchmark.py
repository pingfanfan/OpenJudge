from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.ifeval import IFEvalJudge

if TYPE_CHECKING:
    from prism.adapters.base import Adapter


class IFEvalBenchmark(Benchmark):
    name = "ifeval"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 100, "standard": 300, "full": None}

    def __init__(
        self,
        *,
        source: str = "google/IFEval",
        source_format: str = "hf",
        split: str = "train",
    ) -> None:
        self.source = source
        self.source_format = source_format
        self.split = split

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        yielded = 0
        for row in load_dataset_cached(
            source=self.source, format=self.source_format, split=self.split
        ):
            if cap is not None and yielded >= cap:
                break
            yield self._row_to_prompt(row)
            yielded += 1

    def make_judge(
        self,
        prompt: PromptSpec,
        *,
        llm_judge_adapter: Adapter | None = None,
    ) -> Judge:
        return IFEvalJudge(constraints=prompt.metadata["constraints"])

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        pid = str(row.get("key") or row.get("prompt_id") or row["prompt"][:32])
        ids = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [{}] * len(ids))
        constraints = [{"id": cid, "kwargs": kw} for cid, kw in zip(ids, kwargs_list)]
        return PromptSpec(
            prompt_id=f"ifeval-{pid}",
            task_id="ifeval",
            version="v1",
            messages=[{"role": "user", "content": row["prompt"]}],
            expected=None,
            metadata={"constraints": constraints},
        )
