from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.llm import LLMJudge
from prism.judges.safety_rubrics import HARM_REFUSAL_RUBRIC

if TYPE_CHECKING:
    from prism.adapters.base import Adapter


class HarmBenchBenchmark(Benchmark):
    """HarmBench — tests whether the model refuses actionable harmful requests.

    Scoring convention: higher = safer behavior (1.0 = refused, 0.0 = complied).
    """

    name = "harmbench"
    track = "limit"
    version = "v1"
    needs_llm_judge = True
    subset_caps = {"quick": 100, "standard": 400, "full": None}

    def __init__(
        self,
        *,
        source: str = "walledai/HarmBench",
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
        if llm_judge_adapter is None:
            raise ValueError(
                "HarmBench requires an llm_judge_adapter — pass --judge-model on the CLI "
                "or judge_adapter=... to LimitRunner.run()"
            )
        return LLMJudge(adapter=llm_judge_adapter, rubric=HARM_REFUSAL_RUBRIC)

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        bid = str(row.get("BehaviorID") or row.get("id") or row["Behavior"][:32])
        behavior = row["Behavior"]
        category = row.get("Category", "unknown")
        return PromptSpec(
            prompt_id=f"harmbench-{bid}",
            task_id="harmbench",
            version="v1",
            messages=[{"role": "user", "content": behavior}],
            expected=category,
            metadata={"category": category},
        )
