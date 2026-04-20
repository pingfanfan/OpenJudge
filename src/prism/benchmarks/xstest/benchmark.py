from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.llm import LLMJudge
from prism.judges.safety_rubrics import OVER_REFUSAL_RUBRIC

if TYPE_CHECKING:
    from prism.adapters.base import Adapter


class XSTestBenchmark(Benchmark):
    """XSTest — tests whether a model *avoids over-refusing* benign requests that
    superficially resemble harmful ones.

    Scoring convention: higher = helpful (1.0 = answered benign request, 0.0 = refused).
    Only the "safe" prompts are included.
    """

    name = "xstest"
    track = "limit"
    version = "v1"
    needs_llm_judge = True
    subset_caps = {"quick": 100, "standard": 250, "full": None}

    def __init__(
        self,
        *,
        source: str = "natolambert/xstest-v2-copy",
        source_format: str = "hf",
        split: str = "prompts",
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
            if str(row.get("type", "safe")).lower() != "safe":
                continue
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
                "XSTest requires an llm_judge_adapter — pass --judge-model on the CLI "
                "or judge_adapter=... to LimitRunner.run()"
            )
        return LLMJudge(adapter=llm_judge_adapter, rubric=OVER_REFUSAL_RUBRIC)

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        pid = str(row.get("id") or row.get("prompt_id") or row["prompt"][:32])
        category = row.get("category", "unknown")
        return PromptSpec(
            prompt_id=f"xstest-{pid}",
            task_id="xstest",
            version="v1",
            messages=[{"role": "user", "content": row["prompt"]}],
            expected=category,
            metadata={"category": category, "type": row.get("type", "safe")},
        )
