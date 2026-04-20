from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec

if TYPE_CHECKING:
    from prism.adapters.base import Adapter
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.rules import NumericJudge

_PROMPT_TEMPLATE = (
    "You are solving an AIME (American Invitational Mathematics Examination) problem."
    " The answer is always a non-negative integer between 0 and 999.\n"
    "\n"
    "Problem:\n"
    "{problem}\n"
    "\n"
    "Work through the problem step by step. Give your final answer on the last line"
    " as a single integer (no units, no words, just the number)."
)


class AIMEBenchmark(Benchmark):
    name = "aime"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 30, "standard": 60, "full": None}

    def __init__(
        self,
        *,
        source: str = "Maxwell-Jia/AIME_2024",
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
        return NumericJudge()

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        rid = str(row.get("id") or f"{row.get('year', 'x')}-{row.get('problem_number', 'x')}")
        content = _PROMPT_TEMPLATE.format(problem=row["problem"])
        return PromptSpec(
            prompt_id=f"aime-{rid}",
            task_id="aime",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=str(row["answer"]),
            metadata={"year": row.get("year"), "problem_number": row.get("problem_number")},
        )
