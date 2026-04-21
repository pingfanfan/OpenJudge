from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.llm import LLMJudge

if TYPE_CHECKING:
    from prism.adapters.base import Adapter

_PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning step by step,"
    " then give the final answer on the last line as a single number"
    " (or simplified expression).\n\nProblem:\n{problem}\n\nFinal answer on the last line."
)


_JUDGE_RUBRIC = (
    "Decide if MODEL_OUTPUT's final answer is mathematically equivalent to REFERENCE. "
    "Accept any equivalent form: LaTeX vs plain text, \\frac{a}{b} vs a/b, "
    "3\\sqrt{13} vs 3*sqrt(13), (3, π/2) vs \\left(3, \\frac{\\pi}{2}\\right), etc. "
    "Score 1.0 if clearly equivalent. Score 0.0 if clearly not. "
    "Score 0.5 only if genuinely ambiguous."
)


class MATH500Benchmark(Benchmark):
    name = "math500"
    track = "limit"
    version = "v1"
    needs_llm_judge = True
    subset_caps = {"quick": 100, "standard": 250, "full": None}

    def __init__(
        self,
        *,
        source: str = "HuggingFaceH4/MATH-500",
        source_format: str = "hf",
        split: str = "test",
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
                "MATH-500 requires an llm_judge_adapter — pass --judge-model on the CLI. "
                "Math answers contain LaTeX / fractions / text that a pure numeric judge "
                "cannot match reliably."
            )
        return LLMJudge(adapter=llm_judge_adapter, rubric=_JUDGE_RUBRIC)

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        rid = str(row.get("problem_id") or row.get("id") or row["problem"][:32])
        content = _PROMPT_TEMPLATE.format(problem=row["problem"])
        return PromptSpec(
            prompt_id=f"math500-{rid}",
            task_id="math500",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=str(row["answer"]),
            metadata={"level": row.get("level"), "subject": row.get("subject")},
        )
