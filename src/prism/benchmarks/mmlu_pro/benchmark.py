from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec

if TYPE_CHECKING:
    from prism.adapters.base import Adapter
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.rules import RegexJudge

_PROMPT_TEMPLATE = """Answer the following multiple-choice question.
Respond with ONLY the letter (A/B/C/D) on the last line, prefixed by "Answer:".

Question: {question}

Choices:
{choices_block}

Think step by step, then give your final answer as "Answer: X"."""

_JUDGE_PATTERN = r"Answer:\s*([A-D])\b"


class MMLUProBenchmark(Benchmark):
    name = "mmlu_pro"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 200, "standard": 1000, "full": None}

    def __init__(
        self,
        *,
        source: str = "TIGER-Lab/MMLU-Pro",
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
        return RegexJudge(pattern=_JUDGE_PATTERN)

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        qid = str(row.get("question_id") or row.get("id") or row["question"][:32])
        options = row["options"]
        choices_block = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))
        content = _PROMPT_TEMPLATE.format(question=row["question"], choices_block=choices_block)
        return PromptSpec(
            prompt_id=f"mmlu_pro-{qid}",
            task_id="mmlu_pro",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=row["answer"],
            metadata={"category": row.get("category", "unknown")},
        )
