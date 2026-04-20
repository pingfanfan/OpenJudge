from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.llm import LLMJudge

if TYPE_CHECKING:
    from prism.adapters.base import Adapter

_PROMPT_TEMPLATE = """Answer the following factual question concisely in 1-2 sentences.

Question: {question}"""

_RUBRIC = (
    "Score 1.0 if the MODEL_OUTPUT is factually correct"
    " (matches the REFERENCE answer, allowing for phrasing differences or extra context).\n"
    "Score 0.0 if the MODEL_OUTPUT is factually incorrect or refuses to answer.\n"
    "If the output is partially correct (e.g., right person but wrong year), score 0.5."
)


class SimpleQABenchmark(Benchmark):
    name = "simpleqa"
    track = "limit"
    version = "v1"
    needs_llm_judge = True
    subset_caps = {"quick": 100, "standard": 300, "full": None}

    def __init__(
        self,
        *,
        source: str = "basicv8vc/SimpleQA",
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
                "SimpleQA requires an llm_judge_adapter — pass --judge-model on the CLI "
                "or judge_adapter=... to LimitRunner.run()"
            )
        return LLMJudge(adapter=llm_judge_adapter, rubric=_RUBRIC)

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        qid = str(row.get("id") or row["question"][:32])
        content = _PROMPT_TEMPLATE.format(question=row["question"])
        return PromptSpec(
            prompt_id=f"simpleqa-{qid}",
            task_id="simpleqa",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=str(row["answer"]),
            metadata={},
        )
