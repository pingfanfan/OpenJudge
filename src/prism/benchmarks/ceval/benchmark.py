from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.rules import RegexJudge

if TYPE_CHECKING:
    from prism.adapters.base import Adapter

_PROMPT_TEMPLATE = """请回答下面的选择题。只在最后一行以 "Answer: X" 的形式给出字母（A/B/C/D）。

问题：{question}

选项：
{choices_block}

请先给出推理，然后在最后一行输出答案。"""

_JUDGE_PATTERN = r"Answer:\s*([A-D])\b"


class CEvalBenchmark(Benchmark):
    name = "ceval"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 200, "standard": 500, "full": None}

    def __init__(
        self,
        *,
        source: str = "ceval/ceval-exam",
        source_format: str = "hf",
        split: str = "val",
        subset_name: str | None = None,
    ) -> None:
        self.source = source
        self.source_format = source_format
        self.split = split
        self.subset_name = subset_name

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        yielded = 0
        load_kwargs: dict[str, Any] = {}
        if self.source_format == "hf" and self.subset_name:
            load_kwargs["name"] = self.subset_name
        for row in load_dataset_cached(
            source=self.source, format=self.source_format, split=self.split, **load_kwargs
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
        qid = str(row.get("id") or row["question"][:32])
        choices_block = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
        content = _PROMPT_TEMPLATE.format(question=row["question"], choices_block=choices_block)
        return PromptSpec(
            prompt_id=f"ceval-{qid}",
            task_id="ceval",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=row["answer"],
            metadata={},
        )
