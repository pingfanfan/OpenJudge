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


class SuperCLUEBenchmark(Benchmark):
    name = "superclue"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 200, "standard": 500, "full": None}

    def __init__(
        self,
        *,
        source: str = "CLUEbenchmark/SuperCLUE",
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
        qid = str(row.get("id") or row["question"][:32])

        # Accept two row shapes:
        # 1. {"A": "opt1", "B": "opt2", ...}  — per-letter keys
        # 2. {"choices": ["opt1", "opt2", ...]}  — list form
        if "choices" in row:
            choices = row["choices"]
            if len(choices) != 4:
                raise ValueError(
                    f"SuperCLUE expects 4 choices, got {len(choices)} for row {qid!r}"
                )
            choices_block = "\n".join(
                f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(choices)
            )
        elif all(letter in row for letter in "ABCD"):
            choices_block = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
        else:
            raise ValueError(
                f"SuperCLUE row {qid!r} missing choices: expected either "
                f"'A/B/C/D' keys or a 'choices' list"
            )

        answer = row.get("answer") or row.get("label")
        if answer is None:
            raise ValueError(f"SuperCLUE row {qid!r} missing 'answer' or 'label' field")

        content = _PROMPT_TEMPLATE.format(question=row["question"], choices_block=choices_block)
        return PromptSpec(
            prompt_id=f"superclue-{qid}",
            task_id="superclue",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=str(answer),
            metadata={},
        )
