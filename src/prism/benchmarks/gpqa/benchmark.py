from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.rules import RegexJudge

if TYPE_CHECKING:
    from prism.adapters.base import Adapter

_PROMPT_TEMPLATE = (
    "Answer the following graduate-level multiple-choice question."
    ' Respond with ONLY the letter (A/B/C/D) on the last line, prefixed by "Answer:".\n'
    "\nQuestion: {question}\n\nChoices:\n{choices_block}"
    '\n\nReason carefully, then give your final answer as "Answer: X".'
)

_JUDGE_PATTERN = r"Answer:\s*([A-D])\b"


class GPQABenchmark(Benchmark):
    name = "gpqa"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 100, "standard": 300, "full": None}

    def __init__(
        self,
        *,
        source: str = "Idavidrein/gpqa",
        source_format: str = "hf",
        split: str = "train",
        subset_name: str = "gpqa_diamond",
    ) -> None:
        self.source = source
        self.source_format = source_format
        self.split = split
        self.subset_name = subset_name

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        yielded = 0
        load_kwargs: dict[str, Any] = {}
        if self.source_format == "hf":
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
        # Fixture form (our sample jsonl): {id, question, choices: [...], correct_index}
        # HF form (Idavidrein/gpqa):         {Record ID, Question, Correct Answer,
        #                                      Incorrect Answer 1, Incorrect Answer 2,
        #                                      Incorrect Answer 3, ...}
        if "choices" in row and "correct_index" in row:
            # Fixture form — use as-is.
            qid = str(row.get("id") or row["question"][:32])
            question = row["question"]
            choices = list(row["choices"])
            correct_index = int(row["correct_index"])
        else:
            # HF form — build choices by shuffling the correct answer among 3 incorrect.
            import random
            qid = str(row.get("Record ID") or row.get("id") or row["Question"][:32])
            question = row["Question"]
            correct = row["Correct Answer"]
            incorrect = [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [correct, *incorrect]
            # Deterministic shuffle seeded by Record ID so results are reproducible.
            rng = random.Random(qid)
            rng.shuffle(choices)
            correct_index = choices.index(correct)

        choices_block = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(choices))
        expected = chr(ord("A") + correct_index)
        content = _PROMPT_TEMPLATE.format(question=question, choices_block=choices_block)
        return PromptSpec(
            prompt_id=f"gpqa-{qid}",
            task_id="gpqa",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=expected,
            metadata={},
        )
