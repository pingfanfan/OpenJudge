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

# A representative spread across STEM / humanities / practical / language
# (8 subjects, ~20 questions each = ~160 prompts total on quick subset).
_DEFAULT_SUBJECTS = [
    "advanced_mathematics",
    "college_physics",
    "high_school_chinese",
    "chinese_language_and_literature",
    "modern_chinese_history",
    "law",
    "college_programming",
    "marxism",
]


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
        subjects: list[str] | None = None,
    ) -> None:
        """
        C-Eval is partitioned into 52 subjects. Strategy:
          * `subjects` explicitly set → iterate each in turn.
          * `subset_name` set (single subject) → use just that one.
          * Neither → use _DEFAULT_SUBJECTS (broad coverage).
        """
        self.source = source
        self.source_format = source_format
        self.split = split
        self.subset_name = subset_name
        self.subjects = subjects

    def _resolve_subjects(self) -> list[str | None]:
        """Return the list of subject configs to iterate over. `None` means
        'no name kwarg' (for fixture / jsonl paths)."""
        if self.source_format != "hf":
            # Fixture / jsonl doesn't use HF subset config.
            return [None]
        if self.subjects is not None:
            return list(self.subjects)
        if self.subset_name is not None:
            return [self.subset_name]
        return list(_DEFAULT_SUBJECTS)

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        yielded = 0
        for subject in self._resolve_subjects():
            load_kwargs: dict[str, Any] = {}
            if subject is not None:
                load_kwargs["name"] = subject
            for row in load_dataset_cached(
                source=self.source, format=self.source_format, split=self.split,
                **load_kwargs,
            ):
                if cap is not None and yielded >= cap:
                    return
                yield self._row_to_prompt(row, subject=subject)
                yielded += 1

    def make_judge(
        self,
        prompt: PromptSpec,
        *,
        llm_judge_adapter: Adapter | None = None,
    ) -> Judge:
        return RegexJudge(pattern=_JUDGE_PATTERN)

    @staticmethod
    def _row_to_prompt(row: dict[str, Any], *, subject: str | None = None) -> PromptSpec:
        qid = str(row.get("id") or row["question"][:32])
        # Prefix prompt_id with subject so multi-subject runs don't collide on
        # per-subject id=0, 1, 2, ... that C-Eval uses.
        full_id = f"{subject}-{qid}" if subject else qid
        choices_block = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
        content = _PROMPT_TEMPLATE.format(question=row["question"], choices_block=choices_block)
        return PromptSpec(
            prompt_id=f"ceval-{full_id}",
            task_id="ceval",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=row["answer"],
            metadata={"subject": subject} if subject else {},
        )
