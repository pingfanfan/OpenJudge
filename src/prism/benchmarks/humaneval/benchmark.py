from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec

if TYPE_CHECKING:
    from prism.adapters.base import Adapter
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.code_exec import PytestJudge

_PROMPT_TEMPLATE = (
    "Complete the following Python function. Return ONLY the complete function"
    " implementation in a ```python code block. Do not include any explanation text.\n"
    "\n"
    "{prompt}"
)

# Wrap HumanEval's check(<entry_point>) trailing call in a pytest test for the PytestJudge harness.
_TEST_WRAPPER_TEMPLATE = """
from solution import {entry_point}

{raw_test}

def test_humaneval():
    # The raw test ends with `check({entry_point})` — it already asserts when imported.
    # We run it again here to trigger pytest discovery.
    check({entry_point})
"""


class HumanEvalBenchmark(Benchmark):
    name = "humaneval"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 50, "standard": 164, "full": None}

    def __init__(
        self,
        *,
        source: str = "evalplus/humanevalplus",
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
        return PytestJudge(
            test_code=prompt.metadata["test_code"],
            timeout_sec=30.0,
        )

    @staticmethod
    def _row_to_prompt(row: dict[str, Any]) -> PromptSpec:
        task_id = row["task_id"]
        entry_point = row["entry_point"]
        raw_test = row["test"]
        test_code = _TEST_WRAPPER_TEMPLATE.format(entry_point=entry_point, raw_test=raw_test)
        content = _PROMPT_TEMPLATE.format(prompt=row["prompt"])
        return PromptSpec(
            prompt_id=f"humaneval-{task_id}",
            task_id="humaneval",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=row.get("canonical_solution"),
            metadata={
                "entry_point": entry_point,
                "test_code": test_code,
                "raw_test": raw_test,
            },
        )
