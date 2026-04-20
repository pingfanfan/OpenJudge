from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.benchmarks.dataset_cache import load_dataset_cached
from prism.judges.base import Judge
from prism.judges.code_exec import PytestJudge

if TYPE_CHECKING:
    from prism.adapters.base import Adapter

_PROMPT_TEMPLATE = """# {title}

{description}

Return ONLY the complete Python function implementation in a ```python code block. Do not include any explanation text."""


class LiveCodeBenchBenchmark(Benchmark):
    name = "livecodebench"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 50, "standard": 150, "full": None}

    def __init__(
        self,
        *,
        source: str = "livecodebench/code_generation_lite",
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
        pid = str(row.get("problem_id") or row.get("id") or row["title"][:32])
        entry_point = row["entry_point"]
        test_cases = row["test_cases"]

        assertions = "\n".join(
            f"    assert {entry_point}({', '.join(repr(a) for a in args)}) == {repr(expected)}"
            for args, expected in test_cases
        )
        test_code = (
            f"from solution import {entry_point}\n\n"
            f"def test_livecodebench():\n{assertions}\n"
        )
        content = _PROMPT_TEMPLATE.format(
            title=row.get("title", pid), description=row["description"]
        )
        return PromptSpec(
            prompt_id=f"livecodebench-{pid}",
            task_id="livecodebench",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=None,
            metadata={"entry_point": entry_point, "test_code": test_code},
        )
