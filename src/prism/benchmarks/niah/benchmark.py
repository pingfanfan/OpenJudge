"""Needle-in-a-Haystack (single-needle) benchmark.

Each benchmark instance emits N×M prompts where
  N = len(lengths) is the number of context-length points on the staircase, and
  M = len(depths) is the number of needle-depth positions tested.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING

from prism.benchmarks.base import Benchmark, PromptSpec
from prism.judges.base import Judge
from prism.judges.rules import RegexJudge
from prism.utils.haystack import build_haystack

if TYPE_CHECKING:
    from prism.adapters.base import Adapter


_PROMPT_TEMPLATE = (
    "Below is a long document. Read it carefully, then answer the question"
    " at the end.\n\n<document>\n{haystack}\n</document>\n\n"
    "Question: What is the special passcode mentioned in the document?\n\n"
    'Respond with just the passcode on the last line, prefixed by "Answer:".'
    ' For example: "Answer: ABC123".'
)

_JUDGE_PATTERN = r"Answer:\s*([A-Z0-9_-]{6,})\b"

_NEEDLE_CODES = [
    "HONEYBADGER_9412",
    "PURPLE_ELEPHANT_42",
    "QUANTUM_LIZARD_7731",
    "SILVER_TURTLE_88",
    "NEON_DRAGON_5521",
    "BRONZE_PHOENIX_33",
    "CRYSTAL_WHALE_616",
    "OBSIDIAN_HAWK_909",
    "EMERALD_JAGUAR_1024",
    "CARBON_FALCON_2048",
    "PLATINUM_SHARK_4096",
    "TITANIUM_LYNX_8192",
]


class NIAHBenchmark(Benchmark):
    name = "niah"
    track = "limit"
    version = "v1"
    subset_caps = {"quick": 6, "standard": 12, "full": None}

    def __init__(
        self,
        *,
        lengths: list[int] | None = None,
        depths: list[float] | None = None,
        seed: int = 42,
    ) -> None:
        self.lengths = (
            lengths if lengths is not None
            else [1024, 4096, 16384, 65536, 262144, 1048576]
        )
        self.depths = depths if depths is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        self.seed = seed

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        rng = random.Random(self.seed)
        yielded = 0
        for length in self.lengths:
            for depth in self.depths:
                if cap is not None and yielded >= cap:
                    return
                needle_code = _NEEDLE_CODES[rng.randrange(len(_NEEDLE_CODES))]
                yield self._build_prompt(length=length, depth=depth, needle_code=needle_code)
                yielded += 1

    def make_judge(
        self,
        prompt: PromptSpec,
        *,
        llm_judge_adapter: Adapter | None = None,
    ) -> Judge:
        return RegexJudge(pattern=_JUDGE_PATTERN)

    @staticmethod
    def _build_prompt(*, length: int, depth: float, needle_code: str) -> PromptSpec:
        needle_phrase = f"The special passcode is {needle_code}."
        haystack = build_haystack(
            target_tokens=length, needle=needle_phrase, needle_depth=depth
        )
        content = _PROMPT_TEMPLATE.format(haystack=haystack)
        depth_pct = int(depth * 100)
        prompt_id = f"niah-len{length}-depth{depth_pct:02d}"
        return PromptSpec(
            prompt_id=prompt_id,
            task_id="niah",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=needle_code,
            metadata={"context_tokens": length, "needle_depth": depth},
        )
