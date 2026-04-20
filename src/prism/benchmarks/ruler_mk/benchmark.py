"""RULER Multi-Key NIAH benchmark.

The haystack embeds 3 distinct "key: value" records at spread-out positions,
and the question asks for the value of one specific key.
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


_PROMPT_TEMPLATE = """Below is a long document containing multiple key-value records. Read it, then answer the question.

<document>
{haystack}
</document>

Question: What is the value associated with the key "{queried_key}"?

Respond with just the value on the last line, prefixed by "Answer:". For example: "Answer: XYZ123"."""

_JUDGE_PATTERN = r"Answer:\s*([A-Z0-9_-]{4,})\b"

_KEYS = [
    "alpha_key", "beta_key", "gamma_key", "delta_key", "epsilon_key",
    "zeta_key", "eta_key", "theta_key",
]
_VALUES = [
    "REDFOX_01", "BLUEJAY_02", "GREENOWL_03", "YELLOWBAT_04",
    "BLACKWOLF_05", "WHITECROW_06", "ORANGEMOTH_07", "SILVERCAT_08",
    "GOLDBEAR_09", "COPPERDEER_10", "BRONZEEAGLE_11", "CARBONMOOSE_12",
]


class RulerMKBenchmark(Benchmark):
    name = "ruler_mk"
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
        self.lengths = lengths if lengths is not None else [1024, 4096, 16384, 65536, 262144, 1048576]
        self.depths = depths if depths is not None else [0.25, 0.5, 0.75]
        self.seed = seed

    def load_prompts(self, *, subset: str | None = None) -> Iterable[PromptSpec]:
        cap = self._cap_for(subset)
        rng = random.Random(self.seed)
        yielded = 0
        for length in self.lengths:
            for depth in self.depths:
                if cap is not None and yielded >= cap:
                    return
                yield self._build_prompt(length=length, depth=depth, rng=rng)
                yielded += 1

    def make_judge(
        self,
        prompt: PromptSpec,
        *,
        llm_judge_adapter: "Adapter | None" = None,
    ) -> Judge:
        return RegexJudge(pattern=_JUDGE_PATTERN)

    def _build_prompt(
        self, *, length: int, depth: float, rng: random.Random
    ) -> PromptSpec:
        keys = rng.sample(_KEYS, 3)
        values = rng.sample(_VALUES, 3)
        pairs = list(zip(keys, values, strict=True))
        queried_key, queried_value = pairs[0]

        needle_block = "\n".join(f"The {k}'s value is {v}." for k, v in pairs)
        haystack = build_haystack(
            target_tokens=length, needle=needle_block, needle_depth=depth
        )
        content = _PROMPT_TEMPLATE.format(haystack=haystack, queried_key=queried_key)
        depth_pct = int(depth * 100)
        prompt_id = f"ruler_mk-len{length}-depth{depth_pct:02d}-{queried_key}"
        return PromptSpec(
            prompt_id=prompt_id,
            task_id="ruler_mk",
            version="v1",
            messages=[{"role": "user", "content": content}],
            expected=queried_value,
            metadata={
                "context_tokens": length,
                "needle_depth": depth,
                "queried_key": queried_key,
            },
        )
