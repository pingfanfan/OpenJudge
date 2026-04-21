from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prism.benchmarks.math500.benchmark import MATH500Benchmark
from prism.judges.llm import LLMJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "math500_sample.jsonl"
    bm = MATH500Benchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    assert prompts[0].expected == "13"
    assert "f(x) = 2x + 3" in prompts[0].messages[0]["content"]
    assert prompts[0].prompt_id == "math500-math-1"


def test_needs_llm_judge():
    assert MATH500Benchmark.needs_llm_judge is True


def test_make_judge_requires_adapter():
    fixture = Path(__file__).parent.parent / "fixtures" / "math500_sample.jsonl"
    bm = MATH500Benchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    with pytest.raises(ValueError, match="llm_judge_adapter"):
        bm.make_judge(prompt, llm_judge_adapter=None)


def test_make_judge_returns_llm_judge():
    fixture = Path(__file__).parent.parent / "fixtures" / "math500_sample.jsonl"
    bm = MATH500Benchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt, llm_judge_adapter=MagicMock())
    assert isinstance(judge, LLMJudge)
