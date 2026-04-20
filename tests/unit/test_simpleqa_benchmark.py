from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prism.benchmarks.simpleqa.benchmark import SimpleQABenchmark
from prism.judges.llm import LLMJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "simpleqa_sample.jsonl"
    bm = SimpleQABenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    assert prompts[0].expected == "Paris"
    assert "capital of France" in prompts[0].messages[0]["content"]


def test_needs_llm_judge():
    assert SimpleQABenchmark.needs_llm_judge is True


def test_make_judge_requires_adapter():
    fixture = Path(__file__).parent.parent / "fixtures" / "simpleqa_sample.jsonl"
    bm = SimpleQABenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    with pytest.raises(ValueError, match="llm_judge_adapter"):
        bm.make_judge(prompt, llm_judge_adapter=None)


def test_make_judge_returns_llm_judge():
    fixture = Path(__file__).parent.parent / "fixtures" / "simpleqa_sample.jsonl"
    bm = SimpleQABenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    fake_adapter = MagicMock()
    judge = bm.make_judge(prompt, llm_judge_adapter=fake_adapter)
    assert isinstance(judge, LLMJudge)
    assert judge.adapter is fake_adapter
