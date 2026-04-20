from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prism.benchmarks.xstest.benchmark import XSTestBenchmark
from prism.judges.llm import LLMJudge
from prism.judges.safety_rubrics import OVER_REFUSAL_RUBRIC


def test_load_prompts_filters_safe_only_by_default():
    fixture = Path(__file__).parent.parent / "fixtures" / "xstest_sample.jsonl"
    bm = XSTestBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "xstest"
    assert "kill a Python process" in first.messages[0]["content"]
    assert first.expected == "programming"


def test_needs_llm_judge():
    assert XSTestBenchmark.needs_llm_judge is True


def test_make_judge_uses_over_refusal_rubric():
    fixture = Path(__file__).parent.parent / "fixtures" / "xstest_sample.jsonl"
    bm = XSTestBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt, llm_judge_adapter=MagicMock())
    assert isinstance(judge, LLMJudge)
    assert judge.rubric == OVER_REFUSAL_RUBRIC


def test_make_judge_requires_adapter():
    fixture = Path(__file__).parent.parent / "fixtures" / "xstest_sample.jsonl"
    bm = XSTestBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    with pytest.raises(ValueError, match="llm_judge_adapter"):
        bm.make_judge(prompt, llm_judge_adapter=None)
