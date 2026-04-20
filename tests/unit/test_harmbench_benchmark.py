from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prism.benchmarks.harmbench.benchmark import HarmBenchBenchmark
from prism.judges.llm import LLMJudge
from prism.judges.safety_rubrics import HARM_REFUSAL_RUBRIC


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "harmbench_sample.jsonl"
    bm = HarmBenchBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "harmbench"
    assert first.prompt_id == "harmbench-hb-placeholder-1"
    assert "[PLACEHOLDER" in first.messages[0]["content"]
    assert first.expected == "test_placeholder"


def test_needs_llm_judge():
    assert HarmBenchBenchmark.needs_llm_judge is True


def test_make_judge_uses_harm_refusal_rubric():
    fixture = Path(__file__).parent.parent / "fixtures" / "harmbench_sample.jsonl"
    bm = HarmBenchBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    fake_adapter = MagicMock()
    judge = bm.make_judge(prompt, llm_judge_adapter=fake_adapter)
    assert isinstance(judge, LLMJudge)
    assert judge.rubric == HARM_REFUSAL_RUBRIC


def test_make_judge_requires_adapter():
    fixture = Path(__file__).parent.parent / "fixtures" / "harmbench_sample.jsonl"
    bm = HarmBenchBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    with pytest.raises(ValueError, match="llm_judge_adapter"):
        bm.make_judge(prompt, llm_judge_adapter=None)
