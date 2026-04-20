from pathlib import Path

import pytest

from prism.benchmarks.aime.benchmark import AIMEBenchmark
from prism.judges.rules import NumericJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "aime_sample.jsonl"
    bm = AIMEBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts())
    assert len(prompts) == 2
    assert prompts[0].task_id == "aime"
    assert prompts[0].expected == "42"
    assert "smallest positive integer" in prompts[0].messages[0]["content"]
    assert prompts[0].prompt_id == "aime-aime-2024-1"


def test_judge_is_numeric():
    fixture = Path(__file__).parent.parent / "fixtures" / "aime_sample.jsonl"
    bm = AIMEBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts()))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, NumericJudge)


@pytest.mark.asyncio
async def test_judge_accepts_trailing_answer():
    fixture = Path(__file__).parent.parent / "fixtures" / "aime_sample.jsonl"
    bm = AIMEBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts()))
    judge = bm.make_judge(prompt)
    # The numeric judge extracts the LAST number in the output, so trailing reasoning works.
    output = "After computation we find n = 41, but actually rechecking gives 42."
    result = await judge.judge(output=output, expected=prompt.expected)
    assert result.score == 1.0
