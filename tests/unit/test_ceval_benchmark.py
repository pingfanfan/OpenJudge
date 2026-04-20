from pathlib import Path

from prism.benchmarks.ceval.benchmark import CEvalBenchmark
from prism.judges.rules import RegexJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "ceval_sample.jsonl"
    bm = CEvalBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "ceval"
    assert first.expected == "B"
    assert "原子序数" in first.messages[0]["content"]
    assert "A. 氢" in first.messages[0]["content"]
    assert first.prompt_id == "ceval-ceval-1"


def test_judge_is_regex():
    fixture = Path(__file__).parent.parent / "fixtures" / "ceval_sample.jsonl"
    bm = CEvalBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)
