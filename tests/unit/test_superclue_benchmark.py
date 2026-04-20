from pathlib import Path

from prism.benchmarks.superclue.benchmark import SuperCLUEBenchmark
from prism.judges.rules import RegexJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "superclue_sample.jsonl"
    bm = SuperCLUEBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "superclue"
    assert first.expected == "B"
    assert "中国的首都" in first.messages[0]["content"]
    assert "A. 上海" in first.messages[0]["content"]


def test_judge_is_regex():
    fixture = Path(__file__).parent.parent / "fixtures" / "superclue_sample.jsonl"
    bm = SuperCLUEBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)
