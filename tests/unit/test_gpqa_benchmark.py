from pathlib import Path

from prism.benchmarks.gpqa.benchmark import GPQABenchmark
from prism.judges.rules import RegexJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "gpqa_sample.jsonl"
    bm = GPQABenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "gpqa"
    assert first.expected == "B"  # correct_index 1 → letter B
    assert "electrons in orbit" in first.messages[0]["content"]
    assert "A. Gravitational" in first.messages[0]["content"]
    assert first.prompt_id == "gpqa-gpqa-q1"


def test_judge_is_regex():
    fixture = Path(__file__).parent.parent / "fixtures" / "gpqa_sample.jsonl"
    bm = GPQABenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)
