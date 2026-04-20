from pathlib import Path

from prism.benchmarks.truthfulqa.benchmark import TruthfulQABenchmark
from prism.judges.rules import RegexJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "truthfulqa_sample.jsonl"
    bm = TruthfulQABenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "truthfulqa"
    # First choice is labeled 1 (correct) → expected = "A"
    assert first.expected == "A"
    assert "watermelon seeds" in first.messages[0]["content"]
    # Verify all 4 choices are rendered
    for letter in "ABCD":
        assert f"{letter}." in first.messages[0]["content"]


def test_judge_is_regex():
    fixture = Path(__file__).parent.parent / "fixtures" / "truthfulqa_sample.jsonl"
    bm = TruthfulQABenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)
