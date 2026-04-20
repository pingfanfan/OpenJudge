from pathlib import Path

from prism.benchmarks.math500.benchmark import MATH500Benchmark
from prism.judges.rules import NumericJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "math500_sample.jsonl"
    bm = MATH500Benchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    assert prompts[0].expected == "13"
    assert "f(x) = 2x + 3" in prompts[0].messages[0]["content"]
    assert prompts[0].prompt_id == "math500-math-1"


def test_judge_is_numeric():
    fixture = Path(__file__).parent.parent / "fixtures" / "math500_sample.jsonl"
    bm = MATH500Benchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, NumericJudge)
