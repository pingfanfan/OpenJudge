from pathlib import Path

from prism.benchmarks.livecodebench.benchmark import LiveCodeBenchBenchmark
from prism.judges.code_exec import PytestJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "livecodebench_sample.jsonl"
    bm = LiveCodeBenchBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 1
    p = prompts[0]
    assert p.task_id == "livecodebench"
    assert "Sum Two Numbers" in p.messages[0]["content"]
    assert p.metadata["entry_point"] == "sum_two"
    assert "sum_two(1, 2) == 3" in p.metadata["test_code"]


def test_judge_is_pytest():
    fixture = Path(__file__).parent.parent / "fixtures" / "livecodebench_sample.jsonl"
    bm = LiveCodeBenchBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, PytestJudge)
