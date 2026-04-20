from pathlib import Path

from prism.benchmarks.humaneval.benchmark import HumanEvalBenchmark
from prism.judges.code_exec import PytestJudge


def test_load_prompts():
    fixture = Path(__file__).parent.parent / "fixtures" / "humaneval_sample.jsonl"
    bm = HumanEvalBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts())
    assert len(prompts) == 1
    p = prompts[0]
    assert p.task_id == "humaneval"
    assert p.prompt_id == "humaneval-HumanEval/0"
    assert "def add(a, b)" in p.messages[0]["content"]
    assert p.metadata["entry_point"] == "add"
    assert "def check" in p.metadata["test_code"]


def test_judge_is_pytest():
    fixture = Path(__file__).parent.parent / "fixtures" / "humaneval_sample.jsonl"
    bm = HumanEvalBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts()))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, PytestJudge)
