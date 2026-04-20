from pathlib import Path

from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark
from prism.judges.rules import RegexJudge


def test_load_prompts_from_local_fixture():
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "mmlu_pro"
    assert first.expected == "B"
    assert "What is 2+2?" in first.messages[0]["content"]
    assert "A. 3" in first.messages[0]["content"]
    assert "D. 6" in first.messages[0]["content"]
    assert first.prompt_id == "mmlu_pro-q1"


def test_make_judge_is_regex():
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    prompt = prompts[0]
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)


def test_subset_size_limits_output():
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    # Override subset_caps via a tiny subclass to cap at 1.
    class OneCap(MMLUProBenchmark):
        subset_caps = {"quick": 1, "standard": 1, "full": None}
    bm = OneCap(source=str(fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="quick"))
    assert len(prompts) == 1
