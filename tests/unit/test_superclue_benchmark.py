import json
from pathlib import Path

import pytest

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


def test_load_prompts_accepts_choices_list_form(tmp_path):
    """SuperCLUE HF subsets sometimes use 'choices: [...]' instead of A/B/C/D keys."""
    alt_fixture = tmp_path / "alt.jsonl"
    alt_fixture.write_text(
        json.dumps({
            "id": "sclue-alt-1",
            "question": "测试题",
            "choices": ["甲", "乙", "丙", "丁"],
            "answer": "A",
        }) + "\n"
    )
    bm = SuperCLUEBenchmark(source=str(alt_fixture), source_format="jsonl")
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 1
    assert prompts[0].expected == "A"
    assert "A. 甲" in prompts[0].messages[0]["content"]
    assert "D. 丁" in prompts[0].messages[0]["content"]


def test_load_prompts_missing_schema_raises(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "x", "question": "Q", "answer": "A"}) + "\n"  # no choices
    )
    bm = SuperCLUEBenchmark(source=str(bad), source_format="jsonl")
    with pytest.raises(ValueError, match="missing choices"):
        list(bm.load_prompts(subset="full"))
