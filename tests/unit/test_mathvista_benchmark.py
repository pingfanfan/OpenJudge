from pathlib import Path

from prism.benchmarks.mathvista.benchmark import MathVistaBenchmark
from prism.judges.rules import NumericJudge


def test_load_prompts_filters_to_free_form_numeric():
    fixture = Path(__file__).parent.parent / "fixtures" / "mathvista_sample.jsonl"
    bm = MathVistaBenchmark(
        source=str(fixture), source_format="jsonl", fixture_root=fixture.parent.parent
    )
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    ids = [p.prompt_id for p in prompts]
    assert "mathvista-mv-1" in ids
    assert "mathvista-mv-2" in ids
    assert "mathvista-mv-skip-1" not in ids


def test_emits_multimodal_content():
    fixture = Path(__file__).parent.parent / "fixtures" / "mathvista_sample.jsonl"
    bm = MathVistaBenchmark(
        source=str(fixture), source_format="jsonl", fixture_root=fixture.parent.parent
    )
    first = next(iter(bm.load_prompts(subset="full")))
    content = first.messages[0]["content"]
    assert isinstance(content, list)
    types = [p["type"] for p in content]
    assert "text" in types
    assert "image_url" in types


def test_judge_is_numeric():
    fixture = Path(__file__).parent.parent / "fixtures" / "mathvista_sample.jsonl"
    bm = MathVistaBenchmark(
        source=str(fixture), source_format="jsonl", fixture_root=fixture.parent.parent
    )
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, NumericJudge)
