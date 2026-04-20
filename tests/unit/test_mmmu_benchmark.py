from pathlib import Path

from prism.benchmarks.mmmu.benchmark import MMMUBenchmark
from prism.judges.rules import RegexJudge


def test_load_prompts_emits_multimodal_content():
    fixture = Path(__file__).parent.parent / "fixtures" / "mmmu_sample.jsonl"
    bm = MMMUBenchmark(
        source=str(fixture), source_format="jsonl", fixture_root=fixture.parent.parent
    )
    prompts = list(bm.load_prompts(subset="full"))
    assert len(prompts) == 2
    first = prompts[0]
    assert first.task_id == "mmmu"
    assert first.expected == "A"
    content = first.messages[0]["content"]
    assert isinstance(content, list)
    types = [p["type"] for p in content]
    assert "text" in types
    assert "image_url" in types
    img_part = next(p for p in content if p["type"] == "image_url")
    assert img_part["image_url"]["url"].startswith("data:image/png;base64,")
    text_part = next(p for p in content if p["type"] == "text")
    assert "What color" in text_part["text"]
    assert "A. Red" in text_part["text"]
    assert "D. Yellow" in text_part["text"]


def test_judge_is_regex():
    fixture = Path(__file__).parent.parent / "fixtures" / "mmmu_sample.jsonl"
    bm = MMMUBenchmark(
        source=str(fixture), source_format="jsonl", fixture_root=fixture.parent.parent
    )
    prompt = next(iter(bm.load_prompts(subset="full")))
    judge = bm.make_judge(prompt)
    assert isinstance(judge, RegexJudge)


def test_accepts_pil_image_from_hf_row():
    """Real HF MMMU returns PIL.Image in 'image' field; benchmark must handle this."""
    from PIL import Image
    bm = MMMUBenchmark()
    pil_img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    row = {
        "id": "q-hf",
        "question": "Test?",
        "options": ["A opt", "B opt", "C opt", "D opt"],
        "answer": "B",
        "image": pil_img,
    }
    spec = bm._row_to_prompt(row)
    content = spec.messages[0]["content"]
    img_part = next(p for p in content if p["type"] == "image_url")
    assert img_part["image_url"]["url"].startswith("data:image/png;base64,")
