import pytest

from prism.benchmarks.base import Benchmark, PromptSpec


def test_prompt_spec_minimal():
    ps = PromptSpec(
        prompt_id="mmlu_pro-0001",
        task_id="mmlu_pro",
        version="v1",
        messages=[{"role": "user", "content": "Q: 2+2?"}],
        expected="4",
    )
    assert ps.prompt_id == "mmlu_pro-0001"
    assert ps.messages[0]["role"] == "user"
    assert ps.metadata == {}


def test_prompt_spec_with_metadata():
    ps = PromptSpec(
        prompt_id="x",
        task_id="x",
        version="v1",
        messages=[],
        expected=None,
        metadata={"choices": ["A", "B", "C", "D"]},
    )
    assert ps.metadata["choices"] == ["A", "B", "C", "D"]


def test_benchmark_is_abstract():
    with pytest.raises(TypeError):
        Benchmark()  # type: ignore[abstract]


def test_benchmark_needs_llm_judge_defaults_false():
    from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark
    assert MMLUProBenchmark.needs_llm_judge is False


def test_make_judge_accepts_llm_judge_adapter_kwarg():
    """All benchmarks must accept (but may ignore) the llm_judge_adapter kwarg."""
    from pathlib import Path
    from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark
    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    bm = MMLUProBenchmark(source=str(fixture), source_format="jsonl")
    prompt = next(iter(bm.load_prompts(subset="full")))
    # Must not raise even with extra kwarg (rule-based benchmarks ignore it).
    judge = bm.make_judge(prompt, llm_judge_adapter=None)
    assert judge is not None
