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
