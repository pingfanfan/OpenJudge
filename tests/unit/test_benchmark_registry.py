import pytest

from prism.benchmarks.base import Benchmark, BenchmarkRegistry, PromptSpec
from prism.judges.base import Judge, JudgeResult
from prism.judges.rules import ExactMatchJudge


class _FakeBenchmark(Benchmark):
    name = "fake"
    version = "v1"

    def load_prompts(self, *, subset=None):
        return iter([])

    def make_judge(self, prompt: PromptSpec) -> Judge:
        return ExactMatchJudge()


def test_register_and_lookup():
    reg = BenchmarkRegistry()
    reg.register(_FakeBenchmark)
    bm = reg.get("fake")
    assert isinstance(bm, _FakeBenchmark)


def test_lookup_missing_raises():
    reg = BenchmarkRegistry()
    with pytest.raises(KeyError, match="unknown"):
        reg.get("unknown")


def test_duplicate_register_raises():
    reg = BenchmarkRegistry()
    reg.register(_FakeBenchmark)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_FakeBenchmark)


def test_list_names():
    reg = BenchmarkRegistry()
    reg.register(_FakeBenchmark)
    assert list(reg.names()) == ["fake"]
