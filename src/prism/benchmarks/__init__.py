from prism.benchmarks.base import Benchmark, BenchmarkRegistry, PromptSpec


def default_registry() -> BenchmarkRegistry:
    """Build a fresh registry pre-populated with P2a benchmarks."""
    from prism.benchmarks.aime.benchmark import AIMEBenchmark
    from prism.benchmarks.humaneval.benchmark import HumanEvalBenchmark
    from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark

    reg = BenchmarkRegistry()
    reg.register(MMLUProBenchmark)
    reg.register(AIMEBenchmark)
    reg.register(HumanEvalBenchmark)
    return reg


__all__ = ["Benchmark", "BenchmarkRegistry", "PromptSpec", "default_registry"]
