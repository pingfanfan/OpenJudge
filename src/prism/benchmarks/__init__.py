from prism.benchmarks.base import Benchmark, BenchmarkRegistry, PromptSpec


def default_registry() -> BenchmarkRegistry:
    """Build a fresh registry pre-populated with all shipped benchmarks."""
    from prism.benchmarks.aime.benchmark import AIMEBenchmark
    from prism.benchmarks.ceval.benchmark import CEvalBenchmark
    from prism.benchmarks.gpqa.benchmark import GPQABenchmark
    from prism.benchmarks.harmbench.benchmark import HarmBenchBenchmark
    from prism.benchmarks.humaneval.benchmark import HumanEvalBenchmark
    from prism.benchmarks.ifeval.benchmark import IFEvalBenchmark
    from prism.benchmarks.livecodebench.benchmark import LiveCodeBenchBenchmark
    from prism.benchmarks.math500.benchmark import MATH500Benchmark
    from prism.benchmarks.mathvista.benchmark import MathVistaBenchmark
    from prism.benchmarks.mmlu_pro.benchmark import MMLUProBenchmark
    from prism.benchmarks.mmmu.benchmark import MMMUBenchmark
    from prism.benchmarks.niah.benchmark import NIAHBenchmark
    from prism.benchmarks.ruler_mk.benchmark import RulerMKBenchmark
    from prism.benchmarks.simpleqa.benchmark import SimpleQABenchmark
    from prism.benchmarks.superclue.benchmark import SuperCLUEBenchmark
    from prism.benchmarks.truthfulqa.benchmark import TruthfulQABenchmark
    from prism.benchmarks.xstest.benchmark import XSTestBenchmark

    reg = BenchmarkRegistry()
    reg.register(MMLUProBenchmark)
    reg.register(AIMEBenchmark)
    reg.register(HumanEvalBenchmark)
    reg.register(GPQABenchmark)
    reg.register(MATH500Benchmark)
    reg.register(LiveCodeBenchBenchmark)
    reg.register(IFEvalBenchmark)
    reg.register(CEvalBenchmark)
    reg.register(SimpleQABenchmark)
    reg.register(TruthfulQABenchmark)
    reg.register(HarmBenchBenchmark)
    reg.register(XSTestBenchmark)
    reg.register(SuperCLUEBenchmark)
    reg.register(MMMUBenchmark)
    reg.register(MathVistaBenchmark)
    reg.register(NIAHBenchmark)
    reg.register(RulerMKBenchmark)
    return reg


__all__ = ["Benchmark", "BenchmarkRegistry", "PromptSpec", "default_registry"]
