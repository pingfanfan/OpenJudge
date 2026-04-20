from prism.benchmarks import default_registry


def test_default_registry_has_thirteen_benchmarks():
    names = default_registry().names()
    assert set(names) == {
        "mmlu_pro", "aime", "humaneval",
        "gpqa", "math500", "livecodebench",
        "ifeval", "ceval", "simpleqa", "truthfulqa",
        "harmbench", "xstest", "superclue",
    }


def test_default_registry_returns_fresh_instance_each_time():
    r1 = default_registry()
    r2 = default_registry()
    # default_registry() constructs fresh registry each call (no shared mutable state).
    r1_bm = r1.get("mmlu_pro")
    r2_bm = r2.get("mmlu_pro")
    assert r1_bm is not r2_bm
