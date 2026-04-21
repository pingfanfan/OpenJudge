from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_list_benchmarks_exits_0():
    result = runner.invoke(app, ["list-benchmarks"])
    assert result.exit_code == 0


def test_list_benchmarks_shows_all_17_limit():
    result = runner.invoke(app, ["list-benchmarks"])
    for name in ("mmlu_pro", "aime", "humaneval", "gpqa", "math500",
                 "livecodebench", "ifeval", "ceval", "simpleqa", "truthfulqa",
                 "harmbench", "xstest", "superclue", "mmmu", "mathvista",
                 "niah", "ruler_mk"):
        assert name in result.stdout.lower(), f"missing {name}"


def test_list_benchmarks_shows_agent_track():
    result = runner.invoke(app, ["list-benchmarks"])
    assert "toy_agent" in result.stdout.lower()
    assert "agent" in result.stdout.lower()


def test_list_benchmarks_marks_judge_needed():
    """Benchmarks with needs_llm_judge=True should be visibly marked."""
    result = runner.invoke(app, ["list-benchmarks"])
    # simpleqa / harmbench / xstest need judge
    lines = result.stdout.split("\n")
    judge_marked_lines = [line for line in lines if "judge" in line.lower() and "simpleqa" in line.lower()]
    assert judge_marked_lines, "simpleqa should be marked as needing a judge"
