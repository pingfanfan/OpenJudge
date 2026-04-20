from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "prism" in result.stdout.lower()


def test_doctor_reports_python():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "python" in result.stdout.lower()
    assert "litellm" in result.stdout.lower()
    # P2a additions:
    assert "benchmarks" in result.stdout.lower()
    # P2b: doctor lists all 10 benchmarks
    for name in ("mmlu_pro", "aime", "humaneval", "gpqa", "math500",
                 "livecodebench", "ifeval", "ceval", "simpleqa", "truthfulqa"):
        assert name in result.stdout.lower(), f"missing benchmark {name} in doctor output"


def test_version_flag():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1" in result.stdout
