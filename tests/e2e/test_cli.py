from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "prism" in result.stdout.lower()


def test_doctor_reports_python():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)  # may be 1 if missing API keys
    assert "python" in result.stdout.lower()
    assert "litellm" in result.stdout.lower()


def test_version_flag():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1" in result.stdout
