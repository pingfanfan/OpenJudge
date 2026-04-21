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
    # P2e: doctor lists all 17 benchmarks
    for name in ("mmlu_pro", "aime", "humaneval", "gpqa", "math500",
                 "livecodebench", "ifeval", "ceval", "simpleqa", "truthfulqa",
                 "harmbench", "xstest", "superclue", "mmmu", "mathvista",
                 "niah", "ruler_mk"):
        assert name in result.stdout.lower(), f"missing benchmark {name} in doctor output"


def test_version_flag():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1" in result.stdout


def test_doctor_reports_hf_login_status():
    result = runner.invoke(app, ["doctor"])
    assert "huggingface" in result.stdout.lower() or "hf_token" in result.stdout.lower()


def test_doctor_reports_all_providers():
    result = runner.invoke(app, ["doctor"])
    # All 7 provider API keys should be mentioned
    for env in ("anthropic", "openai", "google", "deepseek", "xai", "kimi", "qwen"):
        assert env in result.stdout.lower(), f"missing provider: {env}"


def test_doctor_offers_export_hint_for_missing_keys(monkeypatch):
    """When a provider API key is unset, the doctor output should contain an
    export hint (`export XXX_API_KEY=...`) that the user can copy-paste."""
    # Conftest strips API keys by default — so we should see the hint.
    result = runner.invoke(app, ["doctor"])
    # Pick any one provider we know is unset and check its hint appears.
    assert "export" in result.stdout.lower()
