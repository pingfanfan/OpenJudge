from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_run_command_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--benchmark" in result.stdout
    assert "--model" in result.stdout


def test_run_command_rejects_unknown_benchmark(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text(
        "id: test\nprovider: openai\nmodel: gpt-4o\n"
    )
    result = runner.invoke(app, [
        "run", "--track", "limit", "--benchmark", "nonexistent",
        "--model", str(model_cfg),
    ])
    assert result.exit_code != 0
    try:
        stderr_text = result.stderr or ""
    except ValueError:
        stderr_text = ""
    assert "nonexistent" in (result.stdout + stderr_text)


def test_run_command_invokes_limit_runner(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text(
        "id: test\nprovider: openai\nmodel: gpt-4o\n"
    )
    fake_result = {"run_id": "run-x", "prompt_count": 2, "pass_at_1": 0.5, "total_cost_usd": 0.0}

    fixture = Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"
    with patch("prism.cli.LimitRunner") as mock_runner:
        instance = mock_runner.return_value
        instance.run = AsyncMock(return_value=fake_result)

        result = runner.invoke(app, [
            "run", "--track", "limit", "--benchmark", "mmlu_pro",
            "--model", str(model_cfg),
            "--work-dir", str(tmp_path),
            "--benchmark-source", str(fixture),
            "--benchmark-format", "jsonl",
        ])

    assert result.exit_code == 0
    assert "prompt_count" in result.stdout
    assert "0.5" in result.stdout


def test_run_command_accepts_judge_model(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text("id: test\nprovider: openai\nmodel: gpt-4o\n")
    judge_cfg = tmp_path / "j.yaml"
    judge_cfg.write_text("id: judge\nprovider: openai\nmodel: gpt-5\n")

    from unittest.mock import AsyncMock, patch
    fake_result = {"run_id": "run-x", "prompt_count": 1, "pass_at_1": 1.0, "total_cost_usd": 0.0}

    with patch("prism.cli.LimitRunner") as MockRunner, \
         patch("prism.cli.LiteLLMAdapter") as MockAdapter:  # noqa: N806
        instance = MockRunner.return_value
        instance.run = AsyncMock(return_value=fake_result)

        result = runner.invoke(app, [
            "run", "--track", "limit", "--benchmark", "mmlu_pro",
            "--model", str(model_cfg),
            "--judge-model", str(judge_cfg),
            "--work-dir", str(tmp_path),
            "--benchmark-source",
            str(Path(__file__).parent.parent / "fixtures" / "mmlu_pro_sample.jsonl"),
            "--benchmark-format", "jsonl",
        ])

    assert result.exit_code == 0
    # LiteLLMAdapter was constructed twice: once for subject model, once for judge model.
    assert MockAdapter.call_count == 2
    # LimitRunner.run was called with judge_adapter kwarg.
    call_kwargs = instance.run.call_args.kwargs
    assert "judge_adapter" in call_kwargs
    assert call_kwargs["judge_adapter"] is not None


def test_run_agent_track_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_run_agent_track_invokes_agent_runner(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text("id: test\nprovider: openai\nmodel: x\n")

    from unittest.mock import AsyncMock, patch
    fake_result = {"run_id": "run-x", "task_count": 2, "success_rate": 1.0, "total_cost_usd": 0.0}

    with patch("prism.cli.AgentRunner") as MockRunner, \
         patch("prism.cli.LiteLLMAdapter") as _MockAdapter:
        instance = MockRunner.return_value
        instance.run = AsyncMock(return_value=fake_result)

        result = runner.invoke(app, [
            "run", "--track", "agent", "--benchmark", "toy_agent",
            "--model", str(model_cfg),
            "--work-dir", str(tmp_path),
        ])

    assert result.exit_code == 0, result.stdout
    assert "success_rate" in result.stdout
    assert "1.0" in result.stdout


def test_run_agent_unknown_benchmark(tmp_path: Path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text("id: test\nprovider: openai\nmodel: x\n")
    result = runner.invoke(app, [
        "run", "--track", "agent", "--benchmark", "nonexistent",
        "--model", str(model_cfg),
        "--work-dir", str(tmp_path),
    ])
    assert result.exit_code != 0
