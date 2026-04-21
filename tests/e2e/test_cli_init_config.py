from pathlib import Path

import yaml
from typer.testing import CliRunner

from prism.cli import app

runner = CliRunner()


def test_init_config_basic_openai(tmp_path: Path):
    output = tmp_path / "m.yaml"
    result = runner.invoke(app, [
        "init-config",
        "--provider", "openai",
        "--model", "gpt-5",
        "--output", str(output),
    ])
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    data = yaml.safe_load(output.read_text())
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-5"
    assert "id" in data


def test_init_config_anthropic_includes_thinking_block(tmp_path: Path):
    output = tmp_path / "m.yaml"
    result = runner.invoke(app, [
        "init-config",
        "--provider", "anthropic",
        "--model", "claude-opus-4-7",
        "--output", str(output),
    ])
    assert result.exit_code == 0
    data = yaml.safe_load(output.read_text())
    assert data["provider"] == "anthropic"
    assert "thinking" in data


def test_init_config_with_api_base(tmp_path: Path):
    output = tmp_path / "m.yaml"
    result = runner.invoke(app, [
        "init-config",
        "--provider", "anthropic",
        "--model", "custom-model",
        "--api-base", "https://my-proxy.example.com/v1",
        "--id", "my-custom",
        "--output", str(output),
    ])
    assert result.exit_code == 0
    data = yaml.safe_load(output.read_text())
    assert data["api_base"] == "https://my-proxy.example.com/v1"
    assert data["id"] == "my-custom"


def test_init_config_refuses_to_overwrite(tmp_path: Path):
    output = tmp_path / "existing.yaml"
    output.write_text("id: existing\n")
    result = runner.invoke(app, [
        "init-config",
        "--provider", "openai",
        "--model", "gpt-5",
        "--output", str(output),
    ])
    # Should refuse without --force
    assert result.exit_code != 0


def test_init_config_force_overwrites(tmp_path: Path):
    output = tmp_path / "existing.yaml"
    output.write_text("id: existing\n")
    result = runner.invoke(app, [
        "init-config",
        "--provider", "openai",
        "--model", "gpt-5",
        "--output", str(output),
        "--force",
    ])
    assert result.exit_code == 0
    data = yaml.safe_load(output.read_text())
    assert data["provider"] == "openai"


def test_init_config_generated_yaml_loads_as_valid_profile(tmp_path: Path):
    """The generated YAML must be loadable by load_model_profile without errors."""
    from prism.config.loader import load_model_profile

    output = tmp_path / "m.yaml"
    runner.invoke(app, [
        "init-config",
        "--provider", "openai",
        "--model", "gpt-5",
        "--output", str(output),
    ])
    profile = load_model_profile(output)
    assert profile.provider == "openai"
    assert profile.model == "gpt-5"
