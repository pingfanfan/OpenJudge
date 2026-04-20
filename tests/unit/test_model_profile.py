from pathlib import Path

import pytest
from pydantic import ValidationError

from prism.config.loader import load_model_profile
from prism.config.model_profile import Cost, ModelProfile, Thinking


def test_minimal_profile_parses():
    profile = ModelProfile(
        id="gpt-5@high",
        provider="openai",
        model="gpt-5",
        reasoning_effort="high",
    )
    assert profile.id == "gpt-5@high"
    assert profile.reasoning_effort == "high"
    assert profile.thinking is None


def test_anthropic_thinking_profile():
    profile = ModelProfile(
        id="claude-opus-4-7@max",
        provider="anthropic",
        model="claude-opus-4-7",
        thinking=Thinking(enabled=True, effort="max"),
    )
    assert profile.thinking.enabled is True
    assert profile.thinking.effort == "max"


def test_invalid_effort_rejected():
    with pytest.raises(ValidationError, match="reasoning_effort"):
        ModelProfile(
            id="x",
            provider="openai",
            model="x",
            reasoning_effort="super-mega",
        )


def test_negative_cost_rejected():
    with pytest.raises(ValidationError, match="input_per_mtok"):
        Cost(input_per_mtok=-1.0)


def test_load_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "opus.yaml"
    yaml_path.write_text(
        "id: claude-opus-4-7@max\n"
        "display_name: Claude Opus 4.7 (max)\n"
        "provider: anthropic\n"
        "model: claude-opus-4-7\n"
        "thinking:\n"
        "  enabled: true\n"
        "  effort: max\n"
        "rate_limit:\n"
        "  rpm: 50\n"
        "  tpm: 400000\n"
        "cost:\n"
        "  input_per_mtok: 15.0\n"
        "  output_per_mtok: 75.0\n"
    )
    profile = load_model_profile(yaml_path)
    assert profile.id == "claude-opus-4-7@max"
    assert profile.rate_limit.rpm == 50
    assert profile.cost.output_per_mtok == 75.0
