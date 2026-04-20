from prism.adapters.reasoning_translator import translate
from prism.config.model_profile import ModelProfile, Thinking


def test_anthropic_thinking_max():
    profile = ModelProfile(
        id="x", provider="anthropic", model="claude-opus-4-7",
        thinking=Thinking(enabled=True, effort="max"),
    )
    extra = translate(profile)
    assert extra["thinking"] == {"type": "enabled"}
    assert extra["output_config"] == {"effort": "max"}


def test_anthropic_thinking_disabled():
    profile = ModelProfile(
        id="x", provider="anthropic", model="claude-opus-4-7",
        thinking=Thinking(enabled=False, effort="high"),
    )
    extra = translate(profile)
    assert extra["thinking"] == {"type": "disabled"}
    assert "output_config" not in extra


def test_openai_reasoning_effort():
    profile = ModelProfile(
        id="x", provider="openai", model="gpt-5",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra["reasoning_effort"] == "high"


def test_openai_no_effort_empty():
    profile = ModelProfile(
        id="x", provider="openai", model="gpt-4o",
    )
    extra = translate(profile)
    assert extra == {}


def test_google_thinking_budget():
    profile = ModelProfile(
        id="x", provider="google", model="gemini-2.5-pro",
        reasoning_effort="max",
    )
    extra = translate(profile)
    assert "thinkingConfig" in extra
    assert extra["thinkingConfig"]["thinkingBudget"] >= 32768


def test_deepseek_reasoning_flag():
    profile = ModelProfile(
        id="x", provider="deepseek", model="deepseek-r1",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra.get("reasoning") is True


def test_unsupported_provider_returns_empty():
    profile = ModelProfile(
        id="x", provider="custom", model="my-local",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra == {}
