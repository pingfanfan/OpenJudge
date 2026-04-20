from prism.adapters.reasoning_translator import translate
from prism.config.model_profile import ModelProfile, Thinking


def test_anthropic_thinking_max():
    profile = ModelProfile(
        id="x",
        provider="anthropic",
        model="claude-opus-4-7",
        thinking=Thinking(enabled=True, effort="max"),
    )
    extra = translate(profile)
    assert extra == {
        "thinking": {"type": "enabled"},
        "output_config": {"effort": "max"},
    }


def test_anthropic_thinking_disabled():
    profile = ModelProfile(
        id="x",
        provider="anthropic",
        model="claude-opus-4-7",
        thinking=Thinking(enabled=False, effort="high"),
    )
    extra = translate(profile)
    assert extra == {"thinking": {"type": "disabled"}}


def test_openai_reasoning_effort():
    profile = ModelProfile(
        id="x",
        provider="openai",
        model="gpt-5",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra == {"reasoning_effort": "high"}


def test_openai_no_effort_empty():
    profile = ModelProfile(
        id="x",
        provider="openai",
        model="gpt-4o",
    )
    extra = translate(profile)
    assert extra == {}


def test_google_thinking_budget():
    profile = ModelProfile(
        id="x",
        provider="google",
        model="gemini-2.5-pro",
        reasoning_effort="max",
    )
    extra = translate(profile)
    assert extra == {"thinkingConfig": {"thinkingBudget": 32768}}


def test_deepseek_reasoning_flag():
    profile = ModelProfile(
        id="x",
        provider="deepseek",
        model="deepseek-r1",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra == {"reasoning": True}


def test_unsupported_provider_returns_empty():
    profile = ModelProfile(
        id="x",
        provider="custom",
        model="my-local",
        reasoning_effort="high",
    )
    extra = translate(profile)
    assert extra == {}


def test_anthropic_reasoning_effort_fallback():
    """When no Thinking object is passed but reasoning_effort is set, enable thinking."""
    profile = ModelProfile(
        id="x",
        provider="anthropic",
        model="claude-opus-4-7",
        reasoning_effort="medium",
    )
    extra = translate(profile)
    assert extra == {
        "thinking": {"type": "enabled"},
        "output_config": {"effort": "medium"},
    }


def test_google_default_effort():
    """When no reasoning_effort is set, Google defaults to 'high' (budget 16384)."""
    profile = ModelProfile(
        id="x",
        provider="google",
        model="gemini-2.5-pro",
    )
    extra = translate(profile)
    assert extra == {"thinkingConfig": {"thinkingBudget": 16384}}


def test_google_effort_off():
    """reasoning_effort='off' on Google emits thinkingBudget=0.

    Gemini's 'disable thinking' signal.
    """
    profile = ModelProfile(
        id="x",
        provider="google",
        model="gemini-2.5-pro",
        reasoning_effort="off",
    )
    extra = translate(profile)
    assert extra == {"thinkingConfig": {"thinkingBudget": 0}}


def test_deepseek_effort_off():
    """reasoning_effort='off' on DeepSeek suppresses the reasoning flag."""
    profile = ModelProfile(
        id="x",
        provider="deepseek",
        model="deepseek-r1",
        reasoning_effort="off",
    )
    extra = translate(profile)
    assert extra == {}
