from typing import Any

from prism.config.model_profile import Effort, ModelProfile

_GOOGLE_BUDGET: dict[Effort, int] = {
    "off": 0,
    "low": 1024,
    "medium": 8192,
    "high": 16384,
    "max": 32768,
}


def translate(profile: ModelProfile) -> dict[str, Any]:
    """Return provider-specific extra kwargs to pass through LiteLLM."""
    extra: dict[str, Any] = {}
    provider = profile.provider

    if provider == "anthropic":
        if profile.thinking is not None:
            if profile.thinking.enabled:
                extra["thinking"] = {"type": "enabled"}
                extra["output_config"] = {"effort": profile.thinking.effort}
            else:
                extra["thinking"] = {"type": "disabled"}
        elif profile.reasoning_effort:
            extra["thinking"] = {"type": "enabled"}
            extra["output_config"] = {"effort": profile.reasoning_effort}

    elif provider == "openai":
        if profile.reasoning_effort:
            extra["reasoning_effort"] = profile.reasoning_effort

    elif provider == "google":
        # Budget 0 is how Gemini signals "disable thinking" (no separate boolean switch).
        effort = profile.reasoning_effort or "high"
        extra["thinkingConfig"] = {"thinkingBudget": _GOOGLE_BUDGET[effort]}

    elif provider == "deepseek":
        if profile.reasoning_effort and profile.reasoning_effort != "off":
            extra["reasoning"] = True

    elif provider in ("xai", "kimi", "qwen"):
        if profile.reasoning_effort:
            extra["reasoning_effort"] = profile.reasoning_effort

    return extra
