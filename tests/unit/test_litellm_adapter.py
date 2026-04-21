from unittest.mock import AsyncMock, patch

import pytest

from prism.adapters.base import AdapterRequest
from prism.adapters.litellm_adapter import LiteLLMAdapter
from prism.config.model_profile import Cost, ModelProfile, Thinking


@pytest.fixture
def anthropic_profile() -> ModelProfile:
    return ModelProfile(
        id="claude-opus-4-7@max",
        provider="anthropic",
        model="claude-opus-4-7",
        thinking=Thinking(enabled=True, effort="max"),
        cost=Cost(input_per_mtok=15.0, output_per_mtok=75.0),
    )


@pytest.fixture
def openai_profile() -> ModelProfile:
    return ModelProfile(
        id="gpt-5@high",
        provider="openai",
        model="gpt-5",
        reasoning_effort="high",
        cost=Cost(input_per_mtok=10.0, output_per_mtok=40.0),
    )


class _FakeUsage:
    def __init__(self, pt: int, ct: int) -> None:
        self.prompt_tokens = pt
        self.completion_tokens = ct


class _FakeMessage:
    def __init__(self, content: str, reasoning: str | None = None) -> None:
        self.content = content
        self.reasoning_content = reasoning


class _FakeChoice:
    def __init__(self, content: str, reasoning: str | None = None) -> None:
        self.message = _FakeMessage(content, reasoning)
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content: str, pt: int, ct: int, reasoning: str | None = None) -> None:
        self.choices = [_FakeChoice(content, reasoning)]
        self.usage = _FakeUsage(pt, ct)

    def model_dump(self) -> dict:
        return {"fake": True}


@pytest.mark.asyncio
async def test_anthropic_call_passes_thinking(anthropic_profile):
    adapter = LiteLLMAdapter(anthropic_profile)
    req = AdapterRequest(messages=[{"role": "user", "content": "2+2"}], max_output_tokens=64)
    fake = _FakeResponse("4", pt=10, ct=5, reasoning="let me think")
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        resp = await adapter.complete(req)

    assert resp.text == "4"
    assert resp.reasoning_text == "let me think"
    assert resp.tokens_in == 10
    assert resp.tokens_out == 5
    kwargs = m.call_args.kwargs
    assert kwargs["model"].startswith("anthropic/")
    assert kwargs["thinking"] == {"type": "enabled"}
    assert kwargs["output_config"] == {"effort": "max"}


@pytest.mark.asyncio
async def test_openai_call_passes_reasoning_effort(openai_profile):
    adapter = LiteLLMAdapter(openai_profile)
    req = AdapterRequest(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    fake = _FakeResponse("hello", pt=5, ct=2)
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        resp = await adapter.complete(req)

    kwargs = m.call_args.kwargs
    assert kwargs["model"].startswith("openai/")
    assert kwargs["reasoning_effort"] == "high"
    assert resp.cost_usd == pytest.approx(10.0 * 5 / 1_000_000 + 40.0 * 2 / 1_000_000)


@pytest.mark.asyncio
async def test_stop_passthrough(openai_profile):
    adapter = LiteLLMAdapter(openai_profile)
    req = AdapterRequest(
        messages=[{"role": "user", "content": "x"}],
        max_output_tokens=16,
        stop=["</stop>", "DONE"],
    )
    fake = _FakeResponse("ok", pt=1, ct=1)
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        await adapter.complete(req)
    assert m.call_args.kwargs["stop"] == ["</stop>", "DONE"]


@pytest.mark.asyncio
async def test_seed_passthrough(openai_profile):
    adapter = LiteLLMAdapter(openai_profile)
    req = AdapterRequest(
        messages=[{"role": "user", "content": "x"}],
        max_output_tokens=16,
        seed=42,
    )
    fake = _FakeResponse("ok", pt=1, ct=1)
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        await adapter.complete(req)
    assert m.call_args.kwargs["seed"] == 42


@pytest.mark.asyncio
async def test_tools_passthrough(openai_profile):
    adapter = LiteLLMAdapter(openai_profile)
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    req = AdapterRequest(
        messages=[{"role": "user", "content": "x"}],
        max_output_tokens=16,
        tools=tools,
    )
    fake = _FakeResponse("ok", pt=1, ct=1)
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        await adapter.complete(req)
    assert m.call_args.kwargs["tools"] == tools


@pytest.mark.asyncio
async def test_reasoning_text_none_when_attr_missing(openai_profile):
    """When the response message has no reasoning_content attribute, reasoning_text is None."""
    adapter = LiteLLMAdapter(openai_profile)
    req = AdapterRequest(
        messages=[{"role": "user", "content": "x"}],
        max_output_tokens=16,
    )

    class _MessageNoReasoning:
        content = "hello"

    class _ChoiceNoReasoning:
        message = _MessageNoReasoning()
        finish_reason = "stop"

    class _RespNoReasoning:
        choices = [_ChoiceNoReasoning()]

        class usage:  # noqa: N801
            prompt_tokens = 1
            completion_tokens = 1

        def model_dump(self):
            return {"fake": True}

    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=_RespNoReasoning()),
    ):
        resp = await adapter.complete(req)
    assert resp.reasoning_text is None


@pytest.mark.asyncio
async def test_api_base_is_passed_to_litellm():
    from unittest.mock import AsyncMock, patch
    profile = ModelProfile(
        id="custom", provider="anthropic", model="claude-opus-4-7",
        api_base="https://my-proxy.example.com/v1",
    )
    adapter = LiteLLMAdapter(profile)
    req = AdapterRequest(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    fake = _FakeResponse("ok", pt=1, ct=1)
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        await adapter.complete(req)

    kwargs = m.call_args.kwargs
    assert kwargs.get("api_base") == "https://my-proxy.example.com/v1"


@pytest.mark.asyncio
async def test_api_base_absent_by_default(openai_profile):
    """When api_base is None, the kwarg should NOT appear in the LiteLLM call."""
    from unittest.mock import AsyncMock, patch
    adapter = LiteLLMAdapter(openai_profile)
    req = AdapterRequest(messages=[{"role": "user", "content": "x"}], max_output_tokens=16)
    fake = _FakeResponse("ok", pt=1, ct=1)
    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=fake),
    ) as m:
        await adapter.complete(req)
    assert "api_base" not in m.call_args.kwargs


@pytest.mark.asyncio
async def test_raw_defaults_empty_when_no_model_dump(openai_profile):
    """When the response object has no model_dump(), raw is an empty dict."""
    adapter = LiteLLMAdapter(openai_profile)
    req = AdapterRequest(
        messages=[{"role": "user", "content": "x"}],
        max_output_tokens=16,
    )

    class _MsgNoDump:
        content = "x"
        reasoning_content = None

    class _ChoiceNoDump:
        message = _MsgNoDump()
        finish_reason = "stop"

    class _RespNoDump:
        choices = [_ChoiceNoDump()]

        class usage:  # noqa: N801
            prompt_tokens = 0
            completion_tokens = 0

        # No model_dump method

    with patch(
        "prism.adapters.litellm_adapter.litellm.acompletion",
        new=AsyncMock(return_value=_RespNoDump()),
    ):
        resp = await adapter.complete(req)
    assert resp.raw == {}
