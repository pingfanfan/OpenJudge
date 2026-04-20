import pytest
from pydantic import ValidationError

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse


def test_request_minimal():
    req = AdapterRequest(
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=16,
    )
    assert req.messages[0]["role"] == "user"
    assert req.max_output_tokens == 16
    assert req.temperature == 0.0


def test_response_cost_computed():
    resp = AdapterResponse(
        text="hello",
        reasoning_text=None,
        tokens_in=100,
        tokens_out=50,
        latency_ms=1234.5,
        cost_usd=0.00175,
        raw={},
    )
    assert resp.tokens_in == 100
    assert resp.cost_usd == 0.00175


def test_abstract_adapter_cannot_instantiate():
    with pytest.raises(TypeError):
        Adapter()


def test_request_rejects_extra_fields():
    with pytest.raises(ValidationError, match="extra"):
        AdapterRequest(
            messages=[{"role": "user", "content": "x"}],
            max_output_tokens=16,
            foo="bar",
        )


def test_response_rejects_extra_fields():
    with pytest.raises(ValidationError, match="extra"):
        AdapterResponse(
            text="x",
            tokens_in=0,
            tokens_out=0,
            latency_ms=0.0,
            cost_usd=0.0,
            raw={},
            unknown_field=1,
        )


def test_request_rejects_out_of_range_temperature():
    with pytest.raises(ValidationError, match="temperature"):
        AdapterRequest(
            messages=[{"role": "user", "content": "x"}],
            max_output_tokens=16,
            temperature=3.0,
        )


def test_request_rejects_zero_top_p():
    with pytest.raises(ValidationError, match="top_p"):
        AdapterRequest(
            messages=[{"role": "user", "content": "x"}],
            max_output_tokens=16,
            top_p=0.0,
        )


def test_response_default_reasoning_text_none():
    resp = AdapterResponse(
        text="x",
        tokens_in=0,
        tokens_out=0,
        latency_ms=0.0,
        cost_usd=0.0,
        raw={},
    )
    assert resp.reasoning_text is None
