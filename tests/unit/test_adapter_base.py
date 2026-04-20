import pytest
from prism.adapters.base import AdapterRequest, AdapterResponse, Adapter


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
