
import pytest

from prism.adapters.base import AdapterResponse
from prism.judges.base import JudgeResult
from prism.judges.llm import LLMJudge


class FakeAdapter:
    def __init__(self, content: str) -> None:
        self._content = content

    async def complete(self, request):
        return AdapterResponse(
            text=self._content,
            reasoning_text=None,
            tokens_in=10,
            tokens_out=10,
            latency_ms=5.0,
            cost_usd=0.0,
            raw={},
        )


@pytest.mark.asyncio
async def test_llm_judge_parses_json():
    payload = '{"score": 0.9, "confidence": 0.85, "reasoning": "mostly right"}'
    j = LLMJudge(adapter=FakeAdapter(payload), rubric="Score 0-1.")
    r = await j.judge_async(output="2+2=4", expected="4")
    assert isinstance(r, JudgeResult)
    assert r.score == 0.9
    assert r.confidence == 0.85


@pytest.mark.asyncio
async def test_llm_judge_parses_json_with_extra_text():
    payload = (
        'Here is my analysis.\n```json\n'
        '{"score": 0.0, "confidence": 1.0, "reasoning": "wrong"}\n```\n'
    )
    j = LLMJudge(adapter=FakeAdapter(payload), rubric="Score 0-1.")
    r = await j.judge_async(output="2+2=5", expected="4")
    assert r.score == 0.0


@pytest.mark.asyncio
async def test_llm_judge_malformed_returns_low_confidence():
    j = LLMJudge(adapter=FakeAdapter("I cannot parse this."), rubric="Score 0-1.")
    r = await j.judge_async(output="x", expected="y")
    assert r.confidence < 0.5
    assert r.score == 0.0
