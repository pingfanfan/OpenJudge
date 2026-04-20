import json
import re

from prism.adapters.base import Adapter, AdapterRequest
from prism.judges.base import Judge, JudgeResult

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"score\"[^{}]*\})", re.DOTALL)

_DEFAULT_PROMPT = """You are an impartial grader. Compare the MODEL_OUTPUT to the REFERENCE.
Return STRICT JSON only, with keys: score (0.0-1.0), confidence (0.0-1.0), reasoning (short).

Rubric: {rubric}

MODEL_OUTPUT:
{output}

REFERENCE:
{expected}

JSON:"""


class LLMJudge(Judge):
    name = "llm_judge"

    def __init__(self, *, adapter: Adapter, rubric: str) -> None:
        self.adapter = adapter
        self.rubric = rubric

    async def judge_async(self, *, output: str, expected: str) -> JudgeResult:
        prompt = _DEFAULT_PROMPT.format(rubric=self.rubric, output=output, expected=expected)
        resp = await self.adapter.complete(
            AdapterRequest(
                messages=[{"role": "user", "content": prompt}],
                max_output_tokens=512,
            )
        )
        return self._parse(resp.text)

    def judge(self, *, output: str, expected: str) -> JudgeResult:  # sync not supported
        raise NotImplementedError("Use judge_async")

    @staticmethod
    def _parse(text: str) -> JudgeResult:
        candidates: list[str] = []
        m = _FENCED_JSON_RE.search(text)
        if m:
            candidates.append(m.group(1))
        m2 = _BARE_JSON_RE.search(text)
        if m2:
            candidates.append(m2.group(1))
        candidates.append(text)

        for c in candidates:
            try:
                data = json.loads(c)
                score = float(data.get("score", 0.0))
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning")
                score = max(0.0, min(1.0, score))
                confidence = max(0.0, min(1.0, confidence))
                return JudgeResult(score=score, confidence=confidence, reasoning=reasoning)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return JudgeResult(score=0.0, confidence=0.0, reasoning="unparseable judge output")
