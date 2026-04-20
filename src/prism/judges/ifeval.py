"""IFEvalJudge — scores a response against a list of IFEval constraints.

Each constraint is a dict {id: str, kwargs: dict}. The judge runs each through the
constraint registry. Score = passed / total among supported constraints.
Confidence = supported / total (1.0 if all supported, <1 if any unknown).

Unsupported constraints do not count toward score but lower confidence.
"""
from __future__ import annotations

from typing import Any

from prism.judges.base import Judge, JudgeResult
from prism.judges.ifeval_constraints import check_constraint


class IFEvalJudge(Judge):
    name = "ifeval"

    def __init__(self, *, constraints: list[dict[str, Any]]) -> None:
        self.constraints = constraints

    async def judge(self, *, output: str, expected: str) -> JudgeResult:
        total = len(self.constraints)
        if total == 0:
            return JudgeResult(score=0.0, confidence=0.0, reasoning="no constraints")

        supported: list[bool] = []
        passed: list[bool] = []
        unsupported_ids: list[str] = []

        for c in self.constraints:
            cid = c["id"]
            kwargs = c.get("kwargs", {})
            result = check_constraint(constraint_id=cid, text=output, kwargs=kwargs)
            supported.append(result.supported)
            if result.supported:
                passed.append(result.passed)
            else:
                unsupported_ids.append(cid)

        supported_count = sum(supported)
        if supported_count == 0:
            return JudgeResult(
                score=0.0, confidence=0.0,
                reasoning=f"all constraints unsupported: {unsupported_ids}",
            )

        score = sum(passed) / supported_count
        confidence = supported_count / total
        reasoning = None
        if unsupported_ids:
            reasoning = f"unsupported: {unsupported_ids}"
        return JudgeResult(score=score, confidence=confidence, reasoning=reasoning)
