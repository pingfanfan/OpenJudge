from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None


class Judge(ABC):
    name: str = "judge"

    @abstractmethod
    def judge(self, *, output: str, expected: str) -> JudgeResult: ...
