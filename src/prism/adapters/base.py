from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.config.model_profile import ModelProfile


class AdapterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]]
    max_output_tokens: int = Field(gt=0, default=4096)
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    seed: int | None = None


class AdapterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    reasoning_text: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    raw: dict[str, Any]
    finish_reason: str | None = None


class Adapter(ABC):
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    @abstractmethod
    async def complete(self, request: AdapterRequest) -> AdapterResponse: ...
