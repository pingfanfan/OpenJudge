from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Effort = Literal["off", "low", "medium", "high", "max"]


class Thinking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    effort: Effort = "high"


class RateLimit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rpm: int = Field(gt=0, default=60)
    tpm: int = Field(gt=0, default=200_000)


class Cost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = Field(ge=0.0, default=0.0)
    output_per_mtok: float = Field(ge=0.0, default=0.0)


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    display_name: str | None = None
    provider: Literal[
        "anthropic", "openai", "google", "deepseek", "xai", "kimi", "qwen", "custom"
    ]
    model: str
    thinking: Thinking | None = None
    reasoning_effort: Effort | None = None
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    cost: Cost = Field(default_factory=Cost)
