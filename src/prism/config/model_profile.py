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
    provider: Literal["anthropic", "openai", "google", "deepseek", "xai", "kimi", "qwen", "custom"]
    model: str
    api_base: str | None = None
    thinking: Thinking | None = None
    reasoning_effort: Effort | None = None
    # Override the per-request max_output_tokens. Needed for thinking-heavy
    # benchmarks (AIME, MATH-500) where thinking tokens count toward max_tokens
    # and 4096 often truncates the final answer.
    max_output_tokens: int | None = Field(default=None, gt=0)
    # Skip native tools=[...] and use <tool_use>{"name":..., "arguments":...}</tool_use>
    # text encoding instead. For endpoints that reject OpenAI-format custom tool
    # schemas (e.g., some Anthropic-format proxies that only accept built-in
    # server tools like web_search_20250305).
    prompted_tool_use: bool = False
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    cost: Cost = Field(default_factory=Cost)
