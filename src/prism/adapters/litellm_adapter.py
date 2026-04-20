import time

import litellm

from prism.adapters.base import Adapter, AdapterRequest, AdapterResponse
from prism.adapters.reasoning_translator import translate
from prism.utils.cost import compute_cost


class LiteLLMAdapter(Adapter):
    async def complete(self, request: AdapterRequest) -> AdapterResponse:
        extra = translate(self.profile)
        model_id = f"{self.profile.provider}/{self.profile.model}"

        kwargs: dict = {
            "model": model_id,
            "messages": request.messages,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stop:
            kwargs["stop"] = request.stop
        if request.tools:
            kwargs["tools"] = request.tools
        if request.seed is not None:
            kwargs["seed"] = request.seed
        kwargs.update(extra)

        t0 = time.perf_counter()
        resp = await litellm.acompletion(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = resp.choices[0]
        message = choice.message
        text = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)

        usage = resp.usage
        tokens_in = getattr(usage, "prompt_tokens", 0)
        tokens_out = getattr(usage, "completion_tokens", 0)

        return AdapterResponse(
            text=text,
            reasoning_text=reasoning,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=compute_cost(self.profile.cost, tokens_in=tokens_in, tokens_out=tokens_out),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            finish_reason=getattr(choice, "finish_reason", None),
        )
