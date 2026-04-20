import asyncio
import time

import pytest

from prism.orchestrator.rate_limit import RateLimiter


@pytest.mark.asyncio
async def test_acquire_no_wait_when_under_limit():
    rl = RateLimiter(rpm=60, tpm=1_000_000)
    t0 = time.perf_counter()
    await rl.acquire(tokens=100)
    assert time.perf_counter() - t0 < 0.05


@pytest.mark.asyncio
async def test_rpm_enforcement():
    rl = RateLimiter(rpm=120, tpm=10_000_000)  # 2 rps
    t0 = time.perf_counter()
    for _ in range(4):
        await rl.acquire(tokens=1)
    elapsed = time.perf_counter() - t0
    assert 1.3 <= elapsed <= 2.2


@pytest.mark.asyncio
async def test_negative_tokens_rejected():
    rl = RateLimiter(rpm=60, tpm=1000)
    with pytest.raises(ValueError):
        await rl.acquire(tokens=-1)
