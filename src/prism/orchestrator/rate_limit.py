import asyncio
import time


class _Bucket:
    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self.updated
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        self.updated = now

    def _try_consume(self, amount: float) -> float:
        """Return 0 if consumed, else wait-seconds needed."""
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return 0.0
        deficit = amount - self.tokens
        return deficit / self.rate if self.rate > 0 else float("inf")


class RateLimiter:
    def __init__(self, *, rpm: int, tpm: int) -> None:
        if rpm <= 0 or tpm <= 0:
            raise ValueError("rpm and tpm must be positive")
        self._req = _Bucket(rpm / 60.0, 1.0)
        self._tok = _Bucket(tpm / 60.0, float(tpm))
        self._lock = asyncio.Lock()

    async def acquire(self, *, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        while True:
            async with self._lock:
                wait_r = self._req._try_consume(1.0)
                if wait_r == 0.0:
                    wait_t = self._tok._try_consume(float(tokens))
                    if wait_t == 0.0:
                        return
                    # Refund the request so it isn't double-spent on retry.
                    self._req.tokens += 1.0
                    wait = wait_t
                else:
                    wait = wait_r
            await asyncio.sleep(wait)
