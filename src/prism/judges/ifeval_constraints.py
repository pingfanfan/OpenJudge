"""IFEval constraint checkers.

Each checker verifies a single IFEval-style constraint and returns a ConstraintResult.
The plugin registry maps constraint IDs (like 'length_constraints:number_words') to
callables with signature (text, **kwargs) -> ConstraintResult.

Task 4 ships the registry + one checker. Tasks 5-6 fill in 11 more.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ConstraintResult:
    passed: bool
    supported: bool = True
    reason: str | None = None


Checker = Callable[..., ConstraintResult]
CONSTRAINT_CHECKERS: dict[str, Checker] = {}


def register(constraint_id: str) -> Callable[[Checker], Checker]:
    """Decorator to register a constraint checker."""
    def _decorator(fn: Checker) -> Checker:
        CONSTRAINT_CHECKERS[constraint_id] = fn
        return fn
    return _decorator


def check_constraint(
    *, constraint_id: str, text: str, kwargs: dict[str, Any]
) -> ConstraintResult:
    """Look up and invoke the checker for a constraint_id."""
    fn = CONSTRAINT_CHECKERS.get(constraint_id)
    if fn is None:
        return ConstraintResult(passed=False, supported=False, reason=f"unknown constraint: {constraint_id!r}")
    try:
        return fn(text=text, **kwargs)
    except Exception as e:
        return ConstraintResult(passed=False, supported=False, reason=f"checker raised: {e}")


# ---- Base checkers (Task 4 ships this one; Tasks 5-6 add 11 more) ----

@register("length_constraints:number_words")
def _check_number_words(
    *, text: str, relation: str, num_words: int, **_: Any
) -> ConstraintResult:
    count = len(text.split())
    if relation == "at least":
        ok = count >= num_words
    elif relation == "at most":
        ok = count <= num_words
    elif relation == "exactly":
        ok = count == num_words
    else:
        return ConstraintResult(passed=False, supported=False, reason=f"unknown relation: {relation!r}")
    return ConstraintResult(
        passed=ok,
        reason=f"expected words {relation} {num_words}, got {count}",
    )
