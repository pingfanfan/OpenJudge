"""IFEval constraint checkers.

Each checker verifies a single IFEval-style constraint and returns a ConstraintResult.
The plugin registry maps constraint IDs to callables with signature
(text, **kwargs) -> ConstraintResult.

**Supported constraint IDs (12 of 25 from the IFEval paper):**

- length_constraints:number_words
- length_constraints:number_sentences
- length_constraints:number_paragraphs
- keywords:existence
- keywords:forbidden_words
- change_case:english_lowercase
- change_case:english_capital
- punctuation:no_comma
- startend:quotation
- startend:end_checker
- detectable_content:number_placeholders
- detectable_format:number_bullet_lists

Unknown constraint IDs return ConstraintResult(supported=False). IFEvalJudge
surfaces this as reduced confidence on the JudgeResult, so callers know when
some constraints could not be evaluated.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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
        return ConstraintResult(
            passed=False, supported=False, reason=f"unknown constraint: {constraint_id!r}"
        )
    try:
        return fn(text=text, **kwargs)
    except Exception as e:
        return ConstraintResult(passed=False, supported=False, reason=f"checker raised: {e}")


# ---- Base checkers (Task 4 ships this one; Tasks 5-6 add 11 more) ----

def _relation_check(count: int, relation: str, target: int) -> bool | None:
    """Return True/False per relation, or None if relation is unknown."""
    table = {
        "at least": count >= target,
        "at most": count <= target,
        "exactly": count == target,
        "less than": count < target,
        "more than": count > target,
        "greater than": count > target,
        "fewer than": count < target,
    }
    return table.get(relation)


@register("length_constraints:number_words")
def _check_number_words(
    *, text: str, relation: str, num_words: int, **_: Any
) -> ConstraintResult:
    count = len(text.split())
    ok = _relation_check(count, relation, num_words)
    if ok is None:
        return ConstraintResult(
            passed=False, supported=False, reason=f"unknown relation: {relation!r}"
        )
    return ConstraintResult(
        passed=ok,
        reason=f"expected words {relation} {num_words}, got {count}",
    )


@register("keywords:existence")
def _check_keywords_existence(
    *, text: str, keywords: list[str], **_: Any
) -> ConstraintResult:
    # Case-sensitive word-boundary match to match IFEval reference behavior:
    # "classify" should NOT satisfy keyword "class".
    missing = [k for k in keywords if not re.search(rf"\b{re.escape(k)}\b", text)]
    return ConstraintResult(
        passed=not missing,
        reason=f"missing keywords: {missing}" if missing else None,
    )


@register("keywords:forbidden_words")
def _check_forbidden(
    *, text: str, forbidden_words: list[str], **_: Any
) -> ConstraintResult:
    found = [w for w in forbidden_words if re.search(rf"\b{re.escape(w)}\b", text)]
    return ConstraintResult(
        passed=not found,
        reason=f"forbidden words present: {found}" if found else None,
    )


@register("change_case:english_lowercase")
def _check_lowercase(*, text: str, **_: Any) -> ConstraintResult:
    return ConstraintResult(
        passed=text == text.lower(),
        reason="found uppercase letters" if text != text.lower() else None,
    )


@register("change_case:english_capital")
def _check_uppercase(*, text: str, **_: Any) -> ConstraintResult:
    return ConstraintResult(
        passed=text == text.upper(),
        reason="found lowercase letters" if text != text.upper() else None,
    )


@register("punctuation:no_comma")
def _check_no_comma(*, text: str, **_: Any) -> ConstraintResult:
    return ConstraintResult(
        passed="," not in text,
        reason="commas found" if "," in text else None,
    )


@register("startend:quotation")
def _check_quotation(*, text: str, **_: Any) -> ConstraintResult:
    t = text.strip()
    ok = len(t) >= 2 and t.startswith('"') and t.endswith('"')
    return ConstraintResult(
        passed=ok,
        reason="not wrapped in double quotes" if not ok else None,
    )


@register("length_constraints:number_sentences")
def _check_number_sentences(
    *, text: str, relation: str = "at least", num_sentences: int, **_: Any
) -> ConstraintResult:
    sentences = [s for s in re.split(r"[.!?]+(?:\s+|$)", text) if s.strip()]
    count = len(sentences)
    ok = _relation_check(count, relation, num_sentences)
    if ok is None:
        return ConstraintResult(
            passed=False, supported=False, reason=f"unknown relation: {relation!r}"
        )
    return ConstraintResult(passed=ok, reason=f"sentences {relation} {num_sentences}, got {count}")


@register("length_constraints:number_paragraphs")
def _check_number_paragraphs(
    *, text: str, num_paragraphs: int, **_: Any
) -> ConstraintResult:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return ConstraintResult(
        passed=len(paragraphs) == num_paragraphs,
        reason=f"paragraphs: got {len(paragraphs)}, want {num_paragraphs}",
    )


@register("detectable_content:number_placeholders")
def _check_placeholders(
    *, text: str, num_placeholders: int, **_: Any
) -> ConstraintResult:
    found = re.findall(r"\[[^\]]+\]", text)
    return ConstraintResult(
        passed=len(found) >= num_placeholders,
        reason=f"placeholders: got {len(found)}, want at least {num_placeholders}",
    )


@register("startend:end_checker")
def _check_end(
    *, text: str, end_phrase: str, **_: Any
) -> ConstraintResult:
    return ConstraintResult(
        passed=text.rstrip().endswith(end_phrase.rstrip()),
        reason=f"text does not end with {end_phrase!r}",
    )


@register("detectable_format:number_bullet_lists")
def _check_bullets(
    *, text: str, num_bullets: int, **_: Any
) -> ConstraintResult:
    lines = text.splitlines()
    bullets = [ln for ln in lines if re.match(r"^\s*[\*\-]\s+", ln)]
    return ConstraintResult(
        passed=len(bullets) >= num_bullets,
        reason=f"bullets: got {len(bullets)}, want at least {num_bullets}",
    )


@register("keywords:frequency")
def _check_keyword_frequency(
    *, text: str, keyword: str, frequency: int, relation: str = "at least", **_: Any
) -> ConstraintResult:
    """IFEval keyword-frequency check: keyword appears `frequency` times per `relation`."""
    count = len(re.findall(rf"\b{re.escape(keyword)}\b", text))
    ok = _relation_check(count, relation, frequency)
    if ok is None:
        return ConstraintResult(
            passed=False, supported=False, reason=f"unknown relation: {relation!r}"
        )
    return ConstraintResult(
        passed=ok,
        reason=f"keyword {keyword!r}: got {count}, want {relation} {frequency}",
    )


@register("detectable_format:title")
def _check_title(*, text: str, **_: Any) -> ConstraintResult:
    """IFEval title check: response must contain a title wrapped in << … >>."""
    ok = bool(re.search(r"<<[^<>]+>>", text))
    return ConstraintResult(passed=ok, reason=None if ok else "missing <<title>> marker")


@register("detectable_format:multiple_sections")
def _check_multiple_sections(
    *, text: str, section_spliter: str, num_sections: int, **_: Any
) -> ConstraintResult:
    """Count occurrences of `section_spliter N` (e.g., 'SECTION 1', 'SECTION 2', ...)."""
    # Typical spliters: "SECTION", "Section", "Chapter".
    # IFEval wants N numbered sections. Count unique section numbers in text.
    pattern = rf"{re.escape(section_spliter)}\s+\d+"
    matches = re.findall(pattern, text, re.IGNORECASE)
    ok = len(matches) >= num_sections
    return ConstraintResult(
        passed=ok,
        reason=f"{section_spliter}: got {len(matches)}, want {num_sections}",
    )


@register("detectable_format:json_format")
def _check_json(*, text: str, **_: Any) -> ConstraintResult:
    """Response must be valid JSON (optionally wrapped in a markdown code fence)."""
    import json as _json
    # Strip markdown fences if present.
    stripped = text.strip()
    if stripped.startswith("```"):
        # Find inner content
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
        if m:
            stripped = m.group(1)
    try:
        _json.loads(stripped)
        return ConstraintResult(passed=True)
    except Exception as e:
        return ConstraintResult(passed=False, reason=f"not valid JSON: {e}")


@register("change_case:capital_word_frequency")
def _check_capital_words(
    *, text: str, capital_frequency: int, capital_relation: str = "at least", **_: Any
) -> ConstraintResult:
    """Count ALL-CAPS words (e.g., 'HELLO', 'WORLD')."""
    all_caps = [w for w in re.findall(r"\b[A-Z]{2,}\b", text)]
    count = len(all_caps)
    ok = _relation_check(count, capital_relation, capital_frequency)
    if ok is None:
        return ConstraintResult(
            passed=False, supported=False, reason=f"unknown relation: {capital_relation!r}"
        )
    return ConstraintResult(
        passed=ok,
        reason=f"ALL-CAPS words: got {count}, want {capital_relation} {capital_frequency}",
    )


@register("detectable_content:postscript")
def _check_postscript(*, text: str, postscript_marker: str, **_: Any) -> ConstraintResult:
    """Response must contain a postscript starting with `postscript_marker` (e.g., 'P.S.')."""
    ok = postscript_marker in text
    return ConstraintResult(
        passed=ok,
        reason=None if ok else f"missing postscript marker {postscript_marker!r}",
    )


@register("keywords:letter_frequency")
def _check_letter_frequency(
    *, text: str, letter: str, let_frequency: int, let_relation: str = "at least", **_: Any
) -> ConstraintResult:
    """Count occurrences of a single letter (case-insensitive)."""
    count = len(re.findall(re.escape(letter), text, re.IGNORECASE))
    ok = _relation_check(count, let_relation, let_frequency)
    if ok is None:
        return ConstraintResult(
            passed=False, supported=False, reason=f"unknown relation: {let_relation!r}"
        )
    return ConstraintResult(
        passed=ok,
        reason=f"letter {letter!r}: got {count}, want {let_relation} {let_frequency}",
    )


@register("detectable_format:number_highlighted_sections")
def _check_highlighted_sections(
    *, text: str, num_highlights: int, **_: Any
) -> ConstraintResult:
    """Count markdown highlights (*italic*, **bold**, or *highlighted*)."""
    highlights = re.findall(r"\*[^*\n]+\*", text)
    ok = len(highlights) >= num_highlights
    return ConstraintResult(
        passed=ok,
        reason=f"highlights: got {len(highlights)}, want at least {num_highlights}",
    )


@register("combination:repeat_prompt")
def _check_repeat_prompt(*, text: str, prompt_to_repeat: str, **_: Any) -> ConstraintResult:
    """Response must start with (an exact repeat of) the prompt."""
    ok = text.lstrip().startswith(prompt_to_repeat.strip())
    return ConstraintResult(
        passed=ok,
        reason=None if ok else "response does not begin with the prompt",
    )


@register("combination:two_responses")
def _check_two_responses(*, text: str, **_: Any) -> ConstraintResult:
    """Response must contain two distinct answers separated by '******'."""
    ok = text.count("******") >= 1
    return ConstraintResult(
        passed=ok,
        reason=None if ok else "missing '******' separator",
    )
