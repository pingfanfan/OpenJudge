import pytest

from prism.judges.ifeval_constraints import (
    ConstraintResult,
    check_constraint,
    CONSTRAINT_CHECKERS,
)


def test_registry_has_checkers():
    assert len(CONSTRAINT_CHECKERS) >= 1
    assert "length_constraints:number_words" in CONSTRAINT_CHECKERS


def test_check_length_number_words_pass():
    text = " ".join(["word"] * 50)
    result = check_constraint(
        constraint_id="length_constraints:number_words",
        text=text,
        kwargs={"relation": "at least", "num_words": 40},
    )
    assert result.passed is True


def test_check_length_number_words_fail_below():
    text = "only three words here"
    result = check_constraint(
        constraint_id="length_constraints:number_words",
        text=text,
        kwargs={"relation": "at least", "num_words": 10},
    )
    assert result.passed is False


def test_check_length_number_words_at_most():
    text = " ".join(["word"] * 100)
    result = check_constraint(
        constraint_id="length_constraints:number_words",
        text=text,
        kwargs={"relation": "at most", "num_words": 50},
    )
    assert result.passed is False


def test_unknown_constraint_returns_unsupported():
    result = check_constraint(
        constraint_id="nonexistent:totally_fake",
        text="anything",
        kwargs={},
    )
    assert result.supported is False
    assert result.passed is False


def test_keywords_existence():
    r = check_constraint(
        constraint_id="keywords:existence",
        text="The quick brown fox jumps.",
        kwargs={"keywords": ["fox", "moon"]},
    )
    assert r.passed is False  # "moon" is missing
    r2 = check_constraint(
        constraint_id="keywords:existence",
        text="The quick brown fox jumps over the moon.",
        kwargs={"keywords": ["fox", "moon"]},
    )
    assert r2.passed is True


def test_keywords_forbidden():
    r = check_constraint(
        constraint_id="keywords:forbidden_words",
        text="I shall not say that forbidden word.",
        kwargs={"forbidden_words": ["forbidden", "banned"]},
    )
    assert r.passed is False
    r2 = check_constraint(
        constraint_id="keywords:forbidden_words",
        text="Nothing bad here.",
        kwargs={"forbidden_words": ["forbidden", "banned"]},
    )
    assert r2.passed is True


def test_english_lowercase_pass():
    r = check_constraint(
        constraint_id="change_case:english_lowercase",
        text="all lowercase words here.",
        kwargs={},
    )
    assert r.passed is True


def test_english_lowercase_fail_on_capital():
    r = check_constraint(
        constraint_id="change_case:english_lowercase",
        text="There is a Capital.",
        kwargs={},
    )
    assert r.passed is False


def test_english_capital():
    r = check_constraint(
        constraint_id="change_case:english_capital",
        text="ALL CAPS HERE.",
        kwargs={},
    )
    assert r.passed is True
    r2 = check_constraint(
        constraint_id="change_case:english_capital",
        text="Mixed Case.",
        kwargs={},
    )
    assert r2.passed is False


def test_no_comma():
    r = check_constraint(
        constraint_id="punctuation:no_comma",
        text="A sentence without any.",
        kwargs={},
    )
    assert r.passed is True
    r2 = check_constraint(
        constraint_id="punctuation:no_comma",
        text="Hello, world.",
        kwargs={},
    )
    assert r2.passed is False


def test_startend_quotation():
    r = check_constraint(
        constraint_id="startend:quotation",
        text='"this is quoted"',
        kwargs={},
    )
    assert r.passed is True
    r2 = check_constraint(
        constraint_id="startend:quotation",
        text="not quoted",
        kwargs={},
    )
    assert r2.passed is False
