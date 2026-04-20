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
