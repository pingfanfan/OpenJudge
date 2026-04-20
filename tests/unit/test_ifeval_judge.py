import pytest

from prism.judges.ifeval import IFEvalJudge


@pytest.mark.asyncio
async def test_all_constraints_pass():
    constraints = [
        {"id": "length_constraints:number_words", "kwargs": {"relation": "at least", "num_words": 3}},  # noqa: E501
        {"id": "punctuation:no_comma", "kwargs": {}},
    ]
    judge = IFEvalJudge(constraints=constraints)
    r = await judge.judge(output="hello world here is text", expected="")
    assert r.score == 1.0
    assert r.confidence == 1.0


@pytest.mark.asyncio
async def test_some_constraints_fail():
    constraints = [
        {"id": "length_constraints:number_words", "kwargs": {"relation": "at least", "num_words": 3}},  # noqa: E501
        {"id": "punctuation:no_comma", "kwargs": {}},
    ]
    judge = IFEvalJudge(constraints=constraints)
    r = await judge.judge(output="hi, there", expected="")
    # 2 words (fails first), comma present (fails second) → 0/2
    assert r.score == 0.0
    assert r.confidence == 1.0


@pytest.mark.asyncio
async def test_unsupported_constraint_lowers_confidence():
    constraints = [
        {"id": "length_constraints:number_words", "kwargs": {"relation": "at least", "num_words": 1}},  # noqa: E501
        {"id": "nonexistent:fake", "kwargs": {}},
    ]
    judge = IFEvalJudge(constraints=constraints)
    r = await judge.judge(output="hello", expected="")
    # 1 of 2 constraints supported; the supported one passes → score = 1/1 = 1.0, confidence = 1/2 = 0.5  # noqa: E501
    assert r.score == 1.0
    assert r.confidence == 0.5
    assert r.reasoning is not None
    assert "nonexistent" in r.reasoning
