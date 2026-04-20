import pytest

from prism.judges.rules import ExactMatchJudge, NumericJudge, RegexJudge


@pytest.mark.asyncio
async def test_exact_match_pass():
    j = ExactMatchJudge()
    r = await j.judge(output="hello", expected="hello")
    assert r.score == 1.0
    assert r.confidence == 1.0


@pytest.mark.asyncio
async def test_exact_match_fail():
    j = ExactMatchJudge()
    r = await j.judge(output="hello!", expected="hello")
    assert r.score == 0.0


@pytest.mark.asyncio
async def test_exact_match_case_insensitive_option():
    j = ExactMatchJudge(case_sensitive=False)
    r = await j.judge(output="Hello", expected="hello")
    assert r.score == 1.0


@pytest.mark.asyncio
async def test_numeric_exact():
    j = NumericJudge()
    r = await j.judge(output="The answer is 42.", expected="42")
    assert r.score == 1.0


@pytest.mark.asyncio
async def test_numeric_tolerance():
    j = NumericJudge(tolerance=0.01)
    r = await j.judge(output="3.141", expected="3.14")
    assert r.score == 1.0


@pytest.mark.asyncio
async def test_numeric_no_number_found():
    j = NumericJudge()
    r = await j.judge(output="I don't know", expected="42")
    assert r.score == 0.0


@pytest.mark.asyncio
async def test_regex_pass():
    j = RegexJudge(pattern=r"\bAnswer:\s*([A-D])\b")
    r = await j.judge(output="My analysis leads to Answer: C here.", expected="C")
    assert r.score == 1.0


@pytest.mark.asyncio
async def test_regex_wrong_capture():
    j = RegexJudge(pattern=r"Answer:\s*([A-D])")
    r = await j.judge(output="Answer: B", expected="C")
    assert r.score == 0.0


@pytest.mark.asyncio
async def test_regex_no_match():
    j = RegexJudge(pattern=r"Answer:\s*([A-D])")
    r = await j.judge(output="I refuse", expected="C")
    assert r.score == 0.0
