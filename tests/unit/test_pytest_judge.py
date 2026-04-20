import pytest

from prism.judges.code_exec import PytestJudge


@pytest.mark.asyncio
async def test_passing_code_scores_1():
    code = """
def add(a, b):
    return a + b
"""
    test_code = """
def test_add():
    from solution import add
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
"""
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code, expected="")
    assert r.score == 1.0
    assert r.confidence == 1.0


@pytest.mark.asyncio
async def test_failing_code_scores_0():
    code = """
def add(a, b):
    return a - b   # bug
"""
    test_code = """
def test_add():
    from solution import add
    assert add(2, 3) == 5
"""
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code, expected="")
    assert r.score == 0.0
    assert r.reasoning is not None
    assert "AssertionError" in r.reasoning or "failed" in r.reasoning.lower()


@pytest.mark.asyncio
async def test_code_with_syntax_error_scores_0():
    code = "def add(a, b:"
    test_code = "def test_x():\n    from solution import add"
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code, expected="")
    assert r.score == 0.0


@pytest.mark.asyncio
async def test_extracts_code_block_if_output_wraps_in_fence():
    code_in_fence = "```python\ndef add(a, b):\n    return a + b\n```"
    test_code = """
def test_add():
    from solution import add
    assert add(1, 2) == 3
"""
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code_in_fence, expected="")
    assert r.score == 1.0


@pytest.mark.asyncio
async def test_reasoning_redacts_tmpdir():
    code = """
def add(a, b):
    return a - b   # bug
"""
    test_code = """
def test_add():
    from solution import add
    assert add(2, 3) == 5
"""
    j = PytestJudge(test_code=test_code, timeout_sec=10)
    r = await j.judge(output=code, expected="")
    assert r.score == 0.0
    assert r.reasoning is not None
    # No absolute /var/folders/... or /tmp/... paths should appear in reasoning.
    assert "/var/folders/" not in r.reasoning
    assert "/tmp/" not in r.reasoning or "<tmp>" in r.reasoning
