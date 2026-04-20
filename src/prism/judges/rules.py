import re

from prism.judges.base import Judge, JudgeResult

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class ExactMatchJudge(Judge):
    name = "exact_match"

    def __init__(self, *, case_sensitive: bool = True, strip: bool = True) -> None:
        self.case_sensitive = case_sensitive
        self.strip = strip

    def judge(self, *, output: str, expected: str) -> JudgeResult:
        a = output.strip() if self.strip else output
        b = expected.strip() if self.strip else expected
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        return JudgeResult(score=1.0 if a == b else 0.0, confidence=1.0)


class NumericJudge(Judge):
    name = "numeric"

    def __init__(self, *, tolerance: float = 0.0) -> None:
        self.tolerance = tolerance

    def judge(self, *, output: str, expected: str) -> JudgeResult:
        try:
            exp = float(expected.strip())
        except ValueError:
            return JudgeResult(score=0.0, confidence=0.0, reasoning="bad expected number")

        matches = _NUMBER_RE.findall(output)
        if not matches:
            return JudgeResult(score=0.0, confidence=1.0, reasoning="no number in output")
        try:
            got = float(matches[-1])
        except ValueError:
            return JudgeResult(score=0.0, confidence=0.0, reasoning="unparseable")

        if self.tolerance > 0:
            ok = abs(got - exp) <= self.tolerance
        else:
            ok = got == exp
        return JudgeResult(score=1.0 if ok else 0.0, confidence=1.0)


class RegexJudge(Judge):
    name = "regex"

    def __init__(self, *, pattern: str, flags: int = 0) -> None:
        self._re = re.compile(pattern, flags)

    def judge(self, *, output: str, expected: str) -> JudgeResult:
        m = self._re.search(output)
        if not m:
            return JudgeResult(score=0.0, confidence=1.0, reasoning="no match")
        captured = m.group(1) if m.groups() else m.group(0)
        return JudgeResult(
            score=1.0 if captured.strip() == expected.strip() else 0.0,
            confidence=1.0,
        )
