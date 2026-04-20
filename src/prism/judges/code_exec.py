"""PytestJudge — scores a model-generated Python solution by running pytest in a subprocess.

Behavior:
- Extracts code from ```python ... ``` fence if present (else uses whole output as code).
- Writes code to `solution.py` + the test_code to `test_solution.py` in a tempdir.
- Runs `python -m pytest -q test_solution.py` with timeout.
- Score = 1.0 if exit code 0, else 0.0.
  Reasoning captures the last ~500 chars of stderr/stdout for debugging.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from prism.judges.base import Judge, JudgeResult

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)


class PytestJudge(Judge):
    name = "pytest"

    def __init__(self, *, test_code: str, timeout_sec: float = 30.0) -> None:
        self.test_code = test_code
        self.timeout_sec = timeout_sec

    async def judge(self, *, output: str, expected: str) -> JudgeResult:
        code = self._extract_code(output)
        return await asyncio.to_thread(self._run_pytest_sync, code)

    @staticmethod
    def _extract_code(output: str) -> str:
        m = _CODE_FENCE_RE.search(output)
        return m.group(1) if m else output

    def _run_pytest_sync(self, code: str) -> JudgeResult:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "solution.py").write_text(code, encoding="utf-8")
            (tmp_path / "test_solution.py").write_text(self.test_code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "--no-header", "test_solution.py"],
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return JudgeResult(
                    score=0.0,
                    confidence=1.0,
                    reasoning=f"pytest timed out after {self.timeout_sec}s",
                )

            # Redact the absolute tmpdir path so persisted reasoning doesn't leak local FS info.
            raw = (proc.stdout + "\n" + proc.stderr)
            redacted = raw.replace(str(tmp_path), "<tmp>")
            output_text = redacted.strip()[-800:]

        if proc.returncode == 0:
            return JudgeResult(score=1.0, confidence=1.0, reasoning="all tests passed")
        return JudgeResult(score=0.0, confidence=1.0, reasoning=output_text or "tests failed")
