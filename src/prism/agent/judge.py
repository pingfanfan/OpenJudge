"""Hard judgment for agent tasks — run a command in the workspace and return pass/fail."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JudgeOutcome:
    exit_code: int
    stdout: str
    success: bool


def run_hard_judge(
    *,
    command: list[str],
    workspace: Path,
    timeout_sec: float = 300.0,
) -> JudgeOutcome:
    """Run `command` in `workspace`. Return JudgeOutcome with exit code + combined output."""
    try:
        proc = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        return JudgeOutcome(
            exit_code=124,
            stdout=f"timed out after {timeout_sec}s: {e.cmd}",
            success=False,
        )
    output = (proc.stdout + proc.stderr).strip()
    return JudgeOutcome(
        exit_code=proc.returncode,
        stdout=output,
        success=proc.returncode == 0,
    )
