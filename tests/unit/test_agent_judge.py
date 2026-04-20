from pathlib import Path

from prism.agent.judge import run_hard_judge


def test_judge_succeeds_when_command_exits_0(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hi")
    result = run_hard_judge(
        command=["cat", "hello.txt"],
        workspace=tmp_path,
        timeout_sec=10,
    )
    assert result.exit_code == 0
    assert "hi" in result.stdout
    assert result.success is True


def test_judge_fails_on_nonzero_exit(tmp_path: Path):
    result = run_hard_judge(
        command=["bash", "-c", "exit 3"],
        workspace=tmp_path,
        timeout_sec=10,
    )
    assert result.exit_code == 3
    assert result.success is False


def test_judge_timeout(tmp_path: Path):
    result = run_hard_judge(
        command=["sleep", "10"],
        workspace=tmp_path,
        timeout_sec=1,
    )
    assert result.success is False
    assert "timed out" in result.stdout.lower() or result.exit_code != 0


def test_judge_runs_in_workspace(tmp_path: Path):
    result = run_hard_judge(command=["pwd"], workspace=tmp_path, timeout_sec=5)
    assert str(tmp_path) in result.stdout
