from prism.benchmarks.toy_agent.benchmark import ToyAgentBenchmark


def test_load_tasks_produces_two_tasks():
    bm = ToyAgentBenchmark()
    tasks = list(bm.load_tasks(subset="full"))
    assert len(tasks) == 2
    ids = [t.task_id for t in tasks]
    assert "toy-add" in ids
    assert "toy-reverse" in ids


def test_tasks_have_workspace_and_judge_command():
    bm = ToyAgentBenchmark()
    task = next(iter(bm.load_tasks(subset="full")))
    assert task.workspace_files
    assert task.judge_command
    assert "solution.py" in task.workspace_files


def test_add_task_judge_fails_on_unedited_solution():
    """Seed solution.py has a TODO stub; pytest should fail before agent edits."""
    import subprocess
    import tempfile
    from pathlib import Path

    bm = ToyAgentBenchmark()
    task = next(t for t in bm.load_tasks(subset="full") if t.task_id == "toy-add")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, content in task.workspace_files.items():
            (root / name).write_text(content)
        result = subprocess.run(
            task.judge_command, cwd=str(root),
            capture_output=True, text=True, timeout=30,
        )
    assert result.returncode != 0
