"""Toy agent benchmark — two tiny "implement this function" tasks.

Purpose: prove the P3a agent pipeline end-to-end without requiring a real
benchmark dataset. Academic benchmarks land in P3c.
"""
from __future__ import annotations

import sys
from collections.abc import Iterable

from prism.agent.task import AgentBenchmark, AgentTask

_ADD_TASK = AgentTask(
    task_id="toy-add",
    workspace_files={
        "solution.py": "def add(a, b):\n    # TODO: implement\n    pass\n",
        "test_solution.py": (
            "from solution import add\n"
            "\n"
            "def test_add_positive():\n"
            "    assert add(2, 3) == 5\n"
            "\n"
            "def test_add_negative():\n"
            "    assert add(-1, 1) == 0\n"
        ),
    },
    user_instruction=(
        "Implement the `add(a, b)` function in solution.py so that the tests "
        "in test_solution.py all pass. Run the tests with pytest to verify."
    ),
    judge_command=[sys.executable, "-m", "pytest", "-q", "test_solution.py"],
    timeout_seconds=60,
    max_turns=10,
)

_REVERSE_TASK = AgentTask(
    task_id="toy-reverse",
    workspace_files={
        "solution.py": "def reverse(s):\n    # TODO\n    pass\n",
        "test_solution.py": (
            "from solution import reverse\n"
            "\n"
            "def test_reverse_hello():\n"
            "    assert reverse('hello') == 'olleh'\n"
            "\n"
            "def test_reverse_empty():\n"
            "    assert reverse('') == ''\n"
        ),
    },
    user_instruction=(
        "Implement the `reverse(s)` function in solution.py so that "
        "test_solution.py passes under pytest."
    ),
    judge_command=[sys.executable, "-m", "pytest", "-q", "test_solution.py"],
    timeout_seconds=60,
    max_turns=10,
)


class ToyAgentBenchmark(AgentBenchmark):
    name = "toy_agent"
    track = "agent"
    version = "v1"

    def load_tasks(self, *, subset: str | None = None) -> Iterable[AgentTask]:
        yield _ADD_TASK
        yield _REVERSE_TASK
