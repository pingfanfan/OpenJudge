"""Agent tools: read_file, write_file, bash.

All tools are sandboxed to a given `workspace` Path — paths that resolve
outside the workspace are rejected.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file in the workspace."
                " Returns the file contents as a string, or an error message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the workspace root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write (create or overwrite) a file in the workspace."
                " Parent directories are created as needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path."},
                    "content": {"type": "string", "description": "File contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the workspace directory."
                " Returns combined stdout+stderr and the exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                },
                "required": ["command"],
            },
        },
    },
]


def _resolve_inside(workspace: Path, relative: str) -> Path | None:
    """Resolve a relative path inside workspace. Returns None if it escapes."""
    workspace = workspace.resolve()
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    return candidate


def execute_tool(
    name: str,
    args: dict[str, Any],
    *,
    workspace: Path,
    bash_timeout_sec: float = 60.0,
) -> str:
    """Dispatch to the named tool. Returns a string result (success or error)."""
    if name == "read_file":
        relative = args.get("path", "")
        path = _resolve_inside(workspace, relative)
        if path is None:
            return f"Error: path {relative!r} escapes the workspace."
        if not path.exists():
            return f"Error: file not found: {relative}"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading {relative}: {type(e).__name__}: {e}"

    if name == "write_file":
        relative = args.get("path", "")
        content = args.get("content", "")
        path = _resolve_inside(workspace, relative)
        if path is None:
            return f"Error: path {relative!r} escapes the workspace."
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {relative}."

    if name == "bash":
        command = args.get("command", "")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=bash_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {bash_timeout_sec}s"
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            return f"Exit code {proc.returncode}. Output:\n{output}"
        return output or "(no output)"

    return f"Error: unknown tool {name!r}"
