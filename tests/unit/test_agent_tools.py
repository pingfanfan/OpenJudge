from pathlib import Path

from prism.agent.tools import AGENT_TOOL_SCHEMAS, execute_tool


def test_tool_schemas_shape():
    assert len(AGENT_TOOL_SCHEMAS) == 3
    names = {t["function"]["name"] for t in AGENT_TOOL_SCHEMAS}
    assert names == {"read_file", "write_file", "bash"}


def test_read_file(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("world")
    out = execute_tool("read_file", {"path": "hello.txt"}, workspace=tmp_path)
    assert out == "world"


def test_read_file_missing(tmp_path: Path):
    out = execute_tool("read_file", {"path": "nope.txt"}, workspace=tmp_path)
    assert "Error" in out or "not found" in out.lower()


def test_read_file_path_escape_rejected(tmp_path: Path):
    (tmp_path.parent / "outside.txt").write_text("secret")
    out = execute_tool("read_file", {"path": "../outside.txt"}, workspace=tmp_path)
    assert "Error" in out or "outside" in out.lower()


def test_write_file(tmp_path: Path):
    out = execute_tool(
        "write_file",
        {"path": "new.txt", "content": "hello"},
        workspace=tmp_path,
    )
    assert "success" in out.lower() or "wrote" in out.lower()
    assert (tmp_path / "new.txt").read_text() == "hello"


def test_write_file_creates_parent_dirs(tmp_path: Path):
    execute_tool(
        "write_file",
        {"path": "sub/deep/x.txt", "content": "ok"},
        workspace=tmp_path,
    )
    assert (tmp_path / "sub" / "deep" / "x.txt").read_text() == "ok"


def test_bash_simple(tmp_path: Path):
    (tmp_path / "greet.txt").write_text("hi")
    out = execute_tool("bash", {"command": "cat greet.txt"}, workspace=tmp_path)
    assert "hi" in out


def test_bash_nonzero_exit(tmp_path: Path):
    out = execute_tool("bash", {"command": "exit 7"}, workspace=tmp_path)
    assert "7" in out


def test_bash_timeout(tmp_path: Path):
    out = execute_tool(
        "bash",
        {"command": "sleep 10"},
        workspace=tmp_path,
        bash_timeout_sec=1,
    )
    assert "timed out" in out.lower() or "timeout" in out.lower()


def test_unknown_tool(tmp_path: Path):
    out = execute_tool("nonexistent", {}, workspace=tmp_path)
    assert "unknown" in out.lower() or "error" in out.lower()
