from prism.agent.workspace import workspace_context


def test_workspace_creates_files():
    files = {"a.py": "print(1)", "sub/b.txt": "hello"}
    with workspace_context(files) as ws:
        assert (ws / "a.py").read_text() == "print(1)"
        assert (ws / "sub" / "b.txt").read_text() == "hello"
        assert ws.exists()
    assert not ws.exists()


def test_workspace_empty_files():
    with workspace_context({}) as ws:
        assert ws.exists()
        assert list(ws.iterdir()) == []


def test_workspace_nested_dirs_created():
    files = {"a/b/c/d.txt": "deep"}
    with workspace_context(files) as ws:
        assert (ws / "a" / "b" / "c" / "d.txt").read_text() == "deep"
