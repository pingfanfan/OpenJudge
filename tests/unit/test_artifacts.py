import json
from pathlib import Path

from prism.storage.artifacts import ArtifactStore


def test_put_and_get(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    store.put("run-1", "trace/prompt-1.json", {"messages": [{"role": "user", "content": "hi"}]})
    got = store.get("run-1", "trace/prompt-1.json")
    assert got["messages"][0]["content"] == "hi"


def test_list(tmp_path: Path):
    store = ArtifactStore(tmp_path / "a")
    store.put("r", "a.json", {"x": 1})
    store.put("r", "sub/b.json", {"y": 2})
    assert set(store.list("r")) == {"a.json", "sub/b.json"}


def test_missing_returns_none(tmp_path: Path):
    store = ArtifactStore(tmp_path / "a")
    assert store.get("r", "nope.json") is None


def test_atomic_write(tmp_path: Path):
    store = ArtifactStore(tmp_path / "a")
    store.put("r", "x.json", {"a": 1})
    # File should exist, not .tmp
    assert (tmp_path / "a" / "r" / "x.json").exists()
    assert not any(p.suffix == ".tmp" for p in (tmp_path / "a").rglob("*"))
