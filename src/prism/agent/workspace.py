"""Workspace lifecycle management for agent tasks."""
from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def workspace_context(files: dict[str, str]) -> Iterator[Path]:
    """Create a tempdir, populate with files, yield the Path, cleanup on exit."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        yield root
