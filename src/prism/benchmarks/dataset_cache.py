"""Dataset loading wrapper with a mock-friendly indirection over HuggingFace datasets.

Two supported formats:
- `jsonl` — read rows from a local file (or `file://` URI). Used by tests.
- `hf`    — delegate to `datasets.load_dataset`. Used in real runs.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import datasets  # noqa: F401 — imported at module level so tests can patch it


def load_dataset_cached(
    *, source: str, format: str = "hf", split: str = "test", **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Yield rows from a dataset. Thin wrapper to keep HF dependency test-patchable."""
    if format == "jsonl":
        path = Path(source.removeprefix("file://"))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if format == "hf":
        ds = datasets.load_dataset(source, split=split, **kwargs)
        yield from ds
        return

    raise ValueError(f"unknown format: {format!r}")
