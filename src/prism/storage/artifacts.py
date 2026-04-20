import json
import os
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str, relative: str) -> Path:
        return self.root / run_id / relative

    def put(self, run_id: str, relative: str, data: Any) -> None:
        path = self._path(run_id, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def get(self, run_id: str, relative: str) -> Any | None:
        path = self._path(run_id, relative)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list(self, run_id: str) -> list[str]:
        base = self.root / run_id
        if not base.exists():
            return []
        return [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()]
