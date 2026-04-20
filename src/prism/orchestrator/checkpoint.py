from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import aiosqlite

from prism.orchestrator.matrix import Cell

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoint (
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (run_id, model_id, prompt_id, seed)
);
"""


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(_SCHEMA)
            await db.commit()

    async def mark(self, run_id: str, cell: Cell, status: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO checkpoint(run_id,model_id,prompt_id,seed,status) VALUES (?,?,?,?,?) "
                "ON CONFLICT(run_id,model_id,prompt_id,seed) DO UPDATE SET status=excluded.status",
                (run_id, cell.model_id, cell.prompt_id, cell.seed, status),
            )
            await db.commit()

    async def status(self, *, run_id: str, cell: Cell) -> str:
        _sql = (
            "SELECT status FROM checkpoint"
            " WHERE run_id=? AND model_id=? AND prompt_id=? AND seed=?"
        )
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                _sql,
                (run_id, cell.model_id, cell.prompt_id, cell.seed),
            )
            row = await cursor.fetchone()
            return str(row[0]) if row else "pending"

    async def pending_cells(self, run_id: str, cells: Iterable[Cell]) -> AsyncIterator[Cell]:
        _sql = (
            "SELECT status FROM checkpoint"
            " WHERE run_id=? AND model_id=? AND prompt_id=? AND seed=?"
        )
        async with aiosqlite.connect(self.path) as db:
            for cell in cells:
                cursor = await db.execute(
                    _sql,
                    (run_id, cell.model_id, cell.prompt_id, cell.seed),
                )
                row = await cursor.fetchone()
                status = row[0] if row else "pending"
                if status == "pending":
                    yield cell
