from pathlib import Path

import pytest

from prism.orchestrator.checkpoint import CheckpointStore
from prism.orchestrator.matrix import Cell


@pytest.mark.asyncio
async def test_mark_and_query(tmp_path: Path):
    cp = CheckpointStore(tmp_path / "cp.db")
    await cp.init()
    c = Cell(model_id="m", prompt_id="p1", seed=0)
    assert await cp.status(run_id="r", cell=c) == "pending"
    await cp.mark(run_id="r", cell=c, status="running")
    assert await cp.status(run_id="r", cell=c) == "running"
    await cp.mark(run_id="r", cell=c, status="done")
    assert await cp.status(run_id="r", cell=c) == "done"


@pytest.mark.asyncio
async def test_pending_cells_filter(tmp_path: Path):
    cp = CheckpointStore(tmp_path / "cp.db")
    await cp.init()
    cells = [Cell("m", f"p{i}", 0) for i in range(3)]
    await cp.mark("r", cells[0], "done")
    await cp.mark("r", cells[1], "running")
    pending = [c async for c in cp.pending_cells("r", cells)]
    assert pending == [cells[2]]
