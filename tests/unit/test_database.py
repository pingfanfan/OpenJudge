from pathlib import Path

import pytest
from sqlalchemy import text

from prism.storage.database import Database
from prism.storage.schema import Model, Run


@pytest.mark.asyncio
async def test_init_creates_tables(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    await db.init()
    async with db.session() as s:
        # query should not raise
        await s.execute(text("SELECT 1 FROM runs WHERE 1=0"))


@pytest.mark.asyncio
async def test_upsert_model(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    await db.init()
    async with db.session() as s:
        s.add(Model(id="m1", provider="openai", model="gpt-5"))
        await s.commit()
    async with db.session() as s:
        got = await s.get(Model, "m1")
        assert got is not None
        assert got.provider == "openai"


@pytest.mark.asyncio
async def test_create_run(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    await db.init()
    async with db.session() as s:
        s.add(Run(id="r1", suite="quick", config_hash="abc"))
        await s.commit()
    async with db.session() as s:
        got = await s.get(Run, "r1")
        assert got.status == "pending"
