from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from prism.storage.schema import Base


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self.path}", future=True)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            # WAL journal mode allows concurrent readers + one writer, much
            # friendlier to our multi-task orchestrator than the default DELETE
            # mode. Also bump busy_timeout so short lock contention retries
            # instead of erroring out (e3q8).
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=5000"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_factory() as s:
            yield s

    async def dispose(self) -> None:
        await self._engine.dispose()
