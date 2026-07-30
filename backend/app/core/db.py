from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession

from app.core.config import get_settings

settings = get_settings()
engine = create_async_engine(settings.database_url, echo=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    # expire_on_commit=False: SQLModel's AsyncSession.refresh()/expired-attribute
    # reload path is incompatible with this SQLAlchemy version under aiosqlite
    # (raises MissingGreenlet). Since callers just build response DTOs from
    # ORM attributes right after commit, cached (pre-expire) values are fine.
    async with SQLModelAsyncSession(engine, expire_on_commit=False) as session:
        yield session
