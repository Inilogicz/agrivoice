"""Database engine and session factory (async SQLAlchemy 2)."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

_settings = get_settings()

is_sqlite = _settings.DATABASE_URL.startswith("sqlite")

engine = create_async_engine(
    _settings.DATABASE_URL,
    echo=_settings.DEBUG,
    future=True,
    # Pool sizing only applies to server databases (PostgreSQL)
    **(
        {}
        if is_sqlite
        else {
            "pool_size": _settings.DATABASE_POOL_SIZE,
            "max_overflow": _settings.DATABASE_MAX_OVERFLOW,
        }
    ),
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def create_sqlite_schema() -> None:
    """
    Create tables directly for SQLite (demo deployments without PostgreSQL).
    PostgreSQL uses Alembic migrations instead.
    """
    from app.db import models  # noqa: F401, PLC0415  (register all tables)
    from app.db.base import Base  # noqa: PLC0415

    def _create_and_add_missing_columns(sync_conn) -> None:
        Base.metadata.create_all(sync_conn)
        # create_all skips existing tables; add columns introduced since then.
        # (New columns are nullable, so SQLite's ADD COLUMN is enough.)
        inspector = inspect(sync_conn)
        for table in Base.metadata.sorted_tables:
            existing = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing:
                    col_type = column.type.compile(dialect=sync_conn.dialect)
                    sync_conn.execute(
                        text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}')
                    )

    async with engine.begin() as conn:
        await conn.run_sync(_create_and_add_missing_columns)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
