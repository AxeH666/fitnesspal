"""Synchronous SQLAlchemy engine and session factory for Phase 0."""

from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create the PostgreSQL engine without attempting a connection."""

    return create_engine(database_url or get_settings().database_url, pool_pre_ping=True)


engine = create_database_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db_session() -> Generator[Session, None, None]:
    """Yield a request-scoped database session."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

