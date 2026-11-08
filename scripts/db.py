"""One place that decides which database the project talks to.

DATABASE_URL set  -> PostgreSQL (for example Docker Compose locally, or AWS RDS)
DATABASE_URL unset -> the SQLite file in database/delays.db (what Streamlit Community Cloud uses)

Everything else (loader, features, model, app) calls connect() and does not care which one it got.
"""
import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text  # noqa: F401  (text is re-exported for callers)
from sqlalchemy.engine import Engine, make_url

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = ROOT / "database" / "delays.db"


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return f"sqlite:///{SQLITE_PATH.as_posix()}"
    # Heroku-style and plain "postgres://" URLs -> the driver SQLAlchemy expects
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


def is_postgres() -> bool:
    return database_url().startswith("postgresql")


@lru_cache(maxsize=1)
def _engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        return create_engine(url)
    # pool_pre_ping: RDS drops idle connections; check before reuse
    return create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=2)


def get_engine() -> Engine:
    return _engine(database_url())


def connect():
    """A SQLAlchemy connection. Works with pandas.read_sql and `with closing(...)`."""
    return get_engine().connect()


def describe() -> str:
    """Safe-to-print description of the target (password hidden)."""
    url = make_url(database_url())
    if url.get_backend_name() == "sqlite":
        return f"SQLite file {Path(url.database).name}"
    return f"PostgreSQL {url.host}:{url.port or 5432}/{url.database}"


def table_exists(conn, name: str) -> bool:
    if is_postgres():
        q = text("SELECT 1 FROM information_schema.tables WHERE table_name = :n")
    else:
        q = text("SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = :n")
    return conn.execute(q, {"n": name}).first() is not None
