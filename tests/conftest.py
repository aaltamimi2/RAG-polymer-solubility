"""Shared fixtures for STRAP tests."""

import pytest

from strap.database import Database


@pytest.fixture(scope="session")
def db():
    """Session-scoped Database instance."""
    return Database()


@pytest.fixture(scope="session")
def conn(db):
    """Session-scoped DuckDB connection."""
    return db.get_connection()
