"""Shared fixtures for STRAP tests."""

import pytest

from strap.database import Database
from strap.testing_utils import block_model_access


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_model_calls: opt out of the global deterministic test model-call blocker",
    )


@pytest.fixture(scope="session")
def db():
    """Session-scoped Database instance."""
    return Database()


@pytest.fixture(scope="session")
def conn(db):
    """Session-scoped DuckDB connection."""
    return db.get_connection()


@pytest.fixture(autouse=True)
def _block_external_model_calls(request):
    """Prevent accidental live model/Gemini access in deterministic tests."""
    if request.node.get_closest_marker("allow_model_calls") is not None:
        yield
        return

    with block_model_access():
        yield
