"""Tests for the DuckDB database layer."""

from strap.database import Database


def test_tables_loaded(conn):
    """Both CSV files should be loaded as tables."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
    ]
    assert "common_solvents_database" in tables
    assert "solvent_data" in tables


def test_columns_sanitized(conn):
    """Column names should be lower-cased with special chars replaced."""
    cols = [
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='common_solvents_database'"
        ).fetchall()
    ]
    # Original header has "Solubility (%)" which should become "solubility____"
    assert "solubility____" in cols
    assert "polymer" in cols
    assert "solvent" in cols


def test_sanitize_column_static():
    assert Database._sanitize_column("Temperature (°C)") == "temperature___c_"
    assert Database._sanitize_column("Solubility (%)") == "solubility____"


def test_basic_query(conn):
    """Should be able to query polymer data."""
    result = conn.execute(
        "SELECT COUNT(*) FROM common_solvents_database"
    ).fetchone()
    assert result[0] > 0
