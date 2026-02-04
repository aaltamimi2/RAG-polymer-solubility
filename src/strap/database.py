"""DuckDB database singleton for STRAP solvent/polymer data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class Database:
    """In-memory DuckDB database that loads all CSVs from a data directory."""

    def __init__(self, data_dir: Optional[str | Path] = None):
        self.data_dir = Path(data_dir) if data_dir else self._default_data_dir()
        self.conn = duckdb.connect(":memory:")
        self._load_all_csvs()

    @staticmethod
    def _default_data_dir() -> Path:
        return Path(__file__).resolve().parent.parent.parent / "data"

    @staticmethod
    def _sanitize_column(col: str) -> str:
        return re.sub(r"[^a-z0-9_]", "_", col.lower().strip())

    def _load_all_csvs(self) -> None:
        for csv_path in sorted(self.data_dir.glob("*.csv")):
            table_name = csv_path.stem.lower().replace("-", "_").replace(" ", "_")
            table_name = re.sub(r"[^a-z0-9_]", "_", table_name)
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.columns = [self._sanitize_column(c) for c in df.columns]
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.register("_tmp_df", df)
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_df")
            self.conn.unregister("_tmp_df")

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        return self.conn


_database: Optional[Database] = None


def get_database(data_dir: Optional[str | Path] = None) -> Database:
    """Return (or create) the module-level Database singleton."""
    global _database
    if _database is None:
        _database = Database(data_dir)
    return _database


def get_connection(data_dir: Optional[str | Path] = None) -> duckdb.DuckDBPyConnection:
    """Convenience: return the DuckDB connection directly."""
    return get_database(data_dir).get_connection()
