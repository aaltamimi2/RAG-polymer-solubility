"""DuckDB database singleton for STRAP solvent/polymer data."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _sanitize_table_name(name: str) -> str:
        table = re.sub(r"[^a-z0-9_]", "_", name.lower().strip().replace("-", "_").replace(" ", "_"))
        if not table:
            return "table"
        if table[0].isdigit():
            table = f"t_{table}"
        return table

    def _load_all_csvs(self) -> None:
        if not self.data_dir.exists():
            logger.warning(
                "Data directory not found: %s — database will have no tables",
                self.data_dir,
            )
            return
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", self.data_dir)
            return
        for csv_path in csv_files:
            table_name = self._sanitize_table_name(csv_path.stem)
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.columns = [self._sanitize_column(c) for c in df.columns]
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.register("_tmp_df", df)
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _tmp_df")
            self.conn.unregister("_tmp_df")

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        return self.conn


_database: Optional[Database] = None
_database_cache: dict[Path, Database] = {}


def _resolve_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    return Path(data_dir).resolve() if data_dir else Database._default_data_dir().resolve()


def get_database(data_dir: Optional[str | Path] = None) -> Database:
    """Return a cached Database instance scoped to the resolved data directory."""
    global _database
    resolved_dir = _resolve_data_dir(data_dir)
    database = _database_cache.get(resolved_dir)
    if database is None:
        database = Database(resolved_dir)
        _database_cache[resolved_dir] = database
    _database = database
    return database


def get_connection(data_dir: Optional[str | Path] = None) -> duckdb.DuckDBPyConnection:
    """Convenience: return the DuckDB connection directly."""
    return get_database(data_dir).get_connection()
