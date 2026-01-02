"""
Async Database Wrapper for DuckDB

Provides async wrapper for DuckDB with lock-protected access.
DuckDB connections are not thread-safe, so we use asyncio.Lock to serialize access
while still allowing concurrent execution of other async operations.
"""

import asyncio
import duckdb
import pandas as pd
from typing import List, Union
import logging
from async_utils import run_in_thread

logger = logging.getLogger(__name__)


class AsyncDuckDBWrapper:
    """
    Async wrapper for DuckDB with lock-protected access.

    DuckDB connections are single-threaded, so we use an asyncio.Lock
    to serialize database access while allowing other async operations
    to proceed concurrently.

    Usage:
        async_db = AsyncDuckDBWrapper(sync_conn)

        # Single query
        df = await async_db.execute_async("SELECT * FROM table")

        # Multiple queries in parallel (lock-protected)
        results = await async_db.execute_many_async([
            "SELECT * FROM table1",
            "SELECT * FROM table2",
            "SELECT * FROM table3"
        ])
    """

    def __init__(self, sync_conn: duckdb.DuckDBPyConnection):
        """
        Initialize async DB wrapper.

        Args:
            sync_conn: Existing synchronous DuckDB connection
        """
        self.sync_conn = sync_conn
        self._lock = asyncio.Lock()
        logger.info("AsyncDuckDBWrapper initialized")

    async def execute_async(self, query: str) -> pd.DataFrame:
        """
        Execute single query asynchronously with lock protection.

        The query runs in a thread pool to avoid blocking the event loop,
        and the lock ensures only one query accesses the DB at a time.

        Args:
            query: SQL query string

        Returns:
            DataFrame with query results

        Raises:
            Exception: If query execution fails
        """
        async with self._lock:
            def _execute():
                return self.sync_conn.execute(query).fetchdf()

            try:
                result = await run_in_thread(_execute)
                return result
            except Exception as e:
                logger.error(f"Query failed: {query[:100]}... Error: {e}")
                raise

    async def execute_many_async(self, queries: List[str]) -> List[pd.DataFrame]:
        """
        Execute multiple queries with lock-protected parallel execution.

        While queries are serialized at the DB level (due to DuckDB limitations),
        this method allows other async operations to proceed while waiting for
        DB access, improving overall concurrency.

        Args:
            queries: List of SQL query strings

        Returns:
            List of DataFrames corresponding to each query

        Example:
            queries = [
                "SELECT * FROM table WHERE polymer='PVDF'",
                "SELECT * FROM table WHERE polymer='PLA'",
                "SELECT * FROM table WHERE polymer='PS'"
            ]
            results = await async_db.execute_many_async(queries)
            # All queries execute, event loop free for other tasks
        """
        tasks = [self.execute_async(q) for q in queries]
        return await asyncio.gather(*tasks)

    async def execute_query_dict_async(self, query: str) -> dict:
        """
        Execute query and return result as dictionary with metadata.

        This matches the pattern used in the SQLDatabase class where queries
        return both the DataFrame and metadata like preview strings.

        Args:
            query: SQL query string

        Returns:
            Dictionary with 'dataframe', 'preview', 'row_count' keys
        """
        async with self._lock:
            def _execute():
                df = self.sync_conn.execute(query).fetchdf()
                preview = df.head(50).to_string() if len(df) > 0 else "No results"
                return {
                    "dataframe": df,
                    "preview": preview,
                    "row_count": len(df)
                }

            try:
                result = await run_in_thread(_execute)
                return result
            except Exception as e:
                logger.error(f"Query failed: {query[:100]}... Error: {e}")
                raise

    async def get_table_schema_async(self, table_name: str) -> dict:
        """
        Get table schema asynchronously.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with column names and types
        """
        query = f"DESCRIBE {table_name}"
        df = await self.execute_async(query)

        schema = {}
        for _, row in df.iterrows():
            schema[row['column_name']] = row['column_type']

        return schema

    def close(self):
        """
        Close the underlying DuckDB connection.

        Note: This is synchronous as it's typically called during shutdown.
        """
        try:
            self.sync_conn.close()
            logger.info("DuckDB connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
