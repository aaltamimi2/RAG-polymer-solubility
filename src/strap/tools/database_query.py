"""Database query tools for STRAP deepagents.

Six plain-function tools that let an agent explore and query the
DuckDB-backed solvent / polymer database.  Every function obtains its
own connection lazily via ``get_connection()`` so there is no module-level
database state.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from strap.database import get_connection
from strap.tools._helpers import safe_tool_wrapper, DataValidator, truncate_output

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers (not exported)
# ------------------------------------------------------------------

def _table_schemas(conn) -> Dict[str, dict]:
    """Build a {table_name: schema_dict} mapping from the live connection."""
    tables_df = conn.execute("SHOW TABLES").fetchdf()
    schemas: Dict[str, dict] = {}
    for table_name in tables_df["name"]:
        schema_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        schemas[table_name] = {
            "columns": list(schema_df["column_name"]),
            "types": dict(zip(schema_df["column_name"], schema_df["column_type"])),
            "row_count": row_count,
        }
    return schemas


def _execute_query(conn, query: str, limit: int = 100) -> dict:
    """Execute *query* safely and return a result dict."""
    try:
        query_lower = query.lower().strip()
        dangerous_keywords = [
            "drop", "delete", "insert", "update", "alter", "create", "truncate",
        ]
        if any(kw in query_lower.split() for kw in dangerous_keywords):
            return {"success": False, "error": "Unsafe operation detected", "query": query}

        if "limit" not in query_lower and not query_lower.rstrip().endswith(";"):
            query = f"{query.rstrip(';')} LIMIT {limit}"

        result_df = conn.execute(query).fetchdf()
        preview = (
            result_df.head(10).to_markdown(index=False) if len(result_df) > 0 else "No data"
        )
        return {
            "success": True,
            "query": query,
            "rows": len(result_df),
            "columns": list(result_df.columns),
            "data": result_df.to_dict("records"),
            "dataframe": result_df,
            "preview": preview,
            "dtypes": {str(k): str(v) for k, v in result_df.dtypes.to_dict().items()},
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}


def _get_sample_data(conn, table_name: str, n: int = 3) -> str:
    try:
        df = conn.execute(f"SELECT * FROM {table_name} LIMIT {n}").fetchdf()
        result = df.to_markdown(index=False)
        del df
        return result
    except Exception as e:
        return f"Error: {e}"


def _verify_inputs(
    conn,
    table_name: str,
    columns: Dict[str, str],
    values: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, str]:
    """Comprehensive input verification against the live connection."""
    issues: List[str] = []
    warnings: List[str] = []

    validator = DataValidator(conn)

    # Verify table
    table_val = validator.verify_table_exists(table_name)
    if not table_val.is_valid:
        return False, f"Table '{table_name}' not found. {table_val.warnings}"

    # Get schema once
    try:
        schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        available_cols = set(schema["column_name"].values)
    except Exception as e:
        return False, f"Could not get schema: {e}"

    # Verify all columns
    for purpose, col_name in columns.items():
        if col_name not in available_cols:
            issues.append(f"Column '{col_name}' ({purpose}) not found")
            similar = [c for c in available_cols if col_name.lower() in c.lower()]
            if similar:
                warnings.append(f"Did you mean: {similar}?")

    if issues:
        msg = "Verification failed:\n- " + "\n- ".join(issues)
        if warnings:
            msg += "\n\n" + "\n".join(warnings)
        return False, msg

    # Verify values if provided
    if values:
        for col_name, expected_vals in values.items():
            if col_name not in available_cols:
                continue
            for val in expected_vals:
                val_result = validator.verify_value_exists(table_name, col_name, val)
                if not val_result.is_valid:
                    issues.append(f"Value '{val}' not found in {col_name}")
                    if val_result.warnings:
                        warnings.extend(val_result.warnings[:1])

    if issues:
        msg = "Value verification failed:\n- " + "\n- ".join(issues)
        if warnings:
            msg += "\n\nAvailable: " + str(warnings[0]) if warnings else ""
        return False, msg

    return True, "All inputs verified"


# ==================================================================
# Public tool functions
# ==================================================================


@safe_tool_wrapper
def list_tables() -> str:
    """List all available SQL tables with schemas, row counts, and data quality info.

    WHEN TO USE:
    - "What tables are available?"
    - "Show me the database schema"
    """
    conn = get_connection()
    schemas = _table_schemas(conn)

    if not schemas:
        return "No tables available."

    info_parts = ["Available Tables:\n"]
    for table_name, schema in schemas.items():
        info_parts.append(f"\n**Table: {table_name}** ({schema['row_count']} rows)")
        info_parts.append("Columns:")
        for col, dtype in schema["types"].items():
            try:
                if "INT" in str(dtype).upper() or "DOUBLE" in str(dtype).upper() or "FLOAT" in str(dtype).upper():
                    stats = conn.execute(
                        f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name}"
                    ).fetchone()
                    info_parts.append(
                        f"  - {col}: {dtype} [min={stats[0]:.4f}, max={stats[1]:.4f}, avg={stats[2]:.4f}]"
                    )
                else:
                    unique_count = conn.execute(
                        f"SELECT COUNT(DISTINCT {col}) FROM {table_name}"
                    ).fetchone()[0]
                    info_parts.append(f"  - {col}: {dtype} [{unique_count} unique values]")
            except Exception:
                info_parts.append(f"  - {col}: {dtype}")

    return "\n".join(info_parts)


@safe_tool_wrapper
def describe_table(table_name: str) -> str:
    """Get detailed information about a specific table including sample data and statistics.

    Args:
        table_name: Name of the table to describe

    WHEN TO USE:
    - "Describe the solvents table"
    - "What columns does this table have?"
    """
    conn = get_connection()
    schemas = _table_schemas(conn)

    if table_name not in schemas:
        available = list(schemas.keys())
        return f"Error: Table '{table_name}' not found. Available tables: {available}"

    schema = schemas[table_name]
    output = [f"**Table: {table_name}**\n", f"Rows: {schema['row_count']}\n", "Columns:"]

    for col, dtype in schema["types"].items():
        try:
            if "INT" in str(dtype).upper() or "DOUBLE" in str(dtype).upper() or "FLOAT" in str(dtype).upper():
                stats = conn.execute(
                    f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name}"
                ).fetchone()
                output.append(
                    f"  - {col}: {dtype} [min={stats[0]:.4f}, max={stats[1]:.4f}, avg={stats[2]:.4f}]"
                )
            else:
                unique_count = conn.execute(
                    f"SELECT COUNT(DISTINCT {col}) FROM {table_name}"
                ).fetchone()[0]
                output.append(f"  - {col}: {dtype} [{unique_count} unique values]")
        except Exception:
            output.append(f"  - {col}: {dtype}")

    output.append(f"\n**Sample data (5 rows):**")
    output.append(_get_sample_data(conn, table_name, 5))

    return "\n".join(output)


@safe_tool_wrapper
def check_column_values(table_name: str, column_name: str, limit: int = 50) -> str:
    """Check what values exist in a specific column with frequency counts.

    Args:
        table_name: Table to query
        column_name: Column to inspect
        limit: Max unique values to return (default: 50)

    WHEN TO USE:
    - "What polymers are in the database?"
    - "Show unique values in the solvent column"
    """
    conn = get_connection()

    is_valid, msg = _verify_inputs(conn, table_name, {"column": column_name})
    if not is_valid:
        return msg

    query = f"""
    SELECT {column_name}, COUNT(*) as count
    FROM {table_name}
    GROUP BY {column_name}
    ORDER BY count DESC
    LIMIT {limit}
    """
    result_df = conn.execute(query).fetchdf()

    output = f"**Unique values in {table_name}.{column_name}:**\n\n"
    output += result_df.to_markdown(index=False)
    output += f"\n\nTotal unique values: {len(result_df)}"

    total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    output += f"\nTotal rows in table: {total_rows}"

    del result_df
    return output


@safe_tool_wrapper
def query_database(sql_query: str, export_csv: bool = False) -> str:
    """Execute a SQL query with validation and error reporting.

    Args:
        sql_query: SQL query to execute
        export_csv: Create a CSV export of results (default: False)

    WHEN TO USE:
    - "Query the database for LDPE solubility data"
    - "Run a SQL query to find solvents with high selectivity"
    """
    conn = get_connection()
    result = _execute_query(conn, sql_query)

    if result["success"]:
        df = result["dataframe"]

        # Generate CSV export if requested
        export_id = None
        if export_csv and result["rows"] > 0:
            try:
                from export_manager import export_manager

                export_id = export_manager.create_export(
                    data=df.to_dict(orient="records"),
                    tool_name="query_database",
                    columns=df.columns.tolist(),
                )
            except Exception as e:
                logger.error(f"Failed to create CSV export: {e}")

        # Format output
        output = (
            f"**Query Results**\n\nQuery: `{result['query']}`\n\n"
            f"Rows returned: {result['rows']}\n\n"
        )

        if export_id:
            output += f"**CSV Export Available:** `/api/export/{export_id}`\n\n"

        if result["rows"] > 0:
            output += "**Data:**\n" + result["preview"]
            if result["rows"] > 10:
                output += f"\n\n_(Showing first 10 of {result['rows']} rows)_"
        else:
            output += "No rows matched the query."
        return output
    else:
        return (
            f"**Query Error**\n\nQuery: `{result['query']}`\n\n"
            f"Error: {result['error']}\n\n"
            f"Tip: Use check_column_values() to verify column names and values."
        )


@safe_tool_wrapper
def validate_and_query(
    table_name: str,
    required_columns: str = "",
    filter_column: Optional[str] = None,
    filter_values: Optional[str] = None,
    sql_query: Optional[str] = None,
    filters: Optional[str] = None,
) -> str:
    """Validate inputs BEFORE executing a query to prevent hallucinations.

    Also supports data accuracy verification: when filters is provided and
    required_columns is empty, runs a count query with WHERE clause and
    shows 5 sample rows matching the filter.

    Args:
        table_name: Table to validate against
        required_columns: Comma-separated column names to verify (default: "")
        filter_column: Optional column to check values in
        filter_values: Optional comma-separated values to verify exist
        sql_query: Optional query to run if validation passes
        filters: Optional SQL WHERE clause filter for data verification

    WHEN TO USE:
    - "Validate that these columns exist before querying"
    - "Check if LDPE exists in the polymer column"
    - "Verify the data for LDPE in the solvents table"
    - "Check how many rows match this filter"
    """
    conn = get_connection()

    # --- Data verification mode ---
    if filters is not None and not required_columns.strip():
        where_clause = f"WHERE {filters}" if filters else ""

        count_query = f"SELECT COUNT(*) FROM {table_name} {where_clause}"
        count = conn.execute(count_query).fetchone()[0]

        sample_query = f"SELECT * FROM {table_name} {where_clause} LIMIT 5"
        sample_df = conn.execute(sample_query).fetchdf()

        output = f"**Data Verification for {table_name}**\n\n"
        output += f"Filter: {filters or 'None'}\n"
        output += f"Total matching rows: {count}\n\n"

        if count > 0:
            output += "Sample data:\n"
            output += sample_df.to_markdown(index=False)
        else:
            output += "**No data matches these criteria!**\n"
            output += "Please verify:\n"
            output += "1. Column names are correct\n"
            output += "2. Filter values exist in the data\n"
            output += "3. Data types match (e.g., strings need quotes)\n"

        del sample_df
        return output

    # --- Validation mode ---
    validator = DataValidator(conn)

    output = ["**Input Validation Report**\n"]
    all_valid = True

    columns = [c.strip() for c in required_columns.split(",") if c.strip()]

    table_val = validator.verify_table_exists(table_name)
    if table_val.is_valid:
        output.append(f"Table '{table_name}' exists ({table_val.verified_row_count} rows)")
    else:
        output.append(f"Table issue: {table_val.issues}")
        all_valid = False

    for col in columns:
        col_val = validator.verify_column_exists(table_name, col)
        if col_val.is_valid:
            output.append(f"Column '{col}' exists")
        else:
            output.append(f"Column '{col}': {col_val.issues}")
            if col_val.warnings:
                output.append(f"   {col_val.warnings[0]}")
            all_valid = False

    if filter_column and filter_values:
        values = [v.strip() for v in filter_values.split(",")]
        for val in values:
            val_result = validator.verify_value_exists(table_name, filter_column, val)
            if val_result.is_valid:
                output.append(
                    f"Value '{val}' found in {filter_column} ({val_result.verified_row_count} rows)"
                )
            else:
                output.append(f"Value '{val}' NOT found in {filter_column}")
                if val_result.warnings:
                    output.append(f"   {val_result.warnings[0]}")
                all_valid = False

    if sql_query and all_valid:
        output.append("\n**Query Execution:**")
        result = _execute_query(conn, sql_query)
        if result["success"]:
            output.append(f"Query successful: {result['rows']} rows returned")
            if result["rows"] > 0:
                output.append("\n" + result["preview"])
        else:
            output.append(f"Query failed: {result['error']}")
    elif sql_query and not all_valid:
        output.append("\nQuery not executed due to validation failures")

    return "\n".join(output)
