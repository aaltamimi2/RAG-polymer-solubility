"""Solvent property lookup, ranking, and separation-analysis tools.
Extracted from the monolithic agent SQL source.  Each public function is
decorated with ``@safe_tool_wrapper`` for uniform error handling and output
truncation.  The ``@tool`` (LangChain) decorator has been removed so that
tool registration is handled by the caller.
Database access is **lazy**: every function calls ``get_connection()`` at
invocation time rather than binding to a module-level connection object.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from strap.database import get_connection
from strap.services.tool_response_service import json_tool_error, json_tool_response, json_tool_success
from strap.tools._helpers import (
    get_cross_database_properties,
    normalize_solvent_name,
    safe_tool_wrapper,
    truncate_output,
)
logger = logging.getLogger(__name__)
def _solvent_error(tool_name: str, message: str, *, error_code: str = "lookup_failed", **data) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)
# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_SOLVENT_DATA_TABLE: Optional[str] = None
# ---------------------------------------------------------------------------
# Internal schema helpers (replaces sql_db.table_schemas access)
# ---------------------------------------------------------------------------
def _get_table_names() -> List[str]:
    """Return all table names known to the DuckDB connection."""
    conn = get_connection()
    df = conn.execute("SHOW TABLES").fetchdf()
    return list(df["name"].values)
def _get_table_schema(table_name: str) -> Dict[str, Any]:
    """Return column names, types dict, and row count for *table_name*.
    Returns a dict compatible with the old ``sql_db.table_schemas[name]``
    layout::
        {
            "columns": ["col1", "col2", ...],
            "types":   {"col1": "VARCHAR", "col2": "DOUBLE", ...},
            "row_count": 1234,
        }
    """
    conn = get_connection()
    desc = conn.execute(f"DESCRIBE {table_name}").fetchdf()
    columns = list(desc["column_name"].values)
    types = dict(zip(desc["column_name"], desc["column_type"]))
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return {"columns": columns, "types": types, "row_count": row_count}
def _table_exists(table_name: str) -> bool:
    return table_name in _get_table_names()
def _execute_query(query: str, limit: int = 100) -> Dict[str, Any]:
    """Execute a read-only SQL query and return a result dict.
    The returned dict mirrors the old ``sql_db.execute_query`` interface::
        {
            "success": bool,
            "query": str,
            "rows": int,
            "columns": list,
            "data": list[dict],
            "dataframe": pd.DataFrame,
            "preview": str,          # markdown table
            "dtypes": dict,
            "error": str | None,
        }
    """
    conn = get_connection()
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
# ---------------------------------------------------------------------------
# Helper: auto-detect solvent data table
# ---------------------------------------------------------------------------
def get_solvent_table_name() -> Optional[str]:
    """Auto-detect the solvent data table name."""
    global _SOLVENT_DATA_TABLE
    all_tables = _get_table_names()
    if _SOLVENT_DATA_TABLE and _SOLVENT_DATA_TABLE in all_tables:
        return _SOLVENT_DATA_TABLE
    # Try to find a table with solvent properties
    for table_name in all_tables:
        if "solvent" in table_name.lower() and "solubility" not in table_name.lower():
            schema = _get_table_schema(table_name)
            cols_lower = [c.lower() for c in schema["columns"]]
            if (
                any("bp" in c or "boil" in c for c in cols_lower)
                or any("logp" in c for c in cols_lower)
                or any("energy" in c for c in cols_lower)
            ):
                _SOLVENT_DATA_TABLE = table_name
                logger.info(f"Auto-detected solvent data table: {table_name}")
                return table_name
    return None
# ---------------------------------------------------------------------------
# Helper: identify special columns
# ---------------------------------------------------------------------------
def get_solvent_name_column(table_name: str) -> Optional[str]:
    """Get the column name that contains solvent names."""
    if not _table_exists(table_name):
        return None
    schema = _get_table_schema(table_name)
    cols = schema["columns"]
    # Priority order for solvent name column
    priority_patterns = ["solvent_name", "solvent", "name", "compound"]
    for pattern in priority_patterns:
        for col in cols:
            if pattern in col.lower():
                return col
    # If no match, return first string column
    for col, dtype in schema["types"].items():
        if "VARCHAR" in str(dtype).upper() or "TEXT" in str(dtype).upper():
            return col
    return cols[0] if cols else None
def get_cosmobase_column(table_name: str) -> Optional[str]:
    """Get the 'Solvent name in cosmobase' column for exact matching."""
    if not _table_exists(table_name):
        return None
    schema = _get_table_schema(table_name)
    cols = schema["columns"]
    for col in cols:
        if "cosmobase" in col.lower():
            return col
    return None
# ---------------------------------------------------------------------------
# Helper: robust fuzzy solvent lookup (async)
# ---------------------------------------------------------------------------
async def lookup_solvent_properties(solvent_names: list, solvent_table: str) -> dict:
    """Look up solvent properties for multiple solvents with robust fuzzy matching.
    Uses multiple strategies to match solvent names:
    1. Exact match on COSMOBASE/name column
    2. Common abbreviation mapping
    3. Bidirectional fuzzy matching
    4. Partial substring matching
    Returns a dict mapping solvent names to their properties.
    """
    if not solvent_table or not _table_exists(solvent_table):
        return {}
    from strap.solvent_registry import ABBREVIATION_MAP
    conn = get_connection()
    schema = _get_table_schema(solvent_table)
    cols = schema["columns"]
    cols_lower = {c.lower(): c for c in cols}
    # Get column names
    cosmobase_col = get_cosmobase_column(solvent_table)
    name_col = get_solvent_name_column(solvent_table)
    # Property columns
    logp_col = next((cols_lower[k] for k in cols_lower if "logp" in k), None)
    bp_col = next((cols_lower[k] for k in cols_lower if "bp" in k or "boil" in k), None)
    energy_col = next((cols_lower[k] for k in cols_lower if "energy" in k), None)
    cp_col = next((cols_lower[k] for k in cols_lower if "cp" in k and "logp" not in k), None)
    match_col = cosmobase_col or name_col
    if not match_col:
        return {}
    def _find_solvent_match(solvent: str):
        """Try multiple strategies to find solvent properties."""
        sol_lower = solvent.lower().strip()
        sol_normalized = sol_lower.replace("-", "").replace(" ", "").replace(",", "")
        # Escaped variants for safe SQL interpolation
        sol_lower_safe = sol_lower.replace("'", "''")
        sol_normalized_safe = sol_normalized.replace("'", "''")
        # Strategy 1: Exact match
        query1 = f'SELECT * FROM {solvent_table} WHERE LOWER("{match_col}") = \'{sol_lower_safe}\''
        try:
            df = conn.execute(query1).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass
        # Strategy 2: Try abbreviation mapping
        if sol_lower in ABBREVIATION_MAP:
            full_name = ABBREVIATION_MAP[sol_lower]
            full_name_safe = full_name.replace("'", "''")
            query2 = (
                f'SELECT * FROM {solvent_table} WHERE LOWER("{match_col}") '
                f"LIKE '%{full_name_safe}%' ORDER BY LENGTH(\"{match_col}\")"
            )
            try:
                df = conn.execute(query2).fetchdf()
                if len(df) > 0:
                    return df.iloc[0]
            except Exception:
                pass
        # Strategy 3: Substring match
        query3 = (
            f'SELECT * FROM {solvent_table} WHERE LOWER("{match_col}") '
            f"LIKE '%{sol_lower_safe}%' ORDER BY LENGTH(\"{match_col}\")"
        )
        try:
            df = conn.execute(query3).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass
        # Strategy 4: Normalized match (remove special characters)
        query4 = f"""
        SELECT * FROM {solvent_table}
        WHERE REPLACE(REPLACE(REPLACE(LOWER("{match_col}"), '-', ''), ' ', ''), ',', '') LIKE '%{sol_normalized_safe}%'
        ORDER BY LENGTH("{match_col}")
        """
        try:
            df = conn.execute(query4).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass
        # Strategy 5: Check if full name contains the abbreviation as a word
        for abbrev, full in ABBREVIATION_MAP.items():
            if abbrev in sol_lower or sol_lower in full:
                full_safe = full.replace("'", "''")
                query5 = (
                    f'SELECT * FROM {solvent_table} WHERE LOWER("{match_col}") '
                    f"LIKE '%{full_safe}%' ORDER BY LENGTH(\"{match_col}\")"
                )
                try:
                    df = conn.execute(query5).fetchdf()
                    if len(df) > 0:
                        return df.iloc[0]
                except Exception:
                    pass
        return None
    # Find matches for all solvents (run synchronously; DuckDB is not truly async)
    matches = [_find_solvent_match(solvent) for solvent in solvent_names]
    # Extract properties from matches
    props_map: Dict[str, Dict[str, Any]] = {}
    for solvent, row in zip(solvent_names, matches):
        props: Dict[str, Any] = {"logp": None, "bp": None, "energy": None, "cp": None}
        if row is not None:
            props = {
                "logp": row[logp_col] if logp_col and logp_col in row.index else None,
                "bp": row[bp_col] if bp_col and bp_col in row.index else None,
                "energy": row[energy_col] if energy_col and energy_col in row.index else None,
                "cp": row[cp_col] if cp_col and cp_col in row.index else None,
            }
        props_map[solvent] = props
    return props_map
# ===================================================================
# Tool functions
# ===================================================================
@safe_tool_wrapper(structured_output=True)
def get_solvent_properties(solvent_names: str) -> str:
    """Look up detailed properties (BP, LogP, Cp, energy) for specific solvents by name.
    Args:
        solvent_names: Comma-separated solvent names to look up
    WHEN TO USE:
    - "What are the properties of toluene and DMF?"
    - "Look up the boiling point and LogP of acetone"
    """
    table_name = get_solvent_table_name()
    if not table_name:
        return _solvent_error(
            "get_solvent_properties",
            "No solvent properties table found. Upload Solvent_Data.csv first.",
            error_code="table_not_found",
        )
    name_col = get_solvent_name_column(table_name)
    if not name_col:
        return _solvent_error(
            "get_solvent_properties",
            "Could not identify solvent name column.",
            error_code="name_column_not_found",
            table_name=table_name,
        )
    # Parse solvent names
    solvents = [s.strip() for s in solvent_names.split(",") if s.strip()]
    if not solvents:
        return _solvent_error(
            "get_solvent_properties",
            "No solvent names provided.",
            error_code="missing_solvents",
        )
    from strap.solvent_registry import get_search_terms
    # Build query with fuzzy matching + aliases
    conditions = []
    for solvent in solvents:
        solvent_lower = solvent.lower().strip()
        search_terms = get_search_terms(solvent_lower)
        for term in search_terms:
            safe_term = term.replace("'", "''")
            conditions.append(f"LOWER({name_col}) LIKE '%{safe_term}%'")
    where_clause = " OR ".join(conditions)
    query = f"SELECT * FROM {table_name} WHERE {where_clause}"
    result = _execute_query(query, limit=50)
    if not result["success"]:
        return _solvent_error(
            "get_solvent_properties",
            f"Query error: {result.get('error')}",
            error_code="query_failed",
            query=query,
        )
    if result["rows"] == 0:
        # Try exact match
        exact_conditions = [
            "LOWER({col}) = '{val}'".format(col=name_col, val=s.lower().replace("'", "''"))
            for s in solvents
        ]
        query = f"SELECT * FROM {table_name} WHERE {' OR '.join(exact_conditions)}"
        result = _execute_query(query, limit=50)
        if result["rows"] == 0:
            return _solvent_error(
                "get_solvent_properties",
                f"No solvents found matching: {', '.join(solvents)}\n\n"
                "Use `list_solvent_properties()` to see available solvents."
                ,
                error_code="solvents_not_found",
                requested_solvents=solvents,
            )
    output = ["**Solvent Properties**\n"]
    output.append(f"Requested: {', '.join(solvents)}")
    output.append(f"Found: {result['rows']} match(es)\n")
    output.append(result["preview"])
    # Add interpretation
    df = result["dataframe"]
    output.append("\n**Interpretation:**")
    # Find relevant columns
    cols = {c.lower(): c for c in df.columns}
    logp_col = next((cols[k] for k in cols if "logp" in k), None)
    bp_col = next((cols[k] for k in cols if "bp" in k or "boil" in k), None)
    energy_col = next((cols[k] for k in cols if "energy" in k), None)
    if logp_col:
        output.append("- **LogP** (toxicity): Lower/negative = less toxic, higher = more toxic")
    if bp_col:
        output.append("- **Boiling Point**: Higher = harder to remove/recycle")
    if energy_col:
        output.append("- **Energy**: Higher = more expensive to use")
    display_text = "\n".join(output)
    # Build structured data for extraction
    solvent_data: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        solvent_entry: Dict[str, Any] = {"name": row.get(name_col, "Unknown")}
        for col in df.columns:
            col_lower = col.lower()
            if "logp" in col_lower:
                solvent_entry["logp"] = row[col] if pd.notna(row[col]) else None
            elif "bp" in col_lower or "boil" in col_lower:
                solvent_entry["bp_c"] = row[col] if pd.notna(row[col]) else None
            elif "energy" in col_lower:
                solvent_entry["energy_j_g"] = row[col] if pd.notna(row[col]) else None
            elif "cp" in col_lower or "heat" in col_lower:
                solvent_entry["cp_j_gk"] = row[col] if pd.notna(row[col]) else None
            elif "delta_d" in col_lower or "hansen_d" in col_lower:
                solvent_entry["delta_d"] = row[col] if pd.notna(row[col]) else None
            elif "delta_p" in col_lower or "hansen_p" in col_lower:
                solvent_entry["delta_p"] = row[col] if pd.notna(row[col]) else None
            elif "delta_h" in col_lower or "hansen_h" in col_lower:
                solvent_entry["delta_h"] = row[col] if pd.notna(row[col]) else None
        solvent_data.append(solvent_entry)
    # Return structured JSON for extraction
    return json_tool_success(
        display_text,
        tool_name="get_solvent_properties",
        requested_solvents=solvents,
        found_count=result["rows"],
        table_name=table_name,
        solvents=solvent_data,
        preview=result["preview"],
    )
@safe_tool_wrapper(structured_output=True)
def rank_solvents_by_property(
    property_name: str,
    ascending: bool = True,
    limit: int = 20,
    filter_solvents: Optional[str] = None,
) -> str:
    """Rank solvents by a given property (bp, logp, energy, cp).
    Args:
        property_name: Property to rank by ('bp', 'logp', 'energy', 'cp', or column name)
        ascending: True for lowest first, False for highest first
        limit: Number of results (default 20)
        filter_solvents: Comma-separated solvent names to restrict ranking to
    WHEN TO USE:
    - "Which solvents have the lowest energy cost?"
    - "Rank solvents by boiling point"
    - "What are the least toxic solvents by LogP?"
    """
    table_name = get_solvent_table_name()
    if not table_name:
        return _solvent_error(
            "rank_solvents_by_property",
            "No solvent properties table found.",
            error_code="table_not_found",
        )
    # Map common property names to likely column names
    property_map = {
        "bp": ["bp", "bp_c", "boiling_point", "boilingpoint"],
        "boiling": ["bp", "bp_c", "boiling_point"],
        "logp": ["logp", "log_p", "logp_value"],
        "toxicity": ["logp", "log_p"],  # LogP is proxy for toxicity
        "energy": ["energy", "energy_j_g", "energy_cost"],
        "cost": ["energy", "energy_j_g", "energy_cost"],
        "cp": ["cp", "cp_j_gk", "heat_capacity"],
        "heat_capacity": ["cp", "cp_j_gk", "heat_capacity"],
    }
    # Find the actual column name
    schema = _get_table_schema(table_name)
    cols_lower = {
        c.lower().replace(" ", "_").replace("(", "_").replace(")", ""): c
        for c in schema["columns"]
    }
    target_col = None
    prop_lower = property_name.lower().replace(" ", "_")
    # Direct match
    if prop_lower in cols_lower:
        target_col = cols_lower[prop_lower]
    else:
        # Try mapped names
        search_terms = property_map.get(prop_lower, [prop_lower])
        for term in search_terms:
            for col_key, col_name in cols_lower.items():
                if term in col_key:
                    target_col = col_name
                    break
            if target_col:
                break
    if not target_col:
        available = ", ".join(schema["columns"])
        return _solvent_error(
            "rank_solvents_by_property",
            f"Property '{property_name}' not found.\n\nAvailable columns: {available}",
            error_code="property_not_found",
            property_name=property_name,
            available_columns=schema["columns"],
        )
    name_col = get_solvent_name_column(table_name)
    order = "ASC" if ascending else "DESC"
    # Build query
    if filter_solvents:
        solvents = [s.strip() for s in filter_solvents.split(",")]
        conditions = [
            "LOWER({col}) LIKE '%{val}%'".format(col=name_col, val=s.lower().replace("'", "''"))
            for s in solvents
        ]
        where_clause = f"WHERE ({' OR '.join(conditions)}) AND {target_col} IS NOT NULL"
    else:
        where_clause = f"WHERE {target_col} IS NOT NULL"
    query = f"""
    SELECT * FROM {table_name}
    {where_clause}
    ORDER BY {target_col} {order}
    LIMIT {limit}
    """
    result = _execute_query(query, limit=limit)
    if not result["success"]:
        return _solvent_error(
            "rank_solvents_by_property",
            f"Query error: {result.get('error')}",
            error_code="query_failed",
            query=query,
            property_name=property_name,
        )
    direction = "lowest" if ascending else "highest"
    output = [f"**Solvents Ranked by {target_col}** ({direction} first)\n"]
    if filter_solvents:
        output.append(f"Filtered to: {filter_solvents}")
    output.append(f"Results: {result['rows']}\n")
    output.append(result["preview"])
    # Add context
    output.append("\n**Note:** ")
    if "logp" in target_col.lower():
        output.append(
            "Lower/negative LogP generally indicates lower toxicity and higher water solubility."
        )
    elif "energy" in target_col.lower():
        output.append("Lower energy typically means lower operating cost.")
    elif "bp" in target_col.lower():
        output.append(
            "Lower boiling point means easier solvent recovery but may require pressure vessels."
        )
    ranked_rows: list[dict[str, Any]] = []
    for row in result["data"]:
        ranked_rows.append(
            {
                "solvent_name": row.get(name_col, "Unknown"),
                "property_column": target_col,
                "property_value": row.get(target_col),
            }
        )
    return json_tool_success(
        "\n".join(output),
        tool_name="rank_solvents_by_property",
        property_name=property_name,
        property_column=target_col,
        ascending=ascending,
        limit=limit,
        filter_solvents=[s.strip() for s in filter_solvents.split(",")] if filter_solvents else [],
        rows=result["rows"],
        ranked_solvents=ranked_rows,
        preview=result["preview"],
    )
