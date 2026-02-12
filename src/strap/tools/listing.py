"""Listing tools: discover available polymers and solvents in the database."""

from __future__ import annotations

import logging
import asyncio
from typing import Dict, Any, Optional, List

from strap.database import get_connection
from strap.tools._helpers import safe_tool_wrapper, truncate_output

logger = logging.getLogger(__name__)


def _execute_query(query: str, limit: int = 100) -> Dict[str, Any]:
    """Run a read-only query through the shared DuckDB connection.

    Returns a dict compatible with the legacy ``sql_db.execute_query`` shape
    (keys: *success*, *dataframe*, *error*).
    """
    try:
        conn = get_connection()
        query_lower = query.lower().strip()
        dangerous_keywords = [
            "drop", "delete", "insert", "update", "alter", "create", "truncate",
        ]
        if any(kw in query_lower.split() for kw in dangerous_keywords):
            return {"success": False, "error": "Unsafe operation detected"}

        if "limit" not in query_lower and not query_lower.rstrip().endswith(";"):
            query = f"{query.rstrip(';')} LIMIT {limit}"

        result_df = conn.execute(query).fetchdf()
        return {"success": True, "dataframe": result_df}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _get_solvent_table_name() -> Optional[str]:
    """Auto-detect the solvent data table name from the database."""
    conn = get_connection()
    try:
        tables = conn.execute("SHOW TABLES").fetchdf()
        for table_name in tables["name"].tolist():
            if "solvent" in table_name.lower() and "solubility" not in table_name.lower():
                schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                cols_lower = [c.lower() for c in schema["column_name"].tolist()]
                if (
                    any("bp" in c or "boil" in c for c in cols_lower)
                    or any("logp" in c for c in cols_lower)
                    or any("energy" in c for c in cols_lower)
                ):
                    return table_name
    except Exception as e:
        logger.debug(f"Could not detect solvent table: {e}")
    return None


@safe_tool_wrapper
async def list_available_solvents(include_properties: bool = False) -> str:
    """List available solvents across all databases with counts and examples.

    Args:
        include_properties: If True, also display physical properties (BP, LogP,
            Cp, energy) from the solvent_data table.

    WHEN TO USE:
    - "What solvents are in the database?"
    - "Show me available solvents"
    - "List all solvents"
    - "Show me all available solvent properties"
    """
    try:
        output = ["**Available Solvents Summary**\n"]

        # Count solvents in each table
        solvent_data_query = "SELECT COUNT(DISTINCT solvent_name) as count FROM solvent_data"
        gsk_query = "SELECT COUNT(DISTINCT solvent_common_name) as count FROM gsk_dataset"
        common_db_query = "SELECT COUNT(DISTINCT solvent) as count FROM common_solvents_database"

        solvent_data_count = _execute_query(solvent_data_query)
        gsk_count = _execute_query(gsk_query)
        common_db_count = _execute_query(common_db_query)

        if solvent_data_count["success"]:
            count = solvent_data_count["dataframe"].iloc[0]['count']
            output.append(f"**Solvent Data:** {count} unique solvents")

        if gsk_count["success"]:
            count = gsk_count["dataframe"].iloc[0]['count']
            output.append(f"**GSK Dataset:** {count} unique solvents")

        if common_db_count["success"]:
            count = common_db_count["dataframe"].iloc[0]['count']
            output.append(f"**Common Solvents Database:** {count} unique solvents")

        # Get sample solvents from each database
        sample_solvent_data = """
        SELECT DISTINCT solvent_name
        FROM solvent_data
        ORDER BY solvent_name
        LIMIT 10
        """

        sample_gsk = """
        SELECT DISTINCT solvent_common_name
        FROM gsk_dataset
        ORDER BY solvent_common_name
        LIMIT 10
        """

        solvent_data_sample = _execute_query(sample_solvent_data)
        gsk_sample = _execute_query(sample_gsk)

        if solvent_data_sample["success"] and len(solvent_data_sample["dataframe"]) > 0:
            output.append("\n**Example Solvents (Solvent Data):**")
            solvents = solvent_data_sample["dataframe"]['solvent_name'].tolist()
            for solvent in solvents[:5]:  # Show 5 from each
                output.append(f"- {solvent}")

        if gsk_sample["success"] and len(gsk_sample["dataframe"]) > 0:
            output.append("\n**Example Solvents (GSK Dataset):**")
            solvents = gsk_sample["dataframe"]['solvent_common_name'].tolist()
            for solvent in solvents[:5]:  # Show 5 from each
                output.append(f"- {solvent}")

        # When include_properties is True, query and display solvent_data contents
        if include_properties:
            table_name = _get_solvent_table_name()
            if table_name:
                props_query = f"SELECT * FROM {table_name} ORDER BY 1 LIMIT 100"
                props_result = _execute_query(props_query, limit=100)

                if props_result["success"]:
                    props_df = props_result["dataframe"]
                    conn = get_connection()
                    row_count = conn.execute(
                        f"SELECT COUNT(*) FROM {table_name}"
                    ).fetchone()[0]
                    cols = list(props_df.columns)

                    output.append(f"\n**Solvent Properties Database**\n")
                    output.append(f"Table: `{table_name}`")
                    output.append(f"Total solvents: {row_count}")
                    output.append(f"Columns: {', '.join(cols)}\n")
                    output.append(
                        props_df.head(10).to_markdown(index=False)
                        if len(props_df) > 0
                        else "No data"
                    )
                else:
                    output.append(
                        f"\nCould not retrieve solvent properties: "
                        f"{props_result.get('error')}"
                    )
            else:
                output.append(
                    "\nNo solvent properties table found.\n"
                    "Please upload a CSV file named 'Solvent_Data.csv' with columns:\n"
                    "- Solvent name, CAS number, Bp (C), LogP, Cp (J/gK), Energy (J/g)"
                )

        output.append("\n**Tip:** Use specific solvent names in your queries for best results!")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in list_available_solvents: {e}")
        return f"Error listing solvents: {str(e)}"


@safe_tool_wrapper
async def list_available_polymers() -> str:
    """List available polymers across databases with counts and examples.

    WHEN TO USE:
    - "What polymers are in the database?"
    - "Show me available polymers"
    - "List all polymers"
    """
    try:
        output = ["**Available Polymers Summary**\n"]

        # Count polymers in common_solvents_database
        polymer_query = "SELECT COUNT(DISTINCT polymer) as count FROM common_solvents_database"
        result = _execute_query(polymer_query)

        if result["success"] and len(result["dataframe"]) > 0:
            count = result["dataframe"].iloc[0]['count']
            output.append(f"**Common Solvents Database:** {count} unique polymers")

        # Get 10 common polymers
        sample_query = """
        SELECT DISTINCT polymer
        FROM common_solvents_database
        ORDER BY polymer
        LIMIT 10
        """

        sample_result = _execute_query(sample_query)

        if sample_result["success"] and len(sample_result["dataframe"]) > 0:
            output.append("\n**Example Polymers:**")
            polymers = sample_result["dataframe"]['polymer'].tolist()
            for polymer in polymers:
                output.append(f"- {polymer}")

        output.append("\n**Tip:** Common polymers include HDPE, LDPE, PP, PET, PVC, PS, PVDF, PC, Nylon66, EVOH")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in list_available_polymers: {e}")
        return f"Error listing polymers: {str(e)}"
