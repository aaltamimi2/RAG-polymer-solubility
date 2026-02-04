"""Adaptive separation tools for selective polymer dissolution analysis.

Provides two tool functions extracted from the monolithic agent source:
  - find_optimal_separation_conditions
  - analyze_selective_solubility_enhanced

These are plain functions (no @tool decorator) intended for deep-agent
orchestration.  Each function lazily obtains its database connection via
``get_connection()`` so there is no module-level global state.
"""

from __future__ import annotations

import gc
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from strap.database import get_connection
from strap.tools._helpers import (
    safe_tool_wrapper,
    DataValidator,
    AdaptiveAnalyzer,
    truncate_output,
    get_cross_database_properties,
    normalize_solvent_name,
    save_plot,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _get_plot_url(filepath: str) -> str:
    """Convert a filepath to a displayable string."""
    return f"Plot saved: `{filepath}`"


def _execute_query(conn, query: str, limit: int = 100) -> Dict[str, Any]:
    """Execute a read-only query and return a result dict (mirrors SQLDatabase.execute_query)."""
    try:
        query_lower = query.lower().strip()
        dangerous_keywords = [
            "drop", "delete", "insert", "update", "alter", "create", "truncate",
        ]
        if any(keyword in query_lower.split() for keyword in dangerous_keywords):
            return {"success": False, "error": "Unsafe operation detected", "query": query}

        if "limit" not in query_lower and not query_lower.strip().endswith(";"):
            query = f"{query.rstrip(';')} LIMIT {limit}"

        result_df = conn.execute(query).fetchdf()

        preview = (
            result_df.head(10).to_markdown(index=False)
            if len(result_df) > 0
            else "No data"
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


def _verify_inputs(
    conn,
    validator: DataValidator,
    table_name: str,
    columns: Dict[str, str],
    values: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, str]:
    """Comprehensive input verification (port of the module-level verify_inputs)."""
    issues: List[str] = []
    warnings: List[str] = []

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


# ------------------------------------------------------------------
# Tool 1: find_optimal_separation_conditions
# ------------------------------------------------------------------


@safe_tool_wrapper
def find_optimal_separation_conditions(
    target_polymer: str,
    comparison_polymers: str,
    start_temperature: float = 25.0,
    initial_selectivity: float = 30.0,
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____",
) -> str:
    """Find optimal solvent and temperature to separate a target polymer from others.

    Args:
        target_polymer: Polymer to dissolve (e.g., "LDPE", "PP")
        comparison_polymers: Polymers to NOT dissolve, comma-separated (e.g., "HDPE,PP,PS")
        start_temperature: Starting temperature in °C (default: 25.0)
        initial_selectivity: Required selectivity in percentage points (default: 30.0)

    WHEN TO USE:
    - "Find optimal conditions to separate LDPE from HDPE and PP"
    - "What solvent and temperature selectively dissolves PS?"
    - "Optimal separation conditions for PET from mixed plastics"
    """
    conn = get_connection()
    validator = DataValidator(conn)
    analyzer = AdaptiveAnalyzer(conn, validator)

    # Safely parse comparison_polymers
    if isinstance(comparison_polymers, str):
        comp_polymers = [p.strip() for p in comparison_polymers.split(",") if p.strip()]
    elif isinstance(comparison_polymers, list):
        comp_polymers = comparison_polymers
    else:
        return f"Error: comparison_polymers must be a comma-separated string, got {type(comparison_polymers)}"

    if not comp_polymers:
        return "Error: No comparison polymers specified."

    all_polymers = [target_polymer] + comp_polymers

    is_valid, msg = _verify_inputs(
        conn,
        validator,
        table_name,
        {
            "polymer": polymer_column,
            "solvent": solvent_column,
            "temperature": temperature_column,
            "solubility": solubility_column,
        },
        {polymer_column: all_polymers},
    )

    if not is_valid:
        return f"Input validation failed:\n{msg}"

    output = [f"**Adaptive Separation Analysis**\n"]
    output.append(f"Target: Dissolve {target_polymer}")
    output.append(f"Separate from: {', '.join(comp_polymers)}")
    output.append(
        f"Starting conditions: T={start_temperature}°C, selectivity threshold={initial_selectivity}%\n"
    )

    temp_result = analyzer.explore_temperature_range(
        table_name,
        polymer_column,
        solvent_column,
        temperature_column,
        solubility_column,
        target_polymer,
        comp_polymers,
        start_temp=start_temperature,
        min_selectivity=initial_selectivity,
    )

    opt = temp_result.get("optimal_conditions")
    if opt and opt["selectivity"] >= initial_selectivity:
        output.append("**Separation IS FEASIBLE**\n")
        output.append("**Optimal Conditions:**")
        output.append(f"  - Temperature: {opt['temperature']}°C")
        output.append(f"  - Selectivity: {opt['selectivity']:.1f}%")
        output.append(f"  - Target solubility: {opt['target_solubility']:.1f}%")
        output.append(f"  - Max other solubility: {opt['max_other_solubility']:.1f}%")

        # Show alternative conditions at other temperatures
        alternatives = [
            c for c in temp_result.get("all_conditions", [])
            if c != opt and c["selectivity"] > 0
        ]
        alternatives.sort(key=lambda c: c["selectivity"], reverse=True)
        if alternatives:
            output.append("\n**Alternative Temperatures:**")
            for i, alt in enumerate(alternatives[:3], 1):
                output.append(
                    f"  {i}. T={alt['temperature']}°C "
                    f"(selectivity={alt['selectivity']:.1f}%)"
                )
    elif opt:
        output.append("**Separation MARGINAL** — below selectivity threshold\n")
        output.append(f"  - Best temperature: {opt['temperature']}°C")
        output.append(f"  - Best selectivity: {opt['selectivity']:.1f}%")
        output.append(f"  - Required: {initial_selectivity}%")
    else:
        output.append("**Separation NOT FEASIBLE** with current data\n")

    output.append(f"\n**Recommendation:** {temp_result.get('recommendation', 'N/A')}")
    output.append(f"**Temperatures explored:** {temp_result.get('temperatures_explored', [])}")

    return "\n".join(output)


# ------------------------------------------------------------------
# Tool 2: analyze_selective_solubility_enhanced
# ------------------------------------------------------------------


@safe_tool_wrapper
def analyze_selective_solubility_enhanced(
    target_polymer: str,
    comparison_polymers: Optional[str] = None,
    temperature_range: str = "25-120",
    auto_threshold: bool = True,
    search_mode: str = "full",
    enrich_properties: bool = False,
    rank_by: str = "selectivity",
    table_name: str = "common_solvents_database",
    polymer_column: str = "polymer",
    solvent_column: str = "solvent",
    temperature_column: str = "temperature___c_",
    solubility_column: str = "solubility____",
) -> str:
    """Analyze selective solubility, optionally with threshold search or property enrichment.

    Args:
        target_polymer: Polymer to dissolve (e.g., "LDPE", "PP", "PET")
        comparison_polymers: Polymers to avoid, comma-separated; omit to compare against all
        temperature_range: Range as "min-max" or single value (e.g., "80-120" or "100")
        auto_threshold: Use adaptive thresholds (default: True)
        search_mode: "full" (default) or "threshold_search" for iterative threshold relaxation
        enrich_properties: When True, look up BP, LogP, energy, G-score for each result
        rank_by: Sort criterion: "selectivity" (default), "energy", "logp", "bp"

    WHEN TO USE:
    - "Find a solvent that dissolves LDPE but not HDPE"
    - "Find selective solvents using adaptive threshold search"
    - "Find solvents to separate PET from PE, ranked by cost"
    - "Show selective solvents with their physical properties"
    """
    conn = get_connection()
    validator = DataValidator(conn)

    # Handle single temperature or range
    if "-" in str(temperature_range):
        parts = str(temperature_range).split("-")
        if len(parts) >= 2:
            temp_min, temp_max = float(parts[0]), float(parts[1])
        else:
            temp_min = temp_max = float(parts[0])
    else:
        # Single temperature - use as both min and max (+-5 deg C range)
        temp = float(temperature_range)
        temp_min, temp_max = temp - 5, temp + 5

    # Safely build comp_list
    comp_list: List[str] = []
    if comparison_polymers:
        if isinstance(comparison_polymers, str):
            comp_list = [p.strip() for p in comparison_polymers.split(",") if p.strip()]
        elif isinstance(comparison_polymers, list):
            comp_list = comparison_polymers
        output = ["**Selective Solubility Analysis (Targeted Comparison)**\n"]
    else:
        try:
            all_query = f"SELECT DISTINCT {polymer_column} FROM {table_name}"
            result = conn.execute(all_query).fetchdf()
            if len(result) > 0 and polymer_column in result.columns:
                comp_list = [p for p in result[polymer_column].tolist() if p != target_polymer]
        except Exception as e:
            logger.warning(f"Could not get polymers: {e}")
            return f"Error: Could not retrieve polymer list from '{table_name}'"
        output = ["**Selective Solubility Analysis (All Polymers)**\n"]

    if not comp_list:
        return "Error: No comparison polymers found."

    output.append(f"Target: {target_polymer}")
    output.append(f"Comparing against: {', '.join(comp_list)}")
    output.append(f"Temperature range: {temp_min}°C - {temp_max}°C\n")

    val_result = validator.verify_value_exists(table_name, polymer_column, target_polymer)
    if not val_result.is_valid:
        return f"Target polymer '{target_polymer}' not found. {val_result.warnings}"

    all_polymers = [target_polymer] + comp_list
    polymer_filter = "', '".join(all_polymers)

    query = f"""
    SELECT {solvent_column}, {polymer_column},
           AVG({solubility_column}) as avg_solubility,
           MIN({solubility_column}) as min_solubility,
           MAX({solubility_column}) as max_solubility,
           COUNT(*) as n_points
    FROM {table_name}
    WHERE {polymer_column} IN ('{polymer_filter}')
    AND {temperature_column} >= {temp_min} AND {temperature_column} <= {temp_max}
    GROUP BY {solvent_column}, {polymer_column}
    """

    result = _execute_query(conn, query, limit=10000)
    if not result["success"]:
        return f"Query failed: {result.get('error')}"

    df = result["dataframe"]
    output.append(f"Data points analyzed: {len(df)}\n")

    solvents = df[solvent_column].unique()
    selectivity_data: List[Dict[str, Any]] = []

    for solvent in solvents:
        solvent_data = df[df[solvent_column] == solvent]

        target_sol = solvent_data[solvent_data[polymer_column] == target_polymer]
        if len(target_sol) == 0:
            continue
        target_avg = target_sol["avg_solubility"].values[0]
        target_n = target_sol["n_points"].values[0]

        other_data = solvent_data[solvent_data[polymer_column].isin(comp_list)]
        if len(other_data) == 0:
            max_other = 0
            avg_other = 0
        else:
            max_other = other_data["avg_solubility"].max()
            avg_other = other_data["avg_solubility"].mean()

        selectivity = target_avg - max_other
        selectivity_ratio = target_avg / max_other if max_other > 0.001 else float("inf")

        selectivity_data.append(
            {
                "solvent": solvent,
                "target_solubility": target_avg,
                "max_other_solubility": max_other,
                "avg_other_solubility": avg_other,
                "selectivity_difference": selectivity,
                "selectivity_ratio": selectivity_ratio,
                "n_data_points": target_n,
            }
        )

    selectivity_data.sort(key=lambda x: x["selectivity_difference"], reverse=True)

    # ------------------------------------------------------------------
    # Threshold-search early return (replaces adaptive_threshold_search)
    # ------------------------------------------------------------------
    if search_mode == "threshold_search":
        start_threshold = 0.5
        temp_mid = (temp_min + temp_max) / 2
        thresholds = [t for t in AdaptiveAnalyzer.SELECTIVITY_THRESHOLDS if t <= start_threshold]
        ts_output = ["**Adaptive Threshold Search (inline)**\n"]
        ts_output.append(f"Target: {target_polymer}")
        ts_output.append(f"Comparing against: {', '.join(comp_list)}")
        ts_output.append(f"Temperature: {temp_mid}°C")
        ts_output.append(f"Starting threshold: {start_threshold}\n")

        found = False
        thresholds_tried: List[Tuple[float, int]] = []
        for thr in thresholds:
            matching = [s for s in selectivity_data if s["selectivity_difference"] >= thr]
            thresholds_tried.append((thr, len(matching)))
            if matching:
                ts_output.append(
                    f"**Found {len(matching)} selective solvent(s)** at threshold {thr}\n"
                )
                ts_output.append("**Results:**")
                for i, r in enumerate(matching[:10], 1):
                    ts_output.append(f"  {i}. {r['solvent']}")
                    ts_output.append(f"     Selectivity: {r['selectivity_difference']:.4f}")
                    ts_output.append(f"     {target_polymer} solubility: {r['target_solubility']:.4f}")
                    ts_output.append(f"     Max other solubility: {r['max_other_solubility']:.4f}")
                found = True
                break

        ts_output.insert(5, f"Thresholds tried: {[t[0] for t in thresholds_tried]}")
        if not found:
            ts_output.append(f"\n**No selective solvents found** even at threshold {thresholds[-1]}")
            ts_output.append("\nConsider:")
            ts_output.append("  - Exploring higher temperatures")
            ts_output.append("  - Using find_optimal_separation_conditions for comprehensive search")

        return "\n".join(ts_output)

    # ------------------------------------------------------------------
    # Property enrichment (replaces analyze_separation_with_properties)
    # ------------------------------------------------------------------
    if enrich_properties:
        for entry in selectivity_data:
            props = get_cross_database_properties(entry["solvent"], conn)
            entry["bp"] = props.get("bp")
            entry["logp"] = props.get("logp")
            entry["energy"] = props.get("energy")
            entry["g_score"] = props.get("g_score")

    # ------------------------------------------------------------------
    # Rank-by override
    # ------------------------------------------------------------------
    if rank_by != "selectivity":
        rank_key = rank_by.lower()
        if rank_key in ("energy", "cost"):
            selectivity_data.sort(
                key=lambda x: (x.get("energy") is None, x.get("energy", float("inf")))
            )
        elif rank_key in ("logp", "toxicity"):
            selectivity_data.sort(
                key=lambda x: (x.get("logp") is None, x.get("logp", float("inf")))
            )
        elif rank_key in ("bp", "boiling"):
            selectivity_data.sort(
                key=lambda x: (x.get("bp") is None, x.get("bp", float("inf")))
            )
        else:
            selectivity_data.sort(
                key=lambda x: (x.get(rank_key) is None, x.get(rank_key, float("inf")))
            )

    if not selectivity_data:
        return f"No selectivity data found for {target_polymer}"

    if auto_threshold:
        thresholds_tried = []
        for threshold in AdaptiveAnalyzer.SELECTIVITY_THRESHOLDS:
            selective_solvents = [
                s for s in selectivity_data if s["selectivity_difference"] >= threshold
            ]
            thresholds_tried.append((threshold, len(selective_solvents)))
            if len(selective_solvents) > 0:
                output.append(
                    f"**Adaptive Threshold:** Found {len(selective_solvents)} "
                    f"solvent(s) at threshold {threshold}"
                )
                break

        output.append(f"Thresholds searched: {[t[0] for t in thresholds_tried]}\n")

    output.append(f"**Selective Solvents (ranked by {rank_by}):**\n")
    for i, data in enumerate(selectivity_data[:15], 1):
        if data["selectivity_difference"] > 10:
            sel_symbol = "[GOOD]"
        elif data["selectivity_difference"] > 0:
            sel_symbol = "[MARGINAL]"
        else:
            sel_symbol = "[POOR]"
        output.append(f"{i}. {sel_symbol} **{data['solvent']}**")
        output.append(f"   - {target_polymer} solubility: {data['target_solubility']:.4f}")
        output.append(
            f"   - Max comparison solubility: {data['max_other_solubility']:.4f}"
        )
        output.append(
            f"   - Selectivity: {data['selectivity_difference']:.4f} "
            f"({data['selectivity_ratio']:.1f}x)"
        )
        output.append(f"   - Data points: {data['n_data_points']}")
        if enrich_properties:
            props_parts: List[str] = []
            if data.get("bp") is not None:
                props_parts.append(f"BP: {data['bp']:.1f} C")
            if data.get("logp") is not None:
                props_parts.append(f"LogP: {data['logp']:.2f}")
            if data.get("energy") is not None:
                props_parts.append(f"Energy: {data['energy']:.1f} J/g")
            if data.get("g_score") is not None:
                props_parts.append(f"G-Score: {data['g_score']:.2f}/10")
            if props_parts:
                output.append(f"   - Properties: {' | '.join(props_parts)}")

    # Create visualization
    if len(selectivity_data) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            top_n = min(12, len(selectivity_data))
            solvent_names = [d["solvent"] for d in selectivity_data[:top_n]]
            target_sols = [d["target_solubility"] for d in selectivity_data[:top_n]]
            other_sols = [d["max_other_solubility"] for d in selectivity_data[:top_n]]

            x = np.arange(len(solvent_names))
            width = 0.35

            axes[0].bar(
                x - width / 2, target_sols, width,
                label=target_polymer, color="green", alpha=0.8,
            )
            axes[0].bar(
                x + width / 2, other_sols, width,
                label="Max Comparison", color="red", alpha=0.8,
            )
            axes[0].set_xlabel("Solvent", fontsize=12, fontweight="bold")
            axes[0].set_ylabel("Average Solubility", fontsize=12, fontweight="bold")
            axes[0].set_title(
                f"Selective Solvents for {target_polymer}",
                fontsize=14, fontweight="bold",
            )
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(solvent_names, rotation=45, ha="right")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis="y")

            selectivity_diffs = [
                d["selectivity_difference"] for d in selectivity_data[:top_n]
            ]
            colors = [
                "green" if s > 10 else "orange" if s > 0 else "red"
                for s in selectivity_diffs
            ]
            axes[1].barh(solvent_names, selectivity_diffs, color=colors, alpha=0.8)
            axes[1].axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            axes[1].axvline(
                x=10, color="green", linestyle="--", linewidth=1,
                label="Good selectivity (10%)",
            )
            axes[1].set_xlabel(
                "Selectivity Difference", fontsize=12, fontweight="bold"
            )
            axes[1].set_title("Selectivity Ranking", fontsize=14, fontweight="bold")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis="x")

            plt.tight_layout()
            filepath = save_plot(fig, f"selective_solubility_{target_polymer}", "matplotlib")
            output.append(f"\n{_get_plot_url(filepath)}")
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")

    del df
    gc.collect()

    display = "\n".join(output)

    # Build structured data for programmatic access
    top_solvents = selectivity_data[:10] if selectivity_data else []
    structured_data = {
        "tool_name": "analyze_selective_solubility_enhanced",
        "success": True,
        "polymers_analyzed": [target_polymer] + comp_list,
        "solvents": [s["solvent"] for s in top_solvents],
        "selectivities": [s["selectivity_difference"] for s in top_solvents],
        "best_solvent": top_solvents[0]["solvent"] if top_solvents else None,
        "best_selectivity": (
            top_solvents[0]["selectivity_difference"] if top_solvents else None
        ),
        "temperature": (temp_min + temp_max) / 2,
        "temperature_range": [temp_min, temp_max],
        "target_polymer": target_polymer,
        "comparison_polymers": comp_list,
        "algorithm_used": "selective_solubility",
        "coverage_complete": len(top_solvents) > 0,
    }

    # Return structured JSON
    return json.dumps({"display": display, "data": structured_data}, ensure_ascii=False)
