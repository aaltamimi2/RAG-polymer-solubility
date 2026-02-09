"""Visualization tools for STRAP solubility / polymer analysis.

Provides matplotlib and plotly-based plotting functions extracted from the
original monolithic agent source.  Every public function is wrapped with
:func:`safe_tool_wrapper` for uniform error handling and memory cleanup.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import seaborn as sns                    # noqa: E402
import plotly.graph_objects as go        # noqa: E402
import plotly.express as px              # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from strap.database import get_connection
from strap.tools._helpers import (
    safe_tool_wrapper,
    truncate_output,
    save_plot,
    get_plots_dir,
    get_cross_database_properties,
    normalize_solvent_name,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Solvent-name normalization helpers (local copies from the vendor source)
# ---------------------------------------------------------------------------

SOLVENT_NAME_MAPPING: Dict[str, List[str]] = {
    # Xylene variants - expand to both isomers
    "xylene": ["1,2-dimethylbenzene", "1,4-dimethylbenzene"],
    "xylenes": ["1,2-dimethylbenzene", "1,4-dimethylbenzene"],
    "o-xylene": ["1,2-dimethylbenzene"],
    "p-xylene": ["1,4-dimethylbenzene"],
    "m-xylene": ["1,4-dimethylbenzene"],
    "ortho-xylene": ["1,2-dimethylbenzene"],
    "para-xylene": ["1,4-dimethylbenzene"],
    # Alkanes
    "heptane": ["n-heptane"],
    "n-heptane": ["n-heptane"],
    "hexane": ["hexane"],
    "n-hexane": ["hexane"],
    "pentane": ["pentane"],
    "n-pentane": ["pentane"],
    "octane": ["octane"],
    "n-octane": ["octane"],
    # Polar aprotic
    "dmso": ["dimethylsulfoxide"],
    "dimethyl sulfoxide": ["dimethylsulfoxide"],
    "dmf": ["dimethylformamide"],
    "dimethyl formamide": ["dimethylformamide"],
    "nmp": ["n-methylpyrrolidone"],
    "n-methyl-2-pyrrolidone": ["n-methylpyrrolidone"],
    # Ketones
    "acetone": ["propanone"],
    "2-propanone": ["propanone"],
    "mek": ["butanone"],
    "methyl ethyl ketone": ["butanone"],
    # Alcohols
    "ipa": ["2-propanol"],
    "isopropanol": ["2-propanol"],
    "isopropyl alcohol": ["2-propanol"],
    "n-propanol": ["propanol"],
    "1-propanol": ["propanol"],
    "meoh": ["methanol"],
    "etoh": ["ethanol"],
    # Ethers
    "tetrahydrofuran": ["thf"],
    "tetrahydropyran": ["thp"],
    "dihydropyran": ["2,3-dihydropyran"],
    # Halogenated
    "dcm": ["ch2cl2"],
    "dichloromethane": ["ch2cl2"],
    "methylene chloride": ["ch2cl2"],
    "chloroform": ["chcl3"],
    "trichloromethane": ["chcl3"],
    # Others
    "ethyl acetate": ["ethylacetate"],
    "methyl acetate": ["methylacetate"],
    "water": ["h2o"],
    "ethylene glycol": ["glycol"],
    "propylene glycol": ["propyleneglycol"],
}

SOLVENT_FRAGMENT_RECONSTRUCTION: Dict[Tuple[str, str], str] = {
    ("2", "3-dihydropyran"): "2,3-dihydropyran",
    ("1", "2-dimethylbenzene"): "1,2-dimethylbenzene",
    ("1", "4-dimethylbenzene"): "1,4-dimethylbenzene",
    ("1", "3-dimethylbenzene"): "1,3-dimethylbenzene",
    ("1", "2-dichloroethane"): "1,2-dichloroethane",
    ("1", "1-dichloroethane"): "1,1-dichloroethane",
    ("1", "2-dichlorobenzene"): "1,2-dichlorobenzene",
    ("1", "4-dichlorobenzene"): "1,4-dichlorobenzene",
    ("1", "2-ethanediol"): "1,2-ethanediol",
    ("1", "3-propanediol"): "1,3-propanediol",
    ("1", "4-dioxane"): "1,4-dioxane",
    ("2", "2-dimethylbutane"): "2,2-dimethylbutane",
    ("2", "3-butanediol"): "2,3-butanediol",
    ("2", "4-pentanedione"): "2,4-pentanedione",
}


def _normalize_solvent_names(solvents: List[str]) -> List[str]:
    """Normalize solvent names to match database entries.

    Expands common names (e.g. ``'xylene'``) to actual database names and
    reconstructs names that were incorrectly split on commas.
    """
    # Reconstruct fragmented solvent names
    reconstructed: List[str] = []
    i = 0
    while i < len(solvents):
        solvent = solvents[i].strip().lower()
        if i + 1 < len(solvents):
            next_solvent = solvents[i + 1].strip().lower()
            fragment_key = (solvent, next_solvent)
            if fragment_key in SOLVENT_FRAGMENT_RECONSTRUCTION:
                reconstructed.append(SOLVENT_FRAGMENT_RECONSTRUCTION[fragment_key])
                i += 2
                continue
        reconstructed.append(solvent)
        i += 1

    # Apply the standard normalization mapping
    normalized: List[str] = []
    for solvent in reconstructed:
        solvent_lower = solvent.strip().lower()
        if solvent_lower in SOLVENT_NAME_MAPPING:
            normalized.extend(SOLVENT_NAME_MAPPING[solvent_lower])
        else:
            normalized.append(solvent_lower)
    return normalized


# ---------------------------------------------------------------------------
# Lightweight SQL execution helper (replaces ``sql_db.execute_query``)
# ---------------------------------------------------------------------------

def _execute_query(query: str, limit: int = 100) -> Dict[str, Any]:
    """Execute *query* through the shared DuckDB connection.

    Returns a dict compatible with the legacy ``sql_db.execute_query`` API.
    """
    conn = get_connection()
    try:
        query_lower = query.lower().strip()
        dangerous_keywords = [
            "drop", "delete", "insert", "update", "alter", "create", "truncate",
        ]
        if any(keyword in query_lower.split() for keyword in dangerous_keywords):
            return {"success": False, "error": "Unsafe operation detected", "query": query}

        if "limit" not in query_lower and not query_lower.rstrip().endswith(";"):
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


# ---------------------------------------------------------------------------
# Input verification helper (replaces ``verify_inputs``)
# ---------------------------------------------------------------------------

def _verify_inputs(
    table_name: str,
    columns: Dict[str, str],
    values: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, str]:
    """Comprehensive input verification against the live database."""
    conn = get_connection()
    issues: List[str] = []
    warnings: List[str] = []

    # Verify table exists
    try:
        tables = conn.execute("SHOW TABLES").fetchdf()
        if table_name not in tables["name"].values:
            return False, f"Table '{table_name}' not found. Available: {list(tables['name'].values)}"
    except Exception as e:
        return False, f"Could not list tables: {e}"

    # Get schema
    try:
        schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        available_cols = set(schema["column_name"].values)
    except Exception as e:
        return False, f"Could not get schema: {e}"

    # Verify columns
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
                safe_value = str(val).replace("'", "''")
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {table_name} "
                    f"WHERE LOWER(CAST({col_name} AS VARCHAR)) = LOWER('{safe_value}')"
                ).fetchone()[0]
                if count == 0:
                    issues.append(f"Value '{val}' not found in {col_name}")

    if issues:
        msg = "Value verification failed:\n- " + "\n- ".join(issues)
        return False, msg

    return True, "All inputs verified"


# ---------------------------------------------------------------------------
# Plot URL helper
# ---------------------------------------------------------------------------

def _get_plot_url(filepath: str) -> str:
    """Convert a filepath to a user-facing display string."""
    return f"Plot saved: `{filepath}`"


# ---------------------------------------------------------------------------
# Solvent-property helpers (replaces global ``get_solvent_table_name`` /
# ``lookup_solvent_properties``)
# ---------------------------------------------------------------------------

def _get_solvent_table_name() -> Optional[str]:
    """Auto-detect the solvent-data table in the current database."""
    conn = get_connection()
    try:
        tables = conn.execute("SHOW TABLES").fetchdf()
    except Exception:
        return None

    for tbl in tables["name"].values:
        if "solvent" in tbl.lower() and "solubility" not in tbl.lower():
            try:
                cols_df = conn.execute(f"DESCRIBE {tbl}").fetchdf()
                cols_lower = [c.lower() for c in cols_df["column_name"].values]
                if (
                    any("bp" in c or "boil" in c for c in cols_lower)
                    or any("logp" in c for c in cols_lower)
                    or any("energy" in c for c in cols_lower)
                ):
                    return str(tbl)
            except Exception:
                continue
    return None


def _get_solvent_name_column(table_name: str) -> Optional[str]:
    """Return the column that holds solvent names in *table_name*."""
    conn = get_connection()
    try:
        cols_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        cols = list(cols_df["column_name"].values)
        types_map = dict(zip(cols_df["column_name"], cols_df["column_type"]))
    except Exception:
        return None

    priority_patterns = ["solvent_name", "solvent", "name", "compound"]
    for pattern in priority_patterns:
        for col in cols:
            if pattern in col.lower():
                return col

    for col in cols:
        if "VARCHAR" in str(types_map.get(col, "")).upper() or "TEXT" in str(types_map.get(col, "")).upper():
            return col
    return cols[0] if cols else None


def _get_cosmobase_column(table_name: str) -> Optional[str]:
    """Return the cosmobase column name if present."""
    conn = get_connection()
    try:
        cols_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        for col in cols_df["column_name"].values:
            if "cosmobase" in col.lower():
                return str(col)
    except Exception:
        pass
    return None


async def _lookup_solvent_properties(
    solvent_names: List[str],
    solvent_table: str,
) -> Dict[str, Dict[str, Any]]:
    """Look up solvent properties with robust fuzzy matching (async)."""
    conn = get_connection()

    ABBREVIATION_MAP = {
        "dmf": "dimethylformamide",
        "thf": "tetrahydrofuran",
        "dme": "dimethoxyethane",
        "meoh": "methanol",
        "etoh": "ethanol",
        "ipa": "isopropanol",
        "nmp": "n-methyl-2-pyrrolidone",
        "dmso": "dimethyl sulfoxide",
        "dcm": "dichloromethane",
        "dce": "dichloroethane",
        "mecn": "acetonitrile",
        "etac": "ethyl acetate",
        "acac": "acetylacetone",
        "tfa": "trifluoroacetic acid",
        "tfe": "trifluoroethanol",
        "hfip": "hexafluoroisopropanol",
        "chcl3": "chloroform",
        "ccl4": "carbon tetrachloride",
        "phme": "toluene",
        "phh": "benzene",
        "mtbe": "methyl tert-butyl ether",
        "tbme": "tert-butyl methyl ether",
        "dipa": "diisopropylamine",
        "tea": "triethylamine",
        "dbu": "1,8-diazabicyclo[5.4.0]undec-7-ene",
        "pyr": "pyridine",
        "acn": "acetonitrile",
        "mibk": "methyl isobutyl ketone",
    }

    try:
        cols_df = conn.execute(f"DESCRIBE {solvent_table}").fetchdf()
        cols = list(cols_df["column_name"].values)
    except Exception:
        return {}

    cols_lower = {c.lower(): c for c in cols}

    cosmobase_col = _get_cosmobase_column(solvent_table)
    name_col = _get_solvent_name_column(solvent_table)

    logp_col = next((cols_lower[k] for k in cols_lower if "logp" in k), None)
    bp_col = next((cols_lower[k] for k in cols_lower if "bp" in k or "boil" in k), None)
    energy_col = next((cols_lower[k] for k in cols_lower if "energy" in k), None)
    cp_col = next((cols_lower[k] for k in cols_lower if "cp" in k and "logp" not in k), None)

    match_col = cosmobase_col or name_col
    if not match_col:
        return {}

    def _find_solvent_match(solvent: str):
        """Try multiple strategies to find a solvent row."""
        sol_lower = solvent.lower().strip()
        sol_normalized = sol_lower.replace("-", "").replace(" ", "").replace(",", "")

        # Strategy 1: Exact match
        try:
            df = conn.execute(
                f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") = '{sol_lower}'"
            ).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass

        # Strategy 2: Abbreviation mapping
        if sol_lower in ABBREVIATION_MAP:
            full_name = ABBREVIATION_MAP[sol_lower]
            try:
                df = conn.execute(
                    f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") LIKE '%{full_name}%' "
                    f"ORDER BY LENGTH(\"{match_col}\")"
                ).fetchdf()
                if len(df) > 0:
                    return df.iloc[0]
            except Exception:
                pass

        # Strategy 3: Substring
        try:
            df = conn.execute(
                f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") LIKE '%{sol_lower}%' "
                f"ORDER BY LENGTH(\"{match_col}\")"
            ).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass

        # Strategy 4: Normalized match
        try:
            df = conn.execute(
                f"SELECT * FROM {solvent_table} "
                f"WHERE REPLACE(REPLACE(REPLACE(LOWER(\"{match_col}\"), '-', ''), ' ', ''), ',', '') "
                f"LIKE '%{sol_normalized}%' ORDER BY LENGTH(\"{match_col}\")"
            ).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass

        # Strategy 5: Reverse abbreviation lookup
        for abbrev, full in ABBREVIATION_MAP.items():
            if abbrev in sol_lower or sol_lower in full:
                try:
                    df = conn.execute(
                        f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") LIKE '%{full}%' "
                        f"ORDER BY LENGTH(\"{match_col}\")"
                    ).fetchdf()
                    if len(df) > 0:
                        return df.iloc[0]
                except Exception:
                    pass

        return None

    # Run matches (synchronously under the hood -- DuckDB is not truly async)
    props_map: Dict[str, Dict[str, Any]] = {}
    for solvent in solvent_names:
        row = _find_solvent_match(solvent)
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
# Visualization tool functions
# ===================================================================


@safe_tool_wrapper
def plot_solubility_vs_temperature(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    polymers: str,
    solvents: str,
    plot_title: Optional[str] = None,
    include_confidence_bands: bool = True,
    temperature_min: Optional[float] = None,
    temperature_max: Optional[float] = None,
) -> str:
    """Plot solubility vs temperature curves using the interpolation model.

    Args:
        table_name, polymer_column, solvent_column: DB table and column names (kept for API compat)
        temperature_column, solubility_column: Temperature and solubility columns (kept for API compat)
        polymers: Comma-separated polymer names
        solvents: Comma-separated solvent names
        plot_title: Custom plot title
        include_confidence_bands: Unused (interpolation produces smooth curves)
        temperature_min/temperature_max: Temperature range filter

    WHEN TO USE:
    - "Plot solubility of PS in toluene vs temperature"
    - "Show how solubility changes with temperature for PET"
    """
    from strap.solubility import get_solubility_curve

    polymer_list = [p.strip() for p in polymers.split(",")]
    solvent_list = [s.strip() for s in solvents.split(",")]
    solvent_list = _normalize_solvent_names(solvent_list)

    t_start = max(temperature_min or 25.0, 25.0)
    t_end = min(temperature_max or 160.0, 160.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(polymer_list) * len(solvent_list)))
    color_idx = 0
    total_points = 0

    for polymer in polymer_list:
        for solvent in solvent_list:
            curve = get_solubility_curve(polymer, solvent, t_start, t_end, 5.0)
            if curve:
                temps = [pt["temperature"] for pt in curve]
                sols = [pt["solubility"] for pt in curve]
                total_points += len(curve)

                ax.plot(
                    temps, sols, marker="o", linewidth=2, markersize=6,
                    label=f"{polymer} in {solvent}", color=colors[color_idx],
                )
                color_idx += 1

    if total_points == 0:
        plt.close(fig)
        return "No data found for the specified polymer-solvent combinations."

    ax.set_xlabel("Temperature (C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Solubility (%)", fontsize=12, fontweight="bold")
    title = plot_title or "Solubility vs Temperature"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    if temperature_min is not None or temperature_max is not None:
        current_xlim = ax.get_xlim()
        new_min = temperature_min if temperature_min is not None else current_xlim[0]
        new_max = temperature_max if temperature_max is not None else current_xlim[1]
        ax.set_xlim(new_min, new_max)

    plt.tight_layout()
    filepath = save_plot(fig, "solubility_temp_curve", "matplotlib")
    plt.close(fig)

    output = "Solubility vs Temperature Plot Created\n\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {', '.join(solvent_list)}\n"
    if temperature_min is not None and temperature_max is not None:
        output += f"Temperature range: {temperature_min}C - {temperature_max}C\n"
    elif temperature_min is not None:
        output += f"Temperature range: {temperature_min}C and above\n"
    elif temperature_max is not None:
        output += f"Temperature range: up to {temperature_max}C\n"
    output += f"Data points: {total_points}\n"
    output += f"\n{_get_plot_url(filepath)}"

    gc.collect()
    return output


@safe_tool_wrapper
def plot_solubility_vs_temperature_interactive(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    polymers: str,
    solvents: str,
    plot_title: Optional[str] = None,
    temperature_min: Optional[float] = None,
    temperature_max: Optional[float] = None,
) -> str:
    """Generate an interactive Plotly HTML plot of solubility vs temperature.

    Args:
        table_name, polymer_column, solvent_column: DB table and column names (kept for API compat)
        temperature_column, solubility_column: Temperature and solubility columns (kept for API compat)
        polymers: Comma-separated polymer names
        solvents: Comma-separated solvent names
        plot_title: Custom plot title
        temperature_min/temperature_max: Temperature range filter

    WHEN TO USE:
    - "Create an interactive solubility vs temperature chart"
    - "I want a zoomable plot of PET solubility curves"
    """
    from strap.solubility import get_solubility_curve

    polymer_list = [p.strip() for p in polymers.split(",")]
    solvent_list = [s.strip() for s in solvents.split(",")]
    solvent_list = _normalize_solvent_names(solvent_list)

    t_start = max(temperature_min or 25.0, 25.0)
    t_end = min(temperature_max or 160.0, 160.0)

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    color_idx = 0
    total_points = 0

    for polymer in polymer_list:
        for solvent in solvent_list:
            curve = get_solubility_curve(polymer, solvent, t_start, t_end, 5.0)
            if curve:
                temps = [pt["temperature"] for pt in curve]
                sols = [pt["solubility"] for pt in curve]
                total_points += len(curve)

                fig.add_trace(go.Scatter(
                    x=temps,
                    y=sols,
                    mode="lines+markers",
                    name=f"{polymer} in {solvent}",
                    line=dict(width=3, color=colors[color_idx % len(colors)]),
                    marker=dict(size=8, symbol="circle"),
                    hovertemplate=(
                        f"<b>{polymer} in {solvent}</b><br>"
                        + "Temperature: %{x:.1f}C<br>"
                        + "Solubility: %{y:.2f}%<br>"
                        + "<extra></extra>"
                    ),
                ))
                color_idx += 1

    if total_points == 0:
        return "No data found for the specified polymer-solvent combinations."

    title = plot_title or "Interactive Solubility vs Temperature"
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Arial Black"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Temperature (C)", font=dict(size=16, family="Arial")),
            rangeslider=dict(visible=True, thickness=0.05),
            showgrid=True, gridcolor="lightgray",
        ),
        yaxis=dict(
            title=dict(text="Solubility (%)", font=dict(size=16, family="Arial")),
            showgrid=True, gridcolor="lightgray",
        ),
        hovermode="closest",
        height=700,
        template="plotly_white",
        legend=dict(
            orientation="v", yanchor="top", y=1,
            xanchor="left", x=1.02, font=dict(size=12),
        ),
    )

    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "solubility_vs_temp",
            "height": 700,
            "width": 1200,
            "scale": 2,
        },
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        "displaylogo": False,
    }

    plots_dir = get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interactive_solubility_temp_{timestamp}.html"
    filepath = os.path.join(plots_dir, filename)

    fig.write_html(filepath, config=config)

    output = "Interactive Solubility vs Temperature Visualization Created\n\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {', '.join(solvent_list)}\n"
    if temperature_min is not None and temperature_max is not None:
        output += f"Temperature range: {temperature_min}C - {temperature_max}C\n"
    elif temperature_min is not None:
        output += f"Temperature range: {temperature_min}C and above\n"
    elif temperature_max is not None:
        output += f"Temperature range: up to {temperature_max}C\n"
    output += f"Data points: {total_points}\n\n"

    output += "## Interactive Features:\n"
    output += "- **Click legend items** to show/hide individual curves\n"
    output += "- **Drag the range slider** below the plot to zoom into temperature ranges\n"
    output += "- **Hover over points** to see exact values\n"
    output += "- **Use toolbar** to zoom, pan, reset, or download as PNG\n"
    output += "- **Double-click legend** to isolate a single curve\n\n"

    html_url = f"/plots/{filename}"
    output += f"**[Click here to open the interactive plot]({html_url})**\n"
    output += "(Opens in a new tab with full interactivity)\n"

    gc.collect()
    return output


@safe_tool_wrapper
def plot_selectivity_heatmap(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    target_polymer: Optional[str] = None,
    temperature: float = 120.0,
    temperature_tolerance: float = 10.0,
    show_selectivity: bool = False,
    max_solvents: int = 30,
) -> str:
    """Create a heatmap of solubility across polymer-solvent combinations at a given temperature.

    Args:
        table_name, polymer_column, solvent_column: DB table and column names
        temperature_column, solubility_column: Temperature and solubility columns
        target_polymer: Filter to a single polymer
        temperature: Target temperature in C (default 120)
        temperature_tolerance: Range +/- in C (default 10)
        show_selectivity: Show selectivity view if target_polymer set
        max_solvents: Max solvents displayed (default 30)

    WHEN TO USE:
    - "Show a heatmap of solubility for all polymers"
    - "Which solvents dissolve PET best at 120C?"
    """
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    from strap.solubility import get_solubility, get_available_polymers, get_available_solvents_for_polymer

    if target_polymer:
        polymer_names = [target_polymer.upper()]
    else:
        polymer_names = sorted(get_available_polymers())

    rows: List[Dict[str, Any]] = []
    for poly in polymer_names:
        for sv in get_available_solvents_for_polymer(poly):
            sol = get_solubility(poly, sv, temperature)
            if sol is not None:
                rows.append({"polymer": poly, "solvent": sv, "avg_solubility": sol})

    if not rows:
        return f"No data found at {temperature}C. Try a different temperature."

    df = pd.DataFrame(rows)
    pivot_df = df.pivot(index="polymer", columns="solvent", values="avg_solubility")

    # Limit solvents for readability
    if len(pivot_df.columns) > max_solvents:
        top_solvents = df.groupby("solvent")["avg_solubility"].mean().nlargest(max_solvents).index
        pivot_df = pivot_df[top_solvents]

    n_cells = pivot_df.shape[0] * pivot_df.shape[1]
    show_annot = n_cells <= 150
    annot_fontsize = 10 if n_cells <= 50 else 8 if n_cells <= 100 else 6

    colors_low = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6"]
    colors_high = ["#4292c6", "#2171b5", "#08519c", "#08306b"]
    cmap = LinearSegmentedColormap.from_list(
        "solubility_emphasis", colors_low + colors_high, N=256,
    )

    fig, ax = plt.subplots(
        figsize=(
            max(14, len(pivot_df.columns) * 0.5),
            max(8, len(pivot_df) * 0.8),
        )
    )

    vmax = pivot_df.max().max()
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax) if vmax > 20 else None

    sns.heatmap(
        pivot_df, annot=show_annot, fmt=".1f", cmap=cmap,
        cbar_kws={"label": "Solubility (%)", "shrink": 0.8},
        linewidths=0.3, ax=ax, annot_kws={"size": annot_fontsize},
        norm=norm,
    )

    title = f"Solubility Heatmap at {temperature}C"
    if target_polymer:
        title = f"{target_polymer} Solubility at {temperature}C"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Solvent", fontsize=14, fontweight="bold")
    ax.set_ylabel("Polymer", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

    plt.tight_layout()
    filepath = save_plot(fig, "solubility_heatmap", "matplotlib")

    output = "Heatmap Created\n\n"
    output += f"Temperature: {temperature}C +/- {temperature_tolerance}C\n"
    output += f"Polymers: {len(pivot_df)}\n"
    output += f"Solvents: {len(pivot_df.columns)}\n"

    if target_polymer and target_polymer.upper() in [p.upper() for p in pivot_df.index]:
        idx = [p for p in pivot_df.index if p.upper() == target_polymer.upper()][0]
        row = pivot_df.loc[idx].sort_values(ascending=False)
        output += f"\n**Top solvents for {target_polymer}:**\n"
        for solvent, sol in list(row.items())[:10]:
            if pd.notna(sol):
                symbol = "OK" if sol > 20 else "WARN" if sol > 5 else "LOW"
                output += f"  {symbol} {solvent}: {sol:.1f}%\n"

    output += f"\n{_get_plot_url(filepath)}"
    output += (
        "\n\n*Color scale emphasizes 0-20% range for better "
        "differentiation of low-solubility solvents.*"
    )

    del df
    return output


@safe_tool_wrapper
def plot_multi_panel_analysis(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    target_polymer: str,
    comparison_polymers: str,
    solvent: str,
) -> str:
    """Create a 4-panel separation analysis plot comparing a target polymer against others in a solvent.

    Args:
        table_name: Database table name
        polymer_column: Column with polymer names
        solvent_column: Column with solvent names
        temperature_column: Column with temperature values
        solubility_column: Column with solubility values
        target_polymer: Polymer to dissolve
        comparison_polymers: Comma-separated polymers to separate from
        solvent: Solvent to analyze

    WHEN TO USE:
    - "Can I separate PET from PE using toluene?"
    - "Show a multi-panel separation analysis for PS vs PVDF"
    """

    if isinstance(comparison_polymers, str):
        comp_list = [p.strip() for p in comparison_polymers.split(",") if p.strip()]
    elif isinstance(comparison_polymers, list):
        comp_list = comparison_polymers
    else:
        return "Error: comparison_polymers must be a comma-separated string"

    if not comp_list:
        return "Error: No comparison polymers specified."

    from strap.solubility import get_solubility_curve, get_solubility

    all_polymers = [target_polymer] + comp_list

    # Build curves dict: polymer -> {temps: [...], sols: [...]}
    curves: Dict[str, Dict[str, list]] = {}
    for poly in all_polymers:
        curve = get_solubility_curve(poly, solvent, 25.0, 160.0, 5.0)
        if curve:
            curves[poly] = {
                "temps": [pt["temperature"] for pt in curve],
                "sols": [pt["solubility"] for pt in curve],
            }

    if not curves:
        return f"No solubility data found for any polymer in {solvent}."

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    colors_others = plt.cm.Reds(np.linspace(0.4, 0.8, len(comp_list)))

    # Panel 1: Solubility curves
    ax1 = fig.add_subplot(gs[0, 0])
    if target_polymer in curves:
        ax1.plot(
            curves[target_polymer]["temps"], curves[target_polymer]["sols"],
            "o-", color="green", linewidth=3, markersize=8, label=target_polymer,
        )

    for i, comp in enumerate(comp_list):
        if comp in curves:
            ax1.plot(
                curves[comp]["temps"], curves[comp]["sols"],
                "s--", color=colors_others[i], linewidth=2, markersize=6, label=comp,
            )

    ax1.set_xlabel("Temperature (C)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Solubility (%)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Solubility in {solvent}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Selectivity vs Temperature
    ax2 = fig.add_subplot(gs[0, 1])
    if target_polymer in curves:
        temps = curves[target_polymer]["temps"]
        target_sols = curves[target_polymer]["sols"]

        for i, comp in enumerate(comp_list):
            selectivity = []
            for j, temp in enumerate(temps):
                comp_sol = get_solubility(comp, solvent, temp)
                if comp_sol is not None:
                    selectivity.append(target_sols[j] - comp_sol)
                else:
                    selectivity.append(np.nan)

            ax2.plot(
                temps, selectivity, "o-", color=colors_others[i],
                linewidth=2, markersize=6, label=f"vs {comp}",
            )

        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.axhline(y=10, color="green", linestyle=":", alpha=0.7, label="Good selectivity (10%)")

    ax2.set_xlabel("Temperature (C)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Selectivity (%) (target - other)", fontsize=11, fontweight="bold")
    ax2.set_title("Selectivity vs Temperature", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Separation window
    ax3 = fig.add_subplot(gs[1, 0])
    good_separation_temps: List[float] = []
    if target_polymer in curves:
        temps = curves[target_polymer]["temps"]
        target_sols = curves[target_polymer]["sols"]

        for j, temp in enumerate(temps):
            max_other = 0.0
            for comp in comp_list:
                comp_sol = get_solubility(comp, solvent, temp)
                if comp_sol is not None:
                    max_other = max(max_other, comp_sol)
            if target_sols[j] - max_other > 5:
                good_separation_temps.append(temp)

        all_temps = sorted(temps)
        bar_colors = ["green" if t in good_separation_temps else "lightgray" for t in all_temps]
        ax3.bar(range(len(all_temps)), [1] * len(all_temps), color=bar_colors, edgecolor="black")
        ax3.set_xticks(range(len(all_temps)))
        ax3.set_xticklabels([f"{int(t)}C" for t in all_temps], rotation=45, ha="right")

    ax3.set_ylabel("Separation Feasibility", fontsize=11, fontweight="bold")
    ax3.set_title("Separation Window", fontsize=12, fontweight="bold")
    ax3.set_yticks([])

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Good separation"),
        Patch(facecolor="lightgray", label="Poor separation"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right")

    # Panel 4: Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    summary_text = f"**Analysis Summary**\n\n"
    summary_text += f"Target: {target_polymer}\n"
    summary_text += f"Solvent: {solvent}\n"
    summary_text += f"Comparisons: {', '.join(comp_list)}\n\n"

    if good_separation_temps:
        summary_text += "Separation possible at:\n"
        summary_text += f"   {', '.join([f'{int(t)}C' for t in good_separation_temps])}\n"
    else:
        summary_text += "No clear separation window\n"

    ax4.text(
        0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    plt.tight_layout()
    filepath = save_plot(fig, "multi_panel_analysis", "matplotlib")

    output = "Multi-Panel Analysis Created\n\n"
    output += f"Target: {target_polymer}\n"
    output += f"Solvent: {solvent}\n"
    output += f"Comparisons: {', '.join(comp_list)}\n\n"

    if good_separation_temps:
        output += f"**Separation possible at:** {', '.join([f'{int(t)}C' for t in good_separation_temps])}\n"

    output += f"\n{_get_plot_url(filepath)}"

    return output


@safe_tool_wrapper
def plot_comparison_dashboard(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    temperature_column: str,
    solubility_column: str,
    polymers: str,
    temperature: float = 25.0,
) -> str:
    """Create a 4-panel dashboard comparing solubility of multiple polymers across solvents.

    Args:
        table_name: Database table name
        polymer_column: Column with polymer names
        solvent_column: Column with solvent names
        temperature_column: Column with temperature values
        solubility_column: Column with solubility values
        polymers: Comma-separated polymer names to compare
        temperature: Target temperature in C (default 25)

    WHEN TO USE:
    - "Compare solubility of PET, PS, and PE across solvents"
    - "Show a dashboard ranking these polymers by solubility"
    """
    from strap.solubility import get_solubility, get_available_solvents_for_polymer

    polymer_list = [p.strip() for p in polymers.split(",")]

    # Build solubility data for all polymers × solvents at the target temperature
    rows: List[Dict[str, Any]] = []
    for poly in polymer_list:
        for sv in get_available_solvents_for_polymer(poly):
            sol = get_solubility(poly, sv, temperature)
            if sol is not None:
                rows.append({"polymer": poly, "solvent": sv, "avg_sol": sol})

    if not rows:
        return f"No solubility data found at {temperature}C for the specified polymers."

    df = pd.DataFrame(rows)
    solvents = df["solvent"].unique()

    # Limit number of solvents for readability
    max_solvents = 15
    if len(solvents) > max_solvents:
        solvent_means = df.groupby("solvent")["avg_sol"].mean().sort_values(ascending=False)
        solvents = solvent_means.head(max_solvents).index.tolist()
        df = df[df["solvent"].isin(solvents)]

    fig = plt.figure(figsize=(20, 12))

    # Panel 1: Grouped bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(solvents))
    width = 0.8 / len(polymer_list)
    colors = plt.cm.Set2(np.linspace(0, 1, len(polymer_list)))

    for i, polymer in enumerate(polymer_list):
        poly_data = df[df["polymer"] == polymer]
        values = []
        for sv in solvents:
            sol_data = poly_data[poly_data["solvent"] == sv]["avg_sol"]
            values.append(sol_data.values[0] if len(sol_data) > 0 else 0)
        ax1.bar(
            x + i * width, values, width, label=polymer,
            color=colors[i], edgecolor="black", linewidth=0.5,
        )

    ax1.set_xlabel("Solvent", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Solubility (%)", fontsize=13, fontweight="bold")
    ax1.set_title(f"Solubility Comparison at {temperature}C", fontsize=15, fontweight="bold", pad=10)
    ax1.set_xticks(x + width * (len(polymer_list) - 1) / 2)
    short_labels = [s[:20] + "..." if len(s) > 20 else s for s in solvents]
    ax1.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=10)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    pivot = df.pivot(index="polymer", columns="solvent", values="avg_sol")
    n_cells = pivot.shape[0] * pivot.shape[1]
    annot_size = 11 if n_cells <= 20 else 9 if n_cells <= 40 else 7
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax2,
        annot_kws={"size": annot_size}, linewidths=0.5,
        cbar_kws={"label": "Solubility (%)", "shrink": 0.8},
    )
    ax2.set_title("Solubility Heatmap", fontsize=15, fontweight="bold", pad=10)
    ax2.set_xlabel("Solvent", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Polymer", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=11)

    # Panel 3: Box plot
    ax3 = fig.add_subplot(2, 2, 3)
    data_for_box = [df[df["polymer"] == p]["avg_sol"].values for p in polymer_list]
    bp = ax3.boxplot(data_for_box, labels=polymer_list, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    ax3.set_xlabel("Polymer", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Solubility Distribution (%)", fontsize=13, fontweight="bold")
    ax3.set_title("Solubility Distribution by Polymer", fontsize=15, fontweight="bold", pad=10)
    ax3.tick_params(axis="both", labelsize=11)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Rankings
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    ranking_text = "POLYMER RANKINGS\n" + "=" * 25 + "\n\n"
    mean_sols = {p: df[df["polymer"] == p]["avg_sol"].mean() for p in polymer_list}
    sorted_polymers = sorted(mean_sols.items(), key=lambda x: x[1], reverse=True)

    for i, (polymer, sol) in enumerate(sorted_polymers, 1):
        ranking_text += f"{i}. {polymer}: {sol:.2f}%\n"

    ax4.text(
        0.1, 0.85, ranking_text, transform=ax4.transAxes, fontsize=14,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7, edgecolor="gray"),
    )

    plt.tight_layout()
    filepath = save_plot(fig, "comparison_dashboard", "matplotlib")

    output = "Comparison Dashboard Created\n\n"
    output += f"Temperature: {temperature}C\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {len(solvents)}\n\n"
    output += _get_plot_url(filepath)

    return output


@safe_tool_wrapper
def plot_interpolation_vs_sql(
    polymer_solvent_pairs: str,
    temperature_min: float = 25.0,
    temperature_max: float = 160.0,
) -> str:
    """Compare interpolation model predictions vs raw SQL database values side-by-side.

    Creates a multi-panel figure showing smooth interpolation curves alongside
    raw database data points for each polymer-solvent pair. Useful for validating
    that the interpolation model accurately represents the underlying data.

    Args:
        polymer_solvent_pairs: Comma-separated pairs in "POLYMER:solvent" format.
            E.g. "LDPE:dodecane, EVOH:dimethylsulfoxide, PET:nitrobenzene"
        temperature_min: Start of temperature range (default 25)
        temperature_max: End of temperature range (default 160)

    WHEN TO USE:
    - "Compare interpolation vs raw data for my separation sequence"
    - "Validate that interpolated values match the database"
    - "Show me how accurate the model is for LDPE in dodecane"
    """
    from strap.solubility import get_solubility_curve, INTERPOLATION, SQL

    # Parse pairs
    pairs: List[Tuple[str, str]] = []
    for item in polymer_solvent_pairs.split(","):
        item = item.strip()
        if ":" not in item:
            return f"Invalid pair format: '{item}'. Use 'POLYMER:solvent' format."
        polymer, solvent = item.split(":", 1)
        pairs.append((polymer.strip(), solvent.strip()))

    if not pairs:
        return "No polymer-solvent pairs provided."

    t_start = max(temperature_min, 25.0)
    t_end = min(temperature_max, 160.0)

    n_pairs = len(pairs)
    n_cols = min(n_pairs, 3)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    for idx, (polymer, solvent) in enumerate(pairs):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        # Get interpolation curve (smooth)
        interp_curve = get_solubility_curve(polymer, solvent, t_start, t_end, 1.0, method=INTERPOLATION)
        # Get SQL raw data points
        sql_curve = get_solubility_curve(polymer, solvent, t_start, t_end, 5.0, method=SQL)

        has_data = False

        if interp_curve:
            interp_temps = [pt["temperature"] for pt in interp_curve]
            interp_sols = [pt["solubility"] for pt in interp_curve]
            ax.plot(
                interp_temps, interp_sols, "-", color="#2171b5", linewidth=2.5,
                label="Interpolation model", zorder=2,
            )
            has_data = True

        if sql_curve:
            sql_temps = [pt["temperature"] for pt in sql_curve]
            sql_sols = [pt["solubility"] for pt in sql_curve]
            ax.scatter(
                sql_temps, sql_sols, color="#d94801", s=60, marker="o",
                edgecolors="black", linewidths=0.5, label="Database (raw)", zorder=3,
            )
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")

        ax.set_title(f"{polymer} in {solvent}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Temperature (C)", fontsize=10)
        ax.set_ylabel("Solubility (%)", fontsize=10)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_pairs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Interpolation Model vs Database Comparison",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    filepath = save_plot(fig, "interpolation_vs_sql", "matplotlib")
    plt.close(fig)

    output = "Interpolation vs SQL Comparison Plot Created\n\n"
    output += f"Pairs: {len(pairs)}\n"
    output += f"Temperature range: {t_start}C - {t_end}C\n"
    for polymer, solvent in pairs:
        output += f"  - {polymer} in {solvent}\n"
    output += "\nBlue line = interpolation model (ln(S) = A + B/T + C/T^2)\n"
    output += "Orange dots = raw database values\n"
    output += f"\n{_get_plot_url(filepath)}"

    gc.collect()
    return output


@safe_tool_wrapper
async def plot_solvent_properties(
    table_name: str,
    polymer_column: str,
    solvent_column: str,
    solubility_column: str,
    polymer: str,
    property_to_plot: str,
    temperature_column: Optional[str] = None,
    temperature: Optional[float] = 25.0,
    min_solubility: Optional[float] = 0.0,
    max_solvents: int = 20,
    plot_type: str = "bar",
    y_property: Optional[str] = None,
) -> str:
    """Plot solvent properties for solvents that dissolve a polymer.

    In 1D mode (default), creates a bar or scatter chart of one property.
    When *y_property* is given, creates a 2D scatter (x=property_to_plot,
    y=y_property, colour=solubility).

    Args:
        table_name, polymer_column, solvent_column, solubility_column: DB columns
        polymer: Polymer to analyse
        property_to_plot: 'bp', 'energy', 'logp', 'cp', or 'g_score'
        temperature_column: Temperature column (optional)
        temperature: Target temperature in C (default 25)
        min_solubility: Min solubility threshold (default 0)
        max_solvents: Max solvents to show (default 20)
        plot_type: 'bar' or 'scatter' for 1D mode (default 'bar')
        y_property: Second property for 2D scatter mode (optional)

    WHEN TO USE:
    - "Plot boiling points of solvents that dissolve PET"
    - "Show energy costs for PS solvents"
    - "Plot LogP vs G-score for solvents that dissolve PET at 140C"
    """
    valid_properties = {"bp", "energy", "logp", "cp", "g_score"}
    property_lower = property_to_plot.lower().strip()
    if property_lower not in valid_properties:
        return f"Invalid property '{property_to_plot}'. Must be one of: {', '.join(sorted(valid_properties))}"

    if y_property is not None:
        y_lower = y_property.lower().strip()
        if y_lower not in valid_properties:
            return f"Invalid y_property '{y_property}'. Must be one of: {', '.join(sorted(valid_properties))}"
    else:
        y_lower = None

    property_labels = {
        "bp": "Boiling Point (C)",
        "energy": "Energy Cost (J/g)",
        "logp": "LogP (Lipophilicity)",
        "cp": "Heat Capacity Cp (J/g*K)",
        "g_score": "G-Score (Safety - higher is safer)",
    }

    # Query for solvents that dissolve the polymer via interpolation model
    from strap.solubility import get_solubility, get_available_solvents_for_polymer

    query_temp = temperature if temperature is not None else 100.0
    all_solvents = get_available_solvents_for_polymer(polymer)

    solubility_map: Dict[str, float] = {}
    for sv in all_solvents:
        sol = get_solubility(polymer, sv, query_temp)
        if sol is not None and sol >= (min_solubility or 0.0):
            solubility_map[sv] = sol

    # Sort by solubility desc and limit
    sorted_pairs = sorted(solubility_map.items(), key=lambda x: x[1], reverse=True)[:max_solvents]
    solvents_found = [s for s, _ in sorted_pairs]
    solubility_map = dict(sorted_pairs)

    if not solvents_found:
        return f"No solvents found for {polymer} with solubility >= {min_solubility}%"

    # -----------------------------------------------------------------
    # 2D scatter mode  (y_property supplied)
    # -----------------------------------------------------------------
    if y_lower is not None:
        conn = get_connection()
        prop_data: List[Dict[str, Any]] = []
        for solvent in solvents_found:
            props = get_cross_database_properties(solvent, conn)
            if props:
                prop_data.append({
                    "solvent": solvent,
                    "solubility": solubility_map.get(solvent, 0),
                    "logp": props.get("logp"),
                    "bp": props.get("bp"),
                    "energy": props.get("energy"),
                    "cp": props.get("cp"),
                    "g_score": props.get("g_score"),
                })

        if not prop_data:
            return (
                f"Found {len(solvents_found)} solvents but couldn't retrieve "
                f"properties. Solvent names may not match across databases."
            )

        pdf = pd.DataFrame(prop_data)
        pdf_valid = pdf.dropna(subset=[property_lower, y_lower])

        if len(pdf_valid) == 0:
            avail = [p for p in valid_properties if pdf[p].notna().any()]
            return (
                f"No solvents have both {property_to_plot} and {y_property} data. "
                f"Available properties: {', '.join(avail)}"
            )

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            pdf_valid[property_lower],
            pdf_valid[y_lower],
            c=pdf_valid["solubility"],
            cmap="YlOrRd",
            s=150,
            alpha=0.7,
            edgecolors="black",
            linewidths=1,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(f"Solubility in {polymer} (%)", fontsize=12)

        for _, row in pdf_valid.iterrows():
            ax.annotate(
                row["solvent"],
                (row[property_lower], row[y_lower]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

        ax.set_xlabel(property_labels.get(property_lower, property_lower), fontsize=14, fontweight="bold")
        ax.set_ylabel(property_labels.get(y_lower, y_lower), fontsize=14, fontweight="bold")
        ax.set_title(
            f"{property_to_plot.upper()} vs {y_property.upper()} for {polymer} Solvents at {temperature}C",
            fontsize=16, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Reference lines for G-score thresholds
        if y_lower == "g_score":
            ax.axhline(y=6.0, color="orange", linestyle="--", alpha=0.5, label="Good threshold (6.0)")
            ax.axhline(y=8.0, color="green", linestyle="--", alpha=0.5, label="Excellent threshold (8.0)")
            ax.legend()
        elif property_lower == "g_score":
            ax.axvline(x=6.0, color="orange", linestyle="--", alpha=0.5, label="Good threshold (6.0)")
            ax.axvline(x=8.0, color="green", linestyle="--", alpha=0.5, label="Excellent threshold (8.0)")
            ax.legend()

        plt.tight_layout()
        filepath = save_plot(fig, f"solvent_properties_{property_lower}_vs_{y_lower}")
        plt.close(fig)
        gc.collect()

        output_lines = [f"**{property_to_plot.upper()} vs {y_property.upper()} Scatter Plot for {polymer}**\n"]
        output_lines.append(f"Temperature: {temperature}C")
        output_lines.append(f"Solvents with data: {len(pdf_valid)} (of {len(solvents_found)} found)")
        output_lines.append(f"Min solubility filter: {min_solubility}%\n")
        output_lines.append("**Top Solvents (by solubility):**")
        for _, row in pdf_valid.nlargest(5, "solubility").iterrows():
            line = f"  - **{row['solvent']}**: {row['solubility']:.1f}% solubility"
            if pd.notna(row.get("logp")):
                line += f", LogP={row['logp']:.2f}"
            if pd.notna(row.get("g_score")):
                line += f", G-score={row['g_score']:.1f}"
            if pd.notna(row.get("bp")):
                line += f", BP={row['bp']:.0f}C"
            output_lines.append(line)
        output_lines.append(f"\n{_get_plot_url(filepath)}")

        del pdf, pdf_valid
        gc.collect()
        return "\n".join(output_lines)

    # -----------------------------------------------------------------
    # 1D mode  (original bar / scatter)
    # -----------------------------------------------------------------
    if property_lower == "g_score":
        # g_score is not in the solvent-data table; use cross-database lookup
        conn = get_connection()
        solvent_data_1d: List[Dict[str, Any]] = []
        for solv in solvents_found:
            props = get_cross_database_properties(solv, conn)
            if props and props.get("g_score") is not None:
                solvent_data_1d.append({
                    "solvent": solv,
                    "property_value": props["g_score"],
                    "solubility": solubility_map.get(solv, 0),
                })
    else:
        # Look up properties using robust matching
        solvent_table = _get_solvent_table_name()
        if not solvent_table:
            return "Solvent property database (solvent_data) not found. Cannot retrieve properties."

        logger.info(f"Looking up {property_lower} for {len(solvents_found)} solvents")
        props = await _lookup_solvent_properties(solvents_found, solvent_table)

        # Extract the requested property
        solvent_data_1d = []
        for solv in solvents_found:
            if solv in props and props[solv][property_lower] is not None:
                solubility = df[df[solvent_column] == solv]["avg_solubility"].values[0]
                solvent_data_1d.append({
                    "solvent": solv,
                    "property_value": props[solv][property_lower],
                    "solubility": solubility,
                })

    if not solvent_data_1d:
        return (
            f"No {property_lower.upper()} data found for solvents that dissolve {polymer}.\n\n"
            f"This may be due to naming mismatches between databases. "
            f"Found {len(solvents_found)} solvents but none had {property_lower.upper()} data."
        )

    solvent_data_1d.sort(key=lambda x: x["property_value"])

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    if plot_type.lower() == "bar":
        names = [d["solvent"] for d in solvent_data_1d]
        values = [d["property_value"] for d in solvent_data_1d]
        solubilities = [d["solubility"] for d in solvent_data_1d]

        color_arr = plt.cm.YlOrRd(np.array(solubilities) / max(solubilities))
        bars = ax.bar(range(len(names)), values, color=color_arr, edgecolor="black", linewidth=1.5)

        for i, (bar, sol) in enumerate(zip(bars, solubilities)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height,
                f"{sol:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=11)
        ax.set_ylabel(property_labels[property_lower], fontsize=13, fontweight="bold")
        ax.set_xlabel("Solvent", fontsize=13, fontweight="bold")
        ax.set_title(
            f"{property_labels[property_lower]} for Solvents Dissolving {polymer}\n"
            f"(Color intensity = solubility)",
            fontsize=15, fontweight="bold", pad=15,
        )

    else:  # scatter plot
        values = [d["property_value"] for d in solvent_data_1d]
        solubilities = [d["solubility"] for d in solvent_data_1d]
        names = [d["solvent"] for d in solvent_data_1d]

        ax.scatter(
            values, solubilities, s=150, alpha=0.6,
            edgecolors="black", linewidth=2, c=values, cmap="viridis",
        )

        for x, y, name in zip(values, solubilities, names):
            ax.annotate(
                name, (x, y), xytext=(5, 5), textcoords="offset points",
                fontsize=9, alpha=0.8, fontweight="bold",
            )

        ax.set_xlabel(property_labels[property_lower], fontsize=13, fontweight="bold")
        ax.set_ylabel("Solubility (%)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"{property_labels[property_lower]} vs Solubility for {polymer}",
            fontsize=15, fontweight="bold", pad=15,
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = save_plot(fig, f"solvent_properties_{property_lower}")
    plt.close(fig)
    gc.collect()

    # Build output message
    output_lines = [f"Solvent Property Plot Created\n"]
    output_lines.append(f"**Polymer:** {polymer}")
    output_lines.append(f"**Property:** {property_labels[property_lower]}")
    output_lines.append(f"**Solvents analyzed:** {len(solvent_data_1d)} (from {len(solvents_found)} total)")

    if len(solvents_found) > len(solvent_data_1d):
        missing = len(solvents_found) - len(solvent_data_1d)
        output_lines.append(f"Note: {missing} solvents had no {property_lower.upper()} data")

    output_lines.append(f"\n**Top 5 by {property_labels[property_lower]}:**")
    for i, d in enumerate(solvent_data_1d[:5], 1):
        prop_val = d["property_value"]
        sol = d["solubility"]
        output_lines.append(f"{i}. **{d['solvent']}**: {prop_val:.2f} (solubility: {sol:.1f}%)")

    output_lines.append(f"\n{_get_plot_url(filepath)}")

    return "\n".join(output_lines)
