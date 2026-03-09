"""Shared helper functions for visualization tool adapters."""

from __future__ import annotations

from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strap.database import get_connection

PUB_FONT = "Liberation Sans"  # metrically identical to Arial
PUB_FONTSIZE = 8
PUB_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
]

SOLVENT_NAME_MAPPING: dict[str, list[str]] = {
    "xylene": ["1,2-dimethylbenzene", "1,4-dimethylbenzene"],
    "xylenes": ["1,2-dimethylbenzene", "1,4-dimethylbenzene"],
    "o-xylene": ["1,2-dimethylbenzene"],
    "p-xylene": ["1,4-dimethylbenzene"],
    "m-xylene": ["1,4-dimethylbenzene"],
    "ortho-xylene": ["1,2-dimethylbenzene"],
    "para-xylene": ["1,4-dimethylbenzene"],
    "heptane": ["n-heptane"],
    "n-heptane": ["n-heptane"],
    "hexane": ["hexane"],
    "n-hexane": ["hexane"],
    "pentane": ["pentane"],
    "n-pentane": ["pentane"],
    "octane": ["octane"],
    "n-octane": ["octane"],
    "dmso": ["dimethylsulfoxide"],
    "dimethyl sulfoxide": ["dimethylsulfoxide"],
    "dmf": ["dimethylformamide"],
    "dimethyl formamide": ["dimethylformamide"],
    "nmp": ["n-methylpyrrolidone"],
    "n-methyl-2-pyrrolidone": ["n-methylpyrrolidone"],
    "acetone": ["propanone"],
    "2-propanone": ["propanone"],
    "mek": ["butanone"],
    "methyl ethyl ketone": ["butanone"],
    "ipa": ["2-propanol"],
    "isopropanol": ["2-propanol"],
    "isopropyl alcohol": ["2-propanol"],
    "n-propanol": ["propanol"],
    "1-propanol": ["propanol"],
    "meoh": ["methanol"],
    "etoh": ["ethanol"],
    "tetrahydrofuran": ["thf"],
    "tetrahydropyran": ["thp"],
    "dihydropyran": ["2,3-dihydropyran"],
    "dcm": ["ch2cl2"],
    "dichloromethane": ["ch2cl2"],
    "methylene chloride": ["ch2cl2"],
    "chloroform": ["chcl3"],
    "trichloromethane": ["chcl3"],
    "ethyl acetate": ["ethylacetate"],
    "methyl acetate": ["methylacetate"],
    "water": ["h2o"],
    "ethylene glycol": ["glycol"],
    "propylene glycol": ["propyleneglycol"],
}

SOLVENT_FRAGMENT_RECONSTRUCTION: dict[tuple[str, str], str] = {
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


def apply_pub_style() -> None:
    """Apply publication-style matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [PUB_FONT, "Arial", "DejaVu Sans"],
            "font.size": PUB_FONTSIZE,
            "axes.labelsize": PUB_FONTSIZE,
            "axes.titlesize": PUB_FONTSIZE,
            "xtick.labelsize": PUB_FONTSIZE,
            "ytick.labelsize": PUB_FONTSIZE,
            "legend.fontsize": PUB_FONTSIZE - 1,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def normalize_solvent_names(solvents: list[str]) -> list[str]:
    """Normalize solvent names to match database entries."""
    reconstructed: list[str] = []
    index = 0
    while index < len(solvents):
        solvent = solvents[index].strip().lower()
        if index + 1 < len(solvents):
            next_solvent = solvents[index + 1].strip().lower()
            fragment_key = (solvent, next_solvent)
            if fragment_key in SOLVENT_FRAGMENT_RECONSTRUCTION:
                reconstructed.append(SOLVENT_FRAGMENT_RECONSTRUCTION[fragment_key])
                index += 2
                continue
        reconstructed.append(solvent)
        index += 1

    normalized: list[str] = []
    for solvent in reconstructed:
        solvent_lower = solvent.strip().lower()
        if solvent_lower in SOLVENT_NAME_MAPPING:
            normalized.extend(SOLVENT_NAME_MAPPING[solvent_lower])
        else:
            normalized.append(solvent_lower)
    return normalized


def execute_query(query: str, limit: int = 100) -> dict[str, Any]:
    """Execute a read-only DuckDB query and return a legacy-compatible envelope."""
    conn = get_connection()
    try:
        query_lower = query.lower().strip()
        dangerous_keywords = [
            "drop",
            "delete",
            "insert",
            "update",
            "alter",
            "create",
            "truncate",
        ]
        if any(keyword in query_lower.split() for keyword in dangerous_keywords):
            return {"success": False, "error": "Unsafe operation detected", "query": query}

        if "limit" not in query_lower and not query_lower.rstrip().endswith(";"):
            query = f"{query.rstrip(';')} LIMIT {limit}"

        result_df = conn.execute(query).fetchdf()
        preview = result_df.head(10).to_markdown(index=False) if len(result_df) > 0 else "No data"
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
    except Exception as exc:
        return {"success": False, "error": str(exc), "query": query}


def verify_inputs(
    table_name: str,
    columns: dict[str, str],
    values: Optional[dict[str, list[str]]] = None,
) -> tuple[bool, str]:
    """Verify a table/column/value combination against the live database."""
    conn = get_connection()
    issues: list[str] = []
    warnings: list[str] = []

    try:
        tables = conn.execute("SHOW TABLES").fetchdf()
        if table_name not in tables["name"].values:
            return False, f"Table '{table_name}' not found. Available: {list(tables['name'].values)}"
    except Exception as exc:
        return False, f"Could not list tables: {exc}"

    try:
        schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        available_cols = set(schema["column_name"].values)
    except Exception as exc:
        return False, f"Could not get schema: {exc}"

    for purpose, col_name in columns.items():
        if col_name not in available_cols:
            issues.append(f"Column '{col_name}' ({purpose}) not found")
            similar = [col for col in available_cols if col_name.lower() in col.lower()]
            if similar:
                warnings.append(f"Did you mean: {similar}?")

    if issues:
        message = "Verification failed:\n- " + "\n- ".join(issues)
        if warnings:
            message += "\n\n" + "\n".join(warnings)
        return False, message

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
        return False, "Value verification failed:\n- " + "\n- ".join(issues)

    return True, "All inputs verified"


def get_plot_url(filepath: str) -> str:
    """Convert a local file path to the user-facing status string."""
    return f"Plot saved: `{filepath}`"


def get_solvent_table_name() -> Optional[str]:
    """Auto-detect the solvent property table in the current database."""
    conn = get_connection()
    try:
        tables = conn.execute("SHOW TABLES").fetchdf()
    except Exception:
        return None

    for table in tables["name"].values:
        if "solvent" in table.lower() and "solubility" not in table.lower():
            try:
                cols_df = conn.execute(f"DESCRIBE {table}").fetchdf()
                cols_lower = [col.lower() for col in cols_df["column_name"].values]
                if (
                    any("bp" in col or "boil" in col for col in cols_lower)
                    or any("logp" in col for col in cols_lower)
                    or any("energy" in col for col in cols_lower)
                ):
                    return str(table)
            except Exception:
                continue
    return None


def get_solvent_name_column(table_name: str) -> Optional[str]:
    """Return the likely solvent-name column for a table."""
    conn = get_connection()
    try:
        cols_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        cols = list(cols_df["column_name"].values)
        types_map = dict(zip(cols_df["column_name"], cols_df["column_type"]))
    except Exception:
        return None

    for pattern in ["solvent_name", "solvent", "name", "compound"]:
        for col in cols:
            if pattern in col.lower():
                return col

    for col in cols:
        col_type = str(types_map.get(col, "")).upper()
        if "VARCHAR" in col_type or "TEXT" in col_type:
            return col
    return cols[0] if cols else None


def get_cosmobase_column(table_name: str) -> Optional[str]:
    """Return the cosmobase column if present."""
    conn = get_connection()
    try:
        cols_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        for col in cols_df["column_name"].values:
            if "cosmobase" in col.lower():
                return str(col)
    except Exception:
        pass
    return None


async def lookup_solvent_properties(
    solvent_names: list[str],
    solvent_table: str,
) -> dict[str, dict[str, Any]]:
    """Look up solvent properties with robust fuzzy matching."""
    conn = get_connection()

    from strap.solvent_registry import ABBREVIATION_MAP

    try:
        cols_df = conn.execute(f"DESCRIBE {solvent_table}").fetchdf()
        cols = list(cols_df["column_name"].values)
    except Exception:
        return {}

    cols_lower = {col.lower(): col for col in cols}
    cosmobase_col = get_cosmobase_column(solvent_table)
    name_col = get_solvent_name_column(solvent_table)

    logp_col = next((cols_lower[key] for key in cols_lower if "logp" in key), None)
    bp_col = next((cols_lower[key] for key in cols_lower if "bp" in key or "boil" in key), None)
    energy_col = next((cols_lower[key] for key in cols_lower if "energy" in key), None)
    cp_col = next((cols_lower[key] for key in cols_lower if "cp" in key and "logp" not in key), None)

    match_col = cosmobase_col or name_col
    if not match_col:
        return {}

    def _find_solvent_match(solvent: str):
        sol_lower = solvent.lower().strip()
        sol_normalized = sol_lower.replace("-", "").replace(" ", "").replace(",", "")

        try:
            df = conn.execute(
                f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") = '{sol_lower}'"
            ).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass

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

        try:
            df = conn.execute(
                f"SELECT * FROM {solvent_table} WHERE LOWER(\"{match_col}\") LIKE '%{sol_lower}%' "
                f"ORDER BY LENGTH(\"{match_col}\")"
            ).fetchdf()
            if len(df) > 0:
                return df.iloc[0]
        except Exception:
            pass

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

    props_map: dict[str, dict[str, Any]] = {}
    for solvent in solvent_names:
        row = _find_solvent_match(solvent)
        props: dict[str, Any] = {"logp": None, "bp": None, "energy": None, "cp": None}
        if row is not None:
            props = {
                "logp": row[logp_col] if logp_col and logp_col in row.index else None,
                "bp": row[bp_col] if bp_col and bp_col in row.index else None,
                "energy": row[energy_col] if energy_col and energy_col in row.index else None,
                "cp": row[cp_col] if cp_col and cp_col in row.index else None,
            }
        props_map[solvent] = props
    return props_map
