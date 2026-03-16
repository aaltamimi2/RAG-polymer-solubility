"""GSK safety (G-Score) analysis tools.
Provides tools for querying and visualizing GSK solvent safety scores,
finding family alternatives, and plotting solvent properties against
polymer solubility data.
"""
from __future__ import annotations
import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from strap.database import get_connection
from strap.paths import get_data_path
from strap.services.tool_response_service import json_tool_error, json_tool_response
from strap.solubility import get_logp
from strap.tools._helpers import (
    get_plots_dir,
    safe_tool_wrapper,
)
logger = logging.getLogger(__name__)
def _gsk_response(tool_name: str, display: str, **data) -> str:
    return json_tool_response(display, data, tool_name=tool_name)
def _gsk_error(tool_name: str, message: str, *, error_code: str = "invalid_input", **data) -> str:
    return json_tool_error(message, tool_name=tool_name, error_code=error_code, **data)
# ---------------------------------------------------------------------------
# GreenSolventDB 10k — ML-predicted G-scores (lazy-loaded fallback)
# ---------------------------------------------------------------------------
_GREEN_SOLVENT_DB: dict[str, dict] | None = None  # keyed by lowercase name
_GREEN_SOLVENT_DB_CAS: dict[str, dict] | None = None  # keyed by CAS
_GSK_DATAFRAME: pd.DataFrame | None = None


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())


def _load_gsk_dataframe() -> pd.DataFrame:
    """Lazy-load the curated GSK dataset into a local pandas cache.

    This avoids shared DuckDB connection issues under concurrent safety-tool use.
    """
    global _GSK_DATAFRAME
    if _GSK_DATAFRAME is not None:
        return _GSK_DATAFRAME

    csv_path = get_data_path("GSK_dataset.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [_normalize_column_name(col) for col in df.columns]
    required = {"solvent_common_name", "classification", "g_score", "cas_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GSK dataset missing required columns: {sorted(missing)}")
    df = df[list(required)].copy()
    df["solvent_common_name"] = df["solvent_common_name"].astype(str).str.strip()
    df["classification"] = df["classification"].astype(str).str.strip()
    df["cas_number"] = df["cas_number"].astype(str).str.strip()
    df["g_score"] = pd.to_numeric(df["g_score"], errors="coerce")
    df["solvent_common_name_norm"] = df["solvent_common_name"].str.lower()
    _GSK_DATAFRAME = df
    return _GSK_DATAFRAME


def _lookup_gsk_exact(solvent_name: str) -> pd.DataFrame:
    df = _load_gsk_dataframe()
    key = solvent_name.strip().lower()
    return df[df["solvent_common_name_norm"] == key]


def _fuzzy_match_gsk_name(solvent_name: str, threshold: int = 80) -> Optional[Dict[str, Any]]:
    try:
        from thefuzz import fuzz, process
    except Exception:
        return None

    df = _load_gsk_dataframe()
    names = df["solvent_common_name"].dropna().astype(str).tolist()
    if not names:
        return None
    names_lower = [name.lower() for name in names]
    query = solvent_name.strip().lower()
    match = process.extractOne(query, names_lower, scorer=fuzz.ratio)
    if not match or match[1] < threshold:
        return None
    idx = names_lower.index(match[0])
    return {
        "matched_name": names[idx],
        "score": match[1],
        "dataset": "gsk_dataset",
        "original_query": solvent_name,
    }
def _load_green_solvent_db() -> tuple[dict[str, dict], dict[str, dict]]:
    """Lazy-load GreenSolventDB_10k.csv into name and CAS lookup dicts."""
    global _GREEN_SOLVENT_DB, _GREEN_SOLVENT_DB_CAS
    if _GREEN_SOLVENT_DB is not None:
        return _GREEN_SOLVENT_DB, _GREEN_SOLVENT_DB_CAS
    by_name: dict[str, dict] = {}
    by_cas: dict[str, dict] = {}
    csv_path = get_data_path("GreenSolventDB_10k.csv")
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
                    "name": row.get("Solvents", ""),
                    "cas": row.get("CAS", ""),
                    "smiles": row.get("smiles1", ""),
                    "g_score": float(row.get("G-score prediction", 0)),
                    "uncertainty": float(row.get("G-score uncertainty", 0)),
                    "source": row.get("Source", ""),
                }
                name_key = entry["name"].strip().lower()
                if name_key:
                    by_name[name_key] = entry
                cas_key = entry["cas"].strip()
                if cas_key:
                    by_cas[cas_key] = entry
        logger.info(f"GreenSolventDB loaded: {len(by_name)} names, {len(by_cas)} CAS entries")
    else:
        logger.warning(f"GreenSolventDB not found at {csv_path}")
    _GREEN_SOLVENT_DB = by_name
    _GREEN_SOLVENT_DB_CAS = by_cas
    return by_name, by_cas
def _lookup_green_solvent_db(solvent_name: str) -> dict | None:
    """Look up a solvent in GreenSolventDB by name or CAS."""
    by_name, by_cas = _load_green_solvent_db()
    key = solvent_name.strip().lower()
    # Direct name match
    if key in by_name:
        return by_name[key]
    # CAS match
    if key in by_cas:
        return by_cas[key]
    # Try solvent registry for CAS → GreenSolventDB lookup
    try:
        from strap.solvent_registry import SOLVENT_REGISTRY
        for entry in SOLVENT_REGISTRY.values():
            aliases = [entry["interp_key"]] + entry.get("aliases", [])
            if key in [a.lower() for a in aliases]:
                cas = entry.get("cas")
                if cas and cas in by_cas:
                    return by_cas[cas]
    except Exception:
        pass
    # Substring match (for longer names)
    if len(key) > 4:
        for name, entry in by_name.items():
            if key in name or name in key:
                return entry
    return None
def _interpret_logp(logp: float) -> str:
    """Return a short interpretation of LogP value."""
    if logp < 0:
        return "Hydrophilic (low bioaccumulation risk)"
    elif logp < 2:
        return "Moderate lipophilicity"
    elif logp < 3:
        return "Lipophilic (higher bioaccumulation risk)"
    else:
        return "Highly lipophilic (significant bioaccumulation concern)"
# ---------------------------------------------------------------------------
# Lazy async DB wrapper
# ---------------------------------------------------------------------------
_async_db = None
def _get_async_db():
    """Return (or create) an AsyncDuckDBWrapper around the shared connection."""
    global _async_db
    if _async_db is None:
        from strap.vendor.async_db import AsyncDuckDBWrapper
        _async_db = AsyncDuckDBWrapper(get_connection())
    return _async_db
# ---------------------------------------------------------------------------
# Local fuzzy-matching helpers (use get_connection() instead of global sql_db)
# ---------------------------------------------------------------------------
def _search_fuzzy_match_in_dataset(
    conn,
    query: str,
    column_name: str,
    dataset_name: str,
    solvent_name_clean: str,
    current_best_score: int,
) -> Tuple[Optional[str], int, Optional[str]]:
    """Search a single dataset for the best fuzzy match."""
    try:
        from thefuzz import fuzz, process
        df = conn.execute(query).fetchdf()
        if len(df) > 0:
            names = df[column_name].tolist()
            names_lower = [n.lower() for n in names]
            match = process.extractOne(solvent_name_clean, names_lower, scorer=fuzz.ratio)
            if match and match[1] > current_best_score:
                idx = names_lower.index(match[0])
                return names[idx], match[1], dataset_name
    except Exception as e:
        logger.debug(f"{dataset_name} search failed: {e}")
    return None, current_best_score, None
def _fuzzy_match_solvent_name(
    solvent_name: str,
    dataset: str = "all",
    threshold: int = 80,
) -> Optional[Dict[str, Any]]:
    """Find the best matching solvent name across datasets using fuzzy matching."""
    try:
        best_match = None
        best_score = 0
        best_dataset = None
        solvent_name_clean = solvent_name.strip().lower()
        if dataset in ["gsk", "all"]:
            gsk_match = _fuzzy_match_gsk_name(solvent_name, threshold=threshold)
            if gsk_match:
                best_match = gsk_match["matched_name"]
                best_score = gsk_match["score"]
                best_dataset = gsk_match["dataset"]

        conn = get_connection()
        dataset_configs = [
            ("solvent_data", "SELECT DISTINCT cosmobase_name FROM solvent_data", "cosmobase_name", "solvent_data"),
            ("common_solvents", "SELECT DISTINCT solvent FROM common_solvents_database", "solvent", "common_solvents_database"),
        ]
        for ds_key, query, column, ds_name in dataset_configs:
            if dataset in [ds_key, "all"]:
                match, score, matched_ds = _search_fuzzy_match_in_dataset(
                    conn, query, column, ds_name, solvent_name_clean, best_score
                )
                if match:
                    best_match, best_score, best_dataset = match, score, matched_ds
        if best_score >= threshold:
            return {
                "matched_name": best_match,
                "score": best_score,
                "dataset": best_dataset,
                "original_query": solvent_name,
            }
        return None
    except Exception as e:
        logger.error(f"Fuzzy matching error: {e}")
        return None
# ============================================================
# GSK Safety (G-Score) Analysis Tools
# ============================================================
def _format_green_solvent_result(solvent_name: str, entry: dict) -> str:
    """Format a GreenSolventDB result as JSON response."""
    score = entry["g_score"]
    if score >= 8.0:
        rating = "Excellent (Preferred)"
    elif score >= 6.0:
        rating = "Good (Usable)"
    elif score >= 4.0:
        rating = "Problematic (Use with caution)"
    else:
        rating = "Hazardous (Avoid if possible)"
    output = ["**GSK G-Score Analysis** _(from GreenSolventDB — ML-predicted)_\n"]
    output.append(f"**Solvent:** {entry['name']}")
    output.append(f"**CAS:** {entry['cas']}")
    output.append(f"**G-Score:** {score:.2f} / 10.00 (±{entry['uncertainty']:.2f})")
    output.append(f"**Safety Rating:** {rating}")
    output.append(f"**Data Source:** {entry['source']}")
    output.append("")
    output.append("**Note:** This G-score is an ML prediction from the GreenSolventDB "
                   "(10k solvents), not from the curated GSK guide (272 solvents). "
                   "Uncertainty reflects model confidence.")
    # LogP from local Solvent_Data
    logp_val = get_logp(entry['name'])
    if logp_val is not None:
        output.append(f"\n**LogP:** {logp_val:.2f} — {_interpret_logp(logp_val)}")
    display_str = "\n".join(output)
    data_dict = {
        "found": True,
        "solvent_name": entry["name"],
        "g_score": score,
        "g_score_uncertainty": entry["uncertainty"],
        "safety_rating": rating,
        "cas_number": entry["cas"],
        "source": "GreenSolventDB_10k",
        "data_quality": entry["source"],
        "ml_predicted": True,
    }
    return _gsk_response("get_solvent_gscore", display_str, **data_dict)


def lookup_local_gscore_data(solvent_name: str, use_fuzzy_matching: bool = True) -> dict[str, Any] | None:
    """Return local G-score metadata from curated GSK data or GreenSolventDB fallback."""
    result = _lookup_gsk_exact(solvent_name)
    if len(result) == 0 and use_fuzzy_matching:
        match_result = _fuzzy_match_solvent_name(solvent_name, dataset="gsk", threshold=80)
        if match_result:
            result = _lookup_gsk_exact(match_result["matched_name"])
            if len(result) > 0:
                row = result.iloc[0]
                return {
                    "solvent_name": str(row["solvent_common_name"]),
                    "classification": str(row["classification"]),
                    "g_score": float(row["g_score"]),
                    "source": "gsk_dataset",
                    "ml_predicted": False,
                    "fuzzy_matched": True,
                    "match_confidence": int(match_result["score"]),
                }

    if len(result) > 0:
        row = result.iloc[0]
        return {
            "solvent_name": str(row["solvent_common_name"]),
            "classification": str(row["classification"]),
            "g_score": float(row["g_score"]),
            "source": "gsk_dataset",
            "ml_predicted": False,
            "fuzzy_matched": False,
            "match_confidence": None,
        }

    green_entry = _lookup_green_solvent_db(solvent_name)
    if green_entry:
        return {
            "solvent_name": str(green_entry["name"]),
            "classification": None,
            "g_score": float(green_entry["g_score"]),
            "g_score_uncertainty": float(green_entry["uncertainty"]),
            "source": "GreenSolventDB_10k",
            "data_quality": str(green_entry["source"]),
            "ml_predicted": True,
            "fuzzy_matched": False,
            "match_confidence": None,
        }

    return None


@safe_tool_wrapper(structured_output=True)
async def get_solvent_gscore(solvent_name: str, use_fuzzy_matching: bool = True) -> str:
    """Look up the GSK G-score composite safety rating (0-10) for a solvent.
    Args:
        solvent_name: Name of the solvent to look up
        use_fuzzy_matching: If True, attempt fuzzy name matching if exact match fails
    WHEN TO USE:
    - "What is the G-score for toluene?"
    - "How safe is dichloromethane according to GSK?"
    - "Get the GSK safety rating for ethanol"
    """
    try:
        result = _lookup_gsk_exact(solvent_name)
        # If no exact match and fuzzy matching enabled, try fuzzy match
        _fuzzy_matched = False
        if len(result) == 0 and use_fuzzy_matching:
            match_result = _fuzzy_match_solvent_name(solvent_name, dataset="gsk", threshold=80)
            if match_result:
                matched_name = match_result["matched_name"]
                result = _lookup_gsk_exact(matched_name)
                if len(result) > 0:
                    _fuzzy_matched = True
                    output = [f"**GSK G-Score Analysis**\n"]
                    output.append(f"Fuzzy matched '{solvent_name}' -> '{matched_name}' (confidence: {match_result['score']}%)\n")
            else:
                # Tier 2: Try GreenSolventDB 10k (ML-predicted G-scores)
                green_entry = _lookup_green_solvent_db(solvent_name)
                if green_entry:
                    return _format_green_solvent_result(solvent_name, green_entry)
                not_found_msg = (
                    f"**NOT FOUND**: '{solvent_name}' is not in the GSK dataset "
                    f"(272 solvents) or GreenSolventDB (10k solvents). Do NOT estimate "
                    f"or fabricate a G-score. Instead, call "
                    f"get_pubchem_safety_info('{solvent_name}') to retrieve GHS hazard "
                    f"classification from PubChem as a fallback. "
                    f"Report this solvent as 'Not in GSK database' in your assessment."
                )
                return _gsk_error(
                    "get_solvent_gscore",
                    not_found_msg,
                    error_code="solvent_not_found",
                    found=False,
                    solvent_name=solvent_name,
                )
        if len(result) == 0:
            # Tier 2: Try GreenSolventDB 10k (ML-predicted G-scores)
            green_entry = _lookup_green_solvent_db(solvent_name)
            if green_entry:
                return _format_green_solvent_result(solvent_name, green_entry)
            not_found_msg = (
                f"**NOT FOUND**: '{solvent_name}' is not in the GSK dataset "
                f"(272 solvents) or GreenSolventDB (10k solvents). Do NOT estimate "
                f"or fabricate a G-score. Instead, call "
                f"get_pubchem_safety_info('{solvent_name}') to retrieve GHS hazard "
                f"classification from PubChem as a fallback. "
                f"Report this solvent as 'Not in GSK database' in your assessment."
            )
            return _gsk_error(
                "get_solvent_gscore",
                not_found_msg,
                error_code="solvent_not_found",
                found=False,
                solvent_name=solvent_name,
            )
        # Format output
        if 'output' not in locals():
            output = [f"**GSK G-Score Analysis**\n"]
        row = result.iloc[0]
        output.append(f"**Solvent:** {row['solvent_common_name']}")
        output.append(f"**Family:** {row['classification']}")
        output.append(f"**G-Score:** {row['g_score']:.2f} / 10.00")
        # Interpret G-score
        score = row['g_score']
        if score >= 8.0:
            rating = "Excellent (Preferred)"
            color = "green"
        elif score >= 6.0:
            rating = "Good (Usable)"
            color = "light green"
        elif score >= 4.0:
            rating = "Problematic (Use with caution)"
            color = "yellow"
        else:
            rating = "Hazardous (Avoid if possible)"
            color = "red"
        output.append(f"**Safety Rating:** {rating}")
        output.append(f"**CAS Number:** {row['cas_number']}")
        output.append("")
        # LogP from local Solvent_Data
        logp_val = get_logp(row['solvent_common_name'])
        if logp_val is not None:
            output.append(f"**LogP:** {logp_val:.2f} — {_interpret_logp(logp_val)}")
        else:
            output.append("**LogP:** Not available in local database")
        output.append("")
        output.append("**Note:** G-score is the geometric mean of Environment, Health, Safety, and Waste (EHSW) scores.")
        display_str = "\n".join(output)
        data_dict = {
            "found": True,
            "solvent_name": row['solvent_common_name'],
            "classification": row['classification'],
            "g_score": float(row['g_score']),
            "safety_rating": rating,
            "cas_number": str(row.get('cas_number', '')),
            "fuzzy_matched": _fuzzy_matched,
        }
        return _gsk_response("get_solvent_gscore", display_str, **data_dict)
    except Exception as e:
        logger.error(f"Error in get_solvent_gscore: {e}")
        err_msg = f"Error retrieving G-score: {str(e)}"
        return _gsk_error(
            "get_solvent_gscore",
            err_msg,
            error_code="gscore_lookup_failed",
            found=False,
        )
@safe_tool_wrapper(structured_output=True)
async def get_family_alternatives(
    solvent_name: str,
    min_gscore: Optional[float] = None,
    limit: int = 10,
    use_fuzzy_matching: bool = True,
    family_override: Optional[str] = None,
) -> str:
    """Find safer alternative solvents from the same chemical family, ranked by G-score.
    Args:
        solvent_name: Name of the reference solvent
        min_gscore: Minimum G-score threshold (0-10), or None for all
        limit: Maximum number of alternatives to return
        use_fuzzy_matching: If True, attempt fuzzy name matching
        family_override: If provided, query this chemical family directly instead
            of looking up the solvent's family. Use when the solvent is not in the
            GSK dataset. Valid families: Alcohols, Aromatics, Carbonates,
            Dipolar Aprotics, Esters, Ethers, Halogenated, Hydrocarbons,
            Ketones, Other, water and acids.
    WHEN TO USE:
    - "What are safer alternatives to toluene in the same family?"
    - "Find greener substitutes for DCM"
    - "List alcohols with G-score above 7"
    - "Nitrobenzene is not in GSK -- show me safer Aromatics"
    """
    try:
        gsk_df = _load_gsk_dataframe()
        if family_override is not None:
            # Use the provided family directly (skip solvent lookup)
            family = family_override
        else:
            # First, find the family of the input solvent
            family_result = _lookup_gsk_exact(solvent_name)
            # Try fuzzy matching if no exact match
            if len(family_result) == 0 and use_fuzzy_matching:
                match_result = _fuzzy_match_solvent_name(solvent_name, dataset="gsk", threshold=80)
                if match_result:
                    family_result = _lookup_gsk_exact(match_result["matched_name"])
            if len(family_result) == 0:
                msg = (
                    f"**NOT FOUND**: Could not find solvent '{solvent_name}' in GSK dataset. "
                    f"To browse a family directly, call get_family_alternatives("
                    f"solvent_name='{solvent_name}', family_override='<family>') where "
                    f"<family> is one of: Alcohols, Aromatics, Carbonates, "
                    f"Dipolar Aprotics, Esters, Ethers, Halogenated, Hydrocarbons, "
                    f"Ketones, Other, water and acids."
                )
                return _gsk_error(
                    "get_family_alternatives",
                    msg,
                    error_code="solvent_not_found",
                    found=False,
                    input_solvent=solvent_name,
                )
            family = family_result.iloc[0]['classification']
        # Get all solvents from the same family
        alternatives = gsk_df[gsk_df["classification"] == family].copy()
        if min_gscore is not None:
            alternatives = alternatives[alternatives["g_score"] >= min_gscore]
        alternatives = alternatives.sort_values("g_score", ascending=False).head(limit + 1)
        # Format output
        output = [f"**Family Alternatives for '{solvent_name}'**\n"]
        output.append(f"**Family:** {family}")
        output.append(f"**Alternatives found:** {len(alternatives)}")
        if min_gscore is not None:
            output.append(f"**Min G-score filter:** {min_gscore:.1f}")
        output.append("\n**Ranked by G-Score (Best to Worst):**\n")
        for i, row in alternatives.iterrows():
            is_original = row['solvent_common_name'].lower() == solvent_name.lower()
            marker = ">> " if is_original else f"{i+1}. "
            score = row['g_score']
            if score >= 8.0:
                label = "[Excellent]"
            elif score >= 6.0:
                label = "[Good]"
            elif score >= 4.0:
                label = "[Problematic]"
            else:
                label = "[Hazardous]"
            line = f"{marker}{label} **{row['solvent_common_name']}** - G-score: {score:.2f}"
            if is_original:
                line += " (Your selection)"
            output.append(line)
        # Add recommendation
        if len(alternatives) > 0:
            best = alternatives.iloc[0]
            output.append(f"\n**Recommendation:** For best safety, consider **{best['solvent_common_name']}** (G-score: {best['g_score']:.2f})")
        display_str = "\n".join(output)
        alternatives_list = []
        for _, row in alternatives.iterrows():
            alternatives_list.append({
                "solvent_name": row['solvent_common_name'],
                "g_score": float(row['g_score']),
                "cas_number": str(row.get('cas_number', '')),
            })
        data_dict = {
            "found": True,
            "input_solvent": solvent_name,
            "family": family,
            "alternatives": alternatives_list,
            "count": len(alternatives_list),
        }
        return _gsk_response("get_family_alternatives", display_str, **data_dict)
    except Exception as e:
        logger.error(f"Error in get_family_alternatives: {e}")
        err_msg = f"Error retrieving family alternatives: {str(e)}"
        return _gsk_error(
            "get_family_alternatives",
            err_msg,
            error_code="alternatives_lookup_failed",
            found=False,
        )
@safe_tool_wrapper(structured_output=True)
async def visualize_gscores(
    filter_by: Optional[str] = None,
    family: Optional[str] = None,
    solvent_list: Optional[str] = None,
    min_score: Optional[float] = None,
    plot_type: str = "bar",
    top_k: int = 10
) -> str:
    """Generate bar, scatter, or box plots of GSK G-scores for filtered solvents.
    Args:
        filter_by: Filter mode: "all", "family", "list", or None
        family: Family name when filter_by="family" (e.g., "Alcohols")
        solvent_list: Comma-separated names when filter_by="list"
        min_score: Minimum G-score to include (0-10)
        plot_type: "bar", "scatter", or "box"
        top_k: Maximum number of solvents to show (default: 10)
    WHEN TO USE:
    - "Plot G-scores for the top 10 safest solvents"
    - "Show a box plot of G-scores by solvent family"
    """
    try:
        df = _load_gsk_dataframe().copy()
        # Build query based on filters
        if filter_by == "family" and family:
            df = df[df["classification"] == family]
        elif filter_by == "list" and solvent_list:
            solvents = {s.strip().lower() for s in solvent_list.split(',') if s.strip()}
            df = df[df["solvent_common_name_norm"].isin(solvents)]
        if min_score is not None:
            df = df[df["g_score"] >= min_score]
        df = df.sort_values("g_score", ascending=False).head(top_k)
        if len(df) == 0:
            msg = "No solvents match the specified criteria."
            return _gsk_error(
                "visualize_gscores",
                msg,
                error_code="no_matching_solvents",
            )
        # Create plot
        plots_dir = get_plots_dir()
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if plot_type == "bar":
            fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3)))
            # Color bars by score
            colors = []
            for score in df['g_score']:
                if score >= 8.0:
                    colors.append('#10b981')  # green
                elif score >= 6.0:
                    colors.append('#84cc16')  # light green
                elif score >= 4.0:
                    colors.append('#f59e0b')  # yellow
                else:
                    colors.append('#ef4444')  # red
            ax.barh(df['solvent_common_name'], df['g_score'], color=colors)
            ax.set_xlabel('G-Score (Safety Rating)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Solvent', fontsize=12, fontweight='bold')
            ax.set_title('GSK G-Score Comparison\n(Higher = Safer)', fontsize=14, fontweight='bold')
            ax.axvline(x=6.0, color='gray', linestyle='--', alpha=0.5, label='Good threshold (6.0)')
            ax.axvline(x=8.0, color='green', linestyle='--', alpha=0.5, label='Excellent threshold (8.0)')
            ax.legend()
            ax.set_xlim(0, 10)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            filename = f"gscore_bar_{timestamp}.png"
        elif plot_type == "scatter":
            fig, ax = plt.subplots(figsize=(12, 8))
            # Group by family for color coding
            families = df['classification'].unique()
            colors_map = plt.cm.tab10(np.linspace(0, 1, len(families)))
            for i, fam in enumerate(families):
                family_df = df[df['classification'] == fam]
                ax.scatter(range(len(family_df)), family_df['g_score'],
                          label=fam, alpha=0.7, s=100, color=colors_map[i])
            ax.set_xlabel('Solvent Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('G-Score (Safety Rating)', fontsize=12, fontweight='bold')
            ax.set_title('GSK G-Score Distribution by Family', fontsize=14, fontweight='bold')
            ax.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
            ax.axhline(y=8.0, color='green', linestyle='--', alpha=0.5, label='Excellent threshold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylim(0, 10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            filename = f"gscore_scatter_{timestamp}.png"
        elif plot_type == "box":
            fig, ax = plt.subplots(figsize=(12, 8))
            # Group by family
            families = df['classification'].unique()
            family_data = [df[df['classification'] == fam]['g_score'].values for fam in families]
            bp = ax.boxplot(family_data, labels=families, patch_artist=True)
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('#c77b4a')
                patch.set_alpha(0.6)
            ax.set_xlabel('Solvent Family', fontsize=12, fontweight='bold')
            ax.set_ylabel('G-Score (Safety Rating)', fontsize=12, fontweight='bold')
            ax.set_title('GSK G-Score Distribution by Family', fontsize=14, fontweight='bold')
            ax.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
            ax.axhline(y=8.0, color='green', linestyle='--', alpha=0.5, label='Excellent threshold')
            plt.xticks(rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            filename = f"gscore_box_{timestamp}.png"
        else:
            msg = f"Invalid plot_type '{plot_type}'. Use 'bar', 'scatter', or 'box'."
            return _gsk_error(
                "visualize_gscores",
                msg,
                error_code="invalid_plot_type",
                plot_type=plot_type,
            )
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        output = [f"**G-Score Visualization Created**\n"]
        output.append(f"**Plot type:** {plot_type}")
        output.append(f"**Solvents shown:** {len(df)}")
        output.append(f"**Saved as:** {filename}\n")
        # Statistics
        output.append(f"**Statistics:**")
        output.append(f"- Mean G-score: {df['g_score'].mean():.2f}")
        output.append(f"- Median G-score: {df['g_score'].median():.2f}")
        output.append(f"- Range: {df['g_score'].min():.2f} - {df['g_score'].max():.2f}")
        output.append(f"- Excellent solvents (>=8.0): {len(df[df['g_score'] >= 8.0])}")
        output.append(f"- Good solvents (>=6.0): {len(df[df['g_score'] >= 6.0])}")
        display_str = "\n".join(output)
        data_dict = {
            "success": True,
            "plot_type": plot_type,
            "filepath": filepath,
            "n_solvents": len(df),
            "statistics": {
                "mean": float(df['g_score'].mean()),
                "median": float(df['g_score'].median()),
                "min": float(df['g_score'].min()),
                "max": float(df['g_score'].max()),
            },
        }
        return _gsk_response("visualize_gscores", display_str, **data_dict)
    except Exception as e:
        logger.error(f"Error in visualize_gscores: {e}")
        err_msg = f"Error creating visualization: {str(e)}"
        return _gsk_error(
            "visualize_gscores",
            err_msg,
            error_code="visualization_failed",
        )
