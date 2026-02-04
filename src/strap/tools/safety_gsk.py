"""GSK safety (G-Score) analysis tools.

Provides tools for querying and visualizing GSK solvent safety scores,
finding family alternatives, and plotting solvent properties against
polymer solubility data.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from strap.database import get_connection
from strap.tools._helpers import (
    get_plots_dir,
    safe_tool_wrapper,
)

logger = logging.getLogger(__name__)

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
        conn = get_connection()
        best_match = None
        best_score = 0
        best_dataset = None
        solvent_name_clean = solvent_name.strip().lower()

        dataset_configs = [
            ("gsk", "SELECT DISTINCT solvent_common_name FROM gsk_dataset", "solvent_common_name", "gsk_dataset"),
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


@safe_tool_wrapper
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
        async_db = _get_async_db()

        # Try exact match first
        query = f"""
        SELECT solvent_common_name, classification, g_score, cas_number
        FROM gsk_dataset
        WHERE LOWER(solvent_common_name) = LOWER('{solvent_name}')
        """

        result = await async_db.execute_async(query)

        # If no exact match and fuzzy matching enabled, try fuzzy match
        if len(result) == 0 and use_fuzzy_matching:
            match_result = _fuzzy_match_solvent_name(solvent_name, dataset="gsk", threshold=80)

            if match_result:
                matched_name = match_result["matched_name"]
                query = f"""
                SELECT solvent_common_name, classification, g_score, cas_number
                FROM gsk_dataset
                WHERE LOWER(solvent_common_name) = LOWER('{matched_name}')
                """
                result = await async_db.execute_async(query)

                if len(result) > 0:
                    output = [f"**GSK G-Score Analysis**\n"]
                    output.append(f"Fuzzy matched '{solvent_name}' -> '{matched_name}' (confidence: {match_result['score']}%)\n")
            else:
                return f"No G-score data found for '{solvent_name}'. The GSK dataset contains 153 solvents. Try `list_tables()` to see available solvents."

        if len(result) == 0:
            return f"No G-score data found for '{solvent_name}'. The GSK dataset contains 153 solvents."

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
        output.append(f"**CAS Number:** {row['cas_number']}\n")

        output.append("**Note:** G-score is the geometric mean of Environment, Health, Safety, and Waste (EHSW) scores.")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in get_solvent_gscore: {e}")
        return f"Error retrieving G-score: {str(e)}"


@safe_tool_wrapper
async def get_family_alternatives(
    solvent_name: str,
    min_gscore: Optional[float] = None,
    limit: int = 10,
    use_fuzzy_matching: bool = True
) -> str:
    """Find safer alternative solvents from the same chemical family, ranked by G-score.

    Args:
        solvent_name: Name of the reference solvent
        min_gscore: Minimum G-score threshold (0-10), or None for all
        limit: Maximum number of alternatives to return
        use_fuzzy_matching: If True, attempt fuzzy name matching

    WHEN TO USE:
    - "What are safer alternatives to toluene in the same family?"
    - "Find greener substitutes for DCM"
    - "List alcohols with G-score above 7"
    """
    try:
        async_db = _get_async_db()

        # First, find the family of the input solvent
        query = f"""
        SELECT classification
        FROM gsk_dataset
        WHERE LOWER(solvent_common_name) = LOWER('{solvent_name}')
        """

        family_result = await async_db.execute_async(query)

        # Try fuzzy matching if no exact match
        if len(family_result) == 0 and use_fuzzy_matching:
            match_result = _fuzzy_match_solvent_name(solvent_name, dataset="gsk", threshold=80)
            if match_result:
                query = f"""
                SELECT classification
                FROM gsk_dataset
                WHERE LOWER(solvent_common_name) = LOWER('{match_result["matched_name"]}')
                """
                family_result = await async_db.execute_async(query)

        if len(family_result) == 0:
            return f"Could not find solvent '{solvent_name}' in GSK dataset."

        family = family_result.iloc[0]['classification']

        # Get all solvents from the same family
        min_score_clause = f"AND g_score >= {min_gscore}" if min_gscore is not None else ""

        query = f"""
        SELECT solvent_common_name, g_score, cas_number
        FROM gsk_dataset
        WHERE classification = '{family}'
        {min_score_clause}
        ORDER BY g_score DESC
        LIMIT {limit + 1}
        """

        alternatives = await async_db.execute_async(query)

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

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in get_family_alternatives: {e}")
        return f"Error retrieving family alternatives: {str(e)}"


@safe_tool_wrapper
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
        async_db = _get_async_db()

        # Build query based on filters
        where_clauses = []

        if filter_by == "family" and family:
            where_clauses.append(f"classification = '{family}'")
        elif filter_by == "list" and solvent_list:
            solvents = [s.strip() for s in solvent_list.split(',')]
            solvent_filter = "', '".join(solvents)
            where_clauses.append(f"solvent_common_name IN ('{solvent_filter}')")

        if min_score is not None:
            where_clauses.append(f"g_score >= {min_score}")

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
        SELECT solvent_common_name, g_score, classification
        FROM gsk_dataset
        WHERE {where_clause}
        ORDER BY g_score DESC
        LIMIT {top_k}
        """

        df = await async_db.execute_async(query)

        if len(df) == 0:
            return "No solvents match the specified criteria."

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
            return f"Invalid plot_type '{plot_type}'. Use 'bar', 'scatter', or 'box'."

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

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Error in visualize_gscores: {e}")
        return f"Error creating visualization: {str(e)}"
