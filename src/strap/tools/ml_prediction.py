"""ML-based solubility prediction using Hansen Solubility Parameters."""
from __future__ import annotations
import asyncio
import json
import logging
import math
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from strap.hsp_registry import (
    HspPolymerEntry,
    HspResolverResult,
    HspSolventEntry,
    get_hsp_polymer_asset,
    get_hsp_solvent_asset,
    list_hsp_polymer_entries,
    list_hsp_solvent_entries,
    resolve_hsp_polymer,
    resolve_hsp_polymer_category,
    resolve_hsp_solvent,
    resolve_hsp_solvent_category,
    resolve_hsp_solvent_polarity,
)
from strap.database import get_connection
from strap.services.tool_response_service import json_tool_error, json_tool_success
from strap.tools._helpers import (
    descriptive_plot_name,
    get_plots_dir,
    safe_tool_wrapper,
    save_plot,
    truncate_output,
)
# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
try:
    from strap.vendor.solubility_predictor import get_predictor
except ImportError:
    get_predictor = None
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Local fuzzy-match helper (adapted from monolith to use get_connection())
# ---------------------------------------------------------------------------
def _fuzzy_match_solvent_name(
    solvent_name: str,
    dataset: str = "all",
    threshold: int = 80,
) -> Optional[Dict[str, Any]]:
    """Find the best matching solvent name across datasets using fuzzy matching."""
    try:
        from thefuzz import fuzz, process
    except ImportError:
        return None
    try:
        conn = get_connection()
        best_match: Optional[str] = None
        best_score: int = 0
        best_dataset: Optional[str] = None
        solvent_name_clean = solvent_name.strip().lower()
        dataset_configs: list[tuple[str, str, str, str]] = [
            ("gsk", "SELECT DISTINCT solvent_common_name FROM gsk_dataset", "solvent_common_name", "gsk_dataset"),
            ("solvent_data", "SELECT DISTINCT cosmobase_name FROM solvent_data", "cosmobase_name", "solvent_data"),
            ("common_solvents", "SELECT DISTINCT solvent FROM common_solvents_database", "solvent", "common_solvents_database"),
        ]
        for ds_key, query, column, ds_name in dataset_configs:
            if dataset not in (ds_key, "all"):
                continue
            try:
                df = conn.execute(query).fetchdf()
                if len(df) == 0:
                    continue
                names = df[column].tolist()
                names_lower = [n.lower() for n in names]
                match = process.extractOne(solvent_name_clean, names_lower, scorer=fuzz.ratio)
                if match and match[1] > best_score:
                    idx = names_lower.index(match[0])
                    best_match = names[idx]
                    best_score = match[1]
                    best_dataset = ds_name
            except Exception as exc:
                logger.debug(f"{ds_name} search failed: {exc}")
        if best_score >= threshold and best_match is not None:
            return {
                "matched_name": best_match,
                "score": best_score,
                "dataset": best_dataset,
                "original_query": solvent_name,
            }
        return None
    except Exception as exc:
        logger.error(f"Fuzzy matching error: {exc}")
        return None


def _coerce_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _resolution_error(tool_name: str, result: HspResolverResult, *, category_tool: str) -> str:
    if result.status == "category":
        return json_tool_error(
            (
                f"`{result.query}` resolved to the HSP category "
                f"`{result.category_label or result.category_id}`. Use "
                f"`{category_tool}` for category-level HSP screening instead "
                "of a single-pair prediction."
            ),
            tool_name=tool_name,
            error_code="hsp_category_requires_matrix_screen",
            resolution=result.to_dict(),
        )
    if result.status == "ambiguous":
        names = ", ".join(entry.display_name for entry in result.matches) or "none"
        return json_tool_error(
            f"Ambiguous HSP {result.kind} query `{result.query}`. Choose one of: {names}.",
            tool_name=tool_name,
            error_code=f"ambiguous_hsp_{result.kind}",
            resolution=result.to_dict(),
        )
    return json_tool_error(
        result.unsupported_reason or f"Unsupported HSP {result.kind}: {result.query}",
        tool_name=tool_name,
        error_code=f"unsupported_hsp_{result.kind}",
        resolution=result.to_dict(),
    )


def _polymer_hsp(entry: HspPolymerEntry) -> tuple[dict[str, float], float, dict[str, Any]]:
    asset = get_hsp_polymer_asset(entry)
    return (
        {
            "Dispersion": float(asset["dispersion"]),
            "Polar": float(asset["polar"]),
            "Hydrogen": float(asset["hydrogen_bonding"]),
        },
        float(asset["interaction_radius"]),
        asset,
    )


def _solvent_hsp(entry: HspSolventEntry) -> tuple[dict[str, float], float, dict[str, Any]]:
    asset = get_hsp_solvent_asset(entry)
    return (
        {
            "Dispersion": float(asset["dispersion"]),
            "Polar": float(asset["polar"]),
            "Hydrogen": float(asset["hydrogen_bonding"]),
        },
        float(asset.get("molar_volume", 100.0)),
        asset,
    )


def _prediction_fields(prediction: dict[str, Any]) -> dict[str, Any]:
    probability_soluble = float(prediction["probability"])
    soluble = bool(prediction["soluble"])
    threshold = float(prediction.get("threshold", 0.85))
    probability_predicted_class = probability_soluble if soluble else 1.0 - probability_soluble
    decision_margin = abs(probability_soluble - threshold)
    return {
        "soluble": soluble,
        "probability_soluble": probability_soluble,
        "probability_predicted_class": probability_predicted_class,
        "classification_threshold": threshold,
        "decision_margin": decision_margin,
        "confidence": float(prediction.get("confidence", decision_margin)),
        "red": float(prediction["red"]),
        "ra": float(prediction["ra"]),
        "r0": float(prediction["r0"]),
    }


def _temperature_warnings(temperature_c: float) -> list[str]:
    if abs(float(temperature_c) - 25.0) < 1e-9:
        return []
    return [
        "HSP/RED screening is temperature-independent in this implementation; "
        f"the requested {temperature_c:g} C is reported but not used by the model. "
        "Use fitted Apelblat/interpolation tools for quantitative solubility-vs-temperature queries."
    ]


def _entity_warnings(*entries: HspPolymerEntry | HspSolventEntry) -> list[str]:
    warnings: list[str] = []
    for entry in entries:
        warnings.extend(getattr(entry, "warnings", ()))
        qualifier = getattr(entry, "qualifier", None)
        quality = getattr(entry, "quality", "canonical")
        if qualifier:
            warnings.append(str(qualifier))
        if quality in {"swelling_or_permeation", "conditioned", "ambiguous"}:
            warnings.append(f"{entry.display_name} has HSP quality `{quality}` and should not be treated as a clean dissolution record.")
    return list(dict.fromkeys(warnings))


def _hsp_result_row(
    polymer: HspPolymerEntry,
    solvent: HspSolventEntry,
    prediction: dict[str, Any],
    *,
    temperature_c: float,
) -> dict[str, Any]:
    fields = _prediction_fields(prediction)
    return {
        "polymer": polymer.display_name,
        "polymer_raw_hsp_name": polymer.raw_hsp_name,
        "polymer_family": polymer.polymer_family,
        "polymer_quality": polymer.quality,
        "polymer_qualifier": polymer.qualifier,
        "polymer_tags": list(polymer.polymer_tags),
        "solvent": solvent.display_name,
        "solvent_raw_hsp_name": solvent.raw_hsp_name,
        "chemical_family": solvent.chemical_family,
        "polarity_class": solvent.polarity_class,
        "solvent_tags": list(solvent.solvent_tags),
        "temperature_c_requested": float(temperature_c),
        "temperature_used_by_model": False,
        **fields,
        "warnings": _entity_warnings(polymer, solvent) + _temperature_warnings(float(temperature_c)),
    }


def _format_resolution_table(entries: list[dict[str, Any]], *, kind: str) -> str:
    if kind == "polymer":
        header = "| Polymer | Raw HSP entry | Family | Quality | Default | Notes |\n| --- | --- | --- | --- | ---: | --- |"
        rows = [
            f"| {e['display_name']} | `{e['raw_hsp_name']}` | {e['polymer_family']} | {e['quality']} | {e['default_include']} | {e.get('qualifier') or ''} |"
            for e in entries
        ]
    else:
        header = "| Solvent | Raw HSP entry | Family | Polarity | Default | Tags |\n| --- | --- | --- | --- | ---: | --- |"
        rows = [
            f"| {e['display_name']} | `{e['raw_hsp_name']}` | {e['chemical_family']} | {e['polarity_class']} | {e['default_include']} | {', '.join(e.get('solvent_tags', []))} |"
            for e in entries
        ]
    return "\n".join([header, *rows])


def _plot_single_hsp_summary(
    *,
    polymer: HspPolymerEntry,
    solvent: HspSolventEntry,
    polymer_hsp: dict[str, float],
    solvent_hsp: dict[str, float],
    row: dict[str, Any],
) -> str | None:
    """Fallback single-pair HSP visual when the optional viz package is absent."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = ["Dispersion", "Polar", "Hydrogen"]
    polymer_values = [float(polymer_hsp[label]) for label in labels]
    solvent_values = [float(solvent_hsp[label]) for label in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    polymer_values += polymer_values[:1]
    solvent_values += solvent_values[:1]

    fig = plt.figure(figsize=(8.0, 4.2))
    radar = fig.add_subplot(1, 2, 1, polar=True)
    radar.plot(angles, polymer_values, color="#2f5597", linewidth=2, label=polymer.display_name)
    radar.fill(angles, polymer_values, color="#2f5597", alpha=0.12)
    radar.plot(angles, solvent_values, color="#c55a11", linewidth=2, label=solvent.display_name)
    radar.fill(angles, solvent_values, color="#c55a11", alpha=0.12)
    radar.set_xticks(angles[:-1])
    radar.set_xticklabels(labels, fontsize=9)
    radar.set_title("HSP Components", fontsize=11)
    radar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)

    gauge = fig.add_subplot(1, 2, 2)
    gauge.set_xlim(0, 2.5)
    gauge.set_ylim(0, 1)
    gauge.axvspan(0, 1, color="#63be7b", alpha=0.35)
    gauge.axvspan(1, 2.5, color="#f8696b", alpha=0.25)
    gauge.axvline(1, color="black", linewidth=1.2)
    red = float(row["red"])
    gauge.barh([0.5], [min(red, 2.5)], height=0.22, color="#444444")
    gauge.text(
        min(red, 2.45),
        0.68,
        f"RED = {red:.2f}",
        ha="right" if red > 1.6 else "left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    gauge.text(0.5, 0.15, "compatible", ha="center", fontsize=8)
    gauge.text(1.75, 0.15, "outside sphere", ha="center", fontsize=8)
    gauge.set_yticks([])
    gauge.set_xlabel("RED (lower is more compatible)")
    gauge.set_title("RED Gauge", fontsize=11)
    gauge.spines[["left", "right", "top"]].set_visible(False)

    fig.suptitle(f"{polymer.display_name} / {solvent.display_name} HSP Screen", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    plot_name = descriptive_plot_name(
        "hsp_pair_summary",
        polymers=[polymer.display_name],
        solvents=[solvent.display_name],
    )
    return save_plot(fig, plot_name, "matplotlib")
# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
async def predict_solubility_ml(
    polymer_name: str,
    solvent_name: str,
    temperature: float = 25.0,
    generate_visualizations: bool = True,
) -> str:
    """Predict polymer-solvent solubility using an ML model trained on Hansen Solubility Parameters.
    Args:
        polymer_name: Name of polymer (e.g., "HDPE", "PET", "PVDF")
        solvent_name: Name of solvent (e.g., "Toluene", "Water", "Acetone")
        temperature: Temperature in Celsius (default: 25.0)
        generate_visualizations: Whether to create visualization files (default: True)
    WHEN TO USE:
    - "Use HSP to check whether PET is compatible with toluene"
    - "Predict solubility of HDPE in acetone using Hansen parameters"
    - "Is PVDF compatible with DMF by RED/HSP?"

    Do not use this tool for quantitative solubility-vs-temperature queries
    unless the user explicitly asks for HSP/RED screening.
    """
    try:
        if get_predictor is None:
            return json_tool_error(
                "ML predictor unavailable - strap.vendor.solubility_predictor could not be imported.",
                tool_name="predict_solubility_ml",
                error_code="predictor_unavailable",
            )
        PLOTS_DIR = get_plots_dir()
        predictor = get_predictor()

        try:
            polymer_resolution = resolve_hsp_polymer(polymer_name)
            if polymer_resolution.status != "resolved" or polymer_resolution.selected is None:
                return _resolution_error(
                    "predict_solubility_ml",
                    polymer_resolution,
                    category_tool="screen_hsp_solubility_matrix",
                )
            solvent_resolution = resolve_hsp_solvent(solvent_name)
            if solvent_resolution.status != "resolved" or solvent_resolution.selected is None:
                return _resolution_error(
                    "predict_solubility_ml",
                    solvent_resolution,
                    category_tool="screen_hsp_solubility_matrix",
                )

            polymer = polymer_resolution.selected
            solvent = solvent_resolution.selected
            assert isinstance(polymer, HspPolymerEntry)
            assert isinstance(solvent, HspSolventEntry)

            polymer_hsp, r0, _polymer_asset = _polymer_hsp(polymer)
            solvent_hsp, molar_volume, _solvent_asset = _solvent_hsp(solvent)
        except Exception as asset_error:
            logger.error(f"Error loading ML HSP assets: {asset_error}")
            return json_tool_error(
                f"Error loading Hansen parameters: {str(asset_error)}",
                tool_name="predict_solubility_ml",
                error_code="hsp_data_load_failed",
            )

        prediction = predictor.predict(polymer_hsp, solvent_hsp, r0, molar_volume)
        row = _hsp_result_row(polymer, solvent, prediction, temperature_c=float(temperature))

        output = ["**HSP Binary Solubility Screen**\n"]
        output.append(f"**Polymer:** {polymer.display_name} (`{polymer.raw_hsp_name}`)")
        output.append(f"**Solvent:** {solvent.display_name} (`{solvent.raw_hsp_name}`)")
        output.append(f"**Requested temperature:** {float(temperature):g} C")
        output.append("**Temperature used by HSP model:** no - HSP/RED is temperature-independent.\n")
        output.append(f"**Prediction:** {'SOLUBLE' if row['soluble'] else 'NON-SOLUBLE'}")
        output.append(f"**Probability soluble:** {row['probability_soluble'] * 100:.1f}%")
        output.append(f"**Predicted-class probability:** {row['probability_predicted_class'] * 100:.1f}%")
        output.append(f"**Decision margin from threshold:** {row['decision_margin']:.3f}")
        output.append(f"**RED Value:** {row['red']:.3f} (Hansen distance/R0)")
        output.append(f"**Ra (Hansen distance):** {row['ra']:.3f}")
        output.append(f"**R0 (Interaction radius):** {row['r0']:.3f}\n")
        output.append("**Interpretation:**")
        if row["red"] < 1.0:
            output.append("- RED < 1.0: polymer and solvent are HSP-compatible.")
        else:
            output.append("- RED > 1.0: polymer and solvent are outside the HSP sphere.")
        output.append("- This is binary HSP compatibility screening, not quantitative wt% solubility.")
        if row["warnings"]:
            output.append("\n**Warnings:**")
            for warning in row["warnings"]:
                output.append(f"- {warning}")

        generated_artifacts: list[str] = []
        if generate_visualizations:
            try:
                from visualization_library_v2 import generate_all_visualizations
                from datetime import datetime
                # Create subdirectory for full viz set
                safe_dirname = re.sub(r'[^\w\-]', '_', f"{polymer.display_name}_{solvent.display_name}")
                viz_dir = os.path.join(PLOTS_DIR, safe_dirname)
                os.makedirs(viz_dir, exist_ok=True)
                # Generate all visualizations in subdirectory
                viz_paths = generate_all_visualizations(
                    polymer_hsp=polymer_hsp,
                    solvent_hsp=solvent_hsp,
                    r0=r0,
                    polymer_name=polymer.display_name,
                    solvent_name=solvent.display_name,
                    prediction=row["soluble"],
                    probability=row["probability_soluble"],
                    output_dir=viz_dir
                )
                # Copy radar plot and RED gauge to root plots directory (so they auto-display)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = re.sub(r'[^\w\-]', '_', f"{polymer.display_name}_{solvent.display_name}")[:30]
                radar_src = viz_paths.get('Radar Plot')
                gauge_src = viz_paths.get('RED Gauge')
                if radar_src and os.path.exists(radar_src):
                    radar_dest = os.path.join(PLOTS_DIR, f"ml_radar_{safe_name}_{timestamp}.png")
                    shutil.copy(radar_src, radar_dest)
                    generated_artifacts.append(radar_dest)
                if gauge_src and os.path.exists(gauge_src):
                    gauge_dest = os.path.join(PLOTS_DIR, f"ml_gauge_{safe_name}_{timestamp}.png")
                    shutil.copy(gauge_src, gauge_dest)
                    generated_artifacts.append(gauge_dest)
                # Copy 3D sphere HTML to root plots directory for easy access
                sphere_src = viz_paths.get('3D Sphere (Interactive HTML)')
                if sphere_src and os.path.exists(sphere_src):
                    sphere_dest = os.path.join(PLOTS_DIR, f"ml_sphere_{safe_name}_{timestamp}.html")
                    shutil.copy(sphere_src, sphere_dest)
                    generated_artifacts.append(sphere_dest)
                    # Add link to 3D sphere (opens in new tab)
                    import urllib.parse
                    sphere_filename = os.path.basename(sphere_dest)
                    sphere_url = f"/plots/{sphere_filename}"
                    # Use markdown link syntax (not HTML) for proper rendering
                    output.append(f"\n**Interactive 3D Visualization:** [Click to open Hansen Sphere]({sphere_url})")
                    output.append(f"\n**Tip:** The 3D sphere opens in a new tab - you can rotate, zoom, and explore the Hansen space!")
            except Exception as viz_error:
                logger.warning(f"Visualization generation failed: {viz_error}")
                fallback_path = _plot_single_hsp_summary(
                    polymer=polymer,
                    solvent=solvent,
                    polymer_hsp=polymer_hsp,
                    solvent_hsp=solvent_hsp,
                    row=row,
                )
                if fallback_path:
                    generated_artifacts.append(fallback_path)
                    output.append("\n**Fallback HSP Summary Visualization:** generated radar/RED summary PNG.")
                else:
                    output.append(f"\nNote: Visualization generation encountered an issue: {str(viz_error)}")
        return json_tool_success(
            "\n".join(output),
            tool_name="predict_solubility_ml",
            analysis_type="hsp_binary_screen",
            hsp_only=True,
            polymer_name=polymer.display_name,
            solvent_name=solvent.display_name,
            polymer_resolution=polymer_resolution.to_dict(),
            solvent_resolution=solvent_resolution.to_dict(),
            **row,
            probability=float(row["probability_soluble"]),
            generate_visualizations=generate_visualizations,
            artifacts=generated_artifacts,
        )
    except Exception as e:
        logger.error(f"Error in predict_solubility_ml: {e}")
        return json_tool_error(
            f"Error making ML prediction: {str(e)}",
            tool_name="predict_solubility_ml",
            error_code="prediction_failed",
        )


@safe_tool_wrapper(structured_output=True)
def list_hsp_supported_polymers(
    category: str | None = None,
    include_quality_flags: bool = True,
    curated_only: bool = True,
    include_excluded: bool = False,
) -> str:
    """List curated polymers supported by the HSP/RED screening registry."""
    if not curated_only:
        return json_tool_error(
            "Raw full-catalog HSP listing is not exposed yet; use curated_only=True.",
            tool_name="list_hsp_supported_polymers",
            error_code="raw_hsp_catalog_not_exposed",
        )
    if category:
        category_result = resolve_hsp_polymer_category(category, include_excluded=include_excluded)
        if category_result.status != "category":
            return json_tool_error(
                category_result.unsupported_reason or f"Unsupported HSP polymer category: {category}",
                tool_name="list_hsp_supported_polymers",
                error_code="unsupported_hsp_polymer_category",
                resolution=category_result.to_dict(),
            )
        entries = [entry.to_dict() for entry in category_result.category_members]
        excluded = [entry.to_dict() for entry in category_result.excluded_members]
        title = f"**Curated HSP Polymers: {category_result.category_label}**"
    else:
        entries = [entry.to_dict() for entry in list_hsp_polymer_entries(include_excluded=include_excluded)]
        excluded = []
        title = "**Curated HSP Polymers**"

    display = [title, ""]
    display.append(_format_resolution_table(entries, kind="polymer") if entries else "No entries found.")
    if excluded:
        display.append("\n**Excluded by default:**")
        display.append(_format_resolution_table(excluded, kind="polymer"))
    return json_tool_success(
        "\n".join(display),
        tool_name="list_hsp_supported_polymers",
        category=category,
        include_quality_flags=include_quality_flags,
        curated_only=curated_only,
        include_excluded=include_excluded,
        entries=entries,
        excluded_entries=excluded,
    )


@safe_tool_wrapper(structured_output=True)
def list_hsp_supported_solvents(
    category: str | None = None,
    polarity_class: str | None = None,
    include_quality_flags: bool = True,
    curated_only: bool = True,
) -> str:
    """List curated solvents supported by the HSP/RED screening registry."""
    if not curated_only:
        return json_tool_error(
            "Raw full-catalog HSP listing is not exposed yet; use curated_only=True.",
            tool_name="list_hsp_supported_solvents",
            error_code="raw_hsp_catalog_not_exposed",
        )
    entries = [
        entry.to_dict()
        for entry in list_hsp_solvent_entries(category=category, polarity_class=polarity_class)
    ]
    display = ["**Curated HSP Solvents**", ""]
    if category:
        display.append(f"Chemical-family/category filter: `{category}`")
    if polarity_class:
        display.append(f"Polarity filter: `{polarity_class}`")
    display.append("")
    display.append(_format_resolution_table(entries, kind="solvent") if entries else "No entries found.")
    display.append(
        "\nUnsupported common HSP solvent requests remain explicit: GVL, GBL, DMA, diethyl ether, and 1,4-dioxane."
    )
    return json_tool_success(
        "\n".join(display),
        tool_name="list_hsp_supported_solvents",
        category=category,
        polarity_class=polarity_class,
        include_quality_flags=include_quality_flags,
        curated_only=curated_only,
        entries=entries,
        unsupported_common=["GVL", "GBL", "DMA", "diethyl ether", "1,4-dioxane"],
    )


def _resolve_polymer_inputs(
    *,
    polymers: list[str],
    polymer_category: str | None,
    include_excluded: bool,
) -> tuple[list[HspPolymerEntry], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    entries: list[HspPolymerEntry] = []
    resolutions: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    warnings: list[str] = []
    if polymer_category:
        result = resolve_hsp_polymer_category(polymer_category, include_excluded=include_excluded)
        resolutions.append(result.to_dict())
        if result.status == "category":
            entries.extend(entry for entry in result.category_members if isinstance(entry, HspPolymerEntry))
            if result.excluded_members:
                warnings.append(
                    "Some category members are excluded by default because their HSP entries are qualified, conditioned, or not clean dissolution records."
                )
        else:
            unsupported.append(result.to_dict())
    for item in polymers:
        result = resolve_hsp_polymer(item, include_excluded=include_excluded)
        resolutions.append(result.to_dict())
        if result.status == "resolved" and isinstance(result.selected, HspPolymerEntry):
            entries.append(result.selected)
        elif result.status == "category":
            entries.extend(entry for entry in result.category_members if isinstance(entry, HspPolymerEntry))
        else:
            unsupported.append(result.to_dict())
    deduped = list({entry.id: entry for entry in entries}.values())
    return deduped, resolutions, unsupported, warnings


def _resolve_solvent_inputs(
    *,
    solvents: list[str],
    solvent_category: str | None,
    solvent_polarity: str | None,
    include_excluded: bool,
) -> tuple[list[HspSolventEntry], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    entries: list[HspSolventEntry] = []
    resolutions: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    warnings: list[str] = []
    if solvent_category:
        result = resolve_hsp_solvent_category(solvent_category, include_excluded=include_excluded)
        resolutions.append(result.to_dict())
        if result.status == "category":
            entries.extend(entry for entry in result.category_members if isinstance(entry, HspSolventEntry))
        else:
            unsupported.append(result.to_dict())
    if solvent_polarity:
        result = resolve_hsp_solvent_polarity(solvent_polarity, include_excluded=include_excluded)
        resolutions.append(result.to_dict())
        if result.status == "category":
            entries.extend(entry for entry in result.category_members if isinstance(entry, HspSolventEntry))
        else:
            unsupported.append(result.to_dict())
    for item in solvents:
        result = resolve_hsp_solvent(item, include_excluded=include_excluded)
        resolutions.append(result.to_dict())
        if result.status == "resolved" and isinstance(result.selected, HspSolventEntry):
            entries.append(result.selected)
        elif result.status == "category":
            entries.extend(entry for entry in result.category_members if isinstance(entry, HspSolventEntry))
        else:
            unsupported.append(result.to_dict())
    deduped = list({entry.id: entry for entry in entries}.values())
    return deduped, resolutions, unsupported, warnings


def _plot_hsp_matrix(rows: list[dict[str, Any]], polymers: list[HspPolymerEntry], solvents: list[HspSolventEntry]) -> str | None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    if not rows or not polymers or not solvents:
        return None
    polymer_ids = [
        f"{entry.display_name}*" if entry.quality not in {"canonical", "representative"} else entry.display_name
        for entry in polymers
    ]
    polymer_lookup = {entry.display_name: label for entry, label in zip(polymers, polymer_ids)}
    solvent_ids = [entry.display_name for entry in solvents]
    values = np.full((len(polymer_ids), len(solvent_ids)), np.nan)
    soluble_mask = np.zeros((len(polymer_ids), len(solvent_ids)), dtype=bool)
    for row in rows:
        try:
            i = polymer_ids.index(polymer_lookup[str(row["polymer"])])
            j = solvent_ids.index(str(row["solvent"]))
        except ValueError:
            continue
        values[i, j] = float(row["red"])
        soluble_mask[i, j] = float(row["red"]) < 1.0

    width = max(7.0, 0.55 * len(solvent_ids) + 2.5)
    height = max(4.0, 0.45 * len(polymer_ids) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(
        left=0.16,
        right=0.86,
        top=0.86,
        bottom=0.32 if len(solvent_ids) <= 6 else 0.38,
    )
    image = ax.imshow(np.clip(values, 0.0, 2.0), cmap="RdYlGn_r", vmin=0.0, vmax=2.0, aspect="auto")
    ax.set_xticks(range(len(solvent_ids)))
    ax.set_xticklabels(solvent_ids, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(polymer_ids)))
    ax.set_yticklabels(polymer_ids, fontsize=9)
    compatible_count = int(np.nansum(values < 1.0))
    total_count = int(np.sum(~np.isnan(values)))
    ax.set_title(f"HSP RED Matrix: {compatible_count}/{total_count} pairs inside RED < 1")
    for i in range(len(polymer_ids)):
        for j in range(len(solvent_ids)):
            if np.isnan(values[i, j]):
                continue
            label = f"{values[i, j]:.2f}"
            color = "white" if values[i, j] > 1.35 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
            if soluble_mask[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="black", linewidth=1.4))
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("RED (lower is more compatible)")
    fig.text(
        0.16,
        0.04,
        "Outlined cells: RED < 1 HSP-compatible. * qualified/proxy HSP entry. This is not wt% solubility.",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    plot_name = descriptive_plot_name("hsp_red_matrix", polymers=polymer_ids, solvents=solvent_ids)
    return save_plot(fig, plot_name, "matplotlib")


@safe_tool_wrapper(structured_output=True)
def screen_hsp_solubility_matrix(
    polymers: list[str] | str | None = None,
    polymer_category: str | None = None,
    solvents: list[str] | str | None = None,
    solvent_category: str | None = None,
    solvent_polarity: str | None = None,
    curated_only: bool = True,
    include_qualified: bool = True,
    include_excluded: bool = False,
    temperature_c: float = 25.0,
    generate_visualization: bool = True,
) -> str:
    """Screen a polymer category/list against solvent categories/lists with HSP/RED."""
    if get_predictor is None:
        return json_tool_error(
            "ML predictor unavailable - strap.vendor.solubility_predictor could not be imported.",
            tool_name="screen_hsp_solubility_matrix",
            error_code="predictor_unavailable",
        )
    if not curated_only:
        return json_tool_error(
            "Raw full-catalog HSP matrix screening is not exposed yet; use curated_only=True.",
            tool_name="screen_hsp_solubility_matrix",
            error_code="raw_hsp_catalog_not_exposed",
        )

    polymer_entries, polymer_resolutions, unsupported_poly, polymer_warnings = _resolve_polymer_inputs(
        polymers=_coerce_items(polymers),
        polymer_category=polymer_category,
        include_excluded=include_excluded,
    )
    solvent_entries, solvent_resolutions, unsupported_solvents, solvent_warnings = _resolve_solvent_inputs(
        solvents=_coerce_items(solvents),
        solvent_category=solvent_category,
        solvent_polarity=solvent_polarity,
        include_excluded=include_excluded,
    )
    if not include_qualified:
        polymer_entries = [entry for entry in polymer_entries if entry.quality in {"canonical", "representative", "proxy"}]

    unsupported = [*unsupported_poly, *unsupported_solvents]
    warnings = [
        *polymer_warnings,
        *solvent_warnings,
        *_temperature_warnings(float(temperature_c)),
        "HSP matrix output is binary compatibility screening, not quantitative wt% solubility.",
    ]

    if not polymer_entries or not solvent_entries:
        return json_tool_error(
            "No resolvable curated HSP polymers or solvents were found for this matrix request.",
            tool_name="screen_hsp_solubility_matrix",
            error_code="empty_hsp_screen",
            polymer_resolution=polymer_resolutions,
            solvent_resolution=solvent_resolutions,
            unsupported=unsupported,
            warnings=list(dict.fromkeys(warnings)),
        )

    predictor = get_predictor()
    rows: list[dict[str, Any]] = []
    for polymer in polymer_entries:
        polymer_hsp, r0, _poly_asset = _polymer_hsp(polymer)
        for solvent in solvent_entries:
            solvent_hsp, molar_volume, _solv_asset = _solvent_hsp(solvent)
            prediction = predictor.predict(polymer_hsp, solvent_hsp, r0, molar_volume)
            rows.append(_hsp_result_row(polymer, solvent, prediction, temperature_c=float(temperature_c)))

    artifacts: list[str] = []
    if generate_visualization:
        path = _plot_hsp_matrix(rows, polymer_entries, solvent_entries)
        if path:
            artifacts.append(path)

    display = ["**HSP Solubility Matrix Screen**", ""]
    display.append("This is temperature-independent HSP/RED compatibility screening, not wt% solubility.")
    if abs(float(temperature_c) - 25.0) > 1e-9:
        display.append(f"Requested temperature {float(temperature_c):g} C was not used by the HSP model.")
    display.append("")
    display.append("| Polymer | Solvent | Prediction | RED | Polymer quality | Solvent class |")
    display.append("| --- | --- | --- | ---: | --- | --- |")
    for row in rows[:80]:
        display.append(
            f"| {row['polymer']} | {row['solvent']} | {'soluble' if row['soluble'] else 'non-soluble'} | "
            f"{row['red']:.3f} | {row['polymer_quality']} | {row['polarity_class']} / {row['chemical_family']} |"
        )
    if len(rows) > 80:
        display.append(f"| ... | ... | {len(rows) - 80} additional rows omitted from display | ... | ... | ... |")
    if unsupported:
        display.append("\n**Unsupported or ambiguous requests:**")
        for item in unsupported:
            display.append(f"- {item.get('query')}: {item.get('unsupported_reason') or item.get('status')}")
    if warnings:
        display.append("\n**Warnings:**")
        for warning in list(dict.fromkeys(warnings))[:10]:
            display.append(f"- {warning}")

    return json_tool_success(
        "\n".join(display),
        tool_name="screen_hsp_solubility_matrix",
        analysis_type="hsp_binary_screen",
        hsp_only=True,
        temperature_used_by_model=False,
        temperature_c_requested=float(temperature_c),
        polymer_resolution=polymer_resolutions,
        solvent_resolution=solvent_resolutions,
        unsupported=unsupported,
        warnings=list(dict.fromkeys(warnings)),
        results=rows,
        artifacts=artifacts,
    )
