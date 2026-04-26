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
import textwrap
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
from strap.services.visualization_service import (
    PUB_COLORS as _PUB_COLORS,
    PUB_FONTSIZE as _PUB_FONTSIZE,
    apply_pub_style as _apply_pub_style,
    execute_query as _execute_query,
    get_cosmobase_column as _get_cosmobase_column,
    get_plot_url as _get_plot_url,
    get_solvent_name_column as _get_solvent_name_column,
    get_solvent_table_name as _get_solvent_table_name,
    lookup_solvent_properties as _lookup_solvent_properties,
    normalize_solvent_names as _normalize_solvent_names,
    verify_inputs as _verify_inputs,
)
from strap.tools._helpers import (
    safe_tool_wrapper,
    save_plot,
    get_plots_dir,
    get_cross_database_properties,
)

logger = logging.getLogger(__name__)
# ===================================================================
# Visualization tool functions
# ===================================================================


def _stringify_point_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _humanize_stage_code(value: str) -> str:
    mapping = {
        "lf": "Landfill",
        "we": "Waste-to-energy",
        "gas_er": "Gasification",
        "gas": "Gasification",
        "py": "Pyrolysis",
        "st1": "Wash 1",
        "st2": "Wash 2",
    }
    text = str(value).strip()
    return mapping.get(text, text.replace("_", " ").title())


def _display_solvent_label(solvent: str) -> str:
    mapping = {
        "1,2-dimethylbenzene": "o-Xylene",
        "1,3-dimethylbenzene": "m-Xylene",
        "1,4-dimethylbenzene": "p-Xylene",
        "ch2cl2": "Dichloromethane",
        "chcl3": "Chloroform",
        "dimethylformamide": "DMF",
        "dimethylsulfoxide": "DMSO",
        "thf": "THF",
        "thp": "THP",
    }
    normalized = str(solvent).strip().lower()
    if normalized in mapping:
        return mapping[normalized]
    if "-" in normalized or "," in normalized:
        return normalized
    return normalized[:1].upper() + normalized[1:]


def _extract_point_design(row: pd.Series) -> Dict[str, Any]:
    equivalent_designs = row.get("equivalent_designs")
    if isinstance(equivalent_designs, list) and equivalent_designs:
        first = equivalent_designs[0]
        if isinstance(first, dict):
            return first
    return row.to_dict()


def _design_or_row_value(design: Dict[str, Any], row: pd.Series, key: str) -> Any:
    value = design.get(key)
    if value not in (None, "", []):
        return value
    if key in row.index:
        return row.get(key)
    return value


def _format_pareto_point_legend_entry(
    row: pd.Series,
    *,
    x_metric: str,
    y_column: str,
) -> str:
    point_id = int(row["point_id"])
    design = _extract_point_design(row)
    components: list[str] = []

    wash1 = _stringify_point_values(design.get("wash1_selection"))
    wash2 = _stringify_point_values(design.get("wash2_selection"))
    if wash1:
        components.append(f"W1: {', '.join(wash1)}")
    if wash2:
        components.append(f"W2: {', '.join(wash2)}")

    residual_polymers = _stringify_point_values(_design_or_row_value(design, row, "residual_polymers"))
    residual_destination = [
        _humanize_stage_code(value)
        for value in _stringify_point_values(_design_or_row_value(design, row, "residual_destination_tech"))
    ]
    has_residual_metadata = (
        "residual_polymers" in design
        or "residual_polymers" in row.index
        or "residual_destination_tech" in design
        or "residual_destination_tech" in row.index
    )
    if residual_polymers:
        destination = ", ".join(residual_destination) if residual_destination else "Downstream waste"
        components.append(f"Waste: {', '.join(residual_polymers)} -> {destination}")
    elif has_residual_metadata and (wash1 or wash2):
        components.append("Waste: none")

    stage1 = [_humanize_stage_code(value) for value in _stringify_point_values(design.get("stage1_tech"))]
    stage2 = [_humanize_stage_code(value) for value in _stringify_point_values(design.get("stage2_tech"))]
    stage3 = [_humanize_stage_code(value) for value in _stringify_point_values(row.get("stage3_variants"))]
    if not stage3:
        stage3 = [_humanize_stage_code(value) for value in _stringify_point_values(design.get("stage3_tech"))]

    if not has_residual_metadata:
        if not wash1 and not wash2:
            final_stage = next((stage for stage in stage1 + stage2 + stage3 if stage not in {"Wash 1", "Wash 2"}), "")
            if final_stage:
                components.append(f"Baseline: {final_stage}")
        else:
            final_stage = next((stage for stage in stage3 + stage2 + stage1 if stage not in {"Wash 1", "Wash 2"}), "")
            if final_stage:
                components.append(f"End: {final_stage}")

    polymer_solvent_map = design.get("polymer_solvent_map") or row.get("polymer_solvent_map")
    if polymer_solvent_map and not wash1 and not wash2 and isinstance(polymer_solvent_map, dict):
        ordered = [f"{polymer}-{solvent}" for polymer, solvent in polymer_solvent_map.items()]
        if ordered:
            components.append(f"Route: {', '.join(ordered)}")

    selection_origin = str(row.get("selection_origin") or design.get("selection_origin") or "").strip()
    if selection_origin and selection_origin != "exact_route":
        components.append(f"Origin: {selection_origin.replace('_', ' ')}")

    matched_route_id = str(
        row.get("matched_route_id")
        or design.get("matched_route_id")
        or row.get("route_id")
        or design.get("route_id")
        or ""
    ).strip()
    if matched_route_id:
        components.append(f"Route: {matched_route_id}")

    x_value = row.get(x_metric)
    y_value = row.get(y_column)
    if x_metric == "total_cost":
        if float(x_value) >= 1_000_000:
            x_text = f"Cost: ${float(x_value)/1_000_000:.2f}M"
        elif float(x_value) >= 1_000:
            x_text = f"Cost: ${float(x_value)/1_000:.0f}k"
        else:
            x_text = f"Cost: ${float(x_value):,.0f}"
    else:
        x_text = f"{x_metric}: {float(x_value):,.3g}"
    if y_column == "emissions":
        y_text = f"Emissions: {float(y_value):,.1f}"
    else:
        y_text = f"Circularity: {float(y_value):.3f}"
    components.append(f"{x_text} | {y_text}")

    body = " | ".join(components)
    wrapped = textwrap.fill(body, width=28, subsequent_indent="    ")
    return f"P{point_id}: {wrapped}"


def _coerce_optimization_payload(raw_payload: Dict[str, Any] | str | None) -> Dict[str, Any]:
    if isinstance(raw_payload, str):
        parsed = json.loads(raw_payload)
        if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], dict):
            return parsed["data"]
        if isinstance(parsed, dict):
            return parsed
        raise TypeError("optimization_result_json must decode to a mapping")
    if isinstance(raw_payload, dict):
        if "data" in raw_payload and isinstance(raw_payload["data"], dict):
            return raw_payload["data"]
        return raw_payload
    raise TypeError("optimization_result_json must be a JSON string or mapping")


def _load_optimization_payload_from_handoff(source_handoff_id: str) -> Dict[str, Any]:
    from strap.handoffs import get_handoff

    record = get_handoff(source_handoff_id)
    if record is None:
        raise ValueError(f"source_handoff_id '{source_handoff_id}' was not found")
    raw_payload = record.payload
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("pareto_result_json"), dict):
        return raw_payload["pareto_result_json"]
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("optimization_result_json"), dict):
        return raw_payload["optimization_result_json"]
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("source_payload"), dict):
        return raw_payload["source_payload"]
    if isinstance(raw_payload, dict):
        return raw_payload
    raise TypeError("source handoff payload must be a mapping")


@safe_tool_wrapper(structured_output=True)
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
    annotate_model_limits: bool = False,
    temperature_min: Optional[float] = None,
    temperature_max: Optional[float] = None,
    y_axis_max: Optional[float] = 100.0,
    y_axis_ranges_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Plot solubility vs temperature curves using the interpolation model.

    Args:
        table_name, polymer_column, solvent_column: DB table and column names (kept for API compat)
        temperature_column, solubility_column: Temperature and solubility columns (kept for API compat)
        polymers: Comma-separated polymer names
        solvents: Comma-separated solvent names
        plot_title: Custom plot title
        include_confidence_bands: Unused (interpolation produces smooth curves)
        annotate_model_limits: If true, mark extrapolated/sensitivity regions on the plot.
            Defaults false so saved PNGs are clean enough for slides/publication drafts.
        temperature_min/temperature_max: Temperature range filter
        y_axis_max: Default shared y-axis maximum. Defaults to 100 for normalized solubility panels.
        y_axis_ranges_json: Optional JSON mapping of polymer -> max, [min, max], or
            {"min": value, "max": value} for explicit panel-specific ranges.
        output_dir: Directory to save the PNG. Accepts Linux paths and WSL UNC paths.
        output_path: Full PNG path, or a directory path, for the saved plot.

    WHEN TO USE:
    - "Plot solubility of PS in toluene vs temperature"
    - "Show how solubility changes with temperature for PET"
    - "Plot EVOH in DMSO up to 170C"
    - For "at 170C" curve requests, use temperature_max=170 unless the user explicitly asks for a single-point lookup.
    """
    from strap.services.tool_response_service import json_tool_error, json_tool_success
    from strap.solubility import (
        FITTED_TEMP_MAX_C,
        SENSITIVITY_EXTRAPOLATION_MAX_C,
        get_entry,
        get_solubility_curve,
        get_solubility_pair_exclusion_reason,
        temperature_basis_note,
        temperature_use_regime,
    )

    polymer_list = [p.strip() for p in polymers.split(",")]
    solvent_list = [s.strip() for s in solvents.split(",")]
    solvent_list = _normalize_solvent_names(solvent_list)

    t_start = max(temperature_min or 25.0, 25.0)
    requested_t_end = temperature_max or FITTED_TEMP_MAX_C
    if t_start > SENSITIVITY_EXTRAPOLATION_MAX_C:
        return json_tool_error(
            (
                f"Start temperature {t_start} C is above the supported sensitivity limit "
                f"of {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} C for Apelblat extrapolation."
            ),
            tool_name="plot_solubility_vs_temperature",
            error_code="temperature_above_supported_extrapolation",
            temperature_c=t_start,
            max_temperature_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
        )
    range_was_capped = requested_t_end > SENSITIVITY_EXTRAPOLATION_MAX_C
    t_end = min(requested_t_end, SENSITIVITY_EXTRAPOLATION_MAX_C)
    y_axis_ranges: dict[str, tuple[float, float]] = {}
    if y_axis_ranges_json:
        raw_ranges = json.loads(y_axis_ranges_json)
        if not isinstance(raw_ranges, dict):
            raise TypeError("y_axis_ranges_json must decode to a mapping")
        for raw_polymer, raw_range in raw_ranges.items():
            key = str(raw_polymer).strip().lower()
            if isinstance(raw_range, dict):
                bottom = float(raw_range.get("min", 0.0))
                top = float(raw_range["max"])
            elif isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                bottom = float(raw_range[0])
                top = float(raw_range[1])
            else:
                bottom = 0.0
                top = float(raw_range)
            if top <= bottom:
                raise ValueError(f"Invalid y-axis range for {raw_polymer}: max must be greater than min")
            y_axis_ranges[key] = (bottom, top)

    series_records: list[dict[str, Any]] = []
    total_points = 0
    extrapolated_points = 0
    sensitivity_points = 0
    min_pair_t_max: float | None = None
    excluded_pairs: list[str] = []
    plotted_polymers: list[str] = []
    plotted_solvents: list[str] = []

    for polymer in polymer_list:
        for solvent in solvent_list:
            if get_solubility_pair_exclusion_reason(polymer, solvent):
                excluded_pairs.append(f"{polymer}/{solvent}")
                continue
            curve = get_solubility_curve(polymer, solvent, t_start, t_end, 5.0)
            if curve:
                if polymer not in plotted_polymers:
                    plotted_polymers.append(polymer)
                if solvent not in plotted_solvents:
                    plotted_solvents.append(solvent)
                entry = get_entry(polymer, solvent)
                pair_t_max = float(entry.get("t_max_c", FITTED_TEMP_MAX_C)) if entry else FITTED_TEMP_MAX_C
                min_pair_t_max = pair_t_max if min_pair_t_max is None else min(min_pair_t_max, pair_t_max)
                temps = [pt["temperature"] for pt in curve]
                sols = [pt["solubility"] for pt in curve]
                total_points += len(curve)
                extrapolated_points += sum(1 for temp in temps if temp > pair_t_max)
                sensitivity_points += sum(
                    1 for temp in temps if temperature_use_regime(float(temp)) == "sensitivity_extrapolation"
                )
                series_records.append(
                    {
                        "polymer": polymer,
                        "solvent": solvent,
                        "temps": temps,
                        "solubilities": sols,
                        "pair_t_max": pair_t_max,
                    }
                )

    if total_points == 0:
        return "No data found for the specified polymer-solvent combinations."

    _apply_pub_style()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "legend.frameon": False,
        }
    )
    solvent_colors = {
        solvent: _PUB_COLORS[index % len(_PUB_COLORS)]
        for index, solvent in enumerate(plotted_solvents)
    }
    solvent_markers = {
        solvent: marker
        for solvent, marker in zip(plotted_solvents, ["o", "s", "^", "D", "v", "P", "X", "*"])
    }

    def _plot_curve(
        ax,
        record: dict[str, Any],
        *,
        label: str,
        color_override: str | None = None,
        marker_override: str | None = None,
    ) -> None:
        temps = record["temps"]
        sols = record["solubilities"]
        pair_t_max = record["pair_t_max"]
        solvent = record["solvent"]
        markevery = max(1, len(temps) // 7)
        line_kwargs = {
            "color": color_override or solvent_colors[solvent],
            "linewidth": 1.55,
            "marker": marker_override or solvent_markers.get(solvent, "o"),
            "markersize": 3.0,
            "markeredgewidth": 0.0,
            "markevery": markevery,
            "label": label,
        }
        if annotate_model_limits:
            solid = [(temp, sol) for temp, sol in zip(temps, sols) if temp <= pair_t_max]
            dashed = [(temp, sol) for temp, sol in zip(temps, sols) if temp > pair_t_max]
            if solid:
                ax.plot([temp for temp, _ in solid], [sol for _, sol in solid], **line_kwargs)
            if dashed:
                bridge = []
                if solid:
                    bridge.append(solid[-1])
                bridge.extend(dashed)
                dashed_kwargs = dict(line_kwargs)
                dashed_kwargs["linestyle"] = "--"
                dashed_kwargs["label"] = f"{label} extrapolated" if not solid else None
                ax.plot([temp for temp, _ in bridge], [sol for _, sol in bridge], **dashed_kwargs)
        else:
            ax.plot(temps, sols, **line_kwargs)

    def _format_axis(ax, values: list[float], *, polymer: str | None = None) -> None:
        ymax = max(values) if values else 1.0
        override = y_axis_ranges.get(str(polymer or "").strip().lower())
        if override is not None:
            bottom, top = override
        elif y_axis_max is not None:
            bottom = 0.0
            top = max(float(y_axis_max), ymax)
        elif ymax >= 80:
            top = 105
            bottom = -0.02 * top
        elif ymax >= 20:
            top = min(105, np.ceil(ymax / 5.0) * 5.0 + 5.0)
            bottom = -0.02 * top
        elif ymax >= 5:
            top = np.ceil(ymax / 2.0) * 2.0 + 2.0
            bottom = -0.02 * top
        else:
            top = max(1.0, np.ceil(max(ymax, 0.1) * 10.0) / 10.0 + 0.2)
            bottom = -0.02 * top
        ax.set_ylim(bottom=bottom, top=top)
        ax.set_xlim(t_start, t_end)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.tick_params(
            axis="both",
            which="major",
            length=3,
            width=0.6,
            direction="out",
            top=False,
            right=False,
        )

    use_facets = len(plotted_polymers) > 1 and (len(plotted_solvents) > 1 or bool(y_axis_ranges))
    if use_facets:
        n_panels = len(plotted_polymers)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig_w = max(5.2, 2.35 * ncols)
        fig_h = 2.45 * nrows + 0.45
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, squeeze=False)
        flat_axes = list(axes.ravel())
        for ax, polymer in zip(flat_axes, plotted_polymers):
            panel_records = [record for record in series_records if record["polymer"] == polymer]
            for record in panel_records:
                _plot_curve(ax, record, label=_display_solvent_label(record["solvent"]))
            panel_values = [value for record in panel_records for value in record["solubilities"]]
            _format_axis(ax, panel_values, polymer=polymer)
            ax.set_title(polymer, loc="left", fontweight="bold", pad=4)
        for ax in flat_axes[len(plotted_polymers):]:
            ax.axis("off")
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=solvent_colors[solvent],
                marker=solvent_markers.get(solvent, "o"),
                linewidth=1.55,
                markersize=3.5,
                label=_display_solvent_label(solvent),
            )
            for solvent in plotted_solvents
        ]
        fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), bbox_to_anchor=(0.5, 1.0))
        fig.supxlabel("Temperature (\u00b0C)", y=0.02)
        fig.supylabel("Solubility (%)", x=0.01)
        if plot_title:
            fig.suptitle(plot_title, y=1.08, fontweight="bold")
        fig.subplots_adjust(left=0.09, right=0.99, top=0.82, bottom=0.18, wspace=0.34, hspace=0.42)
    else:
        n_curves = len(series_records)
        fig_w = 3.5 if n_curves <= 3 else 6.4
        fig_h = 3.0 if n_curves <= 3 else 4.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        all_values: list[float] = []
        for index, record in enumerate(series_records):
            label = f"{record['polymer']} in {_display_solvent_label(record['solvent'])}"
            color_override = _PUB_COLORS[index % len(_PUB_COLORS)] if len(plotted_solvents) == 1 else None
            marker_override = ["o", "s", "^", "D", "v", "P", "X", "*"][index % 8] if len(plotted_solvents) == 1 else None
            _plot_curve(
                ax,
                record,
                label=label,
                color_override=color_override,
                marker_override=marker_override,
            )
            all_values.extend(record["solubilities"])
        single_polymer = plotted_polymers[0] if len(plotted_polymers) == 1 else None
        _format_axis(ax, all_values, polymer=single_polymer)
        ax.set_xlabel("Temperature (\u00b0C)")
        ax.set_ylabel("Solubility (%)")
        if plot_title:
            ax.set_title(plot_title)
        ax.legend(loc="best", fontsize=max(_PUB_FONTSIZE - 1, 6))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.14)

    if annotate_model_limits and extrapolated_points and min_pair_t_max is not None and t_end > min_pair_t_max:
        for axis in fig.axes:
            axis.axvspan(min_pair_t_max, t_end, color="#f1c40f", alpha=0.08)
            axis.axvline(min_pair_t_max, color="#a66f00", linestyle=":", linewidth=0.8)
    if annotate_model_limits and sensitivity_points:
        sensitivity_start = max(180.0, t_start)
        if t_end > sensitivity_start:
            for axis in fig.axes:
                axis.axvspan(sensitivity_start, t_end, color="#e67e22", alpha=0.08)

    from strap.tools._helpers import descriptive_plot_name
    plot_name = descriptive_plot_name(
        "solubility_vs_temp",
        polymers=plotted_polymers or polymer_list,
        solvents=plotted_solvents or solvent_list,
    )
    filepath = save_plot(
        fig,
        plot_name,
        "matplotlib",
        output_dir=output_dir,
        output_path=output_path,
    )
    plt.close(fig)

    output = "Solubility vs Temperature Plot Created\n\n"
    output += f"Polymers: {', '.join(plotted_polymers or polymer_list)}\n"
    output += f"Solvents: {', '.join(_display_solvent_label(s) for s in (plotted_solvents or solvent_list))}\n"
    if excluded_pairs:
        output += (
            "Excluded data-quality pair(s): "
            + ", ".join(excluded_pairs)
            + "\n"
        )
    if temperature_min is not None and temperature_max is not None:
        output += f"Temperature range: {temperature_min}C - {temperature_max}C\n"
    elif temperature_min is not None:
        output += f"Temperature range: {temperature_min}C and above\n"
    elif temperature_max is not None:
        output += f"Temperature range: up to {temperature_max}C\n"
    if range_was_capped:
        output += f"Requested upper temperature {requested_t_end}C was capped at {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f}C.\n"
    if y_axis_ranges:
        output += "Y-axis ranges: custom per-polymer override(s)\n"
    elif y_axis_max is not None:
        output += f"Y-axis range: 0-{float(y_axis_max):g}%\n"
    basis_note = temperature_basis_note(t_end)
    if basis_note:
        output += f"Temperature basis: {basis_note}\n"
    if extrapolated_points:
        if annotate_model_limits:
            output += f"Extrapolated points: {extrapolated_points} (annotated on plot)\n"
        else:
            output += (
                f"Model-limit note: {extrapolated_points} point(s) are outside the fitted range; "
                "this caveat is reported in text only so the saved plot remains presentation-ready.\n"
            )
    if sensitivity_points:
        output += "180-200C points are sensitivity-only screening data, not validated operating recommendations.\n"
    output += (
        "Exact SQL/database grid-point values can also be provided on request; "
        "they should be similar to the fitted curve near measured temperatures.\n"
    )
    output += f"Data points: {total_points}\n"
    output += f"\n{_get_plot_url(filepath)}"

    gc.collect()
    return json_tool_success(
        output,
        tool_name="plot_solubility_vs_temperature",
        polymers=plotted_polymers or polymer_list,
        solvents=plotted_solvents or solvent_list,
        excluded_pairs=excluded_pairs,
        temperature_min_c=t_start,
        temperature_max_c=t_end,
        requested_temperature_max_c=requested_t_end,
        range_was_capped=range_was_capped,
        data_points=total_points,
        extrapolated_points=extrapolated_points,
        sensitivity_points=sensitivity_points,
        model_limit_annotations_on_plot=annotate_model_limits,
        y_axis_max=y_axis_max,
        y_axis_ranges=y_axis_ranges,
        exact_sql_values_available=True,
        plot_filepath=filepath,
        output_dir=output_dir,
        output_path=output_path,
        plot_url=_get_plot_url(filepath),
    )


@safe_tool_wrapper(structured_output=True)
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
    y_axis_max: Optional[float] = 100.0,
    output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Generate an interactive Plotly HTML plot of solubility vs temperature.

    Args:
        table_name, polymer_column, solvent_column: DB table and column names (kept for API compat)
        temperature_column, solubility_column: Temperature and solubility columns (kept for API compat)
        polymers: Comma-separated polymer names
        solvents: Comma-separated solvent names
        plot_title: Custom plot title
        temperature_min/temperature_max: Temperature range filter
        y_axis_max: Default y-axis maximum. Defaults to 100 for normalized solubility.
        output_dir: Directory to save the HTML. Accepts Linux paths and WSL UNC paths.
        output_path: Full HTML path, or a directory path, for the saved plot.

    WHEN TO USE:
    - "Create an interactive solubility vs temperature chart"
    - "I want a zoomable plot of PET solubility curves"
    - For "at 170C" curve requests, use temperature_max=170 unless the user explicitly asks for a single-point lookup.
    """
    from strap.services.tool_response_service import json_tool_error, json_tool_success
    from strap.solubility import (
        FITTED_TEMP_MAX_C,
        SENSITIVITY_EXTRAPOLATION_MAX_C,
        get_entry,
        get_solubility_curve,
        temperature_basis_note,
        temperature_use_regime,
    )

    polymer_list = [p.strip() for p in polymers.split(",")]
    solvent_list = [s.strip() for s in solvents.split(",")]
    solvent_list = _normalize_solvent_names(solvent_list)

    t_start = max(temperature_min or 25.0, 25.0)
    requested_t_end = temperature_max or FITTED_TEMP_MAX_C
    if t_start > SENSITIVITY_EXTRAPOLATION_MAX_C:
        return json_tool_error(
            (
                f"Start temperature {t_start} C is above the supported sensitivity limit "
                f"of {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f} C for Apelblat extrapolation."
            ),
            tool_name="plot_solubility_vs_temperature_interactive",
            error_code="temperature_above_supported_extrapolation",
            temperature_c=t_start,
            max_temperature_c=SENSITIVITY_EXTRAPOLATION_MAX_C,
        )
    range_was_capped = requested_t_end > SENSITIVITY_EXTRAPOLATION_MAX_C
    t_end = min(requested_t_end, SENSITIVITY_EXTRAPOLATION_MAX_C)

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    color_idx = 0
    total_points = 0
    extrapolated_points = 0
    sensitivity_points = 0
    min_pair_t_max: float | None = None

    for polymer in polymer_list:
        for solvent in solvent_list:
            curve = get_solubility_curve(polymer, solvent, t_start, t_end, 5.0)
            if curve:
                entry = get_entry(polymer, solvent)
                pair_t_max = float(entry.get("t_max_c", FITTED_TEMP_MAX_C)) if entry else FITTED_TEMP_MAX_C
                min_pair_t_max = pair_t_max if min_pair_t_max is None else min(min_pair_t_max, pair_t_max)
                temps = [pt["temperature"] for pt in curve]
                sols = [pt["solubility"] for pt in curve]
                total_points += len(curve)
                extrapolated_points += sum(1 for temp in temps if temp > pair_t_max)
                sensitivity_points += sum(
                    1 for temp in temps if temperature_use_regime(float(temp)) == "sensitivity_extrapolation"
                )

                c = colors[color_idx % len(colors)]
                customdata = [
                    "sensitivity-only" if temperature_use_regime(float(temp)) == "sensitivity_extrapolation"
                    else "extrapolated" if temp > pair_t_max
                    else "fitted"
                    for temp in temps
                ]
                fig.add_trace(go.Scatter(
                    x=temps,
                    y=sols,
                    mode="lines+markers",
                    name=f"{polymer} in {solvent}",
                    line=dict(width=3, color=c),
                    marker=dict(size=8, symbol="circle"),
                    customdata=customdata,
                    hovertemplate=(
                        f"<b>{polymer} in {solvent}</b><br>"
                        + "Temperature: %{x:.1f}C<br>"
                        + "Solubility: %{y:.2f}%<br>"
                        + "Basis: %{customdata}<br>"
                        + "<extra></extra>"
                    ),
                ))
                color_idx += 1

    if total_points == 0:
        return "No data found for the specified polymer-solvent combinations."

    title = plot_title or "Interactive Solubility vs Temperature"
    shapes = []
    annotations = []
    if extrapolated_points and min_pair_t_max is not None and t_end > min_pair_t_max:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=min_pair_t_max,
                x1=t_end,
                y0=0,
                y1=1,
                fillcolor="rgba(241,196,15,0.12)",
                line_width=0,
                layer="below",
            )
        )
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=min_pair_t_max,
                x1=min_pair_t_max,
                y0=0,
                y1=1,
                line=dict(color="rgba(166,111,0,0.8)", width=1, dash="dot"),
            )
        )
        annotations.append(
            dict(
                x=min_pair_t_max,
                y=1.02,
                xref="x",
                yref="paper",
                text="extrapolated",
                showarrow=False,
                font=dict(size=11),
            )
        )
    if sensitivity_points:
        sensitivity_start = max(180.0, t_start)
        if t_end > sensitivity_start:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=sensitivity_start,
                    x1=t_end,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(230,126,34,0.14)",
                    line_width=0,
                    layer="below",
                )
            )
            annotations.append(
                dict(
                    x=sensitivity_start,
                    y=0.93,
                    xref="x",
                    yref="paper",
                    text="sensitivity only",
                    showarrow=False,
                    font=dict(size=11),
                )
            )
    yaxis_config = dict(
        title=dict(text="Solubility (%)", font=dict(size=16, family="Arial")),
        showgrid=True,
        gridcolor="lightgray",
    )
    if y_axis_max is not None:
        yaxis_config["range"] = [0, float(y_axis_max)]

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
        yaxis=yaxis_config,
        hovermode="closest",
        height=700,
        template="plotly_white",
        shapes=shapes,
        annotations=annotations,
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interactive_solubility_temp_{timestamp}.html"
    filepath = save_plot(
        fig,
        filename,
        "plotly",
        output_dir=output_dir,
        output_path=output_path,
        write_html_kwargs={"config": config},
    )

    output = "Interactive Solubility vs Temperature Visualization Created\n\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {', '.join(solvent_list)}\n"
    if temperature_min is not None and temperature_max is not None:
        output += f"Temperature range: {temperature_min}C - {temperature_max}C\n"
    elif temperature_min is not None:
        output += f"Temperature range: {temperature_min}C and above\n"
    elif temperature_max is not None:
        output += f"Temperature range: up to {temperature_max}C\n"
    if range_was_capped:
        output += f"Requested upper temperature {requested_t_end}C was capped at {SENSITIVITY_EXTRAPOLATION_MAX_C:.0f}C.\n"
    if y_axis_max is not None:
        output += f"Y-axis range: 0-{float(y_axis_max):g}%\n"
    basis_note = temperature_basis_note(t_end)
    if basis_note:
        output += f"Temperature basis: {basis_note}\n"
    if extrapolated_points:
        output += f"Extrapolated points: {extrapolated_points}\n"
    if sensitivity_points:
        output += "180-200C points are sensitivity-only screening data, not validated operating recommendations.\n"
    output += f"Data points: {total_points}\n\n"

    output += "## Interactive Features:\n"
    output += "- **Click legend items** to show/hide individual curves\n"
    output += "- **Drag the range slider** below the plot to zoom into temperature ranges\n"
    output += "- **Hover over points** to see exact values\n"
    output += "- **Use toolbar** to zoom, pan, reset, or download as PNG\n"
    output += "- **Double-click legend** to isolate a single curve\n\n"

    output += f"{_get_plot_url(filepath)}\n"

    gc.collect()
    return json_tool_success(
        output,
        tool_name="plot_solubility_vs_temperature_interactive",
        polymers=polymer_list,
        solvents=solvent_list,
        temperature_min_c=t_start,
        temperature_max_c=t_end,
        requested_temperature_max_c=requested_t_end,
        range_was_capped=range_was_capped,
        data_points=total_points,
        extrapolated_points=extrapolated_points,
        sensitivity_points=sensitivity_points,
        y_axis_max=y_axis_max,
        plot_filepath=filepath,
        output_dir=output_dir,
        output_path=output_path,
        plot_url=_get_plot_url(filepath),
    )


@safe_tool_wrapper(structured_output=True)
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
    annot_fontsize = 7 if n_cells <= 50 else 6 if n_cells <= 100 else 5

    colors_low = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6"]
    colors_high = ["#4292c6", "#2171b5", "#08519c", "#08306b"]
    cmap = LinearSegmentedColormap.from_list(
        "solubility_emphasis", colors_low + colors_high, N=256,
    )

    _apply_pub_style()
    fig, ax = plt.subplots(
        figsize=(
            max(7.0, len(pivot_df.columns) * 0.35),
            max(3.5, len(pivot_df) * 0.5),
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

    ax.set_xlabel("Solvent")
    ax.set_ylabel("Polymer")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels())

    plt.tight_layout()
    from strap.tools._helpers import descriptive_plot_name
    _polymers = list(pivot_df.index)
    _solvents = list(pivot_df.columns)
    plot_name = descriptive_plot_name("selectivity_heatmap", polymers=_polymers, solvents=_solvents)
    filepath = save_plot(fig, plot_name, "matplotlib")

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


@safe_tool_wrapper(structured_output=True)
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
        curve = get_solubility_curve(poly, solvent, 25.0, 250.0, 5.0)
        if curve:
            curves[poly] = {
                "temps": [pt["temperature"] for pt in curve],
                "sols": [pt["solubility"] for pt in curve],
            }

    if not curves:
        return f"No solubility data found for any polymer in {solvent}."

    _apply_pub_style()
    fig = plt.figure(figsize=(7.0, 5.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

    colors_others = [_PUB_COLORS[i + 1] for i in range(len(comp_list))]

    # Panel 1: Solubility curves
    ax1 = fig.add_subplot(gs[0, 0])
    if target_polymer in curves:
        ax1.plot(
            curves[target_polymer]["temps"], curves[target_polymer]["sols"],
            "o-", color=_PUB_COLORS[0], linewidth=1.2, markersize=3, label=target_polymer,
        )

    for i, comp in enumerate(comp_list):
        if comp in curves:
            ax1.plot(
                curves[comp]["temps"], curves[comp]["sols"],
                "s--", color=colors_others[i], linewidth=1.0, markersize=3, label=comp,
            )

    ax1.set_xlabel("Temperature (\u00b0C)")
    ax1.set_ylabel("Solubility (%)")
    ax1.set_title(f"Solubility in {solvent}")
    ax1.legend(frameon=True, edgecolor="none", facecolor="white", framealpha=0.8)
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
                linewidth=1.0, markersize=3, label=f"vs {comp}",
            )

        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.axhline(y=10, color=_PUB_COLORS[2], linestyle=":", alpha=0.7, label="Good selectivity (10%)")

    ax2.set_xlabel("Temperature (\u00b0C)")
    ax2.set_ylabel("Selectivity (%)")
    ax2.set_title("Selectivity vs Temperature")
    ax2.legend(frameon=True, edgecolor="none", facecolor="white", framealpha=0.8)
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

    ax3.set_ylabel("Separation Feasibility")
    ax3.set_title("Separation Window")
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
        0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=_PUB_FONTSIZE,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    plt.tight_layout()
    from strap.tools._helpers import descriptive_plot_name
    plot_name = descriptive_plot_name("multi_panel", polymers=all_polymers, solvents=[solvent])
    filepath = save_plot(fig, plot_name, "matplotlib")

    output = "Multi-Panel Analysis Created\n\n"
    output += f"Target: {target_polymer}\n"
    output += f"Solvent: {solvent}\n"
    output += f"Comparisons: {', '.join(comp_list)}\n\n"

    if good_separation_temps:
        output += f"**Separation possible at:** {', '.join([f'{int(t)}C' for t in good_separation_temps])}\n"

    output += f"\n{_get_plot_url(filepath)}"

    return json_tool_success(
        output,
        tool_name="plot_solubility_vs_temperature_interactive",
        polymers=polymer_list,
        solvents=solvent_list,
        temperature_min_c=t_start,
        temperature_max_c=t_end,
        requested_temperature_max_c=requested_t_end,
        range_was_capped=range_was_capped,
        data_points=total_points,
        extrapolated_points=extrapolated_points,
        sensitivity_points=sensitivity_points,
        plot_filepath=filepath,
        plot_url=_get_plot_url(filepath),
    )


@safe_tool_wrapper(structured_output=True)
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

    _apply_pub_style()
    fig = plt.figure(figsize=(7.0, 5.5))

    # Panel 1: Grouped bar chart
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(solvents))
    width = 0.8 / len(polymer_list)
    colors = [_PUB_COLORS[i % len(_PUB_COLORS)] for i in range(len(polymer_list))]

    for i, polymer in enumerate(polymer_list):
        poly_data = df[df["polymer"] == polymer]
        values = []
        for sv in solvents:
            sol_data = poly_data[poly_data["solvent"] == sv]["avg_sol"]
            values.append(sol_data.values[0] if len(sol_data) > 0 else 0)
        ax1.bar(
            x + i * width, values, width, label=polymer,
            color=colors[i], edgecolor="black", linewidth=0.3,
        )

    ax1.set_xlabel("Solvent")
    ax1.set_ylabel("Solubility (%)")
    ax1.set_title(f"Solubility at {temperature}\u00b0C")
    ax1.set_xticks(x + width * (len(polymer_list) - 1) / 2)
    short_labels = [s[:15] + ".." if len(s) > 15 else s for s in solvents]
    ax1.set_xticklabels(short_labels, rotation=55, ha="right")
    ax1.legend(frameon=True, edgecolor="none", facecolor="white", framealpha=0.8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    pivot = df.pivot(index="polymer", columns="solvent", values="avg_sol")
    n_cells = pivot.shape[0] * pivot.shape[1]
    annot_size = 7 if n_cells <= 20 else 6 if n_cells <= 40 else 5
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax2,
        annot_kws={"size": annot_size}, linewidths=0.5,
        cbar_kws={"label": "Solubility (%)", "shrink": 0.8},
    )
    ax2.set_title("Solubility Heatmap")
    ax2.set_xlabel("Solvent")
    ax2.set_ylabel("Polymer")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_yticklabels(ax2.get_yticklabels())

    # Panel 3: Box plot
    ax3 = fig.add_subplot(2, 2, 3)
    data_for_box = [df[df["polymer"] == p]["avg_sol"].values for p in polymer_list]
    bp = ax3.boxplot(data_for_box, labels=polymer_list, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    ax3.set_xlabel("Polymer")
    ax3.set_ylabel("Solubility Distribution (%)")
    ax3.set_title("Solubility Distribution")
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
        0.1, 0.85, ranking_text, transform=ax4.transAxes, fontsize=_PUB_FONTSIZE,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7, edgecolor="gray"),
    )

    plt.tight_layout()
    from strap.tools._helpers import descriptive_plot_name
    plot_name = descriptive_plot_name("comparison_dashboard", polymers=polymer_list)
    filepath = save_plot(fig, plot_name, "matplotlib")

    output = "Comparison Dashboard Created\n\n"
    output += f"Temperature: {temperature}C\n"
    output += f"Polymers: {', '.join(polymer_list)}\n"
    output += f"Solvents: {len(solvents)}\n\n"
    output += _get_plot_url(filepath)

    return output


@safe_tool_wrapper(structured_output=True)
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
    t_end = min(temperature_max, 250.0)

    n_pairs = len(pairs)
    n_cols = min(n_pairs, 3)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    _apply_pub_style()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows), squeeze=False)

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
                interp_temps, interp_sols, "-", color=_PUB_COLORS[0], linewidth=1.2,
                label="Interpolation", zorder=2,
            )
            has_data = True

        if sql_curve:
            sql_temps = [pt["temperature"] for pt in sql_curve]
            sql_sols = [pt["solubility"] for pt in sql_curve]
            ax.scatter(
                sql_temps, sql_sols, color=_PUB_COLORS[1], s=20, marker="o",
                edgecolors="black", linewidths=0.3, label="Database", zorder=3,
            )
            has_data = True

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

        ax.set_title(f"{polymer} in {solvent}")
        ax.set_xlabel("Temperature (\u00b0C)")
        ax.set_ylabel("Solubility (%)")
        ax.legend(frameon=True, edgecolor="none", facecolor="white", framealpha=0.8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_pairs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    from strap.tools._helpers import descriptive_plot_name
    _p = list(set(p for p, _ in pairs))
    _s = list(set(s for _, s in pairs))
    plot_name = descriptive_plot_name("interp_vs_sql", polymers=_p, solvents=_s)
    filepath = save_plot(fig, plot_name, "matplotlib")
    plt.close(fig)

    output = "Interpolation vs SQL Comparison Plot Created\n\n"
    output += f"Pairs: {len(pairs)}\n"
    output += f"Temperature range: {t_start}C - {t_end}C\n"
    for polymer, solvent in pairs:
        output += f"  - {polymer} in {solvent}\n"
    output += "\nBlue line = interpolation model (ln(S) = A + B/T + C·ln(T), modified Apelblat)\n"
    output += "Orange dots = raw database values\n"
    output += f"\n{_get_plot_url(filepath)}"

    gc.collect()
    return output


@safe_tool_wrapper(structured_output=True)
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

        _apply_pub_style()
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        scatter = ax.scatter(
            pdf_valid[property_lower],
            pdf_valid[y_lower],
            c=pdf_valid["solubility"],
            cmap="YlOrRd",
            s=25,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(f"Solubility in {polymer} (%)")

        for _, row in pdf_valid.iterrows():
            ax.annotate(
                row["solvent"],
                (row[property_lower], row[y_lower]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=_PUB_FONTSIZE - 2,
                alpha=0.8,
            )

        ax.set_xlabel(property_labels.get(property_lower, property_lower))
        ax.set_ylabel(property_labels.get(y_lower, y_lower))
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
                solvent_data_1d.append({
                    "solvent": solv,
                    "property_value": props[solv][property_lower],
                    "solubility": solubility_map.get(solv, 0),
                })

    if not solvent_data_1d:
        return (
            f"No {property_lower.upper()} data found for solvents that dissolve {polymer}.\n\n"
            f"This may be due to naming mismatches between databases. "
            f"Found {len(solvents_found)} solvents but none had {property_lower.upper()} data."
        )

    solvent_data_1d.sort(key=lambda x: x["property_value"])

    # Create visualization
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    if plot_type.lower() == "bar":
        names = [d["solvent"] for d in solvent_data_1d]
        values = [d["property_value"] for d in solvent_data_1d]
        solubilities = [d["solubility"] for d in solvent_data_1d]

        color_arr = plt.cm.YlOrRd(np.array(solubilities) / max(solubilities))
        bars = ax.bar(range(len(names)), values, color=color_arr, edgecolor="black", linewidth=0.3)

        for i, (bar, sol) in enumerate(zip(bars, solubilities)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height,
                f"{sol:.1f}%",
                ha="center", va="bottom", fontsize=_PUB_FONTSIZE - 2,
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(property_labels[property_lower])
        ax.set_xlabel("Solvent")

    else:  # scatter plot
        values = [d["property_value"] for d in solvent_data_1d]
        solubilities = [d["solubility"] for d in solvent_data_1d]
        names = [d["solvent"] for d in solvent_data_1d]

        ax.scatter(
            values, solubilities, s=25, alpha=0.6,
            edgecolors="black", linewidth=0.3, c=values, cmap="viridis",
        )

        for x, y, name in zip(values, solubilities, names):
            ax.annotate(
                name, (x, y), xytext=(3, 3), textcoords="offset points",
                fontsize=_PUB_FONTSIZE - 2, alpha=0.8,
            )

        ax.set_xlabel(property_labels[property_lower])
        ax.set_ylabel("Solubility (%)")
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


@safe_tool_wrapper(structured_output=True)
def plot_optimization_point_result(
    optimization_result_json: Dict[str, Any] | str | None = None,
    plot_title: Optional[str] = None,
    source_handoff_id: Optional[str] = None,
    output_stem: Optional[str] = None,
) -> str:
    """Plot a single-point optimization result as a compact optimization dashboard."""
    if source_handoff_id:
        payload = _load_optimization_payload_from_handoff(source_handoff_id)
    elif optimization_result_json is not None:
        payload = _coerce_optimization_payload(optimization_result_json)
    else:
        raise TypeError("Either optimization_result_json or source_handoff_id must be provided")

    if payload.get("analysis_type") != "point_optimum":
        raise ValueError("optimization_result_json must contain analysis_type='point_optimum'")

    washes = list(payload.get("optimal_washes") or [])
    if not washes:
        wash1 = _stringify_point_values(payload.get("wash1_selection"))
        wash2 = _stringify_point_values(payload.get("wash2_selection"))
        washes = wash1 + wash2
    stage1 = ", ".join(_stringify_point_values(payload.get("stage1_tech"))) or "n/a"
    stage2 = ", ".join(_stringify_point_values(payload.get("stage2_tech"))) or "n/a"
    stage3 = ", ".join(_stringify_point_values(payload.get("stage3_tech"))) or "n/a"
    feed_comp = payload.get("feed_composition") or {}
    feed_comp_text = ", ".join(
        f"{polymer} {float(frac) * 100:.1f}%"
        for polymer, frac in feed_comp.items()
        if frac is not None
    ) or "n/a"
    scenario = str(payload.get("scenario") or "n/a")
    circularity = payload.get("circularity_score", payload.get("ce_score"))
    profit = float(payload.get("profit") or 0.0)
    total_cost = float(payload.get("total_cost") or 0.0)
    emissions = float(payload.get("emissions") or 0.0)

    _apply_pub_style()
    fig = plt.figure(figsize=(12.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.2, 1.0], hspace=0.26, wspace=0.24)
    metrics_ax = fig.add_subplot(gs[0, 0])
    env_ax = fig.add_subplot(gs[0, 1])
    process_ax = fig.add_subplot(gs[1, :])

    money_labels = ["Profit", "Total Cost"]
    money_values = [profit, total_cost]
    metrics_ax.bar(money_labels, money_values, color=[_PUB_COLORS[0], _PUB_COLORS[2]], edgecolor="black", linewidth=0.5)
    metrics_ax.set_ylabel("USD")
    metrics_ax.set_title("Economic Metrics")
    metrics_ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(money_values):
        metrics_ax.text(idx, value, f"${value/1_000_000:.2f}M", ha="center", va="bottom", fontsize=_PUB_FONTSIZE - 1)

    env_labels = ["Emissions", "Circularity"]
    env_values = [emissions, float(circularity or 0.0)]
    env_colors = [_PUB_COLORS[3], _PUB_COLORS[1]]
    env_ax.bar(env_labels, env_values, color=env_colors, edgecolor="black", linewidth=0.5)
    env_ax.set_title("Environmental Metrics")
    env_ax.grid(True, axis="y", alpha=0.25)
    env_ax.set_ylabel("tCO2 / score")
    env_ax.text(0, env_values[0], f"{emissions:,.1f}", ha="center", va="bottom", fontsize=_PUB_FONTSIZE - 1)
    env_ax.text(1, env_values[1], f"{float(circularity or 0.0):.3f}", ha="center", va="bottom", fontsize=_PUB_FONTSIZE - 1)

    process_ax.axis("off")
    summary_lines = [
        f"Scenario: {scenario}",
        f"Feed composition: {feed_comp_text}",
        f"Stage 1: {stage1}",
        f"Stage 2: {stage2}",
        f"Stage 3: {stage3}",
        "Selected washes: " + (", ".join(washes) if washes else "none"),
    ]
    process_ax.text(
        0.01,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=_PUB_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#c5cdd7"},
    )

    fig.suptitle(plot_title or "Optimization Point Result", fontsize=_PUB_FONTSIZE + 1)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.08)

    from strap.tools._helpers import _slugify

    plot_stem = str(output_stem or "").strip()
    if not plot_stem and plot_title:
        plot_stem = _slugify(plot_title)
    if not plot_stem:
        plot_stem = "optimization_point_result"

    filepath = save_plot(fig, plot_stem)
    if not os.path.exists(filepath) and os.path.exists(f"{filepath}.png"):
        filepath = f"{filepath}.png"
    plt.close(fig)
    gc.collect()

    display = "Optimization Point Result Plot Created\n\n"
    display += f"Scenario: {scenario}\n"
    display += f"Selected washes: {', '.join(washes) if washes else 'none'}\n"
    display += f"Profit: ${profit:,.2f}\n"
    display += f"Total cost: ${total_cost:,.2f}\n"
    display += f"Emissions: {emissions:,.2f}\n"
    display += f"Circularity: {float(circularity or 0.0):.4f}\n"
    if source_handoff_id:
        display += f"Source handoff: {source_handoff_id}\n"
    if output_stem:
        display += f"Output stem: {output_stem}\n"
    display += f"\n{_get_plot_url(filepath)}"

    from strap.services.tool_response_service import json_tool_response

    data = {
        "plot_type": "optimization_point_result",
        "plot_paths": [filepath],
        "format": "png",
        "profit": profit,
        "total_cost": total_cost,
        "emissions": emissions,
        "circularity_score": float(circularity or 0.0),
        "optimal_washes": washes,
        "source_handoff_id": source_handoff_id,
        "output_stem": output_stem,
    }
    return json_tool_response(display, data, tool_name="plot_optimization_point_result")


@safe_tool_wrapper(structured_output=True)
def plot_optimization_pareto_front(
    pareto_result_json: Dict[str, Any] | str | None = None,
    color_by: str = "auto",
    plot_mode: str = "frontier_only",
    plot_title: Optional[str] = None,
    source_handoff_id: Optional[str] = None,
    output_stem: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Plot a cost-vs-emissions or cost-vs-circularity Pareto front from optimization output."""
    def _normalize_plot_mode(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "frontier_only"
        if text in {"frontier_only", "landscape"}:
            return text
        if "landscape" in text or "all feasible" in text or "all points" in text:
            return "landscape"
        if "frontier" in text:
            return "frontier_only"
        raise ValueError("plot_mode must be one of: frontier_only, landscape")

    def _coerce_payload(raw_payload: Dict[str, Any] | str | None) -> Dict[str, Any]:
        if isinstance(raw_payload, str):
            parsed = json.loads(raw_payload)
            if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], dict):
                return parsed["data"]
            if isinstance(parsed, dict):
                return parsed
            raise TypeError("pareto_result_json must decode to a mapping")
        if isinstance(raw_payload, dict):
            if "data" in raw_payload and isinstance(raw_payload["data"], dict):
                return raw_payload["data"]
            return raw_payload
        raise TypeError("pareto_result_json must be a JSON string or mapping")

    def _load_sidecar_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        sidecar_path = str(payload.get("pareto_payload_path") or "").strip()
        if not sidecar_path:
            return payload
        candidate_paths = [sidecar_path]
        if not os.path.isabs(sidecar_path):
            candidate_paths.append(os.path.join(os.getcwd(), sidecar_path))
        for candidate_path in candidate_paths:
            if not os.path.exists(candidate_path):
                continue
            with open(candidate_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict) and isinstance(loaded.get("data"), dict):
                loaded = loaded["data"]
            if isinstance(loaded, dict):
                return loaded
        return payload

    payload: Dict[str, Any]
    requested_plot_mode: Any = None
    requested_output_stem: Any = None
    if source_handoff_id:
        from strap.handoffs import get_handoff

        record = get_handoff(source_handoff_id)
        if record is None:
            raise ValueError(f"source_handoff_id '{source_handoff_id}' was not found")
        raw_payload = record.payload
        if isinstance(raw_payload, dict):
            requested_plot_mode = raw_payload.get("requested_plot_mode")
            requested_output_stem = raw_payload.get("requested_output_stem")
        if isinstance(raw_payload, dict) and isinstance(raw_payload.get("pareto_result_json"), dict):
            payload = raw_payload["pareto_result_json"]
        elif isinstance(raw_payload, dict) and isinstance(raw_payload.get("source_payload"), dict):
            payload = raw_payload["source_payload"]
        elif isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            raise TypeError("source handoff payload must be a mapping")
    elif pareto_result_json is not None:
        payload = _coerce_payload(pareto_result_json)
    else:
        raise TypeError("Either pareto_result_json or source_handoff_id must be provided")

    payload = _load_sidecar_payload(payload)

    if payload.get("analysis_type") != "pareto_front":
        raise ValueError("pareto_result_json must contain analysis_type='pareto_front'")

    points = payload.get("points") or []
    if not points:
        raise ValueError("pareto_result_json does not contain any Pareto points to plot")

    if requested_plot_mode is not None and plot_mode == "frontier_only":
        plot_mode = requested_plot_mode
    plot_mode = _normalize_plot_mode(plot_mode)
    if not output_stem and requested_output_stem:
        output_stem = str(requested_output_stem).strip() or None

    x_metric = str(payload.get("x_metric") or "total_cost")
    y_metric = str(payload.get("y_metric") or "emissions")
    y_column = "emissions" if y_metric == "emissions" else "circularity_score"
    if color_by in {"", "auto", None}:
        color_by = y_column
    if color_by not in {"profit", "emissions", "circularity_score", "total_cost"}:
        raise ValueError("color_by must be one of: auto, profit, emissions, circularity_score, total_cost")

    frontier_frame = pd.DataFrame(points)
    if "point_id" not in frontier_frame.columns:
        frontier_frame["point_id"] = np.arange(1, len(frontier_frame) + 1)
    if x_metric not in frontier_frame.columns:
        raise ValueError(f"Pareto points do not contain x_metric column '{x_metric}'")
    if y_column not in frontier_frame.columns:
        raise ValueError(f"Pareto points do not contain y_metric column '{y_column}'")
    frontier_frame = frontier_frame.sort_values(by=[x_metric, y_column], ascending=[True, True]).reset_index(drop=True)
    if color_by not in frontier_frame.columns:
        if y_column in frontier_frame.columns:
            color_by = y_column
        elif x_metric in frontier_frame.columns:
            color_by = x_metric
        else:
            color_by = "point_id"

    landscape_points = payload.get("landscape_points") or []
    if plot_mode == "landscape":
        all_points = payload.get("all_feasible_points") or []
        if landscape_points:
            seen = set()
            merged_points = []
            for source_point in list(all_points) + list(landscape_points):
                key = (
                    round(float(source_point.get(x_metric, 0.0) or 0.0), 6),
                    round(float(source_point.get(y_column, 0.0) or 0.0), 9),
                    tuple(source_point.get("stage1_tech") or []),
                    tuple(source_point.get("stage2_tech") or []),
                    tuple(source_point.get("stage3_tech") or []),
                    tuple(source_point.get("wash1_selection") or []),
                    tuple(source_point.get("wash2_selection") or []),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_points.append(source_point)
            all_points = merged_points
        if not all_points:
            all_points = points
    else:
        all_points = payload.get("all_feasible_points") or points
    all_frame = pd.DataFrame(all_points)
    if not all_frame.empty and "raw_point_id" not in all_frame.columns:
        all_frame["raw_point_id"] = np.arange(1, len(all_frame) + 1)
    if not all_frame.empty:
        if x_metric not in all_frame.columns or y_column not in all_frame.columns:
            all_frame = frontier_frame.copy()
        else:
            all_frame = all_frame.sort_values(by=[x_metric, y_column], ascending=[True, True]).reset_index(drop=True)
    color_label = {
        "profit": "Profit (USD)",
        "emissions": "Emissions (tCO2)",
        "circularity_score": "Circularity (0-1)",
        "total_cost": "Total Cost (USD)",
        "point_id": "Point Index",
    }[color_by]

    legend_entries = [
        _format_pareto_point_legend_entry(row, x_metric=x_metric, y_column=y_column)
        for _, row in frontier_frame.iterrows()
    ]

    _apply_pub_style()
    fig_h = max(4.8, 1.4 + 0.62 * len(frontier_frame))
    fig = plt.figure(figsize=(12.0, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.2], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    if plot_mode == "landscape" and not all_frame.empty:
        dominated_frame = all_frame.copy()
        if "point_status" in dominated_frame.columns:
            dominated_frame = dominated_frame[dominated_frame["point_status"] != "frontier"]
        if not dominated_frame.empty:
            ax.scatter(
                dominated_frame[x_metric],
                dominated_frame[y_column],
                c="#c7cdd4",
                s=26,
                edgecolors="#7a8896",
                linewidth=0.25,
                alpha=0.45,
                zorder=1,
            )

    scatter = ax.scatter(
        frontier_frame[x_metric],
        frontier_frame[y_column],
        c=frontier_frame[color_by],
        cmap="viridis",
        s=70,
        edgecolors="black",
        linewidth=0.4,
        alpha=0.9,
        zorder=3,
    )
    if len(frontier_frame) > 1:
        ax.plot(frontier_frame[x_metric], frontier_frame[y_column], color="#44617b", linewidth=1.2, alpha=0.7, zorder=2)

    for _, row in frontier_frame.iterrows():
        ax.annotate(
            f"P{int(row['point_id'])}",
            (row[x_metric], row[y_column]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=_PUB_FONTSIZE - 1,
        )

    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Emissions (tCO2)" if y_metric == "emissions" else "Circularity (0-1)")
    ax.set_title(plot_title or "Optimization Pareto Front")
    ax.grid(True, alpha=0.25)
    if len(frontier_frame) > 1 or color_by != "point_id":
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label(color_label)

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.set_title("Point Key", loc="left", fontsize=_PUB_FONTSIZE, pad=8)

    cmap = scatter.cmap
    norm = scatter.norm
    legend_fontsize = max(5.8, min(_PUB_FONTSIZE - 2, 8.0 - 0.06 * max(len(frontier_frame) - 6, 0)))
    y_positions = np.linspace(0.96, 0.08, len(frontier_frame))
    for y_pos, (_, row), entry in zip(y_positions, frontier_frame.iterrows(), legend_entries):
        marker_color = cmap(norm(row[color_by])) if color_by in frontier_frame.columns else cmap(0.5)
        legend_ax.scatter(
            [0.04],
            [y_pos],
            s=55,
            c=[marker_color],
            edgecolors="black",
            linewidth=0.4,
            clip_on=False,
        )
        legend_ax.text(
            0.12,
            y_pos,
            entry,
            va="center",
            ha="left",
            fontsize=legend_fontsize,
            linespacing=1.15,
        )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.1, wspace=0.14)

    from strap.tools._helpers import _slugify

    plot_stem = str(output_stem or "").strip()
    if not plot_stem and plot_title:
        plot_stem = _slugify(plot_title)
    if not plot_stem:
        plot_stem = f"optimization_pareto_{y_metric}"

    save_kwargs: dict[str, Any] = {}
    if output_dir is not None:
        save_kwargs["output_dir"] = output_dir
    if output_path is not None:
        save_kwargs["output_path"] = output_path
    filepath = save_plot(fig, plot_stem, **save_kwargs)
    if not os.path.exists(filepath) and os.path.exists(f"{filepath}.png"):
        filepath = f"{filepath}.png"
    plt.close(fig)
    gc.collect()

    display = "Optimization Pareto Front Plot Created\n\n"
    display += f"X metric: {x_metric}\n"
    display += f"Y metric: {y_metric}\n"
    display += f"Plot mode: {plot_mode}\n"
    display += f"Frontier points plotted: {len(frontier_frame)}\n"
    display += f"All feasible points available: {len(all_frame)}\n"
    display += f"Color scale: {color_by}\n"
    if source_handoff_id:
        display += f"Source handoff: {source_handoff_id}\n"
    if output_stem:
        display += f"Output stem: {output_stem}\n"
    display += f"\n{_get_plot_url(filepath)}"

    from strap.services.tool_response_service import json_tool_response

    data = {
        "plot_type": "optimization_pareto_front",
        "plot_paths": [filepath],
        "format": "png",
        "x_metric": x_metric,
        "y_metric": y_metric,
        "color_by": color_by,
        "plot_mode": plot_mode,
        "n_points": int(len(frontier_frame)),
        "n_frontier_points": int(len(frontier_frame)),
        "n_all_feasible_points": int(len(all_frame)),
        "n_landscape_points": int(len(landscape_points)),
        "point_legend": legend_entries,
        "pareto_payload_path": payload.get("pareto_payload_path"),
        "source_handoff_id": source_handoff_id,
        "output_stem": output_stem,
        "output_dir": output_dir,
        "output_path": output_path,
    }
    return json_tool_response(display, data, tool_name="plot_optimization_pareto_front")


@safe_tool_wrapper(structured_output=True)
def plot_optimization_pareto_slices(
    pareto_slices_json: Dict[str, Any] | str | None = None,
    plot_mode: str = "landscape",
    plot_title: Optional[str] = None,
    source_handoff_id: Optional[str] = None,
    output_stem: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Plot multiple fixed-composition optimization Pareto slices.

    Creates one standard Pareto PNG per slice and one combined comparison PNG
    with feasible landscape points faintly shown and each frontier highlighted.
    """

    def _coerce_payload(raw_payload: Dict[str, Any] | str | None) -> Dict[str, Any]:
        if isinstance(raw_payload, str):
            parsed = json.loads(raw_payload)
            if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict):
                return parsed["data"]
            if isinstance(parsed, dict):
                return parsed
            raise TypeError("pareto_slices_json must decode to a mapping")
        if isinstance(raw_payload, dict):
            if isinstance(raw_payload.get("data"), dict):
                return raw_payload["data"]
            return raw_payload
        raise TypeError("pareto_slices_json must be a JSON string or mapping")

    def _load_sidecar(payload: Dict[str, Any]) -> Dict[str, Any]:
        path_text = str(payload.get("pareto_slices_payload_path") or "").strip()
        if not path_text:
            return payload
        candidate_paths = [path_text]
        if not os.path.isabs(path_text):
            candidate_paths.append(os.path.join(os.getcwd(), path_text))
        for candidate_path in candidate_paths:
            if not os.path.exists(candidate_path):
                continue
            with open(candidate_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict) and isinstance(loaded.get("data"), dict):
                loaded = loaded["data"]
            if isinstance(loaded, dict):
                return loaded
        return payload

    if source_handoff_id:
        from strap.handoffs import get_handoff

        record = get_handoff(source_handoff_id)
        if record is None:
            raise ValueError(f"source_handoff_id '{source_handoff_id}' was not found")
        raw_payload = record.payload
        if isinstance(raw_payload, dict) and isinstance(raw_payload.get("pareto_slices_json"), dict):
            payload = raw_payload["pareto_slices_json"]
        elif isinstance(raw_payload, dict) and isinstance(raw_payload.get("source_payload"), dict):
            payload = raw_payload["source_payload"]
        elif isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            raise TypeError("source handoff payload must be a mapping")
    elif pareto_slices_json is not None:
        payload = _coerce_payload(pareto_slices_json)
    else:
        raise TypeError("Either pareto_slices_json or source_handoff_id must be provided")

    payload = _load_sidecar(_coerce_payload(payload))
    if payload.get("analysis_type") != "pareto_slices":
        raise ValueError("pareto_slices_json must contain analysis_type='pareto_slices'")

    slice_payloads = payload.get("slice_payloads") or []
    if not isinstance(slice_payloads, list):
        raise TypeError("pareto_slices payload must include a list `slice_payloads`")
    solved_payloads = [
        item
        for item in slice_payloads
        if isinstance(item, dict) and item.get("analysis_type") == "pareto_front" and item.get("points")
    ]
    if not solved_payloads:
        raise ValueError("No solved Pareto slice payloads were available to plot")

    x_metric = str(payload.get("x_metric") or solved_payloads[0].get("x_metric") or "total_cost")
    y_metric = str(payload.get("y_metric") or solved_payloads[0].get("y_metric") or "emissions")
    y_column = "emissions" if y_metric == "emissions" else "circularity_score"

    from strap.tools._helpers import _slugify

    base_stem = str(output_stem or "").strip() or "optimization_pareto_slices"
    per_slice_plot_paths: list[str] = []
    for index, slice_payload in enumerate(solved_payloads, start=1):
        label = str(slice_payload.get("slice_label") or f"slice_{index}")
        stem = f"{base_stem}_{_slugify(label)}"
        plot_kwargs: dict[str, Any] = {}
        if output_dir is not None:
            plot_kwargs["output_dir"] = output_dir
        raw_plot = plot_optimization_pareto_front(
            pareto_result_json=json.dumps(slice_payload, ensure_ascii=False, default=str),
            plot_mode=plot_mode,
            plot_title=f"{label} Pareto Landscape",
            output_stem=stem,
            **plot_kwargs,
        )
        try:
            plot_env = json.loads(raw_plot)
            for path in (plot_env.get("data") or {}).get("plot_paths") or []:
                if path:
                    per_slice_plot_paths.append(str(path))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, slice_payload in enumerate(solved_payloads):
        color = colors(index % 10)
        label = str(slice_payload.get("slice_label") or f"slice_{index + 1}")
        points = slice_payload.get("points") or []
        landscape_points = slice_payload.get("landscape_points") or []
        all_points = list(slice_payload.get("all_feasible_points") or [])
        if landscape_points:
            seen: set[tuple[float, float, str]] = set()
            merged: list[dict[str, Any]] = []
            for point in all_points + list(landscape_points):
                key = (
                    round(float(point.get(x_metric, 0.0) or 0.0), 6),
                    round(float(point.get(y_column, 0.0) or 0.0), 9),
                    "|".join(str(item) for item in point.get("wash1_selection") or []),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(point)
            all_points = merged
        if not all_points:
            all_points = points
        all_frame = pd.DataFrame(all_points)
        if not all_frame.empty and x_metric in all_frame.columns and y_column in all_frame.columns:
            ax.scatter(
                all_frame[x_metric] / 1_000_000.0,
                all_frame[y_column],
                s=22,
                color=color,
                alpha=0.16,
                edgecolors="none",
                zorder=1,
            )
        frontier_frame = pd.DataFrame(points)
        if frontier_frame.empty or x_metric not in frontier_frame.columns or y_column not in frontier_frame.columns:
            continue
        frontier_frame = frontier_frame.sort_values(by=[x_metric, y_column], ascending=[True, True])
        ax.plot(
            frontier_frame[x_metric] / 1_000_000.0,
            frontier_frame[y_column],
            color=color,
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
        )
        ax.scatter(
            frontier_frame[x_metric] / 1_000_000.0,
            frontier_frame[y_column],
            s=55,
            color=color,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
            label=f"{label} frontier ({len(frontier_frame)})",
        )

    ax.set_title(plot_title or "Cost vs Circularity Pareto Landscapes Across Feed Compositions")
    ax.set_xlabel("Annualized total cost (million USD/yr)")
    ax.set_ylabel("Emissions (tCO2)" if y_metric == "emissions" else "Circularity score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    save_kwargs: dict[str, Any] = {}
    if output_dir is not None:
        save_kwargs["output_dir"] = output_dir
    if output_path is not None:
        save_kwargs["output_path"] = output_path
    combined_path = save_plot(fig, base_stem, **save_kwargs)
    if not os.path.exists(combined_path) and os.path.exists(f"{combined_path}.png"):
        combined_path = f"{combined_path}.png"
    plt.close(fig)
    gc.collect()

    plot_paths = [combined_path] + per_slice_plot_paths
    display = "Optimization Pareto Slice Plots Created\n\n"
    display += f"Slices plotted: {len(solved_payloads)}\n"
    display += f"Combined comparison: {_get_plot_url(combined_path)}\n"
    display += f"Per-slice plots: {len(per_slice_plot_paths)}\n"
    if source_handoff_id:
        display += f"Source handoff: {source_handoff_id}\n"
    if output_stem:
        display += f"Output stem: {output_stem}\n"

    from strap.services.tool_response_service import json_tool_response

    data = {
        "plot_type": "optimization_pareto_slices",
        "plot_paths": plot_paths,
        "combined_plot_path": combined_path,
        "per_slice_plot_paths": per_slice_plot_paths,
        "format": "png",
        "x_metric": x_metric,
        "y_metric": y_metric,
        "plot_mode": plot_mode,
        "n_slices": len(solved_payloads),
        "pareto_slices_payload_path": payload.get("pareto_slices_payload_path"),
        "source_handoff_id": source_handoff_id,
        "output_stem": output_stem,
        "output_dir": output_dir,
        "output_path": output_path,
    }
    return json_tool_response(display, data, tool_name="plot_optimization_pareto_slices")
