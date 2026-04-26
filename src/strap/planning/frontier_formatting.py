"""Compact formatting helpers for optimizer Pareto payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set | frozenset):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _point_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("points", "frontier_points", "pareto_points"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _display_metric_key(metric: Any, point: Mapping[str, Any]) -> str:
    metric_text = str(metric or "").strip()
    if metric_text in point:
        return metric_text
    if metric_text.lower() in {"circularity", "ce"} and "circularity_score" in point:
        return "circularity_score"
    if metric_text.lower() == "cost" and "total_cost" in point:
        return "total_cost"
    return metric_text or "value"


def _metric_value(point: Mapping[str, Any], metric: str) -> Any:
    if metric in point:
        return point.get(metric)
    if metric == "circularity_score":
        return point.get("circularity_score") or point.get("CE")
    return None


def _format_scalar(value: Any) -> str:
    if value in (None, "", [], {}):
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.3g}"
        return f"{value:.3g}"
    text = str(value)
    return text if len(text) <= 72 else text[:69] + "..."


def _join_values(values: Any, *, max_items: int = 3) -> str:
    items = [str(item) for item in _as_list(values) if str(item).strip()]
    if not items:
        return "-"
    rendered = ", ".join(items[:max_items])
    if len(items) > max_items:
        rendered += f", +{len(items) - max_items}"
    return _format_scalar(rendered)


def _stage_values(point: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("stage1_tech", "stage2_tech", "stage3_variants", "stage3_tech"):
        for value in _as_list(point.get(key)):
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
    return values


def _label_stage_value(value: str) -> str:
    aliases = {"lf": "lf/landfill", "wte": "wte", "mech": "mechanical"}
    return aliases.get(value.lower(), value)


def _stage_label(point: Mapping[str, Any]) -> str:
    values = [_label_stage_value(value) for value in _stage_values(point)]
    return _join_values(values, max_items=4)


def _route_label(point: Mapping[str, Any]) -> str:
    route = point.get("route_id") or point.get("matched_route_id") or point.get("selection_origin")
    return _format_scalar(route)


def _wash_label(point: Mapping[str, Any]) -> str:
    wash1 = _join_values(point.get("wash1_selection"))
    wash2 = _join_values(point.get("wash2_selection"))
    if wash1 == "-" and wash2 == "-":
        return "-"
    if wash1 == "-":
        return f"wash2={wash2}"
    if wash2 == "-":
        return f"wash1={wash1}"
    return f"wash1={wash1}; wash2={wash2}"


def pareto_frontier_count(payload: Mapping[str, Any]) -> int:
    """Return the authoritative frontier count from payload fields."""
    explicit = payload.get("n_points_feasible")
    if isinstance(explicit, int):
        return explicit
    if isinstance(explicit, float) and explicit.is_integer():
        return int(explicit)
    return len(_point_rows(payload))


def pareto_landscape_count(payload: Mapping[str, Any]) -> int:
    """Return the feasible/landscape point count from payload fields."""
    for key in ("landscape_points", "all_feasible_points", "epsilon_sweep_points"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    explicit = payload.get("n_points_raw_feasible")
    if isinstance(explicit, int):
        return explicit
    if isinstance(explicit, float) and explicit.is_integer():
        return int(explicit)
    return 0


def pareto_metric_labels(payload: Mapping[str, Any], point: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """Return display keys for x/y metrics, mapped to actual point fields."""
    sample = point or (_point_rows(payload)[0] if _point_rows(payload) else {})
    x_key = _display_metric_key(payload.get("x_metric") or "total_cost", sample)
    y_key = _display_metric_key(payload.get("y_metric") or "circularity", sample)
    return x_key, y_key


def pareto_frontier_table_lines(
    payload: Mapping[str, Any],
    *,
    max_points: int = 6,
) -> list[str]:
    """Render a compact markdown table for frontier points."""
    points = _point_rows(payload)
    if not points:
        return []
    x_key, y_key = pareto_metric_labels(payload, points[0])
    lines = [
        "Frontier points:",
        f"| # | {x_key} | {y_key} | stages | route | washes |",
        "|---:|---:|---:|---|---|---|",
    ]
    for index, point in enumerate(points[:max_points], start=1):
        point_id = point.get("point_id") or point.get("raw_point_id") or index
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_scalar(point_id),
                    _format_scalar(_metric_value(point, x_key)),
                    _format_scalar(_metric_value(point, y_key)),
                    _stage_label(point),
                    _route_label(point),
                    _wash_label(point),
                ]
            )
            + " |"
        )
    if len(points) > max_points:
        lines.append(f"... {len(points) - max_points} more frontier point(s) omitted from this compact view.")
    return lines


def pareto_stage3_mentions(payload: Mapping[str, Any]) -> list[str]:
    """Return distinct stage options appearing on frontier points.

    The historical name is kept for compatibility with callers introduced
    before the table rendered stage 1/2/3 together.
    """
    values: list[str] = []
    for point in _point_rows(payload):
        for text in _stage_values(point):
            if text not in values:
                values.append(text)
    return values
