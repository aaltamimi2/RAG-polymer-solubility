"""Configuration helpers for typed planning enforcement."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, field_validator

from strap.planning.models import PlanningModel


PlannerMode = Literal["off", "shadow", "enforce_selected", "enforce"]
DEFAULT_SELECTED_ENFORCEMENT_ARTIFACTS = frozenset({
    "separation_dp_state_map",
    "separation_tree_plot",
    "separation_selectivity_heatmap",
    "solvent_safety_card",
    "solvent_safety_comparison",
    "hsp_single_pair_summary",
    "hsp_red_heatmap",
    "biosteam_tea_lca_result",
    "biosteam_tea_lca_plot",
    "optimization_pareto_front",
    "optimization_pareto_landscape",
    "optimization_pareto_plot",
    "optimization_pareto_slices",
    "optimization_pareto_slices_plot",
})
DEFAULT_SELECTED_ENFORCEMENT_WORKFLOWS = frozenset({"routed_optimization", "routed_optimization_slices"})


class PlannerConfig(PlanningModel):
    mode: PlannerMode = "off"
    selected_enforcement_artifacts: set[str] = Field(default_factory=set)
    selected_enforcement_workflows: set[str] = Field(default_factory=set)

    @field_validator("selected_enforcement_artifacts", "selected_enforcement_workflows", mode="before")
    @classmethod
    def _normalize_csv_set(cls, value: object) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {
                item.strip()
                for item in value.split(",")
                if item.strip()
            }
        if isinstance(value, (set, list, tuple, frozenset)):
            return {str(item).strip() for item in value if str(item).strip()}
        raise TypeError("expected comma-separated string or iterable of strings")


def get_typed_planner_mode(env: dict[str, str] | None = None) -> PlannerMode:
    source = env if env is not None else os.environ
    raw = source.get("DISSOLVE_TYPED_PLANNER", "off").strip().lower()
    if raw not in {"off", "shadow", "enforce_selected", "enforce"}:
        raise ValueError(f"Invalid DISSOLVE_TYPED_PLANNER mode: {raw!r}")
    return raw  # type: ignore[return-value]


def get_selected_enforcement_artifacts(env: dict[str, str] | None = None) -> set[str]:
    source = env if env is not None else os.environ
    if "DISSOLVE_TYPED_PLANNER_ENFORCE_ARTIFACTS" not in source:
        return set(DEFAULT_SELECTED_ENFORCEMENT_ARTIFACTS)
    return PlannerConfig._normalize_csv_set(  # type: ignore[attr-defined]
        source.get("DISSOLVE_TYPED_PLANNER_ENFORCE_ARTIFACTS", "")
    )


def get_selected_enforcement_workflows(env: dict[str, str] | None = None) -> set[str]:
    source = env if env is not None else os.environ
    if "DISSOLVE_TYPED_PLANNER_ENFORCE_WORKFLOWS" not in source:
        return set(DEFAULT_SELECTED_ENFORCEMENT_WORKFLOWS)
    return PlannerConfig._normalize_csv_set(  # type: ignore[attr-defined]
        source.get("DISSOLVE_TYPED_PLANNER_ENFORCE_WORKFLOWS", "")
    )


def get_planner_config(env: dict[str, str] | None = None) -> PlannerConfig:
    return PlannerConfig(
        mode=get_typed_planner_mode(env),
        selected_enforcement_artifacts=get_selected_enforcement_artifacts(env),
        selected_enforcement_workflows=get_selected_enforcement_workflows(env),
    )
