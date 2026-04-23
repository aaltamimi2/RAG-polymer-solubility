"""
Multi-layer Plastic Waste Optimization Tool.
Integrates BioSTEAM simulations with Pyomo superstructure optimization.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pyomo.environ as pyo

from strap.solvent_registry import resolve_to_biosteam
from strap.tools._helpers import safe_tool_wrapper
from strap.services.biosteam_service import json_tool_response, json_tool_error, build_single_config
from strap.vendor.biosteam_runner import run_single_simulation
from strap.waste_management.data_loader import (
    STRAP_UNIT_COLS,
    get_optimizer_default_sets,
    load_all_data,
)
from strap.waste_management.model import build_model
from strap.waste_management.solver import (
    pareto_cost_vs_ce,
    pareto_cost_vs_emissions,
    solve_single,
)

logger = logging.getLogger(__name__)

_NUMERIC_WORKBOOK_COLUMNS = tuple(STRAP_UNIT_COLS.values())
_OPTIMIZATION_POLYMER_ALIASES = {
    "PE": "PE",
    "HDPE": "PE",
    "LDPE": "PE",
    "EVOH": "EVOH",
}
_VALID_ROUTE_POOL_MODES = {"exact", "slot_independent"}
_COEFFICIENT_SOURCE_COLUMN = "coefficient_source"
_BIOSTEAM_SIM_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_BIOSTEAM_BATCH_PARALLEL = 3
_BIOSTEAM_TIMEOUT_SEC = 45
_BIOSTEAM_RUNTIME_DENYLIST: dict[tuple[str, str], dict[str, str]] = {
    ("PE", "hexamethylphosphoramide"): {
        "failure_class": "undefined_chemical_alias",
        "reason": "thermosteam.exceptions.UndefinedChemicalAlias: <BoilerTurbogenerator: BT> 'P4O10'",
    },
    ("EVOH", "hexamethylphosphoramide"): {
        "failure_class": "undefined_chemical_alias",
        "reason": "thermosteam.exceptions.UndefinedChemicalAlias: <BoilerTurbogenerator: BT> 'P4O10'",
    },
    ("PE", "Tetrahydropyran"): {
        "failure_class": "vapor_pressure_extrapolation_failure",
        "reason": "RuntimeError: <System: PE/Tetrahydropyran> <HXutility: H2> Failed to extrapolate vapor pressure method 'LANDOLT'",
    },
}
_SCIP_CONSTRAINED_OPTION_LADDER: tuple[dict[str, Any] | None, ...] = (
    None,
    {"presolving/maxrounds": 0},
    {"presolving/maxrounds": 0, "randomization/randomseedshift": 1},
)


def _parse_csv_list(values: list[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_items = values.split(",")
    elif isinstance(values, list):
        raw_items = values
    else:
        raise TypeError("candidate solvents must be a comma-separated string or list of strings")

    items: list[str] = []
    seen: set[str] = set()
    for value in raw_items:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def _normalize_optimization_polymer(polymer: Any) -> str | None:
    if polymer is None:
        return None
    text = str(polymer).strip().upper()
    if not text:
        return None
    return _OPTIMIZATION_POLYMER_ALIASES.get(text)


def _parse_polymer_solvent_filters(
    polymer_solvent_filters_json: dict[str, Any] | str | None,
) -> dict[str, list[str]]:
    if polymer_solvent_filters_json in (None, "", {}):
        return {}

    if isinstance(polymer_solvent_filters_json, str):
        payload = json.loads(polymer_solvent_filters_json)
    elif isinstance(polymer_solvent_filters_json, dict):
        payload = polymer_solvent_filters_json
    else:
        raise TypeError("polymer_solvent_filters_json must be a JSON string or mapping")

    if not isinstance(payload, dict):
        raise TypeError("polymer_solvent_filters_json must decode to a mapping")

    parsed: dict[str, list[str]] = {}
    for polymer, values in payload.items():
        normalized_polymer = _normalize_optimization_polymer(polymer)
        if normalized_polymer is None:
            continue
        solvents = _parse_csv_list(values)
        if solvents:
            parsed[normalized_polymer] = solvents
    return parsed


def _parse_stage_candidates(
    stage_candidates_json: dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    if stage_candidates_json in (None, "", {}):
        return None
    if isinstance(stage_candidates_json, str):
        payload = json.loads(stage_candidates_json)
    elif isinstance(stage_candidates_json, dict):
        payload = dict(stage_candidates_json)
    else:
        raise TypeError("stage_candidates_json must be a JSON string or mapping")
    if not isinstance(payload, dict):
        raise TypeError("stage_candidates_json must decode to a mapping")
    return payload


def _normalize_route_pool_mode(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in _VALID_ROUTE_POOL_MODES:
        raise ValueError(
            f"Unsupported route_pool_mode '{value}'. Supported values: "
            f"{', '.join(sorted(_VALID_ROUTE_POOL_MODES))}."
        )
    return text


def _extract_route_candidates(
    stage_candidates_json: dict[str, Any] | str | None,
) -> list[dict[str, Any]]:
    """Return the route_candidates field from a typed stage-candidates payload.

    Each route preserves polymer→solvent coupling from the upstream DP
    separation planner. Unknown or malformed routes are filtered out; the
    caller treats an empty list as "no route enforcement available, fall back
    to per-polymer filters."
    """
    payload = _parse_stage_candidates(stage_candidates_json)
    if payload is None:
        return []
    raw_routes = payload.get("route_candidates") or []
    routes: list[dict[str, Any]] = []
    for entry in raw_routes:
        if not isinstance(entry, dict):
            continue
        mapping_raw = entry.get("polymer_solvent_map") or {}
        if not isinstance(mapping_raw, dict):
            continue
        mapping: dict[str, str] = {}
        for polymer, solvent in mapping_raw.items():
            polymer_key = _normalize_optimization_polymer(polymer)
            solvent_str = str(solvent or "").strip()
            if polymer_key is None or not solvent_str:
                continue
            mapping[polymer_key] = solvent_str
        if not mapping:
            continue
        routes.append(
            {
                "route_id": str(entry.get("route_id") or f"route_{len(routes) + 1}"),
                "rank": entry.get("rank"),
                "sequence": list(entry.get("sequence") or []),
                "source": str(entry.get("source") or ""),
                "polymer_solvent_map": mapping,
            }
        )
    return routes


def _derive_route_pool_mode_from_stage_candidates(
    stage_candidates_json: dict[str, Any] | str | None,
) -> str | None:
    payload = _parse_stage_candidates(stage_candidates_json)
    if payload is None:
        return None
    return _normalize_route_pool_mode(payload.get("route_pool_mode"))


def _derive_filters_from_stage_candidates(
    stage_candidates_json: dict[str, Any] | str | None,
) -> tuple[dict[str, list[str]], list[str], str | None, str | None]:
    payload = _parse_stage_candidates(stage_candidates_json)
    if payload is None:
        return {}, [], None, None

    polymer_filters: dict[str, list[str]] = {}
    global_candidates: list[str] = []
    seen_global: set[str] = set()
    for stage in payload.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        polymer = _normalize_optimization_polymer(stage.get("target_polymer"))
        if polymer is None:
            continue
        stage_candidates = stage.get("candidate_pairs") or []
        solvents: list[str] = []
        seen_solvents: set[str] = set()
        for pair in stage_candidates:
            if not isinstance(pair, dict):
                continue
            solvent = str(pair.get("solvent") or "").strip()
            if not solvent or solvent in seen_solvents:
                continue
            seen_solvents.add(solvent)
            solvents.append(solvent)
            if solvent not in seen_global:
                seen_global.add(solvent)
                global_candidates.append(solvent)
        if solvents:
            polymer_filters[polymer] = solvents

    return (
        polymer_filters,
        global_candidates,
        str(payload.get("constraint_mode")) if payload.get("constraint_mode") is not None else None,
        str(payload.get("fallback_policy")) if payload.get("fallback_policy") is not None else None,
    )


def _normalize_solvent_key(solvent: Any) -> str:
    text = str(solvent or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _build_available_solvent_index(available: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for solvent in available:
        actual = str(solvent).strip()
        if not actual:
            continue
        variants = {
            actual,
            resolve_to_biosteam(actual) or "",
            actual.replace("_", " "),
            actual.replace("-", " "),
        }
        for variant in variants:
            key = _normalize_solvent_key(variant)
            if key:
                index.setdefault(key, actual)
    return index


def _canonicalize_requested_solvents(requested: list[str], available: list[str]) -> list[str]:
    available_index = _build_available_solvent_index(available)
    allowed: list[str] = []
    seen: set[str] = set()
    for solvent in requested:
        candidates = [
            str(solvent or "").strip(),
            resolve_to_biosteam(str(solvent or "").strip()) or "",
        ]
        matched: str | None = None
        for candidate in candidates:
            key = _normalize_solvent_key(candidate)
            if key and key in available_index:
                matched = available_index[key]
                break
        if matched and matched not in seen:
            seen.add(matched)
            allowed.append(matched)
    return allowed


def _build_materialization_allowlist(
    *,
    candidate_solvents: list[str] | str | None = None,
    polymer_solvent_filters_json: dict[str, Any] | str | None = None,
) -> dict[tuple[str, str], list[str]] | None:
    global_candidates = _parse_csv_list(candidate_solvents)
    polymer_filters = _parse_polymer_solvent_filters(polymer_solvent_filters_json)
    if not global_candidates and not polymer_filters:
        return None

    default_sets = get_optimizer_default_sets()
    allowlist: dict[tuple[str, str], list[str]] = {}

    pe_requested = polymer_filters.get("PE") or global_candidates
    if pe_requested:
        pe_allowed = _canonicalize_requested_solvents(pe_requested, default_sets["S_PE"])
        allowlist[("Wash 1", "PE")] = list(pe_allowed)
        allowlist[("Wash 2", "PE")] = list(pe_allowed)

    evoh_requested = polymer_filters.get("EVOH") or global_candidates
    if evoh_requested:
        allowlist[("Wash 1", "EVOH")] = _canonicalize_requested_solvents(
            evoh_requested,
            default_sets["S_EV1"],
        )
        allowlist[("Wash 2", "EVOH")] = _canonicalize_requested_solvents(
            evoh_requested,
            default_sets["S_EV2"],
        )

    return allowlist


def _materialize_optimizer_workbook_rows(
    df: pd.DataFrame,
    *,
    allowed_solvents_by_slot: dict[tuple[str, str], list[str]] | None = None,
) -> pd.DataFrame:
    """Expand the workbook sheet to the shared optimizer solvent catalog."""

    expanded_df = df.copy()
    if _COEFFICIENT_SOURCE_COLUMN not in expanded_df.columns:
        expanded_df[_COEFFICIENT_SOURCE_COLUMN] = "workbook_baseline"
    else:
        expanded_df[_COEFFICIENT_SOURCE_COLUMN] = (
            expanded_df[_COEFFICIENT_SOURCE_COLUMN].fillna("workbook_baseline").astype(str)
        )
    default_sets = get_optimizer_default_sets()
    targets = (
        ("Wash 1", "PE", default_sets["S_PE"]),
        ("Wash 2", "PE", default_sets["S_PE"]),
        ("Wash 1", "EVOH", default_sets["S_EV1"]),
        ("Wash 2", "EVOH", default_sets["S_EV2"]),
    )

    additions: list[pd.Series] = []
    for wash, polymer, solvents in targets:
        mask = expanded_df["Wash number"].eq(wash) & expanded_df["Polymer"].eq(polymer)
        template_rows = expanded_df.loc[mask]
        if template_rows.empty:
            logger.warning(
                "Unable to materialize optimizer rows for %s/%s because the template row is missing.",
                wash,
                polymer,
            )
            continue
        template = template_rows.iloc[0].copy()
        existing = {
            str(solvent).strip()
            for solvent in template_rows["Solvents"].dropna().astype(str)
            if str(solvent).strip()
        }
        candidate_solvents = (
            allowed_solvents_by_slot.get((wash, polymer), solvents)
            if allowed_solvents_by_slot is not None
            else solvents
        )
        for solvent in candidate_solvents:
            if solvent in existing:
                continue
            new_row = template.copy()
            new_row["Wash number"] = wash
            new_row["Polymer"] = polymer
            new_row["Solvents"] = solvent
            new_row[_COEFFICIENT_SOURCE_COLUMN] = "materialized_clone"
            for column in _NUMERIC_WORKBOOK_COLUMNS:
                if column in new_row.index:
                    new_row[column] = 0.0
            additions.append(new_row)

    if additions:
        expanded_df = pd.concat([expanded_df, pd.DataFrame(additions)], ignore_index=True)
    return expanded_df


def _solvent_filter_status(
    requested_filters: dict[str, list[str]],
    applied_filters: dict[str, list[str]],
    warnings: list[str],
) -> str:
    has_requested = any(requested_filters.get(key) for key in requested_filters)
    if not has_requested:
        return "not_requested"
    if applied_filters and warnings:
        return "partially_applied_with_fallback"
    if applied_filters:
        return "applied"
    if warnings:
        return "fallback_to_full_catalog"
    return "requested_no_effect"


def _build_optimization_infeasible_response(
    *,
    failure_reason: str,
    message: str,
    constraint_mode: str,
    fallback_policy: str,
    requested_filters: dict[str, list[str]],
    applied_filters: dict[str, list[str]],
    suggested_relaxation: str,
) -> str:
    payload = {
        "analysis_type": "infeasible",
        "schema_version": "1.0",
        "constraint_mode": constraint_mode,
        "fallback_policy": fallback_policy,
        "failure_reason": failure_reason,
        "message": message,
        "requested_candidate_pairs": requested_filters,
        "applied_candidate_pairs": applied_filters,
        "suggested_relaxation": suggested_relaxation,
        "success": False,
        "tool_name": "run_waste_management_optimization",
    }
    display = (
        "## Waste Optimization Infeasible\n\n"
        f"- **Reason:** {failure_reason}\n"
        f"- **Constraint mode:** {constraint_mode}\n"
        f"- **Fallback policy:** {fallback_policy}\n"
        f"- **Message:** {message}\n"
        f"- **Requested candidate pairs:** {requested_filters}\n"
        f"- **Applied candidate pairs:** {applied_filters}\n"
        f"- **Suggested relaxation:** {suggested_relaxation}\n"
    )
    return json_tool_response(display, payload)


def _apply_solvent_filters(
    df: pd.DataFrame,
    *,
    candidate_solvents: list[str] | str | None = None,
    polymer_solvent_filters_json: dict[str, Any] | str | None = None,
    constraint_mode: str | None = None,
    fallback_policy: str | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str], dict[str, list[str]]]:
    global_candidates = _parse_csv_list(candidate_solvents)
    polymer_filters = _parse_polymer_solvent_filters(polymer_solvent_filters_json)
    requested_filters = {
        "global": global_candidates,
        **polymer_filters,
    }
    if not global_candidates and not polymer_filters:
        return df.copy(), {}, [], requested_filters

    filtered_df = df.copy()
    warnings: list[str] = []
    applied_filters: dict[str, list[str]] = {}
    keep_mask = ~filtered_df["Polymer"].isin(["PE", "EVOH"])

    for polymer in ("PE", "EVOH"):
        polymer_mask = filtered_df["Polymer"].eq(polymer)
        if not polymer_mask.any():
            continue

        available = list(
            dict.fromkeys(
                str(solvent).strip()
                for solvent in filtered_df.loc[polymer_mask, "Solvents"].dropna().astype(str)
                if str(solvent).strip()
            )
        )
        requested = polymer_filters.get(polymer) or global_candidates
        allowed = _canonicalize_requested_solvents(requested, available)

        if requested and allowed:
            keep_mask |= polymer_mask & filtered_df["Solvents"].isin(allowed)
            applied_filters[polymer] = allowed
            continue

        if requested and not allowed and constraint_mode in {"fixed", "hard"} and fallback_policy == "fail_closed":
            return filtered_df.loc[filtered_df["Polymer"].isin([])].copy(), applied_filters, [
                f"No {polymer} solvent overlap between upstream shortlist and the shared optimization catalog under fail-closed semantics."
            ], requested_filters

        keep_mask |= polymer_mask
        if requested and not allowed:
            warnings.append(
                f"No {polymer} solvent overlap between upstream shortlist and the shared optimization catalog; "
                f"falling back to the full {polymer} candidate set."
            )

    return filtered_df.loc[keep_mask].copy(), applied_filters, warnings, requested_filters

# BioSTEAM mapping to Excel metrics. Where exact mappings aren't directly available in standard JSON output, 
# we scale based on capacities or use the primary metric (like GWP for all GHG).
def _map_biosteam_to_strap_row(strap_data_row, res_json, capacity_tons_yr):
    tea = res_json.get("tea", {})
    lca = res_json.get("lca", {})
    ops = res_json.get("operations", {})

    # Energy: MJ/kg -> MJ/yr
    energy_mj_kg = ops.get("total_energy_mj_per_kg") or 0
    strap_data_row["Total Energy Consumed [MJ/yr]"] = energy_mj_kg * capacity_tons_yr * 1000

    # GHG: kg CO2e/kg -> tons CO2e/yr
    gwp_kg = lca.get("gwp_kg_co2e_per_kg") or 0
    gwp_tons_yr = gwp_kg * capacity_tons_yr
    strap_data_row["GWP [tonCO2e/yr]"] = gwp_tons_yr
    strap_data_row["Total Direct GHG emissions [Scope 1] [metric tons CO2 equivalent [tCO2e/yr]]"] = gwp_tons_yr
    strap_data_row["Total Energy indirect GHG emissions (Scope 2) [metric tons CO2 equivalent (t CO2e/yr)]"] = 0

    # Water / Waste
    water_m3_yr = ops.get("water_consumed_m3_yr") or 0
    strap_data_row["Water consumed/discarded [m3/yr]"] = water_m3_yr
    waste_kg_yr = ops.get("waste_generated_kg_yr") or 0
    strap_data_row["Waste generated - Non Hazardous [kg/yr]"] = waste_kg_yr

    # Cost — guard all against None
    capex_usd = tea.get("tci_usd") or 0
    aoc_usd_yr = tea.get("aoc_usd_per_yr") or 0
    if capex_usd > 0:
        strap_data_row["CAPEX [USD/yr]"] = capex_usd / 10  # simple 10-yr annualisation
    if aoc_usd_yr > 0:
        strap_data_row["OPEX [USD/yr]"] = aoc_usd_yr

    # Toxicity
    strap_data_row["Human toxicity cancer [CTUh/yr]"] = (lca.get("htc_ctuh_per_kg") or 0) * capacity_tons_yr * 1000
    strap_data_row["Human toxicity non-cancer [CTUh/yr]"] = (lca.get("htnc_ctuh_per_kg") or 0) * capacity_tons_yr * 1000
    strap_data_row["Ecotoxicity [CTUe/yr]"] = (lca.get("etox_ctue_per_kg") or 0) * capacity_tons_yr * 1000

    return strap_data_row


def _validate_feed_inputs(
    feed: float,
    pe_fraction: float,
    pet_fraction: float,
    n6_fraction: float,
    evoh_fraction: float,
) -> None:
    assert feed > 0, "Feed must be greater than 0"
    total_frac = pe_fraction + pet_fraction + n6_fraction + evoh_fraction
    assert abs(total_frac - 1.0) < 0.01, f"Fractions must sum to 1.0, got {total_frac}"


def _get_scenario_config(
    scenario: str,
    feed: float,
    pe_fraction: float,
    pet_fraction: float,
    n6_fraction: float,
    evoh_fraction: float,
) -> tuple[str, dict[str, Any]]:
    scen_keys = {
        "A": {
            "other_sheet": "Othertech w TransportA",
            "distances": {"strap": 0, "lf": 9.2, "we": 151, "py": 1034, "gas_er": 0, "gas_h2": 2036, "gas_h2cc": 2036},
        },
        "B": {
            "other_sheet": "Othertech w TransportB",
            "distances": {"strap": 0, "lf": 9.2, "we": 151, "py": 76.1, "gas_er": 0, "gas_h2": 76.1, "gas_h2cc": 76.1},
        },
        "C": {
            "other_sheet": "Othertech w TransportA",
            "distances": {"strap": 0, "lf": 9.2, "we": 151, "py": 1034, "gas_er": 0, "gas_h2": 2036, "gas_h2cc": 2036},
        },
    }
    normalized_scenario = scenario if scenario in scen_keys else "A"
    config = {
        "Feed": feed,
        "PE_f": pe_fraction,
        "PET_f": pet_fraction,
        "N6_f": n6_fraction,
        "EV_f": evoh_fraction,
        "Cpe": 1173,
        "Cevoh": 8100,
        "Cwte": 259.57,
        "UB_energy": 6.26e7,
        "UB_ghg": 21303.35985408156,
        "UB_withdrawal": 14468.80855,
        "UB_waste": 1.92e6,
        "fc_t": 3.01,
        "vc_t": 0.07,
        "products_heat": 583.33,
        "products_electricity": 724.3693,
        "price_heat": 0.13,
        "price_elec": 0.0996,
        "Cgas_pw": 110,
        "ce_weights": {"energy": 0.20, "ghg": 0.20, "water": 0.20, "waste": 0.20, "subs": 0.20},
        "distances": scen_keys[normalized_scenario]["distances"],
    }
    return normalized_scenario, {"scenario": normalized_scenario, "other_sheet": scen_keys[normalized_scenario]["other_sheet"], "config": config}


def _classify_sim_failure(result: dict[str, Any] | None) -> tuple[str, str] | None:
    """Return ``(failure_class, reason)`` for a sim-level failure.

    BioSTEAM can return `success: True` while its TEA block is empty or null —
    that's a separate, softer failure mode: the row may still survive if the
    workbook baseline had non-zero CAPEX/OPEX that `_map_biosteam_to_strap_row`
    preserves (it only overwrites when the corresponding sim field is positive).
    Those rows are validated after the update via `_row_has_usable_economics`
    so baseline-only rows aren't dropped just because BioSTEAM couldn't
    compute an augmented TEA.
    """
    if not isinstance(result, dict):
        return ("sim_result_not_a_dict", "sim_result_not_a_dict")
    if result.get("success", False):
        return None

    failure_class = str(result.get("failure_class") or "").strip()
    if failure_class:
        return (failure_class, str(result.get("error") or result.get("reason") or failure_class))

    error_text = str(result.get("error") or "").strip()
    error_type = str(result.get("error_type") or "").strip()
    lower_error = error_text.lower()
    lower_type = error_type.lower()

    if "alias must start with a letter" in lower_error:
        return ("worker_alias_failure", error_text or "alias must start with a letter")
    if "undefinedchemicalalias" in lower_error or "undefinedchemicalalias" in lower_type:
        return ("undefined_chemical_alias", error_text or error_type)
    if "timed out" in lower_error or "timeoutexpired" in lower_type:
        return ("timeout", error_text or error_type or "simulation timed out")
    if result.get("_sim_exception"):
        return ("simulation_exception", str(result["_sim_exception"]))
    if error_type:
        return (error_type, error_text or error_type)
    if error_text:
        return ("sim_success_false", error_text)
    return ("sim_success_false", "sim_success_false")


def _row_failure_class(reason: str | None) -> str | None:
    text = str(reason or "").strip()
    if not text:
        return None
    head, _, _tail = text.partition(":")
    return head or None


def _sanitize_failure_reason(reason: str | None) -> str:
    return str(reason or "").replace("\n", " ").strip()


def _build_runtime_denylist_result(polymer: str, solvent: str) -> dict[str, Any] | None:
    record = _BIOSTEAM_RUNTIME_DENYLIST.get((str(polymer), str(solvent)))
    if record is None:
        return None
    return {
        "success": False,
        "error_type": "RuntimeDenylist",
        "error": record["reason"],
        "failure_class": record["failure_class"],
        "polymer": polymer,
        "solvent": solvent,
    }


def _sim_crashed_reason(result: dict[str, Any] | None) -> str | None:
    failure = _classify_sim_failure(result)
    if failure is None:
        return None
    failure_class, reason = failure
    return f"{failure_class}:{_sanitize_failure_reason(reason)}"


def _failure_record(*, polymer: str, solvent: str, reason: str, source: str) -> dict[str, str]:
    return {
        "polymer": polymer,
        "solvent": solvent,
        "failure_class": _row_failure_class(reason) or "unknown_failure",
        "reason": _sanitize_failure_reason(reason),
        "source": source,
    }


def _skip_record(*, polymer: str, solvent: str, reason: str, source: str) -> dict[str, str]:
    return {
        "polymer": polymer,
        "solvent": solvent,
        "reason": reason,
        "source": source,
    }


def _runtime_denylist_applies(polymer: str, solvent: str) -> bool:
    return (str(polymer), str(solvent)) in _BIOSTEAM_RUNTIME_DENYLIST


def _row_has_usable_economics(row) -> tuple[bool, str | None]:
    """Validate that a strap row has non-zero economics after the sim update pass.

    A row with CAPEX=0 + OPEX=0 + GWP=0 would look like a free, zero-impact
    pathway to the MINLP and dominate any real alternative. We accept the row
    if at least two of the three post-update fields are positive — this admits
    rows where BioSTEAM augmented the workbook baseline AND rows where the
    workbook baseline alone carried the economics, while still dropping
    materialized ghost rows that were never populated from either source.
    """
    def _f(value) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    capex = _f(row.get("CAPEX [USD/yr]"))
    opex = _f(row.get("OPEX [USD/yr]"))
    gwp = _f(row.get("GWP [tonCO2e/yr]"))
    issues: list[str] = []
    if capex <= 0:
        issues.append("CAPEX<=0")
    if opex <= 0:
        issues.append("OPEX<=0")
    if gwp <= 0:
        issues.append("GWP<=0")
    if len(issues) >= 2:
        return False, "row_missing_economics:" + "|".join(issues)
    return True, None


def _row_has_sufficient_baseline_for_skip(row) -> bool:
    """Return True only for fully populated workbook-backed baseline rows."""
    def _f(value) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    return (
        _f(row.get("CAPEX [USD/yr]")) > 0
        and _f(row.get("OPEX [USD/yr]")) > 0
        and _f(row.get("GWP [tonCO2e/yr]")) > 0
    )


def _run_biosteam_updates(
    df: pd.DataFrame,
    capacity_pe: float,
    capacity_evoh: float,
    *,
    pe_fraction_pct: float,
    evoh_fraction_pct: float,
    prefer_baseline_when_sufficient: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, str]], list[dict[str, str]]]:
    """Overwrite workbook metrics with BioSTEAM sim outputs.

    Rows whose BioSTEAM sim fails are dropped rather than kept with zeroed or
    stale metrics, so the MINLP never treats a failed sim as a free, zero-impact
    pathway. A sim is treated as failed if it raised, returned `success: False`,
    OR returned `success: True` with missing/zero TEA or GWP (because the
    MINLP would then see a free option). Returns the updated DataFrame and a
    list of per-(polymer, solvent) failure records with reasons.

    Workbook-backed rows with positive CAPEX/OPEX/GWP can be left untouched
    when `prefer_baseline_when_sufficient=True`. Those are reported separately
    as `simulation_skips` so downstream consumers can distinguish "did not
    simulate because baseline was already sufficient" from "tried to simulate
    and failed."
    """

    unique_simulations = {}
    simulation_skips: list[dict[str, str]] = []
    seen_skips: set[tuple[str, str, float]] = set()
    updated_df = df.copy()
    if _COEFFICIENT_SOURCE_COLUMN not in updated_df.columns:
        updated_df[_COEFFICIENT_SOURCE_COLUMN] = "workbook_baseline"
    else:
        updated_df[_COEFFICIENT_SOURCE_COLUMN] = (
            updated_df[_COEFFICIENT_SOURCE_COLUMN].fillna("workbook_baseline").astype(str)
        )

    rows_to_simulate: set[tuple[str, str, float]] = set()
    for idx, row in updated_df.iterrows():
        wash_number = row.get("Wash number")
        polymer = row.get("Polymer")
        solvent = row.get("Solvents")
        if pd.isna(wash_number) or pd.isna(polymer) or pd.isna(solvent):
            continue

        capacity = capacity_pe if polymer == "PE" else capacity_evoh
        target_plastic_percent = pe_fraction_pct if polymer == "PE" else evoh_fraction_pct
        key = (str(polymer), str(solvent), float(capacity))
        source = str(row.get(_COEFFICIENT_SOURCE_COLUMN) or "workbook_baseline").strip() or "workbook_baseline"

        if (
            prefer_baseline_when_sufficient
            and source == "workbook_baseline"
            and _row_has_sufficient_baseline_for_skip(row)
        ):
            if key not in seen_skips:
                seen_skips.add(key)
                simulation_skips.append(
                    _skip_record(
                        polymer=key[0],
                        solvent=key[1],
                        reason="baseline_sufficient",
                        source=source,
                    )
                )
            continue
        rows_to_simulate.add(key)

    uncached_items: list[tuple[tuple[str, str, float], dict[str, Any], tuple[Any, ...]]] = []
    for polymer, solvent, capacity in sorted(rows_to_simulate):
        target_plastic_percent = pe_fraction_pct if polymer == "PE" else evoh_fraction_pct
        cache_key = (
            polymer,
            solvent,
            round(float(capacity), 6),
            round(float(target_plastic_percent), 6),
            "C1",
            id(run_single_simulation),
        )
        cached_result = _BIOSTEAM_SIM_CACHE.get(cache_key)
        if cached_result is not None:
            unique_simulations[(polymer, solvent, capacity)] = cached_result
            continue
        denylisted_result = _build_runtime_denylist_result(polymer, solvent)
        if denylisted_result is not None:
            _BIOSTEAM_SIM_CACHE[cache_key] = denylisted_result
            unique_simulations[(polymer, solvent, capacity)] = denylisted_result
            continue

        config = build_single_config(
            solvent=solvent,
            target_plastic=polymer,
            target_plastic_percent=target_plastic_percent,
            processing_capacity=capacity,
            energy_case="C1",
        )
        uncached_items.append(((polymer, solvent, capacity), config, cache_key))

    if uncached_items:
        def _call_single_simulation(config: dict[str, Any]) -> dict[str, Any]:
            try:
                return run_single_simulation(config, _BIOSTEAM_TIMEOUT_SEC)
            except TypeError:
                # Test doubles often keep the older single-argument signature.
                return run_single_simulation(config)

        max_workers = min(_BIOSTEAM_BATCH_PARALLEL, len(uncached_items))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_item = {
                pool.submit(_call_single_simulation, config): (sim_key, cache_key)
                for sim_key, config, cache_key in uncached_items
            }
            for future in as_completed(future_to_item):
                sim_key, cache_key = future_to_item[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error(
                        "Failed BioSTEAM simulation for %s in %s: %s",
                        sim_key[0],
                        sim_key[1],
                        exc,
                    )
                    result = {"success": False, "_sim_exception": str(exc)}
                _BIOSTEAM_SIM_CACHE[cache_key] = result
                unique_simulations[sim_key] = result

    keep_indices: list[int] = []
    failed_sims: list[dict[str, str]] = []
    seen_failures: set[tuple[str, str]] = set()
    for idx, row in updated_df.iterrows():
        polymer = row.get("Polymer")
        solvent = row.get("Solvents")
        if pd.isna(polymer) or pd.isna(solvent):
            keep_indices.append(idx)
            continue

        capacity = capacity_pe if polymer == "PE" else capacity_evoh
        pair = (str(polymer), str(solvent))
        key = (pair[0], pair[1], float(capacity))
        source = str(row.get(_COEFFICIENT_SOURCE_COLUMN) or "workbook_baseline").strip() or "workbook_baseline"

        if (
            prefer_baseline_when_sufficient
            and source == "workbook_baseline"
            and _row_has_sufficient_baseline_for_skip(row)
        ):
            keep_indices.append(idx)
            continue

        result = unique_simulations.get(key, {})

        sim_crash_reason = _sim_crashed_reason(result)
        if sim_crash_reason is not None:
            if pair not in seen_failures:
                seen_failures.add(pair)
                failed_sims.append(
                    _failure_record(
                        polymer=pair[0],
                        solvent=pair[1],
                        reason=sim_crash_reason,
                        source="runtime_denylist" if _runtime_denylist_applies(pair[0], pair[1]) else "biosteam_simulation",
                    )
                )
            continue

        # Successful sim — let _map_biosteam_to_strap_row apply whatever fields it
        # can. Because that helper guards each overwrite with a positivity check,
        # a null-TEA sim leaves workbook baseline values intact. The post-update
        # row is then validated against the MINLP-critical economic fields.
        updated_row = _map_biosteam_to_strap_row(row.copy(), result, capacity)
        updated_row[_COEFFICIENT_SOURCE_COLUMN] = "biosteam_updated"
        for column, value in updated_row.items():
            updated_df.at[idx, column] = value

        ok, row_reason = _row_has_usable_economics(updated_df.loc[idx])
        if ok:
            keep_indices.append(idx)
        else:
            if pair not in seen_failures:
                seen_failures.add(pair)
                failed_sims.append(
                    _failure_record(
                        polymer=pair[0],
                        solvent=pair[1],
                        reason=row_reason or "row_missing_economics",
                        source="post_sim_validation",
                    )
                )

    return updated_df.loc[keep_indices].copy(), failed_sims, simulation_skips


def _build_compiled_strap_coefficient_table(
    source_excel_path: Path,
    *,
    capacity_pe: float,
    capacity_evoh: float,
    pe_fraction_pct: float,
    evoh_fraction_pct: float,
    candidate_solvents: list[str] | str | None = None,
    polymer_solvent_filters_json: dict[str, Any] | str | None = None,
    constraint_mode: str | None = None,
    fallback_policy: str | None = None,
    prefer_baseline_when_sufficient: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]], list[str], str, list[dict[str, str]], list[dict[str, str]]]:
    df = pd.read_excel(source_excel_path, sheet_name="StrapScenario3 Units")
    for column in _NUMERIC_WORKBOOK_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    materialization_allowlist = _build_materialization_allowlist(
        candidate_solvents=candidate_solvents,
        polymer_solvent_filters_json=polymer_solvent_filters_json,
    )
    df = _materialize_optimizer_workbook_rows(
        df,
        allowed_solvents_by_slot=materialization_allowlist,
    )
    df, applied_filters, filter_warnings, requested_filters = _apply_solvent_filters(
        df,
        candidate_solvents=candidate_solvents,
        polymer_solvent_filters_json=polymer_solvent_filters_json,
        constraint_mode=constraint_mode,
        fallback_policy=fallback_policy,
    )
    simulation_failures: list[dict[str, str]] = []
    simulation_skips: list[dict[str, str]] = []
    if not df.empty:
        polymers_before = {
            str(p).strip()
            for p in df["Polymer"].dropna().astype(str).tolist()
            if str(p).strip()
        }
        df, simulation_failures, simulation_skips = _run_biosteam_updates(
            df,
            capacity_pe,
            capacity_evoh,
            pe_fraction_pct=pe_fraction_pct,
            evoh_fraction_pct=evoh_fraction_pct,
            prefer_baseline_when_sufficient=prefer_baseline_when_sufficient,
        )
        polymers_after = {
            str(p).strip()
            for p in df["Polymer"].dropna().astype(str).tolist()
            if str(p).strip()
        }
        if simulation_failures:
            failed_pairs = ", ".join(
                f"{item['polymer']}+{item['solvent']} ({item.get('reason', 'unknown')})"
                for item in simulation_failures
            )
            filter_warnings.append(
                "BioSTEAM simulation failed (or returned incomplete TEA/GWP) for these "
                "candidate pairs and the corresponding rows were dropped to prevent "
                f"zero-metric ghost rows from polluting the optimization solve: {failed_pairs}."
            )
        dropped_polymers = sorted(polymers_before - polymers_after)
        for polymer in dropped_polymers:
            filter_warnings.append(
                f"All {polymer} candidate rows were dropped after BioSTEAM "
                "simulation failures. The optimizer has no valid "
                f"{polymer}-stage options under the current shortlist."
            )
    filter_status = _solvent_filter_status(requested_filters, applied_filters, filter_warnings)
    return (
        df,
        requested_filters,
        applied_filters,
        filter_warnings,
        filter_status,
        simulation_failures,
        simulation_skips,
    )


def _prepare_optimization_context(
    *,
    feed: float,
    pe_fraction: float,
    pet_fraction: float,
    n6_fraction: float,
    evoh_fraction: float,
    scenario: str,
    candidate_solvents: list[str] | str | None = None,
    polymer_solvent_filters_json: dict[str, Any] | str | None = None,
    stage_candidates_json: dict[str, Any] | str | None = None,
    constraint_mode: str | None = None,
    fallback_policy: str | None = None,
    route_pool_mode: str | None = None,
    prefer_baseline_when_sufficient: bool = True,
) -> dict[str, Any]:
    _validate_feed_inputs(feed, pe_fraction, pet_fraction, n6_fraction, evoh_fraction)

    base_dir = Path(__file__).resolve().parent.parent / "waste_management"
    source_excel_path = base_dir / "Data for model_Scenarios.xlsx"
    if not source_excel_path.exists():
        raise FileNotFoundError(f"Excel file not found at {source_excel_path}")

    capacity_pe = max(feed * pe_fraction, 1)
    capacity_evoh = max(feed * evoh_fraction, 1)

    stage_candidate_payload = _parse_stage_candidates(stage_candidates_json)
    stage_polymer_filters, stage_global_candidates, stage_constraint_mode, stage_fallback_policy = (
        _derive_filters_from_stage_candidates(stage_candidates_json)
    )
    stage_route_candidates = _extract_route_candidates(stage_candidates_json)
    stage_route_pool_mode = _derive_route_pool_mode_from_stage_candidates(stage_candidates_json)
    has_typed_handoff = stage_candidates_json not in (None, "", {})
    precedence_warnings: list[str] = []
    requested_route_pool_mode = _normalize_route_pool_mode(route_pool_mode)
    if has_typed_handoff:
        legacy_overrides: list[str] = []
        if candidate_solvents is not None:
            legacy_overrides.append("candidate_solvents")
        if polymer_solvent_filters_json is not None:
            legacy_overrides.append("polymer_solvent_filters_json")
        if constraint_mode is not None and constraint_mode != stage_constraint_mode:
            legacy_overrides.append("constraint_mode")
        if fallback_policy is not None and fallback_policy != stage_fallback_policy:
            legacy_overrides.append("fallback_policy")
        if legacy_overrides:
            precedence_warnings.append(
                "Typed stage_candidates_json took precedence over the following legacy "
                f"override(s) supplied alongside it: {', '.join(legacy_overrides)}."
            )
        if (
            requested_route_pool_mode is not None
            and stage_route_pool_mode is not None
            and requested_route_pool_mode != stage_route_pool_mode
        ):
            precedence_warnings.append(
                "Explicit route_pool_mode override took precedence over the handoff-provided "
                f"value ({stage_route_pool_mode} -> {requested_route_pool_mode})."
            )
        effective_constraint_mode = stage_constraint_mode or "soft"
        effective_fallback_policy = stage_fallback_policy or "broaden_disclosed"
        effective_route_pool_mode = (
            requested_route_pool_mode
            or stage_route_pool_mode
            or "exact"
        )
        effective_candidate_solvents = stage_global_candidates
        effective_polymer_filters = stage_polymer_filters
    else:
        effective_constraint_mode = constraint_mode or "soft"
        effective_fallback_policy = fallback_policy or "broaden_disclosed"
        effective_route_pool_mode = _normalize_route_pool_mode(route_pool_mode) or "exact"
        effective_candidate_solvents = candidate_solvents
        effective_polymer_filters = polymer_solvent_filters_json

    compiled_strap_df, requested_filters, applied_filters, filter_warnings, filter_status, simulation_failures, simulation_skips = (
        _build_compiled_strap_coefficient_table(
            source_excel_path,
            capacity_pe=capacity_pe,
            capacity_evoh=capacity_evoh,
            pe_fraction_pct=pe_fraction * 100.0,
            evoh_fraction_pct=evoh_fraction * 100.0,
            candidate_solvents=effective_candidate_solvents,
            polymer_solvent_filters_json=effective_polymer_filters,
            constraint_mode=effective_constraint_mode,
            fallback_policy=effective_fallback_policy,
            prefer_baseline_when_sufficient=prefer_baseline_when_sufficient,
        )
    )

    # fail_closed only short-circuits on actual catalog/simulation-level warnings
    # (generated inside _build_compiled_strap_coefficient_table). Metadata warnings
    # such as precedence notes are appended afterwards so they do not spuriously
    # trigger infeasible outcomes.
    fail_closed_active = (
        effective_constraint_mode in {"fixed", "hard"}
        and effective_fallback_policy == "fail_closed"
    )
    if fail_closed_active and filter_warnings and requested_filters:
        return {
            "temp_dir": None,
            "infeasible_response": _build_optimization_infeasible_response(
                failure_reason="no_candidate_overlap",
                message=filter_warnings[0],
                constraint_mode=effective_constraint_mode,
                fallback_policy=effective_fallback_policy,
                requested_filters=requested_filters,
                applied_filters=applied_filters,
                suggested_relaxation="retry_with_soft_mode",
            ),
        }

    # Post-sim infeasibility short-circuit: if the user asked for a specific
    # shortlist but every matching row got dropped because BioSTEAM couldn't
    # produce usable economics for any of them, falling through to the
    # aggregate solve would silently pick a non-STRAP landfill pathway and
    # report bogus zero-cost, zero-emission results. Return a typed infeasible
    # response instead so the caller can see the real reason.
    #
    # The check runs per-polymer: if the shortlist named PE solvents but no
    # PE row survived the BioSTEAM pass, PE Wash 1 has no feasible option and
    # the MINLP would fall through to non-STRAP pathways for the entire feed.
    polymers_with_shortlist = [
        polymer
        for polymer, solvents in (requested_filters or {}).items()
        if polymer != "global" and solvents
    ]
    if polymers_with_shortlist:
        surviving_polymers = set()
        if compiled_strap_df is not None and not compiled_strap_df.empty:
            surviving_polymers = {
                str(polymer).strip()
                for polymer in compiled_strap_df["Polymer"].dropna().astype(str).tolist()
                if str(polymer).strip()
            }
        polymers_missing_rows = [p for p in polymers_with_shortlist if p not in surviving_polymers]
        if polymers_missing_rows:
            reason_msg = (
                f"After BioSTEAM updates, no workbook rows remain for shortlisted polymer(s): "
                f"{', '.join(polymers_missing_rows)}. The MINLP cannot produce a valid STRAP "
                "pathway for these polymers under the current shortlist."
            )
            return {
                "temp_dir": None,
                "infeasible_response": _build_optimization_infeasible_response(
                    failure_reason="all_shortlisted_sims_failed",
                    message=reason_msg,
                    constraint_mode=effective_constraint_mode,
                    fallback_policy=effective_fallback_policy,
                    requested_filters=requested_filters,
                    applied_filters=applied_filters,
                    suggested_relaxation=(
                        "BioSTEAM could not populate TEA for any shortlisted (polymer, solvent) "
                        "pair in this environment, and the workbook has no baseline CAPEX/OPEX "
                        "for these solvents either. Either (a) expand the optimizer workbook "
                        "with baseline values for these solvents, or (b) ask separation to "
                        "restrict its shortlist to the workbook solvents (Heptane, Toluene, "
                        "Xylene, Methylcyclohexane, etc. for PE; Pyridazine, Ethylene Glycol, "
                        "gamma-butyrolactone, etc. for EVOH)."
                    ),
                ),
            }

    if precedence_warnings:
        filter_warnings.extend(precedence_warnings)

    normalized_scenario, scenario_payload = _get_scenario_config(
        scenario,
        feed,
        pe_fraction,
        pet_fraction,
        n6_fraction,
        evoh_fraction,
    )
    data = load_all_data(
        excel_path=source_excel_path,
        strap_sheet="StrapScenario3 Units",
        other_sheet=scenario_payload["other_sheet"],
        p_strap=1.0,
        strap_df=compiled_strap_df,
    )
    return {
        "temp_dir": None,
        "excel_path": source_excel_path,
        "strap_df": compiled_strap_df,
        "stage_candidate_payload": stage_candidate_payload,
        "scenario": normalized_scenario,
        "config": scenario_payload["config"],
        "data": data,
        "feed": feed,
        "fractions": {
            "PE": pe_fraction,
            "PET": pet_fraction,
            "N6": n6_fraction,
            "EVOH": evoh_fraction,
        },
        "requested_filters": requested_filters,
        "applied_filters": applied_filters,
        "filter_warnings": filter_warnings,
        "filter_status": filter_status,
        "simulation_failures": simulation_failures,
        "simulation_skips": simulation_skips,
        "strap_table_rows": int(len(compiled_strap_df)),
        "constraint_mode": effective_constraint_mode,
        "fallback_policy": effective_fallback_policy,
        "route_pool_mode": effective_route_pool_mode,
        "route_candidates": stage_route_candidates,
        "has_typed_handoff": has_typed_handoff,
    }


def _serialize_optimization_results(results: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (str(value) if not isinstance(value, (int, float, str, list, dict, bool)) else value)
        for key, value in results.items()
    }


def _build_pareto_candidate_summary(context: dict[str, Any], points: list[dict[str, Any]]) -> dict[str, Any]:
    requested_pair_set = {
        (polymer, solvent)
        for polymer, solvents in context["requested_filters"].items()
        if polymer != "global"
        for solvent in solvents
    }
    applied_pair_set = {
        (polymer, solvent)
        for polymer, solvents in context["applied_filters"].items()
        for solvent in solvents
    }
    requested_count = len(requested_pair_set) if requested_pair_set else len(set(context["requested_filters"].get("global", [])))
    applied_count = len(applied_pair_set) if applied_pair_set else 0
    return {
        "status": context["filter_status"],
        "requested_filters": context["requested_filters"],
        "applied_filters": context["applied_filters"],
        "warnings": context["filter_warnings"],
        "n_pairs_requested": requested_count,
        "n_pairs_applied": applied_count,
        "n_pairs_broadened": max(requested_count - applied_count, 0),
        "n_pairs_infeasible": 0 if points else requested_count,
    }


def _build_candidate_telemetry(context: dict[str, Any]) -> dict[str, Any]:
    """Summarize candidate admission and attrition for optimizer runs.

    The optimizer now accepts the broader polymer-aware STRAP/BioSTEAM solvent
    space. This helper makes the downstream result honest about what happened to
    that universe:
      1. what upstream requested,
      2. what overlap was actually applied,
      3. what survived filtering/simulation into the compiled STRAP table, and
      4. what failed or was skipped along the way.
    """

    def _stable_unique(values: list[Any]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _count_map(values_by_key: dict[str, list[str]]) -> dict[str, int]:
        return {
            key: len(values)
            for key, values in values_by_key.items()
            if values
        }

    requested_filters = context.get("requested_filters") or {}
    applied_filters = context.get("applied_filters") or {}
    stage_candidate_payload = context.get("stage_candidate_payload") or {}
    strap_df = context.get("strap_df")
    simulation_failures = list(context.get("simulation_failures") or [])
    simulation_skips = list(context.get("simulation_skips") or [])

    requested_by_polymer = {
        polymer: _stable_unique(solvents)
        for polymer, solvents in requested_filters.items()
        if polymer != "global" and isinstance(solvents, list)
    }
    requested_global = _stable_unique(requested_filters.get("global") or [])
    applied_by_polymer = {
        polymer: _stable_unique(solvents)
        for polymer, solvents in applied_filters.items()
        if isinstance(solvents, list)
    }

    source_candidate_counts = stage_candidate_payload.get("candidate_counts_by_polymer")
    if not isinstance(source_candidate_counts, dict):
        source_candidate_counts = _count_map(requested_by_polymer)
    else:
        source_candidate_counts = {
            str(polymer): int(count)
            for polymer, count in source_candidate_counts.items()
            if str(polymer).strip()
        }

    source_candidate_lists = stage_candidate_payload.get("polymer_solvent_filters")
    if not isinstance(source_candidate_lists, dict):
        source_candidate_lists = requested_by_polymer
    else:
        source_candidate_lists = {
            str(polymer): _stable_unique(solvents if isinstance(solvents, list) else [])
            for polymer, solvents in source_candidate_lists.items()
            if str(polymer).strip()
        }

    surviving_by_stage: dict[str, dict[str, list[str]]] = {"Wash 1": {}, "Wash 2": {}}
    surviving_counts_by_stage: dict[str, dict[str, int]] = {"Wash 1": {}, "Wash 2": {}}
    surviving_by_polymer: dict[str, list[str]] = {}
    if isinstance(strap_df, pd.DataFrame) and not strap_df.empty:
        for wash in ("Wash 1", "Wash 2"):
            wash_mask = strap_df["Wash number"].eq(wash)
            for polymer in ("PE", "EVOH"):
                polymer_mask = strap_df["Polymer"].eq(polymer)
                solvents = _stable_unique(
                    strap_df.loc[wash_mask & polymer_mask, "Solvents"].dropna().astype(str).tolist()
                )
                if solvents:
                    surviving_by_stage[wash][polymer] = solvents
                    surviving_counts_by_stage[wash][polymer] = len(solvents)
                    surviving_by_polymer.setdefault(polymer, [])
                    surviving_by_polymer[polymer] = _stable_unique(
                        [*surviving_by_polymer[polymer], *solvents]
                    )

    failure_counts_by_class: dict[str, int] = {}
    failures_by_polymer: dict[str, list[dict[str, str]]] = {}
    for record in simulation_failures:
        if not isinstance(record, dict):
            continue
        polymer = str(record.get("polymer") or "").strip()
        failure_class = str(record.get("failure_class") or "unknown_failure").strip() or "unknown_failure"
        failure_counts_by_class[failure_class] = failure_counts_by_class.get(failure_class, 0) + 1
        if polymer:
            failures_by_polymer.setdefault(polymer, []).append(record)

    skip_counts_by_reason: dict[str, int] = {}
    skips_by_polymer: dict[str, list[dict[str, str]]] = {}
    for record in simulation_skips:
        if not isinstance(record, dict):
            continue
        polymer = str(record.get("polymer") or "").strip()
        reason = str(record.get("reason") or "unknown_skip").strip() or "unknown_skip"
        skip_counts_by_reason[reason] = skip_counts_by_reason.get(reason, 0) + 1
        if polymer:
            skips_by_polymer.setdefault(polymer, []).append(record)

    return {
        "source": {
            "typed_stage_candidates": bool(context.get("has_typed_handoff")),
            "route_candidates_present": bool(context.get("route_candidates")),
            "route_pool_mode": context.get("route_pool_mode"),
            "constraint_mode": context.get("constraint_mode"),
            "fallback_policy": context.get("fallback_policy"),
        },
        "requested": {
            "global_candidates": requested_global,
            "by_polymer": requested_by_polymer,
            "counts_by_polymer": _count_map(requested_by_polymer),
            "source_counts_by_polymer": source_candidate_counts,
            "source_lists_by_polymer": source_candidate_lists,
        },
        "applied": {
            "by_polymer": applied_by_polymer,
            "counts_by_polymer": _count_map(applied_by_polymer),
        },
        "surviving": {
            "compiled_rows_total": int(context.get("strap_table_rows") or 0),
            "by_stage": surviving_by_stage,
            "counts_by_stage": surviving_counts_by_stage,
            "by_polymer": surviving_by_polymer,
            "counts_by_polymer": _count_map(surviving_by_polymer),
        },
        "simulation": {
            "failures_total": len(simulation_failures),
            "skips_total": len(simulation_skips),
            "failure_counts_by_class": failure_counts_by_class,
            "skip_counts_by_reason": skip_counts_by_reason,
            "failures_by_polymer": failures_by_polymer,
            "skips_by_polymer": skips_by_polymer,
        },
    }


def _frame_to_pareto_points(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    sorted_frame = (
        frame.sort_values(by=["total_cost", "emissions", "CE"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    def _to_text_list(value: Any) -> list[str]:
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item)]
        if value is None or value == "":
            return []
        return [str(value)]

    def _optional_text(value: Any) -> str | None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        text = str(value or "").strip()
        return text or None

    def _optional_number(value: Any) -> int | float | None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            numeric = float(value)
            return int(numeric) if numeric.is_integer() else numeric
        return None

    def _design_signature(row: pd.Series) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        return (
            tuple(_to_text_list(row.get("stage1"))),
            tuple(_to_text_list(row.get("stage2"))),
            tuple(_to_text_list(row.get("stage3"))),
            tuple(_to_text_list(row.get("wash1"))),
            tuple(_to_text_list(row.get("wash2"))),
        )

    def _build_design_variant(row: pd.Series) -> dict[str, Any]:
        return {
            "stage1_tech": _to_text_list(row.get("stage1")),
            "stage2_tech": _to_text_list(row.get("stage2")),
            "stage3_tech": _to_text_list(row.get("stage3")),
            "wash1_selection": _to_text_list(row.get("wash1")),
            "wash2_selection": _to_text_list(row.get("wash2")),
            "route_id": _optional_text(row.get("route_id")) or "",
            "matched_route_id": _optional_text(row.get("matched_route_id")),
            "rank": _optional_number(row.get("rank")),
            "polymer_solvent_map": row.get("polymer_solvent_map"),
            "selection_origin": _optional_text(row.get("selection_origin")) or "exact_route",
            "wash1_origin_route_id": _optional_text(row.get("wash1_origin_route_id")),
            "wash2_origin_route_id": _optional_text(row.get("wash2_origin_route_id")),
            "wash1_origin_route_ids": list(row.get("wash1_origin_route_ids") or []),
            "wash2_origin_route_ids": list(row.get("wash2_origin_route_ids") or []),
        }

    grouped_rows: dict[tuple[float, float, float], dict[str, Any]] = {}
    row_order: list[tuple[float, float, float]] = []
    for _, row in sorted_frame.iterrows():
        key = (
            float(row.get("total_cost", 0.0) or 0.0),
            float(row.get("emissions", 0.0) or 0.0),
            float(row.get("CE", 0.0) or 0.0),
        )
        signature = _design_signature(row)
        if key not in grouped_rows:
            grouped_rows[key] = {
                "row": row,
                "signatures": {signature},
                "design_variants": [_build_design_variant(row)],
            }
            row_order.append(key)
            continue
        if signature not in grouped_rows[key]["signatures"]:
            grouped_rows[key]["signatures"].add(signature)
            grouped_rows[key]["design_variants"].append(_build_design_variant(row))

    points: list[dict[str, Any]] = []
    for idx, key in enumerate(row_order):
        group = grouped_rows[key]
        row = group["row"]
        raw_ce = float(row.get("CE", 0.0) or 0.0)
        equivalent_designs = list(group["design_variants"])
        stage3_variants = sorted(
            {
                stage
                for design in equivalent_designs
                for stage in design.get("stage3_tech", [])
                if stage
            }
        )
        point = {
            "point_id": idx + 1,
            "epsilon": float(row.get("epsilon", 0.0) or 0.0),
            "profit": float(row.get("profit", 0.0) or 0.0),
            "emissions": float(row.get("emissions", 0.0) or 0.0),
            "raw_circularity_score": raw_ce,
            "circularity_score": max(0.0, min(raw_ce / 1_000_000.0, 1.0)),
            "total_cost": float(row.get("total_cost", 0.0) or 0.0),
            "capital_cost": float(row.get("capital_cost", 0.0) or 0.0),
            "operational_cost": float(row.get("operational_cost", 0.0) or 0.0),
            "transportation_cost": float(row.get("transportation_cost", 0.0) or 0.0),
            "stage1_tech": list(row.get("stage1", []) or []),
            "stage2_tech": list(row.get("stage2", []) or []),
            "stage3_tech": list(row.get("stage3", []) or []),
            "wash1_selection": list(row.get("wash1", []) or []),
            "wash2_selection": list(row.get("wash2", []) or []),
            "route_id": _optional_text(row.get("route_id")) or "",
            "matched_route_id": _optional_text(row.get("matched_route_id")),
            "rank": _optional_number(row.get("rank")),
            "polymer_solvent_map": row.get("polymer_solvent_map"),
            "selection_origin": _optional_text(row.get("selection_origin")) or "exact_route",
            "wash1_origin_route_id": _optional_text(row.get("wash1_origin_route_id")),
            "wash2_origin_route_id": _optional_text(row.get("wash2_origin_route_id")),
            "wash1_origin_route_ids": list(row.get("wash1_origin_route_ids") or []),
            "wash2_origin_route_ids": list(row.get("wash2_origin_route_ids") or []),
            "n_equivalent_designs": len(equivalent_designs),
            "stage3_variants": stage3_variants,
            "equivalent_designs": equivalent_designs,
        }
        points.append(point)
    return points


def _extract_tool_data(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    data = parsed.get("data")
    return data if isinstance(data, dict) else {}


def _solve_objective_with_fallback(
    model,
    objective: str,
    *,
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    option_ladder: tuple[dict[str, Any] | None, ...]
    if solver_options is not None:
        option_ladder = (solver_options,)
    else:
        option_ladder = _SCIP_CONSTRAINED_OPTION_LADDER

    def _clone_if_supported(candidate):
        return candidate.clone() if hasattr(candidate, "clone") else candidate

    saw_scip_error = False
    for attempt_options in option_ladder:
        candidate_model = _clone_if_supported(model)
        try:
            if attempt_options is not None:
                return solve_single(
                    candidate_model,
                    objective,
                    solver_name="scip",
                    solver_options=attempt_options,
                )
            return solve_single(candidate_model, objective, solver_name="scip")
        except Exception as exc:
            saw_scip_error = True
            logger.warning(
                "Failed to run SCIP solver for %s with solver_options=%s: %s",
                objective,
                attempt_options or "default",
                exc,
            )

    if not saw_scip_error:
        return None

    logger.warning("Falling back to available solvers for %s after exhausting SCIP retry options.", objective)
    try:
        fallback_model = _clone_if_supported(model)
        if solver_options:
            return solve_single(fallback_model, objective, solver_name=None, solver_options=solver_options)
        return solve_single(fallback_model, objective, solver_name=None)
    except Exception as fallback_exc:
        logger.warning(
            "Fallback solver path also failed for %s with solver_options=%s: %s",
            objective,
            solver_options or "default",
            fallback_exc,
        )
        return None


def _retry_constrained_objective(
    builder,
    objective: str,
    *,
    max_attempts: int = 3,
    option_ladder: tuple[dict[str, Any] | None, ...] = _SCIP_CONSTRAINED_OPTION_LADDER,
) -> tuple[dict[str, Any] | None, str | None]:
    """Solve an objective against freshly-built constrained models.

    SCIP is numerically unstable on some shortlisted MINLPs in this repo. A
    fresh rebuild can succeed even when the previous attempt exited as
    infeasible or errored. This helper retries the objective on new models
    before the caller concludes the constrained route pool is infeasible.
    """
    last_reason: str | None = None
    for attempt in range(1, max_attempts + 1):
        model, enforced, reason = builder()
        if not enforced:
            return None, reason or "route constraints could not be applied"
        solver_options = option_ladder[min(attempt - 1, len(option_ladder) - 1)] if option_ladder else None
        if solver_options:
            result = _solve_objective_with_fallback(model, objective, solver_options=solver_options)
        else:
            result = _solve_objective_with_fallback(model, objective)
        if result is not None:
            return result, None
        options_label = (
            ", ".join(f"{key}={value}" for key, value in sorted(solver_options.items()))
            if solver_options
            else "default"
        )
        last_reason = (
            f"{objective} solve returned no feasible solution on attempt {attempt}"
            f" (solver_options={options_label})"
        )
        logger.warning(
            "Objective %s returned no solution on constrained attempt %d/%d with solver_options=%s.",
            objective,
            attempt,
            max_attempts,
            solver_options or "default",
        )
    return None, last_reason or f"{objective} solve failed"


def _run_constrained_pareto_sweep(
    builder,
    *,
    y_metric: str,
    cost_opt: dict[str, Any],
    y_opt: dict[str, Any],
    n_points: int,
    option_ladder: tuple[dict[str, Any] | None, ...] = _SCIP_CONSTRAINED_OPTION_LADDER,
) -> tuple[pd.DataFrame, str | None]:
    """Run a Pareto sweep against freshly-built constrained models.

    SCIP can mark some constrained route-pool models infeasible during
    presolve even when the identical model solves if presolve is disabled.
    Sweep generation therefore uses the same option ladder as the anchor
    solves and retries on a fresh model before giving up.
    """
    last_reason: str | None = None
    for attempt, solver_options in enumerate(option_ladder or (None,), start=1):
        sweep_model, enforced, reason = builder()
        if not enforced:
            return pd.DataFrame(), reason or "route constraints could not be applied"
        try:
            if y_metric == "emissions":
                frontier = pareto_cost_vs_emissions(
                    sweep_model,
                    emission_ideal=float(y_opt["emissions"]),
                    emission_nonideal=float(cost_opt["emissions"]),
                    n_points=n_points,
                    solver_name="scip",
                    solver_options=solver_options,
                )
            else:
                frontier = pareto_cost_vs_ce(
                    sweep_model,
                    ce_nonideal=float(cost_opt["CE"]),
                    ce_ideal=float(y_opt["CE"]),
                    n_points=n_points,
                    solver_name="scip",
                    solver_options=solver_options,
                )
        except Exception as exc:
            logger.warning(
                "SCIP constrained Pareto sweep attempt %d/%d failed with solver_options=%s: %s",
                attempt,
                len(option_ladder or (None,)),
                solver_options or "default",
                exc,
            )
            last_reason = f"pareto sweep errored on attempt {attempt}"
            continue

        if not frontier.empty:
            return frontier, None

        options_label = (
            ", ".join(f"{key}={value}" for key, value in sorted(solver_options.items()))
            if solver_options
            else "default"
        )
        last_reason = f"pareto sweep returned no feasible points on attempt {attempt} (solver_options={options_label})"
        logger.warning(
            "Constrained Pareto sweep returned no feasible points on attempt %d/%d with solver_options=%s.",
            attempt,
            len(option_ladder or (None,)),
            solver_options or "default",
        )
    return pd.DataFrame(), last_reason or "pareto sweep failed"


def _resolve_route_assignments(
    route_spec: dict[str, Any],
    *,
    polymers_available: list[str],
    all_solvents: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Canonicalize a route candidate into exact wash assignments.

    The upstream route payload may carry aliases (for example LDPE -> PE or
    DMSO -> Dimethyl sulfoxide). This helper normalizes the route against the
    optimizer's active polymer/solvent catalog and returns a concrete Wash 1 /
    Wash 2 assignment tuple that can be enforced in Pyomo.
    """
    mapping_raw = route_spec.get("polymer_solvent_map") or {}
    if not isinstance(mapping_raw, dict):
        return None, "route is missing a usable polymer_solvent_map"

    solvent_index = _build_available_solvent_index(all_solvents)
    mapping: dict[str, str] = {}
    for polymer, solvent in mapping_raw.items():
        polymer_key = _normalize_optimization_polymer(polymer)
        if polymer_key is None or polymer_key not in polymers_available:
            continue
        solvent_key = _normalize_solvent_key(resolve_to_biosteam(solvent) or solvent)
        matched = solvent_index.get(solvent_key)
        if not matched:
            return None, f"Solvent '{solvent}' for polymer '{polymer_key}' not present in optimizer catalog"
        mapping[polymer_key] = matched

    sequence = [_normalize_optimization_polymer(p) for p in route_spec.get("sequence") or []]
    ordered = [p for p in sequence if p is not None and p in mapping and p in polymers_available]
    if not ordered:
        ordered = [p for p in ("PE", "EVOH") if p in mapping and p in polymers_available]
    if not ordered:
        ordered = [p for p in polymers_available if p in mapping]
    if not ordered:
        return None, "route has no polymer-solvent pairs usable by the current optimizer model"

    wash1_polymer = ordered[0]
    wash1_solvent = mapping[wash1_polymer]
    wash2_polymer = ordered[1] if len(ordered) >= 2 else None
    wash2_solvent = mapping[wash2_polymer] if wash2_polymer else None
    has_wash2 = bool(wash2_polymer and wash2_solvent)
    route_id = str(route_spec.get("route_id") or "")

    return {
        "route_id": route_id,
        "rank": route_spec.get("rank"),
        "source": str(route_spec.get("source") or ""),
        "sequence": [p for p in ordered if p],
        "polymer_solvent_map": mapping,
        "wash1_polymer": wash1_polymer,
        "wash1_solvent": wash1_solvent,
        "wash2_polymer": wash2_polymer,
        "wash2_solvent": wash2_solvent,
        "has_wash2": has_wash2,
        "route_signature": (
            f"{wash1_polymer}-{wash1_solvent}",
            f"{wash2_polymer}-{wash2_solvent}" if has_wash2 else "",
        ),
    }, None


def _normalize_route_pool(
    route_candidates: list[dict[str, Any]],
    *,
    polymers_available: list[str],
    all_solvents: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (usable_routes, skipped_route_reports)."""
    usable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str]] = set()

    def _sort_key(route: dict[str, Any]) -> tuple[int, str]:
        rank = route.get("rank")
        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            rank_value = 10**9
        return rank_value, str(route.get("route_id") or "")

    for route in sorted(route_candidates, key=_sort_key):
        normalized, reason = _resolve_route_assignments(
            route,
            polymers_available=polymers_available,
            all_solvents=all_solvents,
        )
        if normalized is None:
            skipped.append(
                {
                    "route_id": str(route.get("route_id") or ""),
                    "rank": route.get("rank"),
                    "status": "skipped",
                    "reason": reason or "route could not be normalized into optimizer assignments",
                    "polymer_solvent_map": route.get("polymer_solvent_map"),
                }
            )
            continue
        signature = normalized["route_signature"]
        if signature in seen_signatures:
            skipped.append(
                {
                    "route_id": normalized["route_id"],
                    "rank": normalized.get("rank"),
                    "status": "skipped",
                    "reason": "duplicate route signature after optimizer canonicalization",
                    "polymer_solvent_map": normalized.get("polymer_solvent_map"),
                }
            )
            continue
        seen_signatures.add(signature)
        usable.append(normalized)

    return usable, skipped


def _apply_route_pool_constraints(
    model,
    route_specs: list[dict[str, Any]],
    *,
    polymers_available: list[str],
    all_solvents: list[str],
) -> tuple[bool, str | None]:
    """Restrict the model to exactly one of the shortlisted upstream routes."""
    if not route_specs:
        return False, "no normalized route candidates were available for pooled Pareto enforcement"

    route_ids = [str(route["route_id"]) for route in route_specs]
    wash1_flags = {
        (route_id, polymer, solvent): 1
        for route_id, polymer, solvent in (
            (str(route["route_id"]), route["wash1_polymer"], route["wash1_solvent"])
            for route in route_specs
        )
    }
    wash2_flags = {
        (str(route["route_id"]), route["wash2_polymer"], route["wash2_solvent"]): 1
        for route in route_specs
        if route.get("has_wash2")
    }
    has_wash2 = {str(route["route_id"]): 1 if route.get("has_wash2") else 0 for route in route_specs}

    model.route_pool_routes = pyo.Set(initialize=route_ids, ordered=True)
    model.route_select = pyo.Var(model.route_pool_routes, within=pyo.Binary)
    model.route_select_one = pyo.Constraint(
        expr=sum(model.route_select[route_id] for route_id in model.route_pool_routes) == 1
    )
    model.route_pool_stage1 = pyo.Constraint(
        expr=model.x["st1"] == sum(model.route_select[route_id] for route_id in model.route_pool_routes)
    )
    model.route_pool_stage2 = pyo.Constraint(
        expr=model.y["st2"] == sum(has_wash2[route_id] * model.route_select[route_id] for route_id in model.route_pool_routes)
    )
    model.route_pool_wash1 = pyo.ConstraintList()
    model.route_pool_wash2 = pyo.ConstraintList()
    for polymer in polymers_available:
        for solvent in all_solvents:
            model.route_pool_wash1.add(
                model.a[polymer, solvent]
                == sum(
                    wash1_flags.get((route_id, polymer, solvent), 0) * model.route_select[route_id]
                    for route_id in route_ids
                )
            )
            model.route_pool_wash2.add(
                model.b[polymer, solvent]
                == sum(
                    wash2_flags.get((route_id, polymer, solvent), 0) * model.route_select[route_id]
                    for route_id in route_ids
                )
            )
    return True, None


def _apply_slot_independent_constraints(
    model,
    route_specs: list[dict[str, Any]],
    *,
    polymers_available: list[str],
    all_solvents: list[str],
) -> tuple[bool, str | None]:
    """Restrict the model to independent wash-slot candidate pools.

    Exact route pooling only allows one upstream (wash1, wash2) tuple at a
    time. Slot-independent pooling expands the feasible region to the cross
    product of shortlisted Wash 1 and Wash 2 candidates, while still keeping
    each slot tied to the upstream solvent shortlist.
    """
    if not route_specs:
        return False, "no normalized route candidates were available for slot-independent pooled enforcement"

    wash1_candidates: list[dict[str, Any]] = []
    wash2_candidates: list[dict[str, Any]] = []
    wash1_seen: set[tuple[str, str]] = set()
    wash2_seen: set[tuple[str, str]] = set()
    allow_no_wash2 = False

    for route in route_specs:
        wash1_key = (str(route["wash1_polymer"]), str(route["wash1_solvent"]))
        if wash1_key not in wash1_seen:
            wash1_seen.add(wash1_key)
            wash1_candidates.append(
                {
                    "candidate_id": f"wash1::{wash1_key[0]}::{wash1_key[1]}",
                    "polymer": wash1_key[0],
                    "solvent": wash1_key[1],
                }
            )

        if route.get("has_wash2"):
            wash2_key = (str(route["wash2_polymer"]), str(route["wash2_solvent"]))
            if wash2_key not in wash2_seen:
                wash2_seen.add(wash2_key)
                wash2_candidates.append(
                    {
                        "candidate_id": f"wash2::{wash2_key[0]}::{wash2_key[1]}",
                        "polymer": wash2_key[0],
                        "solvent": wash2_key[1],
                    }
                )
        else:
            allow_no_wash2 = True

    if not wash1_candidates:
        return False, "slot-independent pooling found no Wash 1 candidates"

    model.slot_pool_wash1_ids = pyo.Set(
        initialize=[candidate["candidate_id"] for candidate in wash1_candidates],
        ordered=True,
    )
    model.slot_pool_wash1_select = pyo.Var(model.slot_pool_wash1_ids, within=pyo.Binary)
    model.slot_pool_wash1_one = pyo.Constraint(
        expr=sum(model.slot_pool_wash1_select[candidate_id] for candidate_id in model.slot_pool_wash1_ids) == 1
    )
    model.slot_pool_stage1 = pyo.Constraint(expr=model.x["st1"] == 1)

    wash1_flags = {
        (candidate["candidate_id"], candidate["polymer"], candidate["solvent"]): 1
        for candidate in wash1_candidates
    }
    model.slot_pool_wash1 = pyo.ConstraintList()
    for polymer in polymers_available:
        for solvent in all_solvents:
            model.slot_pool_wash1.add(
                model.a[polymer, solvent]
                == sum(
                    wash1_flags.get((candidate_id, polymer, solvent), 0) * model.slot_pool_wash1_select[candidate_id]
                    for candidate_id in model.slot_pool_wash1_ids
                )
            )

    if wash2_candidates:
        model.slot_pool_wash2_ids = pyo.Set(
            initialize=[candidate["candidate_id"] for candidate in wash2_candidates],
            ordered=True,
        )
        model.slot_pool_wash2_select = pyo.Var(model.slot_pool_wash2_ids, within=pyo.Binary)
        if allow_no_wash2:
            model.slot_pool_wash2_none = pyo.Var(within=pyo.Binary)
            model.slot_pool_wash2_one = pyo.Constraint(
                expr=sum(model.slot_pool_wash2_select[candidate_id] for candidate_id in model.slot_pool_wash2_ids)
                + model.slot_pool_wash2_none
                == 1
            )
            model.slot_pool_stage2 = pyo.Constraint(
                expr=model.y["st2"]
                == sum(model.slot_pool_wash2_select[candidate_id] for candidate_id in model.slot_pool_wash2_ids)
            )
        else:
            model.slot_pool_wash2_one = pyo.Constraint(
                expr=sum(model.slot_pool_wash2_select[candidate_id] for candidate_id in model.slot_pool_wash2_ids) == 1
            )
            model.slot_pool_stage2 = pyo.Constraint(expr=model.y["st2"] == 1)

        wash2_flags = {
            (candidate["candidate_id"], candidate["polymer"], candidate["solvent"]): 1
            for candidate in wash2_candidates
        }
        model.slot_pool_wash2 = pyo.ConstraintList()
        for polymer in polymers_available:
            for solvent in all_solvents:
                model.slot_pool_wash2.add(
                    model.b[polymer, solvent]
                    == sum(
                        wash2_flags.get((candidate_id, polymer, solvent), 0) * model.slot_pool_wash2_select[candidate_id]
                        for candidate_id in model.slot_pool_wash2_ids
                    )
                )
    else:
        model.slot_pool_stage2 = pyo.Constraint(expr=model.y["st2"] == 0)
        model.slot_pool_wash2 = pyo.ConstraintList()
        for polymer in polymers_available:
            for solvent in all_solvents:
                model.slot_pool_wash2.add(model.b[polymer, solvent] == 0)

    return True, None


def _annotate_frontier_frame_with_routes(
    frame: pd.DataFrame,
    route_specs: list[dict[str, Any]],
    *,
    route_pool_mode: str = "exact",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    def _parse_selection(value: Any) -> tuple[str, str] | tuple[()]:
        if isinstance(value, (list, tuple, set)):
            items = [str(item) for item in value if str(item)]
        elif value is None or value == "":
            items = []
        else:
            items = [str(value)]
        if not items:
            return ()
        if len(items) != 1 or "-" not in items[0]:
            return ()
        polymer, solvent = items[0].split("-", 1)
        return (polymer.strip(), solvent.strip())

    route_by_signature = {
        (
            tuple(str(item) for item in (route.get("wash1_polymer"), route.get("wash1_solvent")) if item),
            tuple(
                str(item)
                for item in (route.get("wash2_polymer"), route.get("wash2_solvent"))
                if item
            ),
        ): route
        for route in route_specs
    }
    wash1_origin_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    wash2_origin_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for route in route_specs:
        wash1_key = (
            str(route.get("wash1_polymer") or ""),
            str(route.get("wash1_solvent") or ""),
        )
        wash1_origin_map.setdefault(wash1_key, []).append(route)
        if route.get("has_wash2"):
            wash2_key = (
                str(route.get("wash2_polymer") or ""),
                str(route.get("wash2_solvent") or ""),
            )
            wash2_origin_map.setdefault(wash2_key, []).append(route)

    def _origin_route_fields(candidates: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
        route_ids = [str(candidate.get("route_id") or "") for candidate in candidates if str(candidate.get("route_id") or "")]
        if not route_ids:
            return None, []
        return route_ids[0], route_ids

    annotated = frame.copy()
    route_ids: list[str] = []
    matched_route_ids: list[str | None] = []
    ranks: list[Any] = []
    polymer_maps: list[dict[str, str] | None] = []
    selection_origins: list[str] = []
    wash1_origin_route_ids: list[list[str]] = []
    wash2_origin_route_ids: list[list[str]] = []
    wash1_origin_route_id: list[str | None] = []
    wash2_origin_route_id: list[str | None] = []
    for _, row in annotated.iterrows():
        wash1_signature = _parse_selection(row.get("wash1"))
        wash2_signature = _parse_selection(row.get("wash2"))
        signature = (wash1_signature, wash2_signature)
        matched = route_by_signature.get(signature)
        wash1_origin_primary, wash1_origin_all = _origin_route_fields(wash1_origin_map.get(wash1_signature, []))
        wash2_origin_primary, wash2_origin_all = _origin_route_fields(wash2_origin_map.get(wash2_signature, []))
        selection_origin = "exact_route"
        matched_route_id = str((matched or {}).get("route_id") or "") or None
        polymer_solvent_map = (matched or {}).get("polymer_solvent_map")
        rank = (matched or {}).get("rank")
        if route_pool_mode == "slot_independent" and matched is None:
            selection_origin = "cross_product"
            polymer_solvent_map = {
                polymer: solvent
                for polymer, solvent in (wash1_signature, wash2_signature)
                if polymer and solvent
            } or None
            rank = None
        route_ids.append(matched_route_id or "")
        matched_route_ids.append(matched_route_id)
        ranks.append(rank)
        polymer_maps.append(polymer_solvent_map)
        selection_origins.append(selection_origin)
        wash1_origin_route_id.append(wash1_origin_primary)
        wash2_origin_route_id.append(wash2_origin_primary)
        wash1_origin_route_ids.append(wash1_origin_all)
        wash2_origin_route_ids.append(wash2_origin_all)

    annotated["route_id"] = route_ids
    annotated["matched_route_id"] = matched_route_ids
    annotated["rank"] = ranks
    annotated["polymer_solvent_map"] = polymer_maps
    annotated["selection_origin"] = selection_origins
    annotated["wash1_origin_route_id"] = wash1_origin_route_id
    annotated["wash2_origin_route_id"] = wash2_origin_route_id
    annotated["wash1_origin_route_ids"] = wash1_origin_route_ids
    annotated["wash2_origin_route_ids"] = wash2_origin_route_ids
    return annotated


def _point_dominates(candidate: dict[str, Any], baseline: dict[str, Any], *, y_key: str) -> bool:
    if y_key == "emissions":
        y_not_worse = float(candidate.get(y_key, 0.0) or 0.0) <= float(baseline.get(y_key, 0.0) or 0.0)
        y_strict = float(candidate.get(y_key, 0.0) or 0.0) < float(baseline.get(y_key, 0.0) or 0.0)
    else:
        y_not_worse = float(candidate.get(y_key, 0.0) or 0.0) >= float(baseline.get(y_key, 0.0) or 0.0)
        y_strict = float(candidate.get(y_key, 0.0) or 0.0) > float(baseline.get(y_key, 0.0) or 0.0)

    cost_candidate = float(candidate.get("total_cost", 0.0) or 0.0)
    cost_baseline = float(baseline.get("total_cost", 0.0) or 0.0)
    cost_not_worse = cost_candidate <= cost_baseline
    cost_strict = cost_candidate < cost_baseline
    return cost_not_worse and y_not_worse and (cost_strict or y_strict)


def _find_dominating_point(
    baseline_points: list[dict[str, Any]],
    frontier_points: list[dict[str, Any]],
    *,
    y_key: str,
) -> int | None:
    best_point_id: int | None = None
    best_score: tuple[float, float] | None = None
    for baseline in baseline_points:
        baseline_cost = float(baseline.get("total_cost", 0.0) or 0.0)
        baseline_y = float(baseline.get(y_key, 0.0) or 0.0)
        for candidate in frontier_points:
            if not _point_dominates(candidate, baseline, y_key=y_key):
                continue
            candidate_cost = float(candidate.get("total_cost", 0.0) or 0.0)
            candidate_y = float(candidate.get(y_key, 0.0) or 0.0)
            cost_gap = baseline_cost - candidate_cost
            y_gap = (baseline_y - candidate_y) if y_key == "emissions" else (candidate_y - baseline_y)
            score = (cost_gap, y_gap)
            if best_score is None or score > best_score:
                best_score = score
                try:
                    best_point_id = int(candidate.get("point_id"))
                except (TypeError, ValueError):
                    best_point_id = None
    return best_point_id


def _apply_route_constraints(
    model,
    route_spec: dict[str, Any],
    *,
    polymers_available: list[str],
    all_solvents: list[str],
) -> tuple[bool, str | None]:
    """Fix Pyomo wash variables to the exact (polymer, solvent) assignments from a route.

    When `route_candidates` are supplied with constraint_mode in {fixed, hard,
    ranked_soft}, the optimizer must be forced to USE the upstream STRAP route
    rather than treating solvent shortlists as loose filters it can ignore by
    selecting a non-STRAP pathway. This helper fixes `m.x["st1"]=1`, the
    `m.a[p, s]` Wash 1 decision variables, `m.y["st2"]`, and the `m.b[p, s]`
    Wash 2 variables according to the route's polymer_solvent_map. Returns
    (enforced, reason) where reason is non-null only if the route could not
    be applied (so the caller can surface it in the result).
    """
    mapping = route_spec.get("polymer_solvent_map") or {}
    sequence = [_normalize_optimization_polymer(p) for p in route_spec.get("sequence") or []]
    ordered = [p for p in sequence if p is not None and p in mapping and p in polymers_available]
    if not ordered:
        # Fallback: use any polymers in the mapping in a stable order (PE first).
        ordered = [p for p in ("PE", "EVOH") if p in mapping and p in polymers_available]
    if not ordered:
        return False, "route has no polymer-solvent pairs usable by the current optimizer model"

    solvent_set = set(all_solvents)
    wash1_polymer = ordered[0]
    wash1_solvent = mapping.get(wash1_polymer)
    if wash1_solvent not in solvent_set:
        return False, f"Wash 1 solvent '{wash1_solvent}' not present in optimizer catalog"

    # Wash 1 enforcement
    model.x["st1"].fix(1)
    for polymer in polymers_available:
        for solvent in all_solvents:
            if polymer == wash1_polymer and solvent == wash1_solvent:
                model.a[polymer, solvent].fix(1)
            else:
                model.a[polymer, solvent].fix(0)

    # Wash 2 enforcement: mirror Wash 1 pattern if a second polymer is declared,
    # otherwise force the entire Wash 2 stage off.
    if len(ordered) >= 2:
        wash2_polymer = ordered[1]
        wash2_solvent = mapping.get(wash2_polymer)
        if wash2_solvent not in solvent_set:
            return False, f"Wash 2 solvent '{wash2_solvent}' not present in optimizer catalog"
        model.y["st2"].fix(1)
        for polymer in polymers_available:
            for solvent in all_solvents:
                if polymer == wash2_polymer and solvent == wash2_solvent:
                    model.b[polymer, solvent].fix(1)
                else:
                    model.b[polymer, solvent].fix(0)
    else:
        model.y["st2"].fix(0)
        for polymer in polymers_available:
            for solvent in all_solvents:
                model.b[polymer, solvent].fix(0)

    return True, None


def _run_pareto_with_route_pool(
    context: dict[str, Any],
    *,
    route_candidates: list[dict[str, Any]],
    feed: float,
    x_metric: str,
    y_metric: str,
    n_points: int,
) -> str:
    """Solve one Pareto sweep over a shortlist-constrained route pool."""
    sets = context["data"].get("sets") or {}
    polymers_available = list(sets.get("P") or ["PE", "EVOH"])
    all_solvents = list(sets.get("S") or [])
    route_pool_mode = str(context.get("route_pool_mode") or "exact")
    candidate_telemetry = _build_candidate_telemetry(context)
    normalized_routes, skipped_reports = _normalize_route_pool(
        route_candidates,
        polymers_available=polymers_available,
        all_solvents=all_solvents,
    )

    if not normalized_routes:
        result_payload = {
            "analysis_type": "pareto_front",
            "schema_version": "1.5",
            "x_metric": x_metric,
            "y_metric": y_metric,
            "scenario": context["scenario"],
            "feed": feed,
            "feed_composition": context["fractions"],
            "constraint_mode": context["constraint_mode"],
            "fallback_policy": context["fallback_policy"],
            "route_pool_mode": route_pool_mode,
            "n_points_requested": n_points,
            "n_points_raw_feasible": 0,
            "n_points_feasible": 0,
            "strap_table_rows": context.get("strap_table_rows"),
            "ideal_points": {},
            "points": [],
            "route_candidates": route_candidates,
            "route_reports": skipped_reports,
            "n_routes_requested": len(route_candidates),
            "n_routes_solved": 0,
            "frontier_summary": {
                "n_routes_on_frontier": 0,
                "route_ids_on_frontier": [],
                "n_distinct_stage3_techs": 0,
                "distinct_stage3_techs": [],
                "n_equivalent_design_variants": 0,
            },
            "requested_solvent_filters": context["requested_filters"],
            "applied_solvent_filters": context["applied_filters"],
            "solvent_filter_warnings": context["filter_warnings"],
            "solvent_filter_status": context["filter_status"],
            "simulation_failures": context.get("simulation_failures", []),
            "simulation_skips": context.get("simulation_skips", []),
            "candidate_telemetry": candidate_telemetry,
            "candidate_summary": _build_pareto_candidate_summary(context, []),
            "tool_name": "run_waste_management_pareto",
            "success": True,
        }
        display = "## Waste Optimization Pareto Front\n\n"
        display += "No shortlisted route candidates could be normalized into the optimizer solvent catalog.\n"
        for report in skipped_reports:
            display += f"- **{report.get('route_id') or 'route'}:** {report.get('reason', 'skipped')}\n"
        return json_tool_response(display, result_payload)

    def _build_and_constrain():
        model = build_model(context["data"], context["config"])
        if route_pool_mode == "slot_independent":
            enforced, reason = _apply_slot_independent_constraints(
                model,
                normalized_routes,
                polymers_available=polymers_available,
                all_solvents=all_solvents,
            )
        else:
            enforced, reason = _apply_route_pool_constraints(
                model,
                normalized_routes,
                polymers_available=polymers_available,
                all_solvents=all_solvents,
            )
        return model, enforced, reason

    cost_model, enforced, reason = _build_and_constrain()
    if not enforced:
        return json_tool_error(
            reason or "Failed to apply pooled route constraints.",
            tool_name="run_waste_management_pareto",
        )
    cost_opt, cost_failure_reason = _retry_constrained_objective(
        _build_and_constrain,
        "min_total_cost",
    )
    if not cost_opt:
        result_payload = {
            "analysis_type": "pareto_front",
            "schema_version": "1.5",
            "x_metric": x_metric,
            "y_metric": y_metric,
            "scenario": context["scenario"],
            "feed": feed,
            "feed_composition": context["fractions"],
            "constraint_mode": context["constraint_mode"],
            "fallback_policy": context["fallback_policy"],
            "route_pool_mode": route_pool_mode,
            "n_points_requested": n_points,
            "n_points_raw_feasible": 0,
            "n_points_feasible": 0,
            "strap_table_rows": context.get("strap_table_rows"),
            "ideal_points": {},
            "points": [],
            "route_candidates": route_candidates,
            "route_reports": skipped_reports + [
                {
                    "route_id": route["route_id"],
                    "rank": route.get("rank"),
                    "status": "infeasible",
                    "reason": cost_failure_reason or "cost-anchor solve failed under pooled route constraints",
                    "polymer_solvent_map": route.get("polymer_solvent_map"),
                }
                for route in normalized_routes
            ],
            "n_routes_requested": len(route_candidates),
            "n_routes_solved": 0,
            "frontier_summary": {
                "n_routes_on_frontier": 0,
                "route_ids_on_frontier": [],
                "n_distinct_stage3_techs": 0,
                "distinct_stage3_techs": [],
                "n_equivalent_design_variants": 0,
            },
            "requested_solvent_filters": context["requested_filters"],
            "applied_solvent_filters": context["applied_filters"],
            "solvent_filter_warnings": context["filter_warnings"],
            "solvent_filter_status": context["filter_status"],
            "simulation_failures": context.get("simulation_failures", []),
            "simulation_skips": context.get("simulation_skips", []),
            "candidate_telemetry": candidate_telemetry,
            "candidate_summary": _build_pareto_candidate_summary(context, []),
            "tool_name": "run_waste_management_pareto",
            "success": True,
        }
        display = "## Waste Optimization Pareto Front\n\nNo feasible pooled route solve was found for the shortlisted routes.\n"
        return json_tool_response(display, result_payload)

    y_objective = "min_emissions" if y_metric == "emissions" else "max_circularity"
    y_opt, y_failure_reason = _retry_constrained_objective(
        _build_and_constrain,
        y_objective,
    )
    if not y_opt:
        result_payload = {
            "analysis_type": "pareto_front",
            "schema_version": "1.5",
            "x_metric": x_metric,
            "y_metric": y_metric,
            "scenario": context["scenario"],
            "feed": feed,
            "feed_composition": context["fractions"],
            "constraint_mode": context["constraint_mode"],
            "fallback_policy": context["fallback_policy"],
            "route_pool_mode": route_pool_mode,
            "n_points_requested": n_points,
            "n_points_raw_feasible": 0,
            "n_points_feasible": 0,
            "strap_table_rows": context.get("strap_table_rows"),
            "ideal_points": {
                "min_total_cost": {
                    "total_cost": float(cost_opt["total_cost"]),
                    "emissions": float(cost_opt["emissions"]),
                    "circularity_score": max(0.0, min(float(cost_opt["CE"]) / 1_000_000.0, 1.0)),
                }
            },
            "points": [],
            "route_candidates": route_candidates,
            "route_reports": skipped_reports + [
                {
                    "route_id": route["route_id"],
                    "rank": route.get("rank"),
                    "status": "infeasible",
                    "reason": y_failure_reason or f"{y_objective} anchor solve failed under pooled route constraints",
                    "polymer_solvent_map": route.get("polymer_solvent_map"),
                }
                for route in normalized_routes
            ],
            "n_routes_requested": len(route_candidates),
            "n_routes_solved": 0,
            "frontier_summary": {
                "n_routes_on_frontier": 0,
                "route_ids_on_frontier": [],
                "n_distinct_stage3_techs": 0,
                "distinct_stage3_techs": [],
                "n_equivalent_design_variants": 0,
            },
            "requested_solvent_filters": context["requested_filters"],
            "applied_solvent_filters": context["applied_filters"],
            "solvent_filter_warnings": context["filter_warnings"],
            "solvent_filter_status": context["filter_status"],
            "simulation_failures": context.get("simulation_failures", []),
            "simulation_skips": context.get("simulation_skips", []),
            "candidate_telemetry": candidate_telemetry,
            "candidate_summary": _build_pareto_candidate_summary(context, []),
            "tool_name": "run_waste_management_pareto",
            "success": True,
        }
        display = (
            "## Waste Optimization Pareto Front\n\n"
            f"No feasible pooled route solve was found for the `{y_objective}` anchor under the shortlisted routes.\n"
        )
        return json_tool_response(display, result_payload)

    frontier, sweep_failure_reason = _run_constrained_pareto_sweep(
        _build_and_constrain,
        y_metric=y_metric,
        cost_opt=cost_opt,
        y_opt=y_opt,
        n_points=n_points,
    )
    if frontier.empty:
        result_payload = {
            "analysis_type": "pareto_front",
            "schema_version": "1.5",
            "x_metric": x_metric,
            "y_metric": y_metric,
            "scenario": context["scenario"],
            "feed": feed,
            "feed_composition": context["fractions"],
            "constraint_mode": context["constraint_mode"],
            "fallback_policy": context["fallback_policy"],
            "route_pool_mode": route_pool_mode,
            "n_points_requested": n_points,
            "n_points_raw_feasible": 0,
            "n_points_feasible": 0,
            "strap_table_rows": context.get("strap_table_rows"),
            "ideal_points": {
                "min_total_cost": {
                    "total_cost": float(cost_opt["total_cost"]),
                    "emissions": float(cost_opt["emissions"]),
                    "circularity_score": max(0.0, min(float(cost_opt["CE"]) / 1_000_000.0, 1.0)),
                },
                y_objective: {
                    "total_cost": float(y_opt["total_cost"]),
                    "emissions": float(y_opt["emissions"]),
                    "circularity_score": max(0.0, min(float(y_opt["CE"]) / 1_000_000.0, 1.0)),
                },
            },
            "points": [],
            "route_candidates": route_candidates,
            "route_reports": skipped_reports + [
                {
                    "route_id": route["route_id"],
                    "rank": route.get("rank"),
                    "status": "infeasible",
                    "reason": sweep_failure_reason or "Pareto sweep returned no feasible points under pooled route constraints",
                    "polymer_solvent_map": route.get("polymer_solvent_map"),
                }
                for route in normalized_routes
            ],
            "n_routes_requested": len(route_candidates),
            "n_routes_solved": 0,
            "frontier_summary": {
                "n_routes_on_frontier": 0,
                "route_ids_on_frontier": [],
                "n_distinct_stage3_techs": 0,
                "distinct_stage3_techs": [],
                "n_equivalent_design_variants": 0,
            },
            "requested_solvent_filters": context["requested_filters"],
            "applied_solvent_filters": context["applied_filters"],
            "solvent_filter_warnings": context["filter_warnings"],
            "solvent_filter_status": context["filter_status"],
            "simulation_failures": context.get("simulation_failures", []),
            "simulation_skips": context.get("simulation_skips", []),
            "candidate_telemetry": candidate_telemetry,
            "candidate_summary": _build_pareto_candidate_summary(context, []),
            "tool_name": "run_waste_management_pareto",
            "success": True,
        }
        display = (
            "## Waste Optimization Pareto Front\n\n"
            "No feasible Pareto points were found for the shortlisted pooled routes.\n"
        )
        return json_tool_response(display, result_payload)

    annotated_frontier = _annotate_frontier_frame_with_routes(
        frontier,
        normalized_routes,
        route_pool_mode=route_pool_mode,
    )
    points = _frame_to_pareto_points(annotated_frontier)
    y_key = "emissions" if y_metric == "emissions" else "circularity_score"
    frontier_points = _non_dominated(points, y_key=y_key)
    for idx, point in enumerate(frontier_points, start=1):
        point["point_id"] = idx

    route_reports = list(skipped_reports)
    raw_exact_route_ids = {
        str(route_id)
        for route_id in annotated_frontier.get("matched_route_id", pd.Series(dtype=str)).astype(str).tolist()
        if route_id and route_id != "None"
    }
    frontier_exact_route_ids = {
        str(point.get("matched_route_id") or "")
        for point in frontier_points
        if str(point.get("matched_route_id") or "")
    }
    contributing_route_ids = {
        route_id
        for point in points
        for route_id in (
            [str(point.get("matched_route_id") or "")]
            + [str(route_id) for route_id in point.get("wash1_origin_route_ids", [])]
            + [str(route_id) for route_id in point.get("wash2_origin_route_ids", [])]
        )
        if route_id
    }
    y_key = "emissions" if y_metric == "emissions" else "circularity_score"
    for route in normalized_routes:
        route_id = str(route["route_id"])
        route_points = [
            point
            for point in points
            if str(point.get("matched_route_id") or "") == route_id
            and str(point.get("selection_origin") or "exact_route") == "exact_route"
        ]
        route_frontier_points = [
            point
            for point in frontier_points
            if str(point.get("matched_route_id") or "") == route_id
            and str(point.get("selection_origin") or "exact_route") == "exact_route"
        ]
        route_contributed_to_frontier = any(
            route_id in point.get("wash1_origin_route_ids", []) or route_id in point.get("wash2_origin_route_ids", [])
            for point in frontier_points
        )
        stage3_techs_explored = sorted(
            {
                stage
                for point in route_points
                for stage in point.get("stage3_variants", []) or point.get("stage3_tech", [])
                if stage
            }
        )
        dominating_point_id = _find_dominating_point(route_points, frontier_points, y_key=y_key) if route_points else None
        if route_frontier_points:
            status = "feasible"
            reason_text = ""
        elif route_points and dominating_point_id is not None:
            status = "dominated"
            reason_text = "exact upstream route is feasible but dominated on the Pareto front"
        elif route_contributed_to_frontier and route_pool_mode == "slot_independent":
            status = "not_selected"
            reason_text = "route contributed slot candidates to the cross-product frontier, but the exact upstream route did not appear"
        elif route_pool_mode == "slot_independent":
            status = "not_selected"
            reason_text = "exact upstream route did not appear on the slot-independent Pareto front"
        else:
            status = "not_selected"
            reason_text = "route was feasible in the shortlist but not selected in the pooled Pareto sweep"
        route_reports.append(
            {
                "route_id": route_id,
                "rank": route.get("rank"),
                "status": status,
                "reason": reason_text,
                "dominating_point_id": dominating_point_id,
                "polymer_solvent_map": route.get("polymer_solvent_map"),
                "n_points_raw": int((annotated_frontier.get("matched_route_id") == route_id).sum()) if "matched_route_id" in annotated_frontier else 0,
                "n_points_unique": len(route_points),
                "n_points_on_frontier": len(route_frontier_points),
                "stage3_techs_explored": stage3_techs_explored,
                "frontier_stage3_techs": sorted(
                    {
                        stage
                        for point in route_frontier_points
                        for stage in point.get("stage3_variants", []) or point.get("stage3_tech", [])
                        if stage
                    }
                ),
            }
        )

    frontier_stage3_techs = sorted(
        {
            stage
            for point in frontier_points
            for stage in point.get("stage3_variants", []) or point.get("stage3_tech", [])
            if stage
        }
    )
    ideal_points = (
        {
            "min_total_cost": {
                "total_cost": float(cost_opt["total_cost"]),
                "emissions": float(cost_opt["emissions"]),
            },
            "min_emissions": {
                "total_cost": float(y_opt["total_cost"]),
                "emissions": float(y_opt["emissions"]),
            },
        }
        if y_metric == "emissions"
        else {
            "min_total_cost": {
                "total_cost": float(cost_opt["total_cost"]),
                "circularity_score": max(0.0, min(float(cost_opt["CE"]) / 1_000_000.0, 1.0)),
            },
            "max_circularity": {
                "total_cost": float(y_opt["total_cost"]),
                "circularity_score": max(0.0, min(float(y_opt["CE"]) / 1_000_000.0, 1.0)),
            },
        }
    )
    result_payload = {
        "analysis_type": "pareto_front",
        "schema_version": "1.5",
        "x_metric": x_metric,
        "y_metric": y_metric,
        "scenario": context["scenario"],
        "feed": feed,
        "feed_composition": context["fractions"],
        "constraint_mode": context["constraint_mode"],
        "fallback_policy": context["fallback_policy"],
        "route_pool_mode": route_pool_mode,
        "n_points_requested": n_points,
        "n_points_raw_feasible": int(len(annotated_frontier)),
        "n_points_feasible": len(frontier_points),
        "strap_table_rows": context.get("strap_table_rows"),
        "ideal_points": ideal_points,
        "points": frontier_points,
        "route_candidates": route_candidates,
        "route_reports": route_reports,
        "n_routes_requested": len(route_candidates),
        "n_routes_solved": len(contributing_route_ids),
        "frontier_summary": {
            "n_routes_on_frontier": len(frontier_exact_route_ids),
            "route_ids_on_frontier": sorted(frontier_exact_route_ids),
            "n_distinct_stage3_techs": len(frontier_stage3_techs),
            "distinct_stage3_techs": frontier_stage3_techs,
            "n_cross_product_points": sum(
                1 for point in frontier_points if str(point.get("selection_origin") or "exact_route") == "cross_product"
            ),
            "n_equivalent_design_variants": sum(
                max(int(point.get("n_equivalent_designs") or 1) - 1, 0)
                for point in frontier_points
            ),
        },
        "requested_solvent_filters": context["requested_filters"],
        "applied_solvent_filters": context["applied_filters"],
        "solvent_filter_warnings": context["filter_warnings"],
        "solvent_filter_status": context["filter_status"],
        "simulation_failures": context.get("simulation_failures", []),
        "simulation_skips": context.get("simulation_skips", []),
        "candidate_telemetry": candidate_telemetry,
        "candidate_summary": _build_pareto_candidate_summary(context, frontier_points),
        "tool_name": "run_waste_management_pareto",
        "success": True,
    }

    display = "## Waste Optimization Pareto Front\n\n"
    display += f"**Scenario:** {context['scenario']} | **X metric:** {x_metric} | **Y metric:** {y_metric}\n"
    display += f"**Feed:** {feed} tonnes/year ({context['fractions']['PE']*100}% PE, {context['fractions']['PET']*100}% PET, {context['fractions']['N6']*100}% N6, {context['fractions']['EVOH']*100}% EVOH)\n"
    display += f"**Feasible Pareto points:** {len(frontier_points)} unique / {len(annotated_frontier)} raw / {n_points} requested\n"
    display += f"**Shortlisted routes:** {len(normalized_routes)} usable of {len(route_candidates)} requested\n"
    display += f"**Routes appearing on frontier:** {len(frontier_exact_route_ids)}\n"
    display += (
        f"**Constraint mode:** {context['constraint_mode']} | **Fallback policy:** {context['fallback_policy']} "
        f"| **Route pool mode:** {route_pool_mode}\n"
    )
    if context.get("strap_table_rows") is not None:
        display += f"**Compiled STRAP rows:** {context['strap_table_rows']}\n"
    requested_counts = candidate_telemetry["requested"]["counts_by_polymer"]
    surviving_counts = candidate_telemetry["surviving"]["counts_by_polymer"]
    if requested_counts:
        display += (
            "**Candidate telemetry:** requested "
            + ", ".join(f"{polymer}={count}" for polymer, count in sorted(requested_counts.items()))
            + "\n"
        )
    if surviving_counts:
        display += (
            "**Surviving candidates:** "
            + ", ".join(f"{polymer}={count}" for polymer, count in sorted(surviving_counts.items()))
            + "\n"
        )
    if context["filter_warnings"]:
        for warning in context["filter_warnings"]:
            display += f"- **Filter note:** {warning}\n"
    if context.get("simulation_skips"):
        display += f"- **BioSTEAM note:** Skipped {len(context['simulation_skips'])} baseline-backed candidate simulation(s) because workbook coefficients were already sufficient.\n"
    failure_counts = candidate_telemetry["simulation"]["failure_counts_by_class"]
    if failure_counts:
        display += (
            "- **BioSTEAM failures by class:** "
            + ", ".join(f"{name}={count}" for name, count in sorted(failure_counts.items()))
            + "\n"
        )
    for report in skipped_reports:
        display += f"- **Route note ({report.get('route_id') or 'route'}):** {report.get('reason', 'skipped')}\n"
    if frontier_points:
        display += "\n### Pareto Points\n"
        for point in frontier_points:
            y_value = point["emissions"] if y_metric == "emissions" else point["circularity_score"]
            y_label = "Emissions (tCO2)" if y_metric == "emissions" else "Circularity (0-1)"
            display += (
                f"- **Point {point['point_id']}:** Route {point.get('route_id') or 'unmatched'}; "
                f"total cost ${point['total_cost']:,.2f}; {y_label} {y_value:,.4f}; "
                f"Washes {point['wash1_selection']} / {point['wash2_selection']}; "
                f"Stage 3 {', '.join(point.get('stage3_variants') or point.get('stage3_tech') or ['none'])}; "
                f"origin {point.get('selection_origin', 'exact_route')}"
            )
            if int(point.get("n_equivalent_designs") or 1) > 1:
                display += f"; equivalent design variants {point['n_equivalent_designs']}"
            display += "\n"
    else:
        display += "\nNo feasible Pareto points were found for the shortlisted pooled route set.\n"

    return json_tool_response(display, result_payload)


def _pareto_sweep_one_route(
    context: dict[str, Any],
    route: dict[str, Any],
    *,
    y_metric: str,
    n_points: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve cost/y anchors and epsilon-constraint sweep for a single enforced route.

    Returns (route_points, route_meta). route_points is a list of per-point
    dicts with route_id and rank tagged on. route_meta captures the status
    ('solved', 'infeasible', 'model_error') and any human-readable reason,
    so the caller can decide whether to keep or drop the route.
    """
    sets = context["data"].get("sets") or {}
    polymers_available = list(sets.get("P") or ["PE", "EVOH"])
    all_solvents = list(sets.get("S") or [])

    def _build_and_constrain():
        model = build_model(context["data"], context["config"])
        enforced, reason = _apply_route_constraints(
            model,
            route,
            polymers_available=polymers_available,
            all_solvents=all_solvents,
        )
        return model, enforced, reason

    cost_opt, cost_failure_reason = _retry_constrained_objective(
        _build_and_constrain,
        "min_total_cost",
    )
    if not cost_opt:
        return [], {
            "route_id": route.get("route_id"),
            "rank": route.get("rank"),
            "status": "infeasible",
            "reason": cost_failure_reason or "cost-anchor solve failed under route enforcement",
            "polymer_solvent_map": route.get("polymer_solvent_map"),
        }

    y_objective = "min_emissions" if y_metric == "emissions" else "max_circularity"
    y_opt, y_failure_reason = _retry_constrained_objective(
        _build_and_constrain,
        y_objective,
    )
    if not y_opt:
        return [], {
            "route_id": route.get("route_id"),
            "rank": route.get("rank"),
            "status": "infeasible",
            "reason": y_failure_reason or f"{y_objective} anchor solve failed under route enforcement",
            "polymer_solvent_map": route.get("polymer_solvent_map"),
        }

    sweep_model, _, _ = _build_and_constrain()
    try:
        if y_metric == "emissions":
            frontier = pareto_cost_vs_emissions(
                sweep_model,
                emission_ideal=float(y_opt["emissions"]),
                emission_nonideal=float(cost_opt["emissions"]),
                n_points=n_points,
                solver_name="scip",
            )
        else:
            frontier = pareto_cost_vs_ce(
                sweep_model,
                ce_nonideal=float(cost_opt["CE"]),
                ce_ideal=float(y_opt["CE"]),
                n_points=n_points,
                solver_name="scip",
            )
    except Exception as exc:
        logger.warning(
            "SCIP Pareto sweep for route %s failed: %s. Retrying with default solver.",
            route.get("route_id"),
            exc,
        )
        sweep_model, _, _ = _build_and_constrain()
        try:
            if y_metric == "emissions":
                frontier = pareto_cost_vs_emissions(
                    sweep_model,
                    emission_ideal=float(y_opt["emissions"]),
                    emission_nonideal=float(cost_opt["emissions"]),
                    n_points=n_points,
                    solver_name=None,
                )
            else:
                frontier = pareto_cost_vs_ce(
                    sweep_model,
                    ce_nonideal=float(cost_opt["CE"]),
                    ce_ideal=float(y_opt["CE"]),
                    n_points=n_points,
                    solver_name=None,
                )
        except Exception as exc2:
            return [], {
                "route_id": route.get("route_id"),
                "rank": route.get("rank"),
                "status": "infeasible",
                "reason": f"both anchor solvers failed: {exc2}",
                "polymer_solvent_map": route.get("polymer_solvent_map"),
            }

    route_points = _frame_to_pareto_points(frontier)
    # Tag each point with route identity for downstream aggregation.
    route_id = str(route.get("route_id") or "")
    rank = route.get("rank")
    for point in route_points:
        point["route_id"] = route_id
        point["rank"] = rank
        point["polymer_solvent_map"] = route.get("polymer_solvent_map")

    stage3_techs_explored = sorted(
        {
            stage
            for point in route_points
            for stage in point.get("stage3_variants", []) or point.get("stage3_tech", [])
            if stage
        }
    )

    return route_points, {
        "route_id": route_id,
        "rank": rank,
        "status": "solved",
        "polymer_solvent_map": route.get("polymer_solvent_map"),
        "n_points_raw": int(len(frontier)),
        "n_points_unique": len(route_points),
        "n_points_on_frontier": 0,
        "stage3_techs_explored": stage3_techs_explored,
        "anchors": {
            "min_total_cost": {
                "total_cost": float(cost_opt.get("total_cost", 0.0)),
                "emissions": float(cost_opt.get("emissions", 0.0)),
                "CE": float(cost_opt.get("CE", 0.0)),
            },
            y_objective: {
                "total_cost": float(y_opt.get("total_cost", 0.0)),
                "emissions": float(y_opt.get("emissions", 0.0)),
                "CE": float(y_opt.get("CE", 0.0)),
            },
        },
    }


def _run_pareto_per_route(
    context: dict[str, Any],
    *,
    route_candidates: list[dict[str, Any]],
    feed: float,
    pe_fraction: float,
    pet_fraction: float,
    n6_fraction: float,
    evoh_fraction: float,
    scenario: str,
    x_metric: str,
    y_metric: str,
    n_points: int,
) -> str:
    """Solve the Pareto sweep once per enforced route and aggregate the frontier.

    Each route's exact (polymer, solvent) wash assignments are fixed in the
    Pyomo model before the anchor solves and epsilon-constraint sweep run, so
    the optimizer cannot silently select a non-STRAP pathway. The aggregated
    result is the union of each route's feasible points filtered to
    non-dominated ones across routes.
    """

    all_points: list[dict[str, Any]] = []
    route_reports: list[dict[str, Any]] = []
    for route in route_candidates:
        points, meta = _pareto_sweep_one_route(
            context,
            route,
            y_metric=y_metric,
            n_points=n_points,
        )
        all_points.extend(points)
        route_reports.append(meta)

    y_key = "emissions" if y_metric == "emissions" else "circularity_score"
    frontier_points = _non_dominated(all_points, y_key=y_key)
    # Re-index point_id to match the aggregated frontier order.
    for idx, point in enumerate(frontier_points, start=1):
        point["point_id"] = idx

    solved_routes = [r for r in route_reports if r.get("status") == "solved"]
    frontier_route_ids = {str(point.get("route_id") or "") for point in frontier_points if str(point.get("route_id") or "")}
    frontier_stage3_techs = sorted(
        {
            stage
            for point in frontier_points
            for stage in point.get("stage3_variants", []) or point.get("stage3_tech", [])
            if stage
        }
    )
    for report in solved_routes:
        route_id = str(report.get("route_id") or "")
        route_frontier_points = [point for point in frontier_points if str(point.get("route_id") or "") == route_id]
        report["n_points_on_frontier"] = len(route_frontier_points)
        report["frontier_stage3_techs"] = sorted(
            {
                stage
                for point in route_frontier_points
                for stage in point.get("stage3_variants", []) or point.get("stage3_tech", [])
                if stage
            }
        )
    result_payload = {
        "analysis_type": "pareto_front",
        "schema_version": "1.1",
        "x_metric": x_metric,
        "y_metric": y_metric,
        "scenario": context["scenario"],
        "feed": feed,
        "feed_composition": context["fractions"],
        "constraint_mode": context["constraint_mode"],
        "fallback_policy": context["fallback_policy"],
        "n_points_requested": n_points,
        "n_points_raw_feasible": sum(int(r.get("n_points_raw", 0) or 0) for r in solved_routes),
        "n_points_feasible": len(frontier_points),
        "strap_table_rows": context.get("strap_table_rows"),
        "ideal_points": {},
        "points": frontier_points,
        "route_candidates": route_candidates,
        "route_reports": route_reports,
        "n_routes_requested": len(route_candidates),
        "n_routes_solved": len(solved_routes),
        "frontier_summary": {
            "n_routes_on_frontier": len(frontier_route_ids),
            "route_ids_on_frontier": sorted(frontier_route_ids),
            "n_distinct_stage3_techs": len(frontier_stage3_techs),
            "distinct_stage3_techs": frontier_stage3_techs,
            "n_equivalent_design_variants": sum(
                max(int(point.get("n_equivalent_designs") or 1) - 1, 0)
                for point in frontier_points
            ),
        },
        "requested_solvent_filters": context["requested_filters"],
        "applied_solvent_filters": context["applied_filters"],
        "solvent_filter_warnings": context["filter_warnings"],
        "solvent_filter_status": context["filter_status"],
        "simulation_failures": context.get("simulation_failures", []),
        "simulation_skips": context.get("simulation_skips", []),
        "candidate_summary": {
            "status": context["filter_status"],
            "warnings": context["filter_warnings"],
            "n_routes_requested": len(route_candidates),
            "n_routes_solved": len(solved_routes),
            "n_points_feasible": len(frontier_points),
        },
        "tool_name": "run_waste_management_pareto",
        "success": bool(frontier_points),
    }

    display = "## Waste Optimization Pareto Front (route-enforced)\n\n"
    display += f"**Scenario:** {context['scenario']} | **X metric:** {x_metric} | **Y metric:** {y_metric}\n"
    display += f"**Feed:** {feed} tonnes/year ({pe_fraction*100}% PE, {pet_fraction*100}% PET, {n6_fraction*100}% N6, {evoh_fraction*100}% EVOH)\n"
    display += (
        f"**Routes:** {len(solved_routes)} of {len(route_candidates)} solved "
        f"under constraint_mode={context['constraint_mode']} | "
        f"fallback_policy={context['fallback_policy']}\n"
    )
    display += f"**Non-dominated Pareto points:** {len(frontier_points)}\n"
    display += (
        f"**Stage-3 technologies on frontier:** "
        f"{', '.join(frontier_stage3_techs) if frontier_stage3_techs else 'none'}\n"
    )

    for report in route_reports:
        if report.get("status") == "solved":
            display += (
                f"- Route {report['route_id']} (rank {report.get('rank')}): "
                f"{report.get('polymer_solvent_map')} → "
                f"{report.get('n_points_unique')} unique points "
                f"({report.get('n_points_raw')} raw), "
                f"{report.get('n_points_on_frontier')} on frontier, "
                f"Stage 3 explored: {', '.join(report.get('stage3_techs_explored') or ['none'])}\n"
            )
        else:
            display += (
                f"- Route {report['route_id']} (rank {report.get('rank')}): "
                f"{report['status']} — {report.get('reason')}\n"
            )

    if frontier_points:
        display += "\n### Aggregated Pareto Points\n"
        for point in frontier_points:
            y_value = point["emissions"] if y_metric == "emissions" else point.get("circularity_score", 0.0)
            y_label = "Emissions (tCO2)" if y_metric == "emissions" else "Circularity (0-1)"
            display += (
                f"- **Point {point['point_id']}** (route {point.get('route_id')}): "
                f"Total cost ${point['total_cost']:,.2f}; "
                f"{y_label} {y_value:,.4f}; "
                f"Stage 3 {', '.join(point.get('stage3_variants') or point.get('stage3_tech') or ['none'])}; "
                f"map {point.get('polymer_solvent_map')}"
            )
            if int(point.get("n_equivalent_designs") or 1) > 1:
                display += f"; equivalent design variants {point['n_equivalent_designs']}"
            display += "\n"
    else:
        display += "\nNo routes produced a feasible Pareto point under the requested sweep.\n"

    if context["filter_warnings"]:
        for warning in context["filter_warnings"]:
            display += f"- **Filter note:** {warning}\n"
    if context.get("simulation_skips"):
        display += f"- **BioSTEAM note:** Skipped {len(context['simulation_skips'])} baseline-backed candidate simulation(s) because workbook coefficients were already sufficient.\n"

    return json_tool_response(display, result_payload)


def _non_dominated(points: list[dict[str, Any]], *, y_key: str) -> list[dict[str, Any]]:
    """Filter to non-dominated points (minimize total_cost, minimize y_key)."""
    if not points:
        return []
    indexed = list(enumerate(points))
    # For circularity we want to MAXIMIZE, not minimize — invert for the test.
    def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        a_cost, b_cost = a.get("total_cost", float("inf")), b.get("total_cost", float("inf"))
        a_y, b_y = a.get(y_key, float("inf")), b.get(y_key, float("inf"))
        if y_key == "circularity_score":
            # Maximize circularity → point a dominates b if a_cost <= b_cost AND a_y >= b_y, strict somewhere
            return a_cost <= b_cost and a_y >= b_y and (a_cost < b_cost or a_y > b_y)
        return a_cost <= b_cost and a_y <= b_y and (a_cost < b_cost or a_y < b_y)

    nondom: list[dict[str, Any]] = []
    for i, pi in indexed:
        dominated = False
        for j, pj in indexed:
            if i == j:
                continue
            if _dominates(pj, pi):
                dominated = True
                break
        if not dominated:
            nondom.append(pi)
    return nondom

@safe_tool_wrapper(structured_output=True)
def run_waste_management_optimization(
    feed: float,
    pe_fraction: float,
    pet_fraction: float,
    n6_fraction: float,
    evoh_fraction: float,
    scenario: str = 'A',
    objective: str = 'max_profit',
    candidate_solvents: list[str] | str | None = None,
    polymer_solvent_filters_json: dict[str, Any] | str | None = None,
    stage_candidates_json: dict[str, Any] | str | None = None,
    constraint_mode: str | None = None,
    fallback_policy: str | None = None,
) -> str:
    """Run the PIW multi-layer plastic waste optimization model.
    This tools recalculates costs and operational parameters using BioSTEAM based on the specified input 
    fractions and total feed, updates the base Excel data, runs the optimization over all simulated pathways 
    using the available solvers, and returns the optimal configuration.
    
    Args:
        feed: Total mixed plastic feed in tonnes/year (e.g. 8000).
        pe_fraction: Fraction of Polyethylene (PE) in the feed (0.0 to 1.0).
        pet_fraction: Fraction of Polyethylene terephthalate (PET) in the feed.
        n6_fraction: Fraction of Nylon-6 (N6) in the feed.
        evoh_fraction: Fraction of Ethylene vinyl alcohol (EVOH) in the feed.
            (Note: fractions must sum to 1.0)
        scenario: Location scenario 'A', 'B', or 'C'. Default is 'A'.
        objective: 'max_profit', 'min_emissions', or 'max_circularity'. Default 'max_profit'.
    
    WHEN TO USE:
    - "Optimize waste management for a plant with 8000 feed composed of 60% PE, 20% PET..."
    - "Evaluate the maximum profit of processing 10000 tons of 50/50 PE/EVOH waste"
    """
    temp_dir: Path | None = None
    try:
        context = _prepare_optimization_context(
            feed=feed,
            pe_fraction=pe_fraction,
            pet_fraction=pet_fraction,
            n6_fraction=n6_fraction,
            evoh_fraction=evoh_fraction,
            scenario=scenario,
            candidate_solvents=candidate_solvents,
            polymer_solvent_filters_json=polymer_solvent_filters_json,
            stage_candidates_json=stage_candidates_json,
            constraint_mode=constraint_mode,
            fallback_policy=fallback_policy,
        )
        temp_dir = context["temp_dir"]
        if "infeasible_response" in context:
            return context["infeasible_response"]

        m = build_model(context["data"], context["config"])
        results = _solve_objective_with_fallback(m, objective)
        if not results:
            return json_tool_error(
                "Optimization model did not return a feasible solution.",
                tool_name="run_waste_management_optimization",
            )
        candidate_telemetry = _build_candidate_telemetry(context)
        # Normalize circularity score (CE) to 0‑1 range as per paper
        raw_ce = results.get("CE", 0)
        circularity_score = max(0.0, min(raw_ce / 1_000_000.0, 1.0))
        results["analysis_type"] = "point_optimum"
        results["schema_version"] = "1.0"
        results["objective"] = objective
        results["raw_circularity_score"] = raw_ce
        results["circularity_score"] = circularity_score
        results["requested_solvent_filters"] = context["requested_filters"]
        results["applied_solvent_filters"] = context["applied_filters"]
        results["solvent_filter_warnings"] = context["filter_warnings"]
        results["solvent_filter_status"] = context["filter_status"]
        results["simulation_failures"] = context.get("simulation_failures", [])
        results["simulation_skips"] = context.get("simulation_skips", [])
        results["candidate_telemetry"] = candidate_telemetry
        results["constraint_mode"] = context["constraint_mode"]
        results["fallback_policy"] = context["fallback_policy"]
        results["scenario"] = context["scenario"]
        results["feed_composition"] = context["fractions"]
        results["strap_table_rows"] = context.get("strap_table_rows")

        display = f"## Multi-layer Plastic Optimization Results\n\n"
        display += f"**Objective:** {objective} | **Scenario:** {context['scenario']}\n"
        display += f"**Feed:** {feed} tonnes/year ({pe_fraction*100}% PE, {pet_fraction*100}% PET, {n6_fraction*100}% N6, {evoh_fraction*100}% EVOH)\n\n"
        
        display += "### Optimal Technology Pathways Selected\n"
        display += f"- **Stage 1 (Separation):** {results.get('stage1_tech', [])}\n"
        display += f"- **Stage 2 (Conversion):** {results.get('stage2_tech', [])}\n"
        display += f"- **Stage 3 (End of Life):** {results.get('stage3_tech', [])}\n"
        
        display += "\n### Chosen STRAP Solvents\n"
        display += f"- **Wash 1 (PE Target):** {results.get('wash1_selection', [])}\n"
        display += f"- **Wash 2 (EVOH Target):** {results.get('wash2_selection', [])}\n"
        
        display += "\n### Economic and Environmental Impact\n"
        display += f"- **Total Profit:** ${results.get('profit', 0):,.2f}\n"
        display += f"- **Emissions:** {results.get('emissions', 0):,.2f} tCO2\n"
        display += f"- **Circularity (0‑1):** {results.get('circularity_score', 0):.4f}\n"
        display += f"- **Capital Cost:** ${results.get('capital_cost', 0):,.2f}\n"
        display += f"- **Operational Cost:** ${results.get('operational_cost', 0):,.2f}\n"
        display += f"- **Total Cost:** ${results.get('total_cost', 0):,.2f}\n"
        display += f"- **Constraint mode:** {context['constraint_mode']}\n"
        display += f"- **Fallback policy:** {context['fallback_policy']}\n"
        display += f"- **Solvent shortlist status:** {context['filter_status']}\n"
        if context.get("strap_table_rows") is not None:
            display += f"- **Compiled STRAP rows:** {context['strap_table_rows']}\n"
        requested_counts = candidate_telemetry["requested"]["counts_by_polymer"]
        surviving_counts = candidate_telemetry["surviving"]["counts_by_polymer"]
        if requested_counts:
            display += (
                "- **Requested solvent candidates:** "
                + ", ".join(f"{polymer}={count}" for polymer, count in sorted(requested_counts.items()))
                + "\n"
            )
        if surviving_counts:
            display += (
                "- **Surviving solvent candidates:** "
                + ", ".join(f"{polymer}={count}" for polymer, count in sorted(surviving_counts.items()))
                + "\n"
            )
        if any(context["requested_filters"].get(key) for key in context["requested_filters"]):
            display += f"- **Requested solvent filters:** {context['requested_filters']}\n"
        if context["applied_filters"]:
            display += f"- **Applied solvent filters:** {context['applied_filters']}\n"
        if context["filter_warnings"]:
            for warning in context["filter_warnings"]:
                display += f"- **Filter note:** {warning}\n"
        failure_counts = candidate_telemetry["simulation"]["failure_counts_by_class"]
        if failure_counts:
            display += (
                "- **BioSTEAM failures by class:** "
                + ", ".join(f"{name}={count}" for name, count in sorted(failure_counts.items()))
                + "\n"
            )

        return json_tool_response(display, _serialize_optimization_results(results))
        
    except Exception as e:
        logger.exception("Error in run_waste_management_optimization")
        return json_tool_error(str(e), tool_name="run_waste_management_optimization")
    finally:
        # temp_dir is retained as a hook for future workflows that need an
        # isolated workbook copy; the current pipeline threads data via a
        # compiled DataFrame and never populates it.
        pass


@safe_tool_wrapper(structured_output=True)
def run_waste_management_pareto(
    feed: float,
    pe_fraction: float,
    pet_fraction: float,
    n6_fraction: float,
    evoh_fraction: float,
    scenario: str = "A",
    x_metric: str = "total_cost",
    y_metric: str = "emissions",
    n_points: int = 100,
    candidate_solvents: list[str] | str | None = None,
    polymer_solvent_filters_json: dict[str, Any] | str | None = None,
    stage_candidates_json: dict[str, Any] | str | None = None,
    constraint_mode: str | None = None,
    fallback_policy: str | None = None,
    route_pool_mode: str | None = None,
) -> str:
    """Run an optimization Pareto sweep on top of the staged solvent-candidate path.

    Args:
        feed: Total mixed plastic feed in tonnes/year.
        pe_fraction: Fraction of PE in the feed.
        pet_fraction: Fraction of PET in the feed.
        n6_fraction: Fraction of N6 in the feed.
        evoh_fraction: Fraction of EVOH in the feed.
        scenario: Location scenario 'A', 'B', or 'C'.
        x_metric: Supported value: 'total_cost'.
        y_metric: Supported values: 'emissions' or 'circularity'.
        n_points: Number of epsilon-constraint points to request. Defaults to
            100 when the caller does not specify a value.
        route_pool_mode: 'exact' keeps upstream route tuples intact; 'slot_independent'
            allows Wash 1 and Wash 2 to be combined independently from the shortlisted pool.
    """
    temp_dir: Path | None = None
    try:
        if x_metric != "total_cost":
            return json_tool_error(
                f"Unsupported x_metric '{x_metric}'. Supported value: total_cost.",
                tool_name="run_waste_management_pareto",
            )
        if y_metric not in {"emissions", "circularity"}:
            return json_tool_error(
                f"Unsupported y_metric '{y_metric}'. Supported values: emissions, circularity.",
                tool_name="run_waste_management_pareto",
            )
        if n_points < 2:
            return json_tool_error(
                "n_points must be at least 2 for a Pareto sweep.",
                tool_name="run_waste_management_pareto",
            )

        context = _prepare_optimization_context(
            feed=feed,
            pe_fraction=pe_fraction,
            pet_fraction=pet_fraction,
            n6_fraction=n6_fraction,
            evoh_fraction=evoh_fraction,
            scenario=scenario,
            candidate_solvents=candidate_solvents,
            polymer_solvent_filters_json=polymer_solvent_filters_json,
            stage_candidates_json=stage_candidates_json,
            constraint_mode=constraint_mode,
            fallback_policy=fallback_policy,
            route_pool_mode=route_pool_mode,
        )
        temp_dir = context["temp_dir"]
        if "infeasible_response" in context:
            return context["infeasible_response"]

        # Route-constrained Pareto path: when the adapter supplied
        # route_candidates and the constraint_mode is not purely soft, solve
        # one pooled Pareto sweep over the shortlisted route set. This keeps
        # polymer↔solvent coupling intact without collapsing the entire front
        # to one exact-route solve per candidate.
        route_candidates = context.get("route_candidates") or []
        route_enforcing_modes = {"fixed", "hard", "ranked_soft"}
        route_enforcement_active = (
            bool(route_candidates)
            and context["constraint_mode"] in route_enforcing_modes
        )
        if route_enforcement_active:
            route_response = _run_pareto_with_route_pool(
                context,
                route_candidates=route_candidates,
                feed=feed,
                x_metric=x_metric,
                y_metric=y_metric,
                n_points=n_points,
            )
            route_payload = _extract_tool_data(route_response)
            n_routes_solved = int(route_payload.get("n_routes_solved") or 0)
            n_points_feasible = int(route_payload.get("n_points_feasible") or 0)

            # Only broaden when pooled route enforcement found no frontier at
            # all but at least one shortlisted route remained usable. If no
            # route survived canonicalization, the honest answer is the
            # route-specific failure state rather than a broadened workbook
            # solve that ignores the user's shortlisted routes.
            should_broaden_after_route_failure = (
                context["constraint_mode"] == "ranked_soft"
                and context["fallback_policy"] == "broaden_disclosed"
                and n_routes_solved > 0
                and n_points_feasible == 0
            )
            if n_routes_solved == 0:
                return route_response

            if not should_broaden_after_route_failure:
                return route_response

            route_reports = route_payload.get("route_reports") or []
            route_statuses = ", ".join(
                f"{report.get('route_id', 'route')}={report.get('status', 'unknown')}"
                for report in route_reports
                if isinstance(report, dict)
            ) or "no solved routes"
            context["filter_warnings"].append(
                "Ranked route-constrained Pareto sweep produced no feasible frontier; "
                "broadening to the optimizer-supported candidate catalog under "
                f"broaden_disclosed fallback. Route statuses: {route_statuses}."
            )
            context["filter_status"] = "broadened_after_route_infeasible"

        cost_model = build_model(context["data"], context["config"])
        cost_opt = _solve_objective_with_fallback(cost_model, "min_total_cost")
        if not cost_opt:
            return json_tool_error(
                "Cost-optimal Pareto anchor could not be solved.",
                tool_name="run_waste_management_pareto",
            )

        if y_metric == "emissions":
            y_model = build_model(context["data"], context["config"])
            y_opt = _solve_objective_with_fallback(y_model, "min_emissions")
            if not y_opt:
                return json_tool_error(
                    "Emissions-optimal Pareto anchor could not be solved.",
                    tool_name="run_waste_management_pareto",
                )
            sweep_model = build_model(context["data"], context["config"])
            try:
                frontier = pareto_cost_vs_emissions(
                    sweep_model,
                    emission_ideal=float(y_opt["emissions"]),
                    emission_nonideal=float(cost_opt["emissions"]),
                    n_points=n_points,
                    solver_name="scip",
                )
            except Exception as exc:
                logger.warning("Failed to run SCIP Pareto sweep for emissions: %s. Falling back to available solvers.", exc)
                sweep_model = build_model(context["data"], context["config"])
                frontier = pareto_cost_vs_emissions(
                    sweep_model,
                    emission_ideal=float(y_opt["emissions"]),
                    emission_nonideal=float(cost_opt["emissions"]),
                    n_points=n_points,
                    solver_name=None,
                )
            ideal_points = {
                "min_total_cost": {
                    "total_cost": float(cost_opt["total_cost"]),
                    "emissions": float(cost_opt["emissions"]),
                },
                "min_emissions": {
                    "total_cost": float(y_opt["total_cost"]),
                    "emissions": float(y_opt["emissions"]),
                },
            }
        else:
            y_model = build_model(context["data"], context["config"])
            y_opt = _solve_objective_with_fallback(y_model, "max_circularity")
            if not y_opt:
                return json_tool_error(
                    "Circularity-optimal Pareto anchor could not be solved.",
                    tool_name="run_waste_management_pareto",
                )
            sweep_model = build_model(context["data"], context["config"])
            try:
                frontier = pareto_cost_vs_ce(
                    sweep_model,
                    ce_nonideal=float(cost_opt["CE"]),
                    ce_ideal=float(y_opt["CE"]),
                    n_points=n_points,
                    solver_name="scip",
                )
            except Exception as exc:
                logger.warning("Failed to run SCIP Pareto sweep for circularity: %s. Falling back to available solvers.", exc)
                sweep_model = build_model(context["data"], context["config"])
                frontier = pareto_cost_vs_ce(
                    sweep_model,
                    ce_nonideal=float(cost_opt["CE"]),
                    ce_ideal=float(y_opt["CE"]),
                    n_points=n_points,
                    solver_name=None,
                )
            ideal_points = {
                "min_total_cost": {
                    "total_cost": float(cost_opt["total_cost"]),
                    "circularity_score": max(0.0, min(float(cost_opt["CE"]) / 1_000_000.0, 1.0)),
                },
                "max_circularity": {
                    "total_cost": float(y_opt["total_cost"]),
                    "circularity_score": max(0.0, min(float(y_opt["CE"]) / 1_000_000.0, 1.0)),
                },
            }

        raw_points = _frame_to_pareto_points(frontier)
        y_key = "emissions" if y_metric == "emissions" else "circularity_score"
        points = _non_dominated(raw_points, y_key=y_key)
        for idx, point in enumerate(points, start=1):
            point["point_id"] = idx
        # Flat filter fields are emitted alongside the nested candidate_summary so the
        # deterministic verifier and any downstream consumers can read the same field
        # names on point_optimum and pareto_front without path-specific branching.
        candidate_telemetry = _build_candidate_telemetry(context)
        result_payload = {
            "analysis_type": "pareto_front",
            "schema_version": "1.1",
            "x_metric": x_metric,
            "y_metric": y_metric,
            "scenario": context["scenario"],
            "feed": feed,
            "feed_composition": context["fractions"],
            "constraint_mode": context["constraint_mode"],
            "fallback_policy": context["fallback_policy"],
            "n_points_requested": n_points,
            "n_points_raw_feasible": int(len(frontier)),
            "n_points_feasible": len(points),
            "strap_table_rows": context.get("strap_table_rows"),
            "ideal_points": ideal_points,
            "points": points,
            "frontier_summary": {
                "n_routes_on_frontier": 0,
                "route_ids_on_frontier": [],
                "n_distinct_stage3_techs": len(
                    {
                        stage
                        for point in points
                        for stage in point.get("stage3_variants", []) or point.get("stage3_tech", [])
                        if stage
                    }
                ),
                "distinct_stage3_techs": sorted(
                    {
                        stage
                        for point in points
                        for stage in point.get("stage3_variants", []) or point.get("stage3_tech", [])
                        if stage
                    }
                ),
                "n_equivalent_design_variants": sum(
                    max(int(point.get("n_equivalent_designs") or 1) - 1, 0)
                    for point in points
                ),
            },
            "requested_solvent_filters": context["requested_filters"],
            "applied_solvent_filters": context["applied_filters"],
            "solvent_filter_warnings": context["filter_warnings"],
            "solvent_filter_status": context["filter_status"],
            "simulation_failures": context.get("simulation_failures", []),
            "simulation_skips": context.get("simulation_skips", []),
            "candidate_telemetry": candidate_telemetry,
            "candidate_summary": _build_pareto_candidate_summary(context, points),
            "tool_name": "run_waste_management_pareto",
            "success": True,
        }

        display = "## Waste Optimization Pareto Front\n\n"
        display += f"**Scenario:** {context['scenario']} | **X metric:** {x_metric} | **Y metric:** {y_metric}\n"
        display += f"**Feed:** {feed} tonnes/year ({pe_fraction*100}% PE, {pet_fraction*100}% PET, {n6_fraction*100}% N6, {evoh_fraction*100}% EVOH)\n"
        display += f"**Feasible Pareto points:** {len(points)} unique / {len(frontier)} raw / {n_points} requested\n"
        display += f"**Constraint mode:** {context['constraint_mode']} | **Fallback policy:** {context['fallback_policy']}\n"
        display += f"**Solvent shortlist status:** {context['filter_status']}\n"
        if context.get("strap_table_rows") is not None:
            display += f"**Compiled STRAP rows:** {context['strap_table_rows']}\n"
        requested_counts = candidate_telemetry["requested"]["counts_by_polymer"]
        surviving_counts = candidate_telemetry["surviving"]["counts_by_polymer"]
        if requested_counts:
            display += (
                "**Candidate telemetry:** requested "
                + ", ".join(f"{polymer}={count}" for polymer, count in sorted(requested_counts.items()))
                + "\n"
            )
        if surviving_counts:
            display += (
                "**Surviving candidates:** "
                + ", ".join(f"{polymer}={count}" for polymer, count in sorted(surviving_counts.items()))
                + "\n"
            )
        if context.get("simulation_skips"):
            display += f"**Baseline-backed BioSTEAM skips:** {len(context['simulation_skips'])}\n"
        failure_counts = candidate_telemetry["simulation"]["failure_counts_by_class"]
        if failure_counts:
            display += (
                "**BioSTEAM failures by class:** "
                + ", ".join(f"{name}={count}" for name, count in sorted(failure_counts.items()))
                + "\n"
            )
        if context["filter_warnings"]:
            for warning in context["filter_warnings"]:
                display += f"- **Filter note:** {warning}\n"
        if points:
            display += "\n### Pareto Points\n"
            for point in points:
                y_value = point["emissions"] if y_metric == "emissions" else point["circularity_score"]
                y_label = "Emissions (tCO2)" if y_metric == "emissions" else "Circularity (0-1)"
                display += (
                    f"- **Point {point['point_id']}:** Total cost ${point['total_cost']:,.2f}; "
                    f"{y_label} {y_value:,.4f}; "
                    f"Washes {point['wash1_selection']} / {point['wash2_selection']}; "
                    f"Stage 3 {', '.join(point.get('stage3_variants') or point.get('stage3_tech') or ['none'])}"
                )
                if int(point.get("n_equivalent_designs") or 1) > 1:
                    display += f"; equivalent design variants {point['n_equivalent_designs']}"
                display += "\n"
        else:
            display += "\nNo feasible Pareto points were found for the requested sweep.\n"

        return json_tool_response(display, result_payload)
    except Exception as e:
        logger.exception("Error in run_waste_management_pareto")
        return json_tool_error(str(e), tool_name="run_waste_management_pareto")
    finally:
        # temp_dir is retained as a hook for future workflows that need an
        # isolated workbook copy; the current pipeline threads data via a
        # compiled DataFrame and never populates it.
        pass
