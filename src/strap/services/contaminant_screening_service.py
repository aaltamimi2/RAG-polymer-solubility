"""Deterministic screening logic for contaminant-removal workflows."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from strap.database import get_connection
from strap.engines.precipitation import PrecipitationAnalyzer
from strap.services.contaminant_data_service import (
    choose_miscibility_regime,
    expand_requested_contaminants,
    get_contaminant_family,
    get_logd_entry,
    get_miscibility_entry,
    normalize_screening_solvent_name,
    get_supported_solvents_for_contaminants,
)
from strap.solubility import (
    get_available_polymers,
    get_available_solvents_for_polymer,
    get_boiling_point,
    get_solubility,
    resolve_polymer,
)
from strap.solvent_registry import resolve_to_interp_key
from strap.tools._helpers import get_cross_database_properties
from strap.tools.safety_gsk import lookup_local_gscore_data
from strap.tools.solvent_lookup import lookup_local_solvent_market_data

_SWELLING_PROXY_MIN_WT_PCT = 1.0
_SWELLING_PROXY_MAX_WT_PCT = 10.0
_MIN_TARGET_DISSOLUTION_WT_PCT = 10.0
_MAX_NON_TARGET_DISSOLUTION_WT_PCT = 1.0
_PRECIPITATION_THRESHOLD_WT_PCT = 1.0
_ATM_BP_MARGIN_C = 1.0
_MAX_SCREEN_TEMP_C = 160.0
_MAX_WASH_STEPS = 3
_MISSING_PRICE_USD_KG = 1.50
_MISSING_G_SCORE = 5.0
_MISSING_MARGIN_C = 10.0
_WASH_PLAN_WEIGHTS = {
    "coverage": 100.0,
    "step_penalty": 12.0,
    "safety": 18.0,
    "cost": 10.0,
    "margin": 6.0,
}


@dataclass
class _PolymerStatus:
    polymer: str
    solvent: str
    temperature_c: float | None
    supported: bool
    solubility_wt_pct: float | None
    status: str


def _parse_csv_list(values: str | list[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    return [item.strip() for item in str(values).split(",") if item.strip()]


def _resolve_polymer_or_none(polymer: str) -> str | None:
    return resolve_polymer(polymer, get_available_polymers())


def _classify_polymer_behavior(polymer: str, solvent: str, temperature_c: float) -> _PolymerStatus:
    resolved = _resolve_polymer_or_none(polymer)
    if resolved is None:
        return _PolymerStatus(polymer=polymer, solvent=solvent, temperature_c=temperature_c, supported=False, solubility_wt_pct=None, status="unsupported_polymer")
    solubility = get_solubility(resolved, solvent, temperature_c)
    if solubility is None:
        return _PolymerStatus(polymer=resolved, solvent=solvent, temperature_c=temperature_c, supported=False, solubility_wt_pct=None, status="unsupported_pair")
    if solubility >= _MIN_TARGET_DISSOLUTION_WT_PCT:
        status = "dissolving"
    elif solubility >= _SWELLING_PROXY_MIN_WT_PCT:
        status = "non_dissolving_proxy_swelling_candidate"
    else:
        status = "non_dissolving_low_swelling_confidence"
    return _PolymerStatus(
        polymer=resolved,
        solvent=solvent,
        temperature_c=temperature_c,
        supported=True,
        solubility_wt_pct=float(solubility),
        status=status,
    )


def _effective_max_temperature(solvent: str, max_temperature_c: float | None) -> float:
    upper = float(max_temperature_c) if max_temperature_c is not None else _MAX_SCREEN_TEMP_C
    bp = get_boiling_point(solvent)
    if bp is not None:
        upper = min(upper, bp - _ATM_BP_MARGIN_C)
    return upper


def _solvent_support_key(solvent: str) -> str:
    return resolve_to_interp_key(solvent) or solvent.lower()


def _choose_leaching_temperature(solvent: str, max_temperature_c: float | None) -> tuple[float, str]:
    effective_max = _effective_max_temperature(solvent, max_temperature_c)
    regime = choose_miscibility_regime(solvent, effective_max)
    if regime == "t_higher":
        return effective_max, regime
    return min(effective_max, 25.0), "rt"


def _screen_contaminants_for_solvent(solvent: str, contaminants: list[str], *, regime: str) -> tuple[list[dict[str, Any]], float | None, bool, bool]:
    contaminant_rows: list[dict[str, Any]] = []
    min_logd: float | None = None
    all_miscible = True
    all_positive_logd = True
    for contaminant in contaminants:
        miscibility = get_miscibility_entry(solvent, contaminant, regime=regime)
        logd = get_logd_entry(solvent, contaminant)
        miscible = miscibility.get("miscible") if miscibility else None
        logd_value = logd.get("logd") if logd else None
        if miscible is not True:
            all_miscible = False
        if logd_value is None or logd_value <= 0:
            all_positive_logd = False
        if logd_value is not None:
            min_logd = logd_value if min_logd is None else min(min_logd, float(logd_value))
        contaminant_rows.append(
            {
                "contaminant": contaminant,
                "miscible": miscible,
                "logd": float(logd_value) if logd_value is not None else None,
                "miscibility_regime": miscibility.get("temperature_regime") if miscibility else regime,
            }
        )
    return contaminant_rows, min_logd, all_miscible, all_positive_logd


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple:
    status = candidate.get("target_polymer_status")
    proxy_rank = 2 if status == "non_dissolving_proxy_swelling_candidate" else 1 if status == "non_dissolving_low_swelling_confidence" else 0
    return (
        int(bool(candidate.get("passes"))),
        int(bool(candidate.get("contaminant_miscibility_pass"))),
        int(bool(candidate.get("contaminant_logd_pass"))),
        proxy_rank,
        candidate.get("contaminant_logd_min") if candidate.get("contaminant_logd_min") is not None else -999.0,
    )


@dataclass(frozen=True)
class _WashCandidateOption:
    solvent: str
    mode: str
    covered_contaminants: tuple[str, ...]
    covered_families: tuple[str, ...]
    operating_temperature_c: float | None
    boiling_point_c: float | None
    bp_margin_c: float | None
    price_usd_kg: float | None
    g_score: float | None
    gsk_class: str | None
    g_score_source: str | None
    g_score_uncertainty: float | None
    candidate_row: dict[str, Any]


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_has_polymer_window(row: dict[str, Any]) -> bool:
    mode = str(row.get("mode") or "").strip()
    target_status = str(row.get("target_polymer_status") or "").strip()
    if mode == "leaching":
        if target_status in {"", "dissolving", "unsupported_pair", "unsupported_polymer"}:
            return False
    elif mode == "strap_contaminant_removal":
        if target_status != "dissolving_then_precipitating":
            return False
    elif target_status in {"unsupported_pair", "unsupported_polymer", "no_feasible_dissolution_precipitation_window"}:
        return False

    for other in (row.get("other_polymer_status") or {}).values():
        status = str((other or {}).get("status") or "").strip()
        if status in {"dissolving", "unsupported_pair", "unsupported_polymer"}:
            return False
    return True


def _covered_contaminants_for_row(row: dict[str, Any]) -> tuple[str, ...]:
    if not _row_has_polymer_window(row):
        return ()

    dissolution_rows = {
        str(item.get("contaminant")): item
        for item in (row.get("contaminants") or [])
        if isinstance(item, dict) and str(item.get("contaminant") or "").strip()
    }
    precipitation_rows = {
        str(item.get("contaminant")): item
        for item in (row.get("precipitation_regime_contaminants") or [])
        if isinstance(item, dict) and str(item.get("contaminant") or "").strip()
    }

    covered: list[str] = []
    mode = str(row.get("mode") or "").strip()
    for contaminant, item in dissolution_rows.items():
        if item.get("miscible") is not True:
            continue
        logd_value = _as_float_or_none(item.get("logd"))
        if logd_value is None or logd_value <= 0:
            continue
        if mode == "strap_contaminant_removal":
            precip_item = precipitation_rows.get(contaminant)
            if precip_item is None or precip_item.get("miscible") is not True:
                continue
        covered.append(contaminant)
    return tuple(sorted(dict.fromkeys(covered)))


def _lookup_solvent_tradeoff_profile(
    solvent: str,
    *,
    conn: Any,
) -> dict[str, Any]:
    market = lookup_local_solvent_market_data(solvent) or {}
    cross_db = get_cross_database_properties(solvent, conn)
    gscore = lookup_local_gscore_data(solvent)
    g_score = _as_float_or_none(cross_db.get("g_score"))
    gsk_class = str(cross_db.get("gsk_class")).strip() if cross_db.get("gsk_class") else None
    g_score_source: str | None = "gsk_dataset" if g_score is not None else None
    g_score_uncertainty: float | None = None

    if gscore is not None:
        g_score = _as_float_or_none(gscore.get("g_score"))
        if gscore.get("classification"):
            gsk_class = str(gscore.get("classification")).strip()
        g_score_source = str(gscore.get("source") or g_score_source or "").strip() or None
        g_score_uncertainty = _as_float_or_none(gscore.get("g_score_uncertainty"))

    return {
        "price_usd_kg": _as_float_or_none(market.get("price_usd_kg")),
        "price_source": market.get("price_source"),
        "g_score": g_score,
        "gsk_class": gsk_class,
        "g_score_source": g_score_source,
        "g_score_uncertainty": g_score_uncertainty,
    }


def _build_wash_candidate_options(
    *,
    mode_result: dict[str, Any],
) -> tuple[list[_WashCandidateOption], dict[str, str | None]]:
    conn = get_connection()
    profile_cache: dict[str, dict[str, Any]] = {}
    family_map = {
        contaminant: get_contaminant_family(contaminant)
        for contaminant in mode_result.get("supported_contaminants", [])
    }
    options: list[_WashCandidateOption] = []

    for row in mode_result.get("candidate_solvents", []):
        if not isinstance(row, dict):
            continue
        solvent = str(row.get("solvent") or "").strip()
        if not solvent:
            continue
        covered_contaminants = _covered_contaminants_for_row(row)
        if not covered_contaminants:
            continue
        if solvent not in profile_cache:
            profile_cache[solvent] = _lookup_solvent_tradeoff_profile(solvent, conn=conn)
        profile = profile_cache[solvent]
        operating_temperature_c = _as_float_or_none(row.get("operating_temperature_c"))
        boiling_point_c = _as_float_or_none(row.get("boiling_point_c"))
        bp_margin_c = (
            boiling_point_c - operating_temperature_c
            if boiling_point_c is not None and operating_temperature_c is not None
            else None
        )
        covered_families = tuple(
            sorted(
                {
                    family_map.get(contaminant)
                    for contaminant in covered_contaminants
                    if family_map.get(contaminant)
                }
            )
        )
        options.append(
            _WashCandidateOption(
                solvent=solvent,
                mode=str(row.get("mode") or mode_result.get("mode") or "").strip(),
                covered_contaminants=covered_contaminants,
                covered_families=covered_families,
                operating_temperature_c=operating_temperature_c,
                boiling_point_c=boiling_point_c,
                bp_margin_c=bp_margin_c,
                price_usd_kg=profile["price_usd_kg"],
                g_score=profile["g_score"],
                gsk_class=profile["gsk_class"],
                g_score_source=profile.get("g_score_source"),
                g_score_uncertainty=profile.get("g_score_uncertainty"),
                candidate_row=row,
            )
        )

    options.sort(
        key=lambda option: (
            len(option.covered_contaminants),
            option.g_score if option.g_score is not None else _MISSING_G_SCORE,
            -(option.price_usd_kg if option.price_usd_kg is not None else _MISSING_PRICE_USD_KG),
        ),
        reverse=True,
    )
    return options, family_map


def _price_score(total_price_usd_kg: float | None) -> float:
    if total_price_usd_kg is None:
        return 0.5
    capped = min(max(total_price_usd_kg, 0.0), 5.0)
    return max(0.0, 1.0 - (capped / 5.0))


def _safety_score(min_g_score: float | None) -> float:
    if min_g_score is None:
        return 0.5
    return min(max(min_g_score / 10.0, 0.0), 1.0)


def _margin_score(min_bp_margin_c: float | None) -> float:
    if min_bp_margin_c is None:
        return 0.5
    return min(max(min_bp_margin_c, 0.0), 20.0) / 20.0


def _plan_rank(plan: dict[str, Any]) -> tuple[float, float, float, float, float]:
    total_price = _as_float_or_none(plan.get("estimated_total_solvent_price_usd_kg"))
    min_g_score = _as_float_or_none(plan.get("min_g_score"))
    return (
        float(plan.get("coverage_fraction") or 0.0),
        float(plan.get("tradeoff_score") or 0.0),
        -float(plan.get("n_steps") or 0),
        -(total_price if total_price is not None else _MISSING_PRICE_USD_KG),
        min_g_score if min_g_score is not None else _MISSING_G_SCORE,
    )


def _annotate_wash_plan_labels(plans: list[dict[str, Any]]) -> None:
    if not plans:
        return
    full_coverage = [plan for plan in plans if plan.get("full_coverage")]
    ranked_pool = full_coverage or plans
    labels_by_id: dict[str, set[str]] = {
        str(plan["plan_id"]): set()
        for plan in plans
    }

    best_overall = max(ranked_pool, key=_plan_rank)
    labels_by_id[best_overall["plan_id"]].add("best_overall")

    single_step = [plan for plan in ranked_pool if int(plan.get("n_steps") or 0) == 1]
    if single_step:
        labels_by_id[max(single_step, key=_plan_rank)["plan_id"]].add("best_single_step")

    multi_step = [plan for plan in ranked_pool if int(plan.get("n_steps") or 0) > 1]
    if multi_step:
        labels_by_id[max(multi_step, key=_plan_rank)["plan_id"]].add("best_multi_step")

    if full_coverage:
        cheapest = min(
            full_coverage,
            key=lambda plan: (
                _as_float_or_none(plan.get("estimated_total_solvent_price_usd_kg"))
                if _as_float_or_none(plan.get("estimated_total_solvent_price_usd_kg")) is not None
                else _MISSING_PRICE_USD_KG * int(plan.get("n_steps") or 1),
                int(plan.get("n_steps") or 0),
                -(_as_float_or_none(plan.get("min_g_score")) or _MISSING_G_SCORE),
            ),
        )
        safest = max(
            full_coverage,
            key=lambda plan: (
                _as_float_or_none(plan.get("min_g_score")) or _MISSING_G_SCORE,
                -int(plan.get("n_steps") or 0),
                -(
                    _as_float_or_none(plan.get("estimated_total_solvent_price_usd_kg"))
                    if _as_float_or_none(plan.get("estimated_total_solvent_price_usd_kg")) is not None
                    else _MISSING_PRICE_USD_KG
                ),
            ),
        )
        labels_by_id[cheapest["plan_id"]].add("cheapest_full_coverage")
        labels_by_id[safest["plan_id"]].add("safest_full_coverage")

    for plan in plans:
        plan["selection_labels"] = sorted(labels_by_id[str(plan["plan_id"])])


def plan_contaminant_wash_steps(
    *,
    mode_result: dict[str, Any],
    max_steps: int = _MAX_WASH_STEPS,
) -> dict[str, Any]:
    """Plan single- or multi-step contaminant washes from partial solvent coverage."""
    supported_contaminants = list(dict.fromkeys(mode_result.get("supported_contaminants", []) or []))
    if not supported_contaminants:
        return {
            "recommended_wash_plan": None,
            "wash_step_plans": [],
            "wash_step_objective": {
                "max_steps_considered": max_steps,
                **_WASH_PLAN_WEIGHTS,
            },
        }

    options, family_map = _build_wash_candidate_options(mode_result=mode_result)
    if not options:
        return {
            "recommended_wash_plan": None,
            "wash_step_plans": [],
            "wash_step_objective": {
                "max_steps_considered": max_steps,
                **_WASH_PLAN_WEIGHTS,
            },
        }

    all_plans: list[dict[str, Any]] = []
    universe = set(supported_contaminants)
    max_considered_steps = max(1, min(max_steps, len(options)))
    for n_steps in range(1, max_considered_steps + 1):
        for combo_index, combo in enumerate(combinations(options, n_steps), start=1):
            covered = sorted(
                {
                    contaminant
                    for option in combo
                    for contaminant in option.covered_contaminants
                }
            )
            covered_set = set(covered)
            uncovered = sorted(universe - covered_set)
            total_price = sum(
                option.price_usd_kg if option.price_usd_kg is not None else _MISSING_PRICE_USD_KG
                for option in combo
            )
            min_g_score = min(
                option.g_score if option.g_score is not None else _MISSING_G_SCORE
                for option in combo
            )
            min_margin_c = min(
                option.bp_margin_c if option.bp_margin_c is not None else _MISSING_MARGIN_C
                for option in combo
            )
            coverage_fraction = len(covered_set) / len(universe) if universe else 0.0
            tradeoff_score = (
                _WASH_PLAN_WEIGHTS["coverage"] * coverage_fraction
                - _WASH_PLAN_WEIGHTS["step_penalty"] * (n_steps - 1)
                + _WASH_PLAN_WEIGHTS["safety"] * _safety_score(min_g_score)
                + _WASH_PLAN_WEIGHTS["cost"] * _price_score(total_price)
                + _WASH_PLAN_WEIGHTS["margin"] * _margin_score(min_margin_c)
            )
            step_payloads = []
            for step_number, option in enumerate(combo, start=1):
                step_payloads.append(
                    {
                        "step": step_number,
                        "solvent": option.solvent,
                        "mode": option.mode,
                        "operating_temperature_c": option.operating_temperature_c,
                        "boiling_point_c": option.boiling_point_c,
                        "bp_margin_c": option.bp_margin_c,
                        "covered_contaminants": list(option.covered_contaminants),
                        "covered_families": list(option.covered_families),
                        "price_usd_kg": option.price_usd_kg,
                        "g_score": option.g_score,
                        "gsk_class": option.gsk_class,
                        "g_score_source": option.g_score_source or "imputed",
                        "g_score_uncertainty": option.g_score_uncertainty,
                        "target_polymer_status": option.candidate_row.get("target_polymer_status"),
                    }
                )
            all_plans.append(
                {
                    "plan_id": f"{mode_result.get('mode', 'wash')}-plan-{n_steps}-{combo_index}",
                    "mode": mode_result.get("mode"),
                    "n_steps": n_steps,
                    "steps": step_payloads,
                    "covered_contaminants": covered,
                    "covered_families": sorted(
                        {
                            family_map.get(contaminant)
                            for contaminant in covered
                            if family_map.get(contaminant)
                        }
                    ),
                    "uncovered_contaminants": uncovered,
                    "uncovered_families": sorted(
                        {
                            family_map.get(contaminant)
                            for contaminant in uncovered
                            if family_map.get(contaminant)
                        }
                    ),
                    "coverage_fraction": round(coverage_fraction, 6),
                    "full_coverage": not uncovered,
                    "estimated_total_solvent_price_usd_kg": round(total_price, 4),
                    "min_g_score": round(min_g_score, 3),
                    "min_bp_margin_c": round(min_margin_c, 3),
                    "tradeoff_score": round(tradeoff_score, 4),
                }
            )

    all_plans.sort(key=_plan_rank, reverse=True)
    _annotate_wash_plan_labels(all_plans)
    selected_ids: set[str] = set()
    selected_plans: list[dict[str, Any]] = []
    for plan in all_plans:
        if plan["selection_labels"] and plan["plan_id"] not in selected_ids:
            selected_ids.add(plan["plan_id"])
            selected_plans.append(plan)
    for plan in all_plans:
        if len(selected_plans) >= 6:
            break
        if plan["plan_id"] in selected_ids:
            continue
        selected_ids.add(plan["plan_id"])
        selected_plans.append(plan)

    recommended = selected_plans[0] if selected_plans else None
    return {
        "recommended_wash_plan": recommended,
        "wash_step_plans": selected_plans,
        "wash_step_objective": {
            "max_steps_considered": max_considered_steps,
            **_WASH_PLAN_WEIGHTS,
            "notes": [
                "coverage is prioritized first; full-contaminant coverage dominates partial coverage",
                "additional wash steps incur a fixed penalty unless they materially improve safety or cost",
                "total solvent price assumes comparable solvent intensity per wash step",
                "missing solvent price or G-score data are imputed to neutral defaults",
                "safety uses curated GSK scores first, then GreenSolventDB ML predictions, then imputation",
            ],
        },
    }


def screen_leaching_candidates(
    *,
    target_polymer: str,
    contaminants: list[str] | str,
    other_polymers: list[str] | str | None = None,
    solvents: list[str] | str | None = None,
    max_temperature_c: float | None = None,
) -> dict[str, Any]:
    requested_contaminants = _parse_csv_list(contaminants)
    supported_contaminants, unsupported_contaminants, contaminant_families = expand_requested_contaminants(requested_contaminants)
    resolved_target = _resolve_polymer_or_none(target_polymer)
    resolved_others = [resolved for item in _parse_csv_list(other_polymers) if (resolved := _resolve_polymer_or_none(item))]
    missing_other_polymers = [item for item in _parse_csv_list(other_polymers) if _resolve_polymer_or_none(item) is None]

    if resolved_target is None:
        raise ValueError(f"Unsupported target polymer: {target_polymer}")
    if not supported_contaminants:
        raise ValueError("None of the requested contaminants are supported by the Zhou workbook.")

    candidate_solvents = [
        normalize_screening_solvent_name(solvent)
        for solvent in (_parse_csv_list(solvents) or get_supported_solvents_for_contaminants(supported_contaminants))
    ]
    candidate_solvents = list(dict.fromkeys(candidate_solvents))
    rows: list[dict[str, Any]] = []
    for solvent in candidate_solvents:
        operating_temperature_c, regime = _choose_leaching_temperature(solvent, max_temperature_c)
        contaminant_rows, min_logd, all_miscible, all_positive_logd = _screen_contaminants_for_solvent(
            solvent,
            supported_contaminants,
            regime=regime,
        )
        target_status = _classify_polymer_behavior(resolved_target, solvent, operating_temperature_c)
        other_statuses: dict[str, dict[str, Any]] = {}
        other_polymers_ok = True
        for polymer in resolved_others:
            other_status = _classify_polymer_behavior(polymer, solvent, operating_temperature_c)
            other_statuses[polymer] = {
                "status": other_status.status,
                "solubility_wt_pct": other_status.solubility_wt_pct,
            }
            if other_status.status == "dissolving" or not other_status.supported:
                other_polymers_ok = False

        caveats: list[str] = []
        if target_status.status == "non_dissolving_proxy_swelling_candidate":
            caveats.append("polymer swelling is proxy-inferred from borderline solubility, not directly measured")
        elif target_status.status == "non_dissolving_low_swelling_confidence":
            caveats.append("polymer remains undissolved, but swelling evidence is weak in the current data")
        if missing_other_polymers:
            caveats.append(
                "other polymers were unsupported in the polymer-solubility dataset: " + ", ".join(missing_other_polymers)
            )

        passes = (
            all_miscible
            and all_positive_logd
            and target_status.status != "dissolving"
            and target_status.supported
            and other_polymers_ok
            and not missing_other_polymers
        )
        bp = get_boiling_point(solvent)
        rows.append(
            {
                "solvent": solvent,
                "passes": passes,
                "mode": "leaching",
                "operating_temperature_c": operating_temperature_c,
                "boiling_point_c": bp,
                "contaminant_miscibility_pass": all_miscible,
                "contaminant_logd_pass": all_positive_logd,
                "contaminant_logd_min": min_logd,
                "contaminants": contaminant_rows,
                "target_polymer_status": target_status.status,
                "target_polymer_solubility_wt_pct": target_status.solubility_wt_pct,
                "other_polymer_status": other_statuses,
                "caveats": caveats,
            }
        )

    rows.sort(key=_candidate_sort_key, reverse=True)
    recommended = [row["solvent"] for row in rows if row["passes"]]
    result_caveats: list[str] = []
    if unsupported_contaminants:
        result_caveats.append(
            "screening only covers supported contaminants: " + ", ".join(supported_contaminants)
        )
    if missing_other_polymers:
        result_caveats.append(
            "non-target polymer exclusion could not be verified for unsupported polymers: " + ", ".join(missing_other_polymers)
        )
    result_caveats.append("leaching-mode swelling is proxy-inferred in v1 and requires experimental validation")
    result = {
        "mode": "leaching",
        "target_polymer": resolved_target,
        "other_polymers": resolved_others,
        "contaminants": supported_contaminants,
        "supported_contaminants": supported_contaminants,
        "unsupported_contaminants": unsupported_contaminants,
        "contaminant_families": contaminant_families,
        "candidate_solvents": rows,
        "recommended_solvents": recommended,
        "decision_basis": [
            "target contaminants must be miscible in the solvent",
            "contaminant logD must stay positive; higher is preferred",
            "target polymer should remain non-dissolved, with borderline compatibility favored as a swelling proxy",
        ],
        "caveats": result_caveats,
    }
    result.update(plan_contaminant_wash_steps(mode_result=result))
    return result


def _find_strap_candidate_temperature(target_polymer: str, solvent: str, other_polymers: list[str], max_temperature_c: float | None) -> dict[str, Any] | None:
    available_for_target = {
        _solvent_support_key(name)
        for name in get_available_solvents_for_polymer(target_polymer)
    }
    if _solvent_support_key(solvent) not in available_for_target:
        return None
    upper = _effective_max_temperature(solvent, max_temperature_c)
    if upper < 25.0:
        return None
    analyzer = PrecipitationAnalyzer(get_connection())
    precip_point = analyzer.analyze_precipitation(target_polymer, solvent, _PRECIPITATION_THRESHOLD_WT_PCT)
    if not precip_point or precip_point.precipitation_temp is None:
        return None

    best: dict[str, Any] | None = None
    for temperature_c in range(25, int(min(upper, _MAX_SCREEN_TEMP_C)) + 1, 5):
        target_solubility = get_solubility(target_polymer, solvent, float(temperature_c))
        if target_solubility is None or target_solubility < _MIN_TARGET_DISSOLUTION_WT_PCT:
            continue
        other_statuses: dict[str, dict[str, Any]] = {}
        other_ok = True
        for polymer in other_polymers:
            sol = get_solubility(polymer, solvent, float(temperature_c))
            if sol is None:
                other_ok = False
                other_statuses[polymer] = {"status": "unsupported_pair", "solubility_wt_pct": None}
                break
            other_statuses[polymer] = {
                "status": "undissolved" if sol <= _MAX_NON_TARGET_DISSOLUTION_WT_PCT else "dissolving",
                "solubility_wt_pct": float(sol),
            }
            if sol > _MAX_NON_TARGET_DISSOLUTION_WT_PCT:
                other_ok = False
        if not other_ok:
            continue
        if precip_point.precipitation_temp >= temperature_c:
            continue
        candidate = {
            "operating_temperature_c": float(temperature_c),
            "target_polymer_solubility_wt_pct": float(target_solubility),
            "precipitation_temperature_c": float(precip_point.precipitation_temp),
            "cloud_point_c": float(precip_point.cloud_point) if precip_point.cloud_point is not None else None,
            "other_polymer_status": other_statuses,
        }
        if best is None or candidate["target_polymer_solubility_wt_pct"] > best["target_polymer_solubility_wt_pct"]:
            best = candidate
    return best


def screen_strap_contaminant_removal_candidates(
    *,
    target_polymer: str,
    contaminants: list[str] | str,
    other_polymers: list[str] | str | None = None,
    solvents: list[str] | str | None = None,
    max_temperature_c: float | None = None,
) -> dict[str, Any]:
    requested_contaminants = _parse_csv_list(contaminants)
    supported_contaminants, unsupported_contaminants, contaminant_families = expand_requested_contaminants(requested_contaminants)
    resolved_target = _resolve_polymer_or_none(target_polymer)
    resolved_others_raw = _parse_csv_list(other_polymers)
    resolved_others = [resolved for item in resolved_others_raw if (resolved := _resolve_polymer_or_none(item))]
    missing_other_polymers = [item for item in resolved_others_raw if _resolve_polymer_or_none(item) is None]

    if resolved_target is None:
        raise ValueError(f"Unsupported target polymer: {target_polymer}")
    if not supported_contaminants:
        raise ValueError("None of the requested contaminants are supported by the Zhou workbook.")

    candidate_solvents = [
        normalize_screening_solvent_name(solvent)
        for solvent in (_parse_csv_list(solvents) or [])
    ]
    if not candidate_solvents:
        candidate_solvents = sorted(
            set(get_supported_solvents_for_contaminants(supported_contaminants))
            & set(get_available_solvents_for_polymer(resolved_target))
        )
    candidate_solvents = list(dict.fromkeys(candidate_solvents))
    rows: list[dict[str, Any]] = []
    for solvent in candidate_solvents:
        operating = _find_strap_candidate_temperature(
            resolved_target,
            solvent,
            resolved_others,
            max_temperature_c,
        )
        bp = get_boiling_point(solvent)
        if operating is None:
            rows.append(
                {
                    "solvent": solvent,
                    "passes": False,
                    "mode": "strap_contaminant_removal",
                    "operating_temperature_c": None,
                    "boiling_point_c": bp,
                    "target_polymer_status": "no_feasible_dissolution_precipitation_window",
                    "other_polymer_status": {},
                    "contaminant_miscibility_pass": False,
                    "contaminant_precipitation_regime_pass": False,
                    "contaminant_logd_pass": False,
                    "contaminant_logd_min": None,
                    "contaminants": [],
                    "caveats": ["no feasible dissolution and cooling-precipitation window was found under the current constraints"],
                }
            )
            continue

        dissolution_regime = choose_miscibility_regime(solvent, operating["operating_temperature_c"])
        precipitation_regime = choose_miscibility_regime(solvent, operating["precipitation_temperature_c"])
        dissolution_rows, min_logd, all_miscible_dissolution, all_positive_logd = _screen_contaminants_for_solvent(
            solvent,
            supported_contaminants,
            regime=dissolution_regime,
        )
        precipitation_rows, _, all_miscible_precipitation, _ = _screen_contaminants_for_solvent(
            solvent,
            supported_contaminants,
            regime=precipitation_regime,
        )
        rows.append(
            {
                "solvent": solvent,
                "passes": (
                    all_miscible_dissolution
                    and all_miscible_precipitation
                    and all_positive_logd
                    and not missing_other_polymers
                ),
                "mode": "strap_contaminant_removal",
                "operating_temperature_c": operating["operating_temperature_c"],
                "boiling_point_c": bp,
                "target_polymer_status": "dissolving_then_precipitating",
                "target_polymer_solubility_wt_pct": operating["target_polymer_solubility_wt_pct"],
                "precipitation_temperature_c": operating["precipitation_temperature_c"],
                "cloud_point_c": operating["cloud_point_c"],
                "other_polymer_status": operating["other_polymer_status"],
                "contaminant_miscibility_pass": all_miscible_dissolution,
                "contaminant_precipitation_regime_pass": all_miscible_precipitation,
                "contaminant_logd_pass": all_positive_logd,
                "contaminant_logd_min": min_logd,
                "contaminants": dissolution_rows,
                "precipitation_regime_contaminants": precipitation_rows,
                "caveats": (
                    [
                        "non-target polymer exclusion could not be verified for unsupported polymers: " + ", ".join(missing_other_polymers)
                    ]
                    if missing_other_polymers
                    else []
                ),
            }
        )

    rows.sort(key=_candidate_sort_key, reverse=True)
    recommended = [row["solvent"] for row in rows if row["passes"]]
    result_caveats: list[str] = []
    if unsupported_contaminants:
        result_caveats.append(
            "screening only covers supported contaminants: " + ", ".join(supported_contaminants)
        )
    if missing_other_polymers:
        result_caveats.append(
            "non-target polymer exclusion could not be verified for unsupported polymers: " + ", ".join(missing_other_polymers)
        )
    result_caveats.append("temperature-swing contaminant removal is screened using a 1 wt% polymer precipitation threshold")
    result = {
        "mode": "strap_contaminant_removal",
        "target_polymer": resolved_target,
        "other_polymers": resolved_others,
        "contaminants": supported_contaminants,
        "supported_contaminants": supported_contaminants,
        "unsupported_contaminants": unsupported_contaminants,
        "contaminant_families": contaminant_families,
        "candidate_solvents": rows,
        "recommended_solvents": recommended,
        "decision_basis": [
            "target polymer must dissolve while non-target polymers remain undissolved",
            "target polymer must precipitate below 1 wt% on cooling",
            "contaminants must stay miscible at both the dissolution and precipitation conditions",
            "contaminant logD must remain positive; higher is preferred",
        ],
        "caveats": result_caveats,
    }
    result.update(plan_contaminant_wash_steps(mode_result=result))
    return result


def compare_contaminant_removal_modes(
    *,
    target_polymer: str,
    contaminants: list[str] | str,
    other_polymers: list[str] | str | None = None,
    solvents: list[str] | str | None = None,
    max_temperature_c: float | None = None,
) -> dict[str, Any]:
    leaching = screen_leaching_candidates(
        target_polymer=target_polymer,
        contaminants=contaminants,
        other_polymers=other_polymers,
        solvents=solvents,
        max_temperature_c=max_temperature_c,
    )
    strap_mode = screen_strap_contaminant_removal_candidates(
        target_polymer=target_polymer,
        contaminants=contaminants,
        other_polymers=other_polymers,
        solvents=solvents,
        max_temperature_c=max_temperature_c,
    )
    leaching_count = len(leaching["recommended_solvents"])
    strap_count = len(strap_mode["recommended_solvents"])
    leaching_plan = leaching.get("recommended_wash_plan")
    strap_plan = strap_mode.get("recommended_wash_plan")
    if isinstance(leaching_plan, dict) and isinstance(strap_plan, dict):
        leaching_key = _plan_rank(leaching_plan)
        strap_key = _plan_rank(strap_plan)
        if strap_key > leaching_key:
            recommended_mode = "strap_contaminant_removal"
        elif leaching_key > strap_key:
            recommended_mode = "leaching"
        else:
            recommended_mode = "tie"
    elif strap_count > leaching_count:
        recommended_mode = "strap_contaminant_removal"
    elif leaching_count > strap_count:
        recommended_mode = "leaching"
    else:
        recommended_mode = "tie"
    result = {
        "mode": "comparison",
        "target_polymer": leaching["target_polymer"],
        "other_polymers": leaching["other_polymers"],
        "contaminants": leaching["contaminants"],
        "supported_contaminants": leaching["supported_contaminants"],
        "unsupported_contaminants": leaching["unsupported_contaminants"],
        "contaminant_families": leaching.get("contaminant_families") or strap_mode.get("contaminant_families"),
        "leaching": leaching,
        "strap_contaminant_removal": strap_mode,
        "recommended_mode": recommended_mode,
        "recommended_solvents": {
            "leaching": leaching["recommended_solvents"],
            "strap_contaminant_removal": strap_mode["recommended_solvents"],
        },
        "recommended_wash_plan": (
            strap_plan
            if recommended_mode == "strap_contaminant_removal"
            else leaching_plan
            if recommended_mode == "leaching"
            else None
        ),
        "recommended_wash_plan_by_mode": {
            "leaching": leaching_plan,
            "strap_contaminant_removal": strap_plan,
        },
        "wash_step_plans": {
            "leaching": leaching.get("wash_step_plans", []),
            "strap_contaminant_removal": strap_mode.get("wash_step_plans", []),
        },
        "decision_basis": [
            "compare the number and quality of passing solvents in each mode",
            "prefer STRAP contaminant removal when a target polymer can dissolve and then precipitate cleanly on cooling",
            "prefer leaching when the polymer should remain intact and a non-dissolving extraction solvent is available",
        ],
        "caveats": list(dict.fromkeys(leaching["caveats"] + strap_mode["caveats"])),
    }
    return result
