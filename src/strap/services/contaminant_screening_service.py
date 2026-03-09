"""Deterministic screening logic for contaminant-removal workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from strap.database import get_connection
from strap.engines.precipitation import PrecipitationAnalyzer
from strap.services.contaminant_data_service import (
    choose_miscibility_regime,
    expand_requested_contaminants,
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

_SWELLING_PROXY_MIN_WT_PCT = 1.0
_SWELLING_PROXY_MAX_WT_PCT = 10.0
_MIN_TARGET_DISSOLUTION_WT_PCT = 10.0
_MAX_NON_TARGET_DISSOLUTION_WT_PCT = 1.0
_PRECIPITATION_THRESHOLD_WT_PCT = 1.0
_ATM_BP_MARGIN_C = 1.0
_MAX_SCREEN_TEMP_C = 160.0


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
    return {
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
    return {
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
    if strap_count > leaching_count:
        recommended_mode = "strap_contaminant_removal"
    elif leaching_count > strap_count:
        recommended_mode = "leaching"
    else:
        recommended_mode = "tie"
    return {
        "mode": "comparison",
        "target_polymer": leaching["target_polymer"],
        "other_polymers": leaching["other_polymers"],
        "contaminants": leaching["contaminants"],
        "supported_contaminants": leaching["supported_contaminants"],
        "unsupported_contaminants": leaching["unsupported_contaminants"],
        "leaching": leaching,
        "strap_contaminant_removal": strap_mode,
        "recommended_mode": recommended_mode,
        "recommended_solvents": {
            "leaching": leaching["recommended_solvents"],
            "strap_contaminant_removal": strap_mode["recommended_solvents"],
        },
        "decision_basis": [
            "compare the number and quality of passing solvents in each mode",
            "prefer STRAP contaminant removal when a target polymer can dissolve and then precipitate cleanly on cooling",
            "prefer leaching when the polymer should remain intact and a non-dissolving extraction solvent is available",
        ],
        "caveats": list(dict.fromkeys(leaching["caveats"] + strap_mode["caveats"])),
    }
