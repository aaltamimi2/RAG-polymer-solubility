"""Normalized access to Zhou contaminant-removal screening data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from strap.solvent_registry import resolve_to_bp_db_key, resolve_to_interp_key

_WORKBOOK_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "zhou_contamintant_removal_SI_Data.xlsx"
)
_RT_TEMPERATURE_C = 25.0
_FAMILY_ALIASES = {
    "pfas": "PFAS",
    "per- and polyfluoroalkyl substances": "PFAS",
    "perfluoroalkyl substances": "PFAS",
    "phthalate": "Phthalates",
    "phthalates": "Phthalates",
}


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def _clean_contaminant_name(value: Any) -> str:
    return _clean_text(value).replace(" .1", "")


def _canonical_contaminant_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _canonical_family_key(name: str) -> str:
    return _canonical_contaminant_key(name)


def _canonical_solvent_key(name: str) -> str:
    return _clean_text(name).lower()


def _normalize_solvent_name(name: str) -> str:
    cleaned = _clean_text(name)
    if not cleaned:
        return ""
    return (
        resolve_to_interp_key(cleaned)
        or resolve_to_bp_db_key(cleaned)
        or cleaned.lower()
    )


def normalize_screening_solvent_name(name: str) -> str:
    """Normalize user/workbook solvent names to the screening namespace."""
    return _clean_text(name).lower()


def _normalize_miscibility(value: Any) -> bool | None:
    text = _clean_text(value).lower()
    if text == "yes":
        return True
    if text == "no":
        return False
    return None


@lru_cache(maxsize=1)
def _load_dataset() -> dict[str, Any]:
    if not _WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Contaminant workbook not found: {_WORKBOOK_PATH}")

    miscibility_records: list[dict[str, Any]] = []
    logd_records: list[dict[str, Any]] = []
    contaminants_by_family: dict[str, set[str]] = {}

    def add_miscibility_record(*, family: str, contaminant: str, solvent: str, value: Any, regime: str, temperature_c: float | None, boiling_point_c: float | None = None, t_higher_c: float | None = None) -> None:
        if not contaminant or not solvent:
            return
        normalized_value = _normalize_miscibility(value)
        if normalized_value is None:
            return
        contaminants_by_family.setdefault(family, set()).add(contaminant)
        miscibility_records.append(
            {
                "family": family,
                "contaminant": contaminant,
                "contaminant_key": _canonical_contaminant_key(contaminant),
                "solvent_raw": solvent,
                "solvent_key": _canonical_solvent_key(solvent),
                "solvent_normalized": _normalize_solvent_name(solvent),
                "temperature_regime": regime,
                "temperature_c": temperature_c,
                "boiling_point_c": boiling_point_c,
                "t_higher_c": t_higher_c,
                "miscible": normalized_value,
            }
        )

    def add_logd_record(*, family: str, contaminant: str, solvent: str, value: Any, boiling_point_c: float | None = None) -> None:
        if not contaminant or not solvent:
            return
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return
        contaminants_by_family.setdefault(family, set()).add(contaminant)
        logd_records.append(
            {
                "family": family,
                "contaminant": contaminant,
                "contaminant_key": _canonical_contaminant_key(contaminant),
                "solvent_raw": solvent,
                "solvent_key": _canonical_solvent_key(solvent),
                "solvent_normalized": _normalize_solvent_name(solvent),
                "logd": float(value),
                "boiling_point_c": boiling_point_c,
            }
        )

    # PFAS wide sheets: row 0 = family group headings, row 1 = contaminant names, rows 2+ = solvents.
    for sheet_name, target in (("PFAS_Miscibility", "miscibility"), ("PFAS_log_D", "logd")):
        raw = pd.read_excel(_WORKBOOK_PATH, sheet_name=sheet_name, header=None)
        family_row = raw.iloc[0].tolist()
        contaminant_row = raw.iloc[1].tolist()
        current_family = "PFAS"
        contaminant_columns: list[tuple[int, str, str]] = []
        for idx in range(2, raw.shape[1]):
            family_cell = _clean_text(family_row[idx])
            if family_cell:
                current_family = family_cell
            contaminant = _clean_contaminant_name(contaminant_row[idx])
            if contaminant:
                contaminant_columns.append((idx, current_family, contaminant))
        for row_idx in range(2, raw.shape[0]):
            solvent = _clean_text(raw.iat[row_idx, 1])
            if not solvent:
                continue
            for col_idx, family, contaminant in contaminant_columns:
                value = raw.iat[row_idx, col_idx]
                normalized_family = "PFAS"
                if target == "miscibility":
                    add_miscibility_record(
                        family=normalized_family,
                        contaminant=contaminant,
                        solvent=solvent,
                        value=value,
                        regime="unspecified",
                        temperature_c=None,
                    )
                else:
                    add_logd_record(
                        family=normalized_family,
                        contaminant=contaminant,
                        solvent=solvent,
                        value=value,
                    )

    # Phthalates miscibility: row 0 = contaminant names, row 1 = RT/T higher regime, rows 2+ = solvents.
    raw = pd.read_excel(_WORKBOOK_PATH, sheet_name="Phthalates_Miscibility", header=None)
    contaminant_row = raw.iloc[0].tolist()
    regime_row = raw.iloc[1].tolist()
    for row_idx in range(2, raw.shape[0]):
        solvent = _clean_text(raw.iat[row_idx, 0])
        if not solvent:
            continue
        boiling_point_c = raw.iat[row_idx, 1]
        t_higher_c = raw.iat[row_idx, 2]
        bp = float(boiling_point_c) if pd.notna(boiling_point_c) else None
        th = float(t_higher_c) if pd.notna(t_higher_c) else None
        for col_idx in range(3, raw.shape[1]):
            contaminant = _clean_contaminant_name(contaminant_row[col_idx])
            regime_label = _clean_text(regime_row[col_idx]).lower()
            if not contaminant or regime_label not in {"rt", "t higher"}:
                continue
            regime = "rt" if regime_label == "rt" else "t_higher"
            temperature_c = _RT_TEMPERATURE_C if regime == "rt" else th
            add_miscibility_record(
                family="Phthalates",
                contaminant=contaminant,
                solvent=solvent,
                value=raw.iat[row_idx, col_idx],
                regime=regime,
                temperature_c=temperature_c,
                boiling_point_c=bp,
                t_higher_c=th,
            )

    # Phthalates logD: row 0 = headers, rows 1+ = solvents.
    raw = pd.read_excel(_WORKBOOK_PATH, sheet_name="Phthalates_log_D", header=None)
    contaminant_row = raw.iloc[0].tolist()
    for row_idx in range(1, raw.shape[0]):
        solvent = _clean_text(raw.iat[row_idx, 0])
        if not solvent:
            continue
        boiling_point_c = raw.iat[row_idx, 1]
        bp = float(boiling_point_c) if pd.notna(boiling_point_c) else None
        for col_idx in range(2, raw.shape[1]):
            contaminant = _clean_contaminant_name(contaminant_row[col_idx])
            if not contaminant:
                continue
            add_logd_record(
                family="Phthalates",
                contaminant=contaminant,
                solvent=solvent,
                value=raw.iat[row_idx, col_idx],
                boiling_point_c=bp,
            )

    contaminants_lookup: dict[str, dict[str, str]] = {}
    for family, names in contaminants_by_family.items():
        for name in names:
            contaminants_lookup[_canonical_contaminant_key(name)] = {
                "name": name,
                "family": family,
            }

    return {
        "families": sorted(contaminants_by_family),
        "contaminants_by_family": {
            family: sorted(names)
            for family, names in contaminants_by_family.items()
        },
        "contaminants_lookup": contaminants_lookup,
        "miscibility_records": tuple(miscibility_records),
        "logd_records": tuple(logd_records),
    }


def list_supported_contaminant_families() -> list[str]:
    return list(_load_dataset()["families"])


def list_supported_contaminants(family: str | None = None) -> list[str]:
    dataset = _load_dataset()
    if family is None:
        return sorted(dataset["contaminants_lookup"][key]["name"] for key in dataset["contaminants_lookup"])
    resolved_family = _FAMILY_ALIASES.get(_canonical_family_key(family), family)
    return list(dataset["contaminants_by_family"].get(resolved_family, []))


def get_contaminant_family(contaminant: str) -> str | None:
    """Return the normalized family name for one supported contaminant or family alias."""
    text = _clean_text(contaminant)
    if not text:
        return None
    family_name = _FAMILY_ALIASES.get(_canonical_family_key(text))
    if family_name:
        return family_name

    info = _load_dataset()["contaminants_lookup"].get(_canonical_contaminant_key(text))
    if info is None:
        return None
    return str(info["family"])


def expand_requested_contaminants(contaminants: list[str]) -> tuple[list[str], list[str], list[str]]:
    dataset = _load_dataset()
    supported: list[str] = []
    unsupported: list[str] = []
    families: set[str] = set()
    for contaminant in contaminants:
        text = _clean_text(contaminant)
        if not text:
            continue
        family_name = _FAMILY_ALIASES.get(_canonical_family_key(text))
        if family_name:
            family_members = dataset["contaminants_by_family"].get(family_name, [])
            if family_members:
                supported.extend(family_members)
                families.add(family_name)
                continue
        info = dataset["contaminants_lookup"].get(_canonical_contaminant_key(text))
        if info is None:
            unsupported.append(text)
            continue
        supported.append(info["name"])
        families.add(info["family"])
    deduped_supported = list(dict.fromkeys(supported))
    deduped_unsupported = list(dict.fromkeys(unsupported))
    return deduped_supported, deduped_unsupported, sorted(families)


def iter_miscibility_records(contaminants: list[str]) -> list[dict[str, Any]]:
    wanted = {_canonical_contaminant_key(name) for name in contaminants}
    return [
        dict(record)
        for record in _load_dataset()["miscibility_records"]
        if record["contaminant_key"] in wanted
    ]


def iter_logd_records(contaminants: list[str]) -> list[dict[str, Any]]:
    wanted = {_canonical_contaminant_key(name) for name in contaminants}
    return [
        dict(record)
        for record in _load_dataset()["logd_records"]
        if record["contaminant_key"] in wanted
    ]


def get_supported_solvents_for_contaminants(contaminants: list[str]) -> list[str]:
    miscibility = iter_miscibility_records(contaminants)
    logd = iter_logd_records(contaminants)
    solvents = {
        record["solvent_key"]
        for record in miscibility + logd
        if record.get("solvent_key")
    }
    return sorted(solvents)


def get_miscibility_entry(solvent: str, contaminant: str, regime: str | None = None) -> dict[str, Any] | None:
    solvent_key = _canonical_solvent_key(solvent)
    solvent_normalized = _normalize_solvent_name(solvent)
    contaminant_key = _canonical_contaminant_key(contaminant)
    candidates = [
        record
        for record in _load_dataset()["miscibility_records"]
        if record["contaminant_key"] == contaminant_key
        and (
            record["solvent_key"] == solvent_key
            or record["solvent_normalized"] == solvent_normalized
        )
    ]
    if not candidates:
        return None
    if regime is not None:
        for record in candidates:
            if record["temperature_regime"] == regime:
                return dict(record)
    return dict(candidates[0])


def get_logd_entry(solvent: str, contaminant: str) -> dict[str, Any] | None:
    solvent_key = _canonical_solvent_key(solvent)
    solvent_normalized = _normalize_solvent_name(solvent)
    contaminant_key = _canonical_contaminant_key(contaminant)
    for record in _load_dataset()["logd_records"]:
        if record["contaminant_key"] != contaminant_key:
            continue
        if record["solvent_key"] == solvent_key or record["solvent_normalized"] == solvent_normalized:
            return dict(record)
    return None


def choose_miscibility_regime(solvent: str, operating_temperature_c: float | None) -> str:
    if operating_temperature_c is None:
        return "rt"
    solvent_key = _canonical_solvent_key(solvent)
    t_higher_values: list[float] = []
    for record in _load_dataset()["miscibility_records"]:
        if record["solvent_key"] != solvent_key:
            continue
        t_higher_c = record.get("t_higher_c")
        if t_higher_c is not None:
            t_higher_values.append(float(t_higher_c))
    if t_higher_values:
        midpoint = (_RT_TEMPERATURE_C + max(t_higher_values)) / 2.0
        return "t_higher" if operating_temperature_c >= midpoint else "rt"
    return "rt"
