"""BioSTEAM service helpers shared by the tool adapter layer.

This module keeps BioSTEAM-specific catalogs, request builders, and response
helpers out of the agent-facing tool module so the tool file can stay focused
on delegation to the runner and presentation logic.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from strap.paths import get_data_path
from strap.solvent_registry import SOLVENT_REGISTRY, resolve_to_biosteam, resolve_to_interp_key
from strap.services.tool_response_service import json_tool_error, json_tool_response

# Chlorinated solvents that fail in BioSTEAM (HCl not in property package)
CHLORINATED_BLOCKLIST = frozenset(
    {
        "Tetrachloroethylene",
        "o-Chlorotoluene",
        "Dichloromethane",
        "Chloroform",
    }
)

# Legacy fallback solvent sets used when the TEA/LCA CSV is unavailable.
_LEGACY_PE_SOLVENTS_CORE = [
    "sec-Butyl Acetate",
    "Isobutyl Acetate",
    "Methylcyclohexane",
    "Dodecanol",
    "Heptane",
    "Toluene",
    "Xylene",
]

# Extended PE/LDPE solvents from COMMON-SOLVENTS-DATABASE (thermosteam-validated)
_LEGACY_PE_SOLVENTS_EXTENDED = [
    "o-Xylene",
    "p-Xylene",
    "Cyclohexane",
    "Dodecane",
    "Hexane",
    "Benzene",
    "Acetone",
    "2-Butanone",
    "Ethyl acetate",
    "Tetrahydrofuran",
    "1-Propanol",
    "Ethanol",
    "Methanol",
    "Isopropanol",
    "tert-Butanol",
    "Cyclohexanol",
    "N,N-Dimethylformamide",
    "Diphenyl ether",
    "Acetylacetone",
    "2,3-Dihydropyran",
    "Tetrahydropyran",
    "Triethylamine",
    "Methyl acetate",
]

_LEGACY_PE_SOLVENTS = _LEGACY_PE_SOLVENTS_CORE + _LEGACY_PE_SOLVENTS_EXTENDED

_LEGACY_EVOH_SOLVENTS = [
    "Ethylene Glycol",
    "Pyridazine",
]

_LEGACY_EVOH_SOLVENTS_E2 = [
    "butane-1,4-diol",
    "Diethanolamine",
    "Diethylene glycol",
    "Ethylene Glycol",
    "Propylene Glycol",
    "Pyridazine",
    "gamma-butyrolactone",
    "Dimethyl sulfoxide",
    "N,N-Dimethylformamide",
    "Triethylamine",
    "Methanol",
    "Ethanol",
    "Isopropanol",
]

_LEGACY_PET_SOLVENTS = [
    "Toluene",
    "Xylene",
    "Acetone",
    "N,N-Dimethylformamide",
    "Tetrahydrofuran",
    "2-Butanone",
    "Benzene",
]

_LEGACY_LDPE_SOLVENTS = list(_LEGACY_PE_SOLVENTS)

_LEGACY_PS_SOLVENTS = [
    "Toluene",
    "Xylene",
    "Tetrahydrofuran",
    "Acetone",
    "2-Butanone",
    "Cyclohexane",
    "Benzene",
    "Ethyl acetate",
]

_LEGACY_PP_SOLVENTS = [
    "Toluene",
    "Xylene",
    "Dodecane",
    "Cyclohexane",
    "Tetrahydrofuran",
]

_LEGACY_PVC_SOLVENTS = [
    "Tetrahydrofuran",
    "2-Butanone",
    "N,N-Dimethylformamide",
    "Acetone",
]

_LEGACY_PC_SOLVENTS = [
    "Dichloromethane",
    "Tetrahydrofuran",
    "N,N-Dimethylformamide",
    "Acetone",
    "Toluene",
]

ALL_ENERGY_CASES = ["C1", "C2", "C3"]

_TEA_LCA_SOLVENT_CSV = get_data_path("60_common_solvents-TEA-LCA.csv")
_TEA_LCA_TH_THRESHOLD = 5.0


def _safe_float(raw: str | None) -> float:
    try:
        return float((raw or "").strip())
    except (AttributeError, ValueError):
        return 0.0


def _normalize_csv_biosteam_name(row: dict[str, str]) -> str | None:
    for candidate in (row.get("name_cosmobase"), row.get("name_biosteam")):
        if not candidate:
            continue
        resolved = resolve_to_biosteam(candidate)
        if resolved:
            return resolved
    raw = (row.get("name_biosteam") or row.get("name_cosmobase") or "").strip()
    return raw or None


def _sanitize_biosteam_solvent_name(solvent: str) -> str:
    """Return a BioSTEAM-safe solvent token for runner configs.

    Some valid catalog names begin with digits or punctuation
    (for example ``2,3-Dihydropyran``). The downstream BioSTEAM process model
    can reject those when it tries to construct internal aliases. When that
    happens, prefer an alphabetic registry alias for the runner config while
    preserving the original solvent identity elsewhere in the workflow.
    """
    text = str(solvent or "").strip()
    if not text:
        return text
    if text[0].isalpha():
        return text

    registry_key = resolve_to_interp_key(text)
    if registry_key:
        info = SOLVENT_REGISTRY.get(registry_key, {})
        for alias in info.get("aliases", []):
            alias_text = str(alias or "").strip()
            if alias_text and alias_text[0].isalpha():
                return alias_text.title()
    # Fall back to a generic alphabetic alias when the registry does not carry
    # one but the solvent is a simple locant-prefixed name like "1-butanol".
    stripped = re.sub(r"^[^A-Za-z]+", "", text)
    stripped = re.sub(r"^[A-Za-z]?(?:,\d+)*-", "", stripped) if not stripped[:1].isalpha() else stripped
    if stripped and stripped[0].isdigit():
        stripped = re.sub(r"^\d+(?:,\d+)*-?", "", stripped)
    if stripped and stripped[0].isalpha():
        return stripped[0].upper() + stripped[1:]
    return text


def _load_csv_solvent_catalog() -> dict[str, list[str]]:
    if not _TEA_LCA_SOLVENT_CSV.exists():
        return {}

    rows = list(csv.DictReader(_TEA_LCA_SOLVENT_CSV.open(encoding="utf-8")))
    scored: dict[str, list[tuple[float, str]]] = {
        "EVOH": [],
        "LDPE": [],
        "HDPE": [],
        "PET": [],
        "PP": [],
        "PS": [],
        "PVC": [],
        "PC": [],
        "PE": [],
    }

    for row in rows:
        normalized_name = _normalize_csv_biosteam_name(row)
        if not normalized_name:
            continue

        th_scores = {
            "EVOH": _safe_float(row.get("EVOH-TH")),
            "LDPE": _safe_float(row.get("LDPE-TH")),
            "HDPE": _safe_float(row.get("HDPE-TH")),
            "PET": _safe_float(row.get("PET-TH")),
            "PP": _safe_float(row.get("PP-TH")),
            "PS": _safe_float(row.get("PS-TH")),
            "PVC": _safe_float(row.get("PVC-TH")),
            "PC": _safe_float(row.get("PC-TH")),
        }
        for polymer, score in th_scores.items():
            if score >= _TEA_LCA_TH_THRESHOLD:
                scored[polymer].append((score, normalized_name))

        pe_score = max(th_scores["LDPE"], th_scores["HDPE"])
        if pe_score >= _TEA_LCA_TH_THRESHOLD:
            scored["PE"].append((pe_score, normalized_name))

    catalog: dict[str, list[str]] = {}
    for polymer, entries in scored.items():
        entries.sort(key=lambda item: (-item[0], item[1]))
        deduped: list[str] = []
        seen: set[str] = set()
        for _score, name in entries:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        catalog[polymer] = deduped
    return catalog


_CSV_SOLVENT_CATALOG = _load_csv_solvent_catalog()

PE_SOLVENTS = _CSV_SOLVENT_CATALOG.get("PE", list(_LEGACY_PE_SOLVENTS))
PE_SOLVENTS_CORE = list(PE_SOLVENTS[:10])
PE_SOLVENTS_EXTENDED = list(PE_SOLVENTS[10:])
EVOH_SOLVENTS = _CSV_SOLVENT_CATALOG.get("EVOH", list(_LEGACY_EVOH_SOLVENTS))
EVOH_SOLVENTS_E2 = list(EVOH_SOLVENTS)
PET_SOLVENTS = _CSV_SOLVENT_CATALOG.get("PET", list(_LEGACY_PET_SOLVENTS))
LDPE_SOLVENTS = _CSV_SOLVENT_CATALOG.get("LDPE", list(_LEGACY_LDPE_SOLVENTS))
PS_SOLVENTS = _CSV_SOLVENT_CATALOG.get("PS", list(_LEGACY_PS_SOLVENTS))
PP_SOLVENTS = _CSV_SOLVENT_CATALOG.get("PP", list(_LEGACY_PP_SOLVENTS))
PVC_SOLVENTS = _CSV_SOLVENT_CATALOG.get("PVC", list(_LEGACY_PVC_SOLVENTS))
PC_SOLVENTS = _CSV_SOLVENT_CATALOG.get("PC", list(_LEGACY_PC_SOLVENTS))

_BATCH_SCREEN_PRIORITY: dict[str, list[str]] = {
    "PE": [
        "Heptane",
        "Toluene",
        "p-Xylene",
        "o-Xylene",
        "Cyclohexane",
        "Dodecane",
        "Hexane",
        "Methylcyclohexane",
        "sec-Butyl Acetate",
        "Isobutyl Acetate",
    ],
    "LDPE": [
        "Heptane",
        "Toluene",
        "p-Xylene",
        "o-Xylene",
        "Cyclohexane",
        "Dodecane",
        "Hexane",
        "Methylcyclohexane",
        "sec-Butyl Acetate",
        "Isobutyl Acetate",
    ],
}

POLYMER_MARKET_VALUES = {
    "PE": 1.10,
    "LDPE": 1.10,
    "HDPE": 1.20,
    "PP": 1.15,
    "PS": 1.30,
    "PVC": 0.90,
    "PET": 1.05,
    "EVOH": 4.50,
    "Nylon6": 2.80,
    "Nylon66": 3.00,
    "PMMA": 2.50,
    "PC": 2.50,
}

SEQUENTIAL_STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "P1": {
        "target_plastic": "PE",
        "target_plastic_percent": 60,
        "processing_capacity": 20000,
        "description": "PE first recovery",
    },
    "E1": {
        "target_plastic": "PE",
        "target_plastic_percent": 10,
        "processing_capacity": 20000,
        "description": "EVOH first recovery (PE target)",
    },
    "E2": {
        "target_plastic": "PE",
        "target_plastic_percent": 25,
        "processing_capacity": 8000,
        "description": "EVOH second recovery (PE target)",
    },
    "P2": {
        "target_plastic": "PE",
        "target_plastic_percent": 66.667,
        "processing_capacity": 18000,
        "description": "PE second recovery",
    },
}

def runner_unavailable_error() -> str:
    """Shared structured response for missing BioSTEAM dependencies."""
    return json_tool_error(
        "BioSTEAM runner module not available. Install BioSTEAM dependencies.",
        tool_name="biosteam_runner",
        error_code="runner_unavailable",
    )


def expand_solvents(solvents_str: str, target_plastic: str) -> list[str]:
    """Parse comma-separated solvent strings or expand shorthand keywords."""
    token = solvents_str.strip().lower()
    if token == "all_pe":
        return list(PE_SOLVENTS)
    if token == "all_ldpe":
        return list(LDPE_SOLVENTS)
    if token == "all_evoh":
        return list(EVOH_SOLVENTS)
    if token == "all_evoh_e2":
        return list(EVOH_SOLVENTS_E2)
    if token == "all_pet":
        return list(PET_SOLVENTS)
    if token == "all_ps":
        return list(PS_SOLVENTS)
    if token == "all_pp":
        return list(PP_SOLVENTS)
    if token == "all_pvc":
        return list(PVC_SOLVENTS)
    if token == "all_pc":
        return list(PC_SOLVENTS)
    if token == "all":
        target = target_plastic.upper()
        if target == "EVOH":
            return list(EVOH_SOLVENTS)
        if target == "PET":
            return list(PET_SOLVENTS)
        if target == "LDPE":
            return list(LDPE_SOLVENTS)
        if target == "PS":
            return list(PS_SOLVENTS)
        if target == "PP":
            return list(PP_SOLVENTS)
        if target == "PVC":
            return list(PVC_SOLVENTS)
        if target == "PC":
            return list(PC_SOLVENTS)
        return list(PE_SOLVENTS)

    alias = resolve_to_biosteam(token)
    if alias:
        return [alias]

    resolved: list[str] = []
    for solvent in (item.strip() for item in solvents_str.split(",")):
        if not solvent:
            continue
        resolved.append(resolve_to_biosteam(solvent) or solvent)
    return resolved


def expand_energy_cases(cases_str: str) -> list[str]:
    """Parse comma-separated energy cases or expand `all`."""
    if cases_str.strip().lower() == "all":
        return list(ALL_ENERGY_CASES)
    return [case.strip().upper() for case in cases_str.split(",") if case.strip()]


def prioritize_batch_solvents(solvents: list[str], target_plastic: str) -> list[str]:
    """Prioritize solvents for large screening batches.

    The goal is to surface decision-quality candidates early so the batch tool
    can return useful partial rankings under a tight wall-clock budget.
    """
    target = target_plastic.upper()
    preferred = _BATCH_SCREEN_PRIORITY.get(target, [])
    seen: set[str] = set()
    ordered: list[str] = []
    for solvent in preferred:
        if solvent in solvents and solvent not in seen:
            ordered.append(solvent)
            seen.add(solvent)
    for solvent in solvents:
        if solvent not in seen:
            ordered.append(solvent)
            seen.add(solvent)
    return ordered


def build_single_config(
    *,
    solvent: str,
    target_plastic: str = "PE",
    energy_case: str = "C1",
    target_plastic_percent: float = 60,
    processing_capacity: float = 20000,
    dissolution_temp_c: float | None = None,
    precipitation_temp_c: float | None = 25,
    solvent_price: float | None = None,
) -> dict[str, Any]:
    """Build one normalized BioSTEAM runner config."""

    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    raw_solvent = str(solvent or "").strip()
    canonical_solvent = resolve_to_biosteam(raw_solvent) or raw_solvent
    safe_solvent = _sanitize_biosteam_solvent_name(canonical_solvent)
    dissolution_temp = _optional_float(dissolution_temp_c)
    precipitation_temp = _optional_float(precipitation_temp_c)
    price = _optional_float(solvent_price)
    config: dict[str, Any] = {
        "solvent": safe_solvent,
        "target_plastic": target_plastic,
        "target_plastic_percent": target_plastic_percent,
        "processing_capacity": processing_capacity,
        "energy_case": energy_case,
    }
    if safe_solvent != raw_solvent:
        config["solvent_input_name"] = raw_solvent
    if dissolution_temp is not None:
        config["dissolution_temperature_c"] = dissolution_temp
    if precipitation_temp is not None:
        config["precipitation_temperature_c"] = precipitation_temp
    if price is not None:
        config["solvent_price"] = price
    return config


def build_manual_batch_configs(
    *,
    solvents: list[str],
    energy_cases: list[str],
    target_plastic: str,
    target_plastic_percent: float,
    processing_capacity: float,
) -> list[dict[str, Any]]:
    """Fallback config builder when the runner helper is unavailable."""
    return [
        build_single_config(
            solvent=solvent,
            target_plastic=target_plastic,
            target_plastic_percent=target_plastic_percent,
            processing_capacity=processing_capacity,
            energy_case=energy_case,
        )
        for solvent in solvents
        for energy_case in energy_cases
    ]


def parse_json_array(raw_json: str, *, field_name: str) -> list[Any]:
    """Parse a JSON array argument and raise a readable ValueError on failure."""
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field_name}: {exc}") from exc
    if not isinstance(parsed, list) or len(parsed) < 1:
        raise ValueError(f"{field_name} must be a JSON array with at least 1 item.")
    return parsed


def extract_successful_results(data: Any) -> list[dict[str, Any]]:
    """Normalize BioSTEAM tool output shapes into a list of successful results."""
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return [result for result in data["results"] if result.get("success", False)]
        if "tea" in data and data.get("success", False):
            return [data]
        if "per_polymer" in data:
            return [
                item.get("result", {})
                for item in data["per_polymer"]
                if item.get("result", {}).get("success", False)
            ]
        return []
    if isinstance(data, list):
        return [result for result in data if result.get("success", False)]
    return []
