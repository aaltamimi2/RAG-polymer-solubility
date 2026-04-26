"""Merged solvent safety profile service.

This module keeps the safety-card logic deterministic: local physical-property
data, curated peroxide-former metadata, GSK scores, optional PubChem enrichment,
and process-temperature flags are merged into one profile object.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import textwrap
import urllib.parse
from functools import lru_cache
from typing import Any

from strap.database import get_connection
from strap.paths import get_data_path
from strap.solvent_registry import ABBREVIATION_MAP, get_search_terms, resolve_to_property_db

logger = logging.getLogger(__name__)


PHYSICAL_HEADINGS = ("Flash Point", "Autoignition Temperature", "Vapor Pressure", "Boiling Point")


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _display_float(value: float | None, unit: str = "") -> str:
    if value is None:
        return "not available"
    suffix = f" {unit}" if unit else ""
    return f"{value:.1f}{suffix}"


@lru_cache(maxsize=1)
def load_curated_safety_profiles() -> dict[str, dict[str, Any]]:
    """Load curated peroxide/SDS-style metadata keyed by name, alias, and CAS."""

    path = get_data_path("solvent_safety_profiles.csv")
    profiles: dict[str, dict[str, Any]] = {}
    if not path.exists():
        logger.warning("Curated solvent safety profile file not found: %s", path)
        return profiles

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entry = {key: (value or "").strip() for key, value in row.items()}
            keys = [
                entry.get("canonical_name", ""),
                entry.get("cas_number", ""),
                *(entry.get("aliases", "").split("|") if entry.get("aliases") else []),
            ]
            for key in keys:
                normalized = _normalize_key(key)
                if normalized:
                    profiles[normalized] = entry
    return profiles


def lookup_curated_safety_profile(solvent_name: str, cas_number: str | None = None) -> dict[str, Any] | None:
    profiles = load_curated_safety_profiles()
    keys = [solvent_name, cas_number or ""]
    resolved = resolve_to_property_db(solvent_name)
    if resolved:
        keys.append(resolved)
    for key in keys:
        entry = profiles.get(_normalize_key(key))
        if entry:
            return dict(entry)
    return None


def _first_solvent_data_row(solvent_name: str) -> dict[str, Any] | None:
    """Return the best local solvent_data row for a solvent query."""

    conn = get_connection()
    candidates: list[str] = [solvent_name]
    resolved = resolve_to_property_db(solvent_name)
    if resolved:
        candidates.append(resolved)
    expanded = ABBREVIATION_MAP.get(solvent_name.strip().lower())
    if expanded:
        candidates.append(expanded)
    candidates.extend(get_search_terms(solvent_name))

    seen: set[str] = set()
    ordered = []
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate or candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        ordered.append(candidate)

    for candidate in ordered:
        row = conn.execute(
            """
            SELECT solvent_name, solvent_name_in_cosmobase, cas_number, cid,
                   bp__oc_, th, logp, cp__j_g_k_, energy__j_g_
            FROM solvent_data
            WHERE lower(solvent_name) = lower(?)
               OR lower(solvent_name_in_cosmobase) = lower(?)
               OR lower(cas_number) = lower(?)
            LIMIT 1
            """,
            [candidate, candidate, candidate],
        ).fetchone()
        if row:
            return {
                "solvent_name": row[0],
                "cosmobase_name": row[1],
                "cas_number": row[2],
                "cid": int(row[3]) if _as_float(row[3]) is not None else None,
                "boiling_point_c": _as_float(row[4]),
                "recommended_temp_c": _as_float(row[5]),
                "logp": _as_float(row[6]),
                "cp_j_gk": _as_float(row[7]),
                "energy_j_g": _as_float(row[8]),
                "source": "solvent_data",
            }

    for candidate in ordered:
        safe_candidate = candidate.lower().replace("'", "''")
        safe = f"%{safe_candidate}%"
        row = conn.execute(
            """
            SELECT solvent_name, solvent_name_in_cosmobase, cas_number, cid,
                   bp__oc_, th, logp, cp__j_g_k_, energy__j_g_
            FROM solvent_data
            WHERE lower(solvent_name) LIKE ?
               OR lower(solvent_name_in_cosmobase) LIKE ?
            ORDER BY length(solvent_name)
            LIMIT 1
            """,
            [safe, safe],
        ).fetchone()
        if row:
            return {
                "solvent_name": row[0],
                "cosmobase_name": row[1],
                "cas_number": row[2],
                "cid": int(row[3]) if _as_float(row[3]) is not None else None,
                "boiling_point_c": _as_float(row[4]),
                "recommended_temp_c": _as_float(row[5]),
                "logp": _as_float(row[6]),
                "cp_j_gk": _as_float(row[7]),
                "energy_j_g": _as_float(row[8]),
                "source": "solvent_data",
                "matched_by": "substring",
            }
    return None


def _extract_info_strings(obj: Any) -> list[str]:
    values: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "Information" in node:
                for info in node.get("Information") or []:
                    value = info.get("Value") or {}
                    values.extend(_value_to_strings(value))
            for child in node.get("Section") or []:
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(obj)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", value).strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def _value_to_strings(value: dict[str, Any]) -> list[str]:
    strings: list[str] = []
    for item in value.get("StringWithMarkup") or []:
        if isinstance(item, dict) and item.get("String"):
            strings.append(str(item["String"]))
    if "Number" in value:
        number = value.get("Number")
        unit = value.get("Unit")
        if isinstance(number, list):
            number_text = ", ".join(str(n) for n in number)
        else:
            number_text = str(number)
        strings.append(f"{number_text} {unit}".strip() if unit else number_text)
    return strings


def fetch_pubchem_heading_strings(cid: int, heading: str) -> list[str]:
    """Fetch raw strings for one PubChem PUG-View heading."""

    from strap.tools.safety_pubchem import _pubchem_request

    encoded = urllib.parse.quote(heading)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading={encoded}"
    raw = _pubchem_request(url, timeout=15)
    if raw is None:
        return []
    data = json.loads(raw.decode())
    return _extract_info_strings(data.get("Record", {}))


def _celsius_values(text: str) -> list[float]:
    values = [float(match.group(1)) for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?C\b", text, re.I)]
    if values:
        return values
    fahrenheit = [float(match.group(1)) for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?F\b", text, re.I)]
    return [(value - 32.0) * 5.0 / 9.0 for value in fahrenheit]


def _first_temperature_c(strings: list[str], *, prefer_lowest: bool = False) -> dict[str, Any] | None:
    candidates: list[tuple[float, str]] = []
    for text in strings:
        for value in _celsius_values(text):
            candidates.append((value, text))
    if not candidates:
        return None
    value, raw = min(candidates, key=lambda item: item[0]) if prefer_lowest else candidates[0]
    return {"value_c": value, "raw": raw}


_PRESSURE_PAIR_RE = re.compile(
    r"(?P<pressure>\d+(?:\.\d+)?)\s*(?P<punit>mmhg|torr|kpa|pa|atm|bar)"
    r"(?:\s*(?:at|@)\s*(?P<temp>-?\d+(?:\.\d+)?)\s*(?P<tunit>°?\s*[cf]))?",
    re.I,
)


def _pressure_to_kpa(value: float, unit: str) -> float:
    normalized = unit.lower().replace(" ", "")
    if normalized in {"mmhg", "torr"}:
        return value * 0.133322
    if normalized == "pa":
        return value / 1000.0
    if normalized == "atm":
        return value * 101.325
    if normalized == "bar":
        return value * 100.0
    return value


def _temp_to_c(value: float, unit: str | None) -> float | None:
    if not unit:
        return None
    normalized = unit.lower().replace(" ", "").replace("°", "")
    if normalized == "f":
        return (value - 32.0) * 5.0 / 9.0
    if normalized == "c":
        return value
    return None


def _select_vapor_pressure(strings: list[str]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for text in strings:
        normalized = text.replace("[", "").replace("]", "")
        for match in _PRESSURE_PAIR_RE.finditer(normalized):
            pressure = _as_float(match.group("pressure"))
            if pressure is None:
                continue
            temp = _as_float(match.group("temp"))
            candidates.append(
                {
                    "value_kpa": _pressure_to_kpa(pressure, match.group("punit")),
                    "temperature_c": _temp_to_c(temp, match.group("tunit")) if temp is not None else None,
                    "raw": text,
                }
            )
    if not candidates:
        return None
    with_temp = [item for item in candidates if item["temperature_c"] is not None]
    if with_temp:
        return min(with_temp, key=lambda item: abs(float(item["temperature_c"]) - 25.0))
    return candidates[0]


def fetch_pubchem_physical_properties(cid: int) -> dict[str, Any]:
    """Fetch selected physical hazard properties from PubChem."""

    raw: dict[str, list[str]] = {}
    for heading in PHYSICAL_HEADINGS:
        try:
            raw[heading] = fetch_pubchem_heading_strings(cid, heading)
        except Exception as exc:
            logger.debug("PubChem physical heading failed for CID %s heading %s: %s", cid, heading, exc)
            raw[heading] = []

    flash = _first_temperature_c(raw.get("Flash Point", []), prefer_lowest=True)
    autoignition = _first_temperature_c(raw.get("Autoignition Temperature", []), prefer_lowest=True)
    boiling = _first_temperature_c(raw.get("Boiling Point", []))
    vapor = _select_vapor_pressure(raw.get("Vapor Pressure", []))
    return {
        "flash_point_c": flash.get("value_c") if flash else None,
        "flash_point_raw": flash.get("raw") if flash else None,
        "autoignition_c": autoignition.get("value_c") if autoignition else None,
        "autoignition_raw": autoignition.get("raw") if autoignition else None,
        "pubchem_boiling_point_c": boiling.get("value_c") if boiling else None,
        "pubchem_boiling_point_raw": boiling.get("raw") if boiling else None,
        "vapor_pressure_kpa": vapor.get("value_kpa") if vapor else None,
        "vapor_pressure_temp_c": vapor.get("temperature_c") if vapor else None,
        "vapor_pressure_raw": vapor.get("raw") if vapor else None,
        "raw_headings": raw,
    }


def volatility_class(vapor_pressure_kpa: float | None) -> str:
    if vapor_pressure_kpa is None:
        return "unknown"
    if vapor_pressure_kpa >= 50.0:
        return "very high"
    if vapor_pressure_kpa >= 10.0:
        return "high"
    if vapor_pressure_kpa >= 1.0:
        return "moderate"
    return "low"


def assess_temperature_risk(
    *,
    operating_temp_c: float | None,
    boiling_point_c: float | None,
    flash_point_c: float | None,
    autoignition_c: float | None,
    vapor_pressure_kpa: float | None,
) -> dict[str, Any]:
    """Compute heating/volatility flags for one solvent."""

    assessment: dict[str, Any] = {
        "operating_temp_c": operating_temp_c,
        "boiling_point_c": boiling_point_c,
        "flash_point_c": flash_point_c,
        "autoignition_c": autoignition_c,
        "boiling_margin_c": None,
        "autoignition_margin_c": None,
        "flags": [],
        "risk_level": "unknown" if operating_temp_c is None else "low",
        "notes": [],
    }
    flags: list[str] = assessment["flags"]
    notes: list[str] = assessment["notes"]

    if operating_temp_c is None:
        notes.append("No operating temperature was supplied; heating-specific flags were not evaluated.")
    if operating_temp_c is not None and boiling_point_c is not None:
        margin = boiling_point_c - operating_temp_c
        assessment["boiling_margin_c"] = margin
        if margin <= 0:
            flags.append("above_normal_boiling_point")
            notes.append("At or above the normal boiling point at 1 atm; use pressure-rated operation or lower the setpoint.")
        elif margin <= 10:
            flags.append("near_normal_boiling_point")
            notes.append("Within 10 C of the normal boiling point; vapor generation and pressure control are major concerns.")
    if operating_temp_c is not None and flash_point_c is not None:
        if operating_temp_c >= flash_point_c:
            flags.append("above_flash_point")
            notes.append("Operating above the flash point; ignition-source control, ventilation, and inerting should be evaluated.")
    if operating_temp_c is not None and autoignition_c is not None:
        margin = autoignition_c - operating_temp_c
        assessment["autoignition_margin_c"] = margin
        if margin <= 0:
            flags.append("at_or_above_autoignition")
            notes.append("At or above reported autoignition temperature; this setpoint is not acceptable without redesign.")
        elif margin <= 50:
            flags.append("low_autoignition_margin")
            notes.append("Within 50 C of reported autoignition temperature; avoid hot surfaces and reassess equipment limits.")
    if autoignition_c is not None and boiling_point_c is not None:
        gap = autoignition_c - boiling_point_c
        assessment["autoignition_boiling_gap_c"] = gap
        if gap <= 75:
            flags.append("autoignition_close_to_boiling_point")
            notes.append("Autoignition temperature is close to the boiling point; boiling/condensing equipment needs stricter controls.")
    vol = volatility_class(vapor_pressure_kpa)
    assessment["volatility_class"] = vol
    if vol in {"high", "very high"}:
        flags.append(f"{vol.replace(' ', '_')}_volatility")
        notes.append("Room-temperature vapor pressure indicates substantial vapor-generation potential.")

    if {"above_normal_boiling_point", "at_or_above_autoignition"} & set(flags):
        assessment["risk_level"] = "critical"
    elif {"near_normal_boiling_point", "above_flash_point", "low_autoignition_margin", "very_high_volatility"} & set(flags):
        assessment["risk_level"] = "high"
    elif flags:
        assessment["risk_level"] = "moderate"
    elif operating_temp_c is not None:
        assessment["risk_level"] = "low"
    return assessment


def _lookup_gscore(solvent_name: str) -> dict[str, Any] | None:
    try:
        from strap.tools.safety_gsk import lookup_local_gscore_data

        return lookup_local_gscore_data(solvent_name)
    except Exception as exc:
        logger.debug("G-score lookup failed for %s: %s", solvent_name, exc)
        return None


def _fetch_pubchem_hazards(cid: int) -> dict[str, Any]:
    try:
        from strap.tools.safety_pubchem import fetch_pubchem_ghs_data, fetch_pubchem_toxicity_data

        ghs = fetch_pubchem_ghs_data(cid) or {}
        tox = fetch_pubchem_toxicity_data(cid) or {}
        if not tox.get("ld50_values") or not tox.get("lc50_values"):
            try:
                values = fetch_pubchem_heading_strings(cid, "Non-Human Toxicity Values")
            except Exception:
                values = []
            if not tox.get("ld50_values"):
                tox["ld50_values"] = [value for value in values if re.search(r"\bLD50\b", value, re.I)][:5]
            if not tox.get("lc50_values"):
                tox["lc50_values"] = [value for value in values if re.search(r"\bLC50\b", value, re.I)][:3]
    except Exception as exc:
        logger.debug("PubChem hazard enrichment failed for CID %s: %s", cid, exc)
        ghs, tox = {}, {}
    return {"ghs": ghs, "toxicity": tox}


def build_solvent_safety_profile(
    solvent_name: str,
    *,
    operating_temp_c: float | None = None,
    include_pubchem: bool = True,
) -> dict[str, Any]:
    """Build a merged solvent safety profile for card rendering."""

    local = _first_solvent_data_row(solvent_name) or {}
    display_name = str(local.get("solvent_name") or resolve_to_property_db(solvent_name) or solvent_name).strip()
    cas_number = str(local.get("cas_number") or "").strip() or None
    cid = local.get("cid")

    if include_pubchem and cid is None:
        try:
            from strap.tools.safety_pubchem import fetch_pubchem_cid

            cid = fetch_pubchem_cid(display_name)
        except Exception:
            cid = None

    curated = lookup_curated_safety_profile(display_name, cas_number)
    gscore = _lookup_gscore(display_name) or _lookup_gscore(solvent_name)

    physical: dict[str, Any] = {}
    hazards: dict[str, Any] = {"ghs": {}, "toxicity": {}}
    if include_pubchem and cid is not None:
        physical = fetch_pubchem_physical_properties(int(cid))
        hazards = _fetch_pubchem_hazards(int(cid))

    boiling_point_c = local.get("boiling_point_c") or physical.get("pubchem_boiling_point_c")
    flash_point_c = physical.get("flash_point_c")
    autoignition_c = physical.get("autoignition_c")
    vapor_pressure_kpa = physical.get("vapor_pressure_kpa")

    temperature_assessment = assess_temperature_risk(
        operating_temp_c=operating_temp_c,
        boiling_point_c=boiling_point_c,
        flash_point_c=flash_point_c,
        autoignition_c=autoignition_c,
        vapor_pressure_kpa=vapor_pressure_kpa,
    )

    data_gaps: list[str] = []
    if flash_point_c is None:
        data_gaps.append("flash_point_c")
    if autoignition_c is None:
        data_gaps.append("autoignition_c")
    if vapor_pressure_kpa is None:
        data_gaps.append("vapor_pressure_kpa")
    if not curated:
        data_gaps.append("peroxide_former_class")
    if not hazards.get("toxicity", {}).get("ld50_values"):
        data_gaps.append("ld50_values")

    peroxide = {
        "peroxide_former_class": (curated or {}).get("peroxide_former_class"),
        "peroxide_former_label": (curated or {}).get("peroxide_former_label"),
        "peroxide_notes": (curated or {}).get("peroxide_notes"),
        "sds_storage_category": (curated or {}).get("sds_storage_category"),
        "source_label": (curated or {}).get("source_label"),
        "source_url": (curated or {}).get("source_url"),
    }
    if not curated:
        peroxide["peroxide_former_class"] = "unknown"
        peroxide["peroxide_former_label"] = "Not found in local curated peroxide-former table"
        peroxide["peroxide_notes"] = "Verify against the supplier SDS and local peroxide-former inventory rules."

    profile = {
        "query": solvent_name,
        "identity": {
            "name": display_name,
            "cas_number": cas_number,
            "pubchem_cid": cid,
            "cosmobase_name": local.get("cosmobase_name"),
        },
        "physical_properties": {
            "boiling_point_c": boiling_point_c,
            "recommended_temp_c": local.get("recommended_temp_c"),
            "flash_point_c": flash_point_c,
            "flash_point_raw": physical.get("flash_point_raw"),
            "autoignition_c": autoignition_c,
            "autoignition_raw": physical.get("autoignition_raw"),
            "vapor_pressure_kpa": vapor_pressure_kpa,
            "vapor_pressure_temp_c": physical.get("vapor_pressure_temp_c"),
            "vapor_pressure_raw": physical.get("vapor_pressure_raw"),
            "volatility_class": volatility_class(vapor_pressure_kpa),
            "logp": local.get("logp"),
            "cp_j_gk": local.get("cp_j_gk"),
            "energy_j_g": local.get("energy_j_g"),
        },
        "gscore": gscore or {},
        "ghs": hazards.get("ghs") or {},
        "toxicity": hazards.get("toxicity") or {},
        "peroxide_risk": peroxide,
        "process_temperature_assessment": temperature_assessment,
        "data_gaps": data_gaps,
        "sources": {
            "local_properties": "data/Solvent_Data.csv" if local else None,
            "gscore": (gscore or {}).get("source"),
            "pubchem": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None,
            "curated_peroxide": peroxide.get("source_url"),
        },
    }
    return profile


_SAFETY_CARD_WIDTH = 104


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate_text(value: Any, width: int) -> str:
    text = _compact_text(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)].rstrip() + "…"


def _truncate_preserving_space(value: Any, width: int) -> str:
    text = str(value or "").replace("\n", " ")
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)].rstrip() + "…"


def _border_line(left: str, fill: str, right: str, *, title: str | None = None, width: int = _SAFETY_CARD_WIDTH) -> str:
    inner_width = width - 2
    if not title:
        return left + fill * inner_width + right
    label = f" {title} "
    if len(label) > inner_width:
        label = label[:inner_width]
    left_fill = max(0, (inner_width - len(label)) // 2)
    right_fill = max(0, inner_width - len(label) - left_fill)
    return left + fill * left_fill + label + fill * right_fill + right


def _box_line(text: str = "", *, width: int = _SAFETY_CARD_WIDTH) -> str:
    content_width = width - 4
    return f"│ {_truncate_preserving_space(text, content_width):<{content_width}} │"


def _box_wrapped(text: str, *, prefix: str = "", width: int = _SAFETY_CARD_WIDTH) -> list[str]:
    content_width = width - 4
    body = _compact_text(text)
    if not body:
        return [_box_line(prefix.rstrip(), width=width)]
    initial = prefix + body
    subsequent_indent = " " * len(prefix)
    wrapped = textwrap.wrap(
        initial,
        width=content_width,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        wrapped = [initial]
    return [_box_line(line, width=width) for line in wrapped]


def _box_split(left_text: str, right_text: str, *, width: int = _SAFETY_CARD_WIDTH) -> list[str]:
    content_width = width - 4
    left = _compact_text(left_text)
    right = _compact_text(right_text)
    if not right:
        return _box_wrapped(left, width=width)
    if len(left) + len(right) + 1 <= content_width:
        spacing = " " * (content_width - len(left) - len(right))
        return [_box_line(f"{left}{spacing}{right}", width=width)]
    return [*_box_wrapped(left, width=width), _box_line(right, width=width)]


def _section_line(title: str, *, width: int = _SAFETY_CARD_WIDTH) -> str:
    return _border_line("├", "─", "┤", title=title, width=width)


def _status_token(label: str, value: Any) -> str:
    value_text = _compact_text(value) or "not available"
    return f"{label}: {value_text}"


def _temp_text(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "not available"
    return f"{number:.1f} C"


def _score_text(gscore: dict[str, Any]) -> str:
    score = gscore.get("g_score")
    if not isinstance(score, (int, float)):
        return "not available"
    source = "ML" if gscore.get("ml_predicted") else "GSK"
    return f"{score:.2f}/10 {source}"


def _pressure_text(props: dict[str, Any]) -> str:
    vp = props.get("vapor_pressure_kpa")
    vp_temp = props.get("vapor_pressure_temp_c")
    if vp is None:
        return "not available"
    suffix = f" @ {float(vp_temp):.1f} C" if vp_temp is not None else ""
    return f"{float(vp):.2f} kPa{suffix}"


def _risk_label(value: Any) -> str:
    text = _compact_text(value or "unknown").upper()
    return text if text else "UNKNOWN"


def _format_flags(flags: list[Any]) -> str:
    labels = {
        "above_normal_boiling_point": "above BP",
        "near_normal_boiling_point": "near BP",
        "above_flash_point": "above flash point",
        "at_or_above_autoignition": "at/above autoignition",
        "low_autoignition_margin": "low autoignition margin",
        "autoignition_close_to_boiling_point": "autoignition near BP",
        "very_high_volatility": "very high volatility",
        "high_volatility": "high volatility",
    }
    values = [labels.get(str(flag), str(flag).replace("_", " ")) for flag in flags]
    return " | ".join(values) if values else "none"


def _format_source_label(source: str) -> str:
    if source.startswith("https://pubchem.ncbi.nlm.nih.gov/compound/"):
        return source
    return source


def format_solvent_safety_card(profile: dict[str, Any]) -> str:
    identity = profile["identity"]
    props = profile["physical_properties"]
    gscore = profile.get("gscore") or {}
    ghs = profile.get("ghs") or {}
    tox = profile.get("toxicity") or {}
    peroxide = profile.get("peroxide_risk") or {}
    temp = profile.get("process_temperature_assessment") or {}

    name = identity.get("name") or profile.get("query") or "Unknown solvent"
    risk = _risk_label(temp.get("risk_level", "unknown"))

    lines = [_border_line("╭", "─", "╮", title="DISSOLVE SAFETY CARD")]
    lines.extend(_box_split(str(name), f"RISK: {risk}"))

    meta = []
    if identity.get("cas_number"):
        meta.append(f"CAS {identity['cas_number']}")
    if identity.get("pubchem_cid"):
        meta.append(f"PubChem CID {identity['pubchem_cid']}")
    if meta:
        lines.extend(_box_wrapped(" | ".join(meta)))

    overview = [_status_token("G-score", _score_text(gscore))]
    if props.get("logp") is not None:
        overview.append(_status_token("LogP", f"{props['logp']:.2f}"))
    lines.extend(_box_wrapped(" | ".join(overview)))

    lines.append(_section_line("Thermal / Volatility"))
    lines.extend(
        _box_wrapped(
            " | ".join(
                [
                    _status_token("Boiling point", _temp_text(props.get("boiling_point_c"))),
                    _status_token("Flash point", _temp_text(props.get("flash_point_c"))),
                    _status_token("Autoignition", _temp_text(props.get("autoignition_c"))),
                ]
            )
        )
    )
    lines.extend(
        _box_wrapped(
            " | ".join(
                [
                    _status_token("Vapor pressure", _pressure_text(props)),
                    _status_token("Volatility", props.get("volatility_class", "unknown")),
                ]
            )
        )
    )

    lines.append(_section_line("Process Temperature"))
    if temp.get("operating_temp_c") is None:
        lines.extend(_box_wrapped("No operating temperature supplied.", prefix="• "))
    else:
        process_values = [_status_token("Operating", _temp_text(temp.get("operating_temp_c")))]
        if temp.get("boiling_margin_c") is not None:
            process_values.append(_status_token("BP margin @ 1 atm", _temp_text(temp.get("boiling_margin_c"))))
        if temp.get("autoignition_margin_c") is not None:
            process_values.append(_status_token("Autoignition margin", _temp_text(temp.get("autoignition_margin_c"))))
        lines.extend(_box_wrapped(" | ".join(process_values)))
    lines.extend(_box_wrapped(_status_token("Heating risk", risk)))
    if temp.get("flags"):
        lines.extend(_box_wrapped(_status_token("Flags", _format_flags(temp.get("flags") or []))))
    for note in temp.get("notes", [])[:4]:
        lines.extend(_box_wrapped(str(note), prefix="• "))

    lines.append(_section_line("Peroxide / Storage"))
    peroxide_values = [
        _status_token("Category", peroxide.get("peroxide_former_class") or "unknown"),
        _status_token("Label", peroxide.get("peroxide_former_label") or "unknown"),
    ]
    if peroxide.get("sds_storage_category"):
        peroxide_values.append(_status_token("SDS/storage", peroxide["sds_storage_category"]))
    lines.extend(_box_wrapped(" | ".join(peroxide_values)))
    if peroxide.get("peroxide_notes"):
        lines.extend(_box_wrapped(str(peroxide["peroxide_notes"]), prefix="• "))

    lines.append(_section_line("GHS / Toxicity"))
    ghs_values = [_status_token("Signal word", ghs.get("signal_word") or "not available")]
    pictograms = ghs.get("pictograms") or []
    if pictograms:
        ghs_values.append(_status_token("Pictograms", ", ".join(pictograms[:6])))
    lines.extend(_box_wrapped(" | ".join(ghs_values)))
    hazards = ghs.get("hazard_statements") or []
    for statement in hazards[:4]:
        lines.extend(_box_wrapped(str(statement), prefix="• "))
    ld50 = tox.get("ld50_values") or []
    if ld50:
        lines.extend(_box_wrapped("LD50 / acute toxicity"))
        for value in ld50[:3]:
            lines.extend(_box_wrapped(str(value), prefix="• "))
    else:
        lines.extend(_box_wrapped("LD50: not available from current PubChem toxicity pull", prefix="• "))

    gaps = profile.get("data_gaps") or []
    if gaps:
        lines.append(_section_line("Data Gaps"))
        lines.extend(_box_wrapped(", ".join(gaps), prefix="• "))

    source_values = [value for value in (profile.get("sources") or {}).values() if value]
    if source_values:
        lines.append(_section_line("Sources"))
        for source in source_values:
            lines.extend(_box_wrapped(_format_source_label(str(source)), prefix="• "))
    lines.append(_border_line("╰", "─", "╯"))
    return "\n".join(lines)


def format_solvent_safety_comparison(
    profiles: list[dict[str, Any]],
    *,
    operating_temp_c: float | None = None,
) -> str:
    """Render a compact terminal comparison card for multiple solvents."""

    lines = [_border_line("╭", "─", "╮", title="DISSOLVE SAFETY COMPARISON")]
    if operating_temp_c is not None:
        lines.extend(_box_wrapped(f"Operating temperature: {operating_temp_c:.1f} C"))

    lines.append(_section_line("Ranked Profiles"))
    header = (
        f"{'Solvent':<24} {'BP C':>7} {'Flash C':>8} {'Vap kPa':>9} "
        f"{'Volatility':<11} {'Peroxide':<11} {'Risk':<9} {'G-score':>7}"
    )
    lines.append(_box_line(header))
    lines.append(_box_line("-" * min(len(header), _SAFETY_CARD_WIDTH - 4)))
    for profile in profiles:
        identity = profile.get("identity") or {}
        props = profile.get("physical_properties") or {}
        peroxide = profile.get("peroxide_risk") or {}
        temp = profile.get("process_temperature_assessment") or {}
        gscore = profile.get("gscore") or {}
        score = gscore.get("g_score")
        vapor = props.get("vapor_pressure_kpa")
        vapor_text = "-" if vapor is None else f"{float(vapor):.2f}"
        score_value = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        row = (
            f"{_truncate_text(identity.get('name') or profile.get('query') or '-', 24):<24} "
            f"{_display_float(props.get('boiling_point_c')):>7} "
            f"{_display_float(props.get('flash_point_c')):>8} "
            f"{vapor_text:>9} "
            f"{_truncate_text(props.get('volatility_class') or 'unknown', 11):<11} "
            f"{_truncate_text(peroxide.get('peroxide_former_class') or 'unknown', 11):<11} "
            f"{_truncate_text(temp.get('risk_level') or 'unknown', 9):<9} "
            f"{score_value:>7}"
        )
        lines.append(_box_line(row))

    if profiles:
        worst = max(
            profiles,
            key=lambda item: {
                "critical": 4,
                "high": 3,
                "moderate": 2,
                "low": 1,
                "unknown": 0,
            }.get(str((item.get("process_temperature_assessment") or {}).get("risk_level", "unknown")), 0),
        )
        worst_name = (worst.get("identity") or {}).get("name") or worst.get("query") or "unknown"
        worst_risk = (worst.get("process_temperature_assessment") or {}).get("risk_level", "unknown")
        lines.append(_section_line("Callout"))
        lines.extend(_box_wrapped(f"Highest heating-risk profile: {worst_name} ({worst_risk}).", prefix="• "))

    lines.append(_border_line("╰", "─", "╯"))
    return "\n".join(lines)
