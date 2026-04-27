"""Solvent safety-card tools."""

from __future__ import annotations

from typing import Any

from strap.services.solvent_safety_service import (
    build_solvent_safety_profile,
    format_solvent_safety_card,
    format_solvent_safety_comparison,
)
from strap.services.tool_response_service import json_tool_error, json_tool_success
from strap.tools._helpers import safe_tool_wrapper


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_sort_key(profile: dict[str, Any]) -> tuple[int, str]:
    order = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "unknown": 0}
    risk = profile.get("process_temperature_assessment", {}).get("risk_level", "unknown")
    return order.get(str(risk), 0), str(profile.get("identity", {}).get("name", ""))


@safe_tool_wrapper(structured_output=True)
def get_solvent_safety_card(
    solvent_name: str,
    operating_temp_c: float | None = None,
    include_pubchem: bool = True,
) -> str:
    """Render a solvent safety card with thermal, volatility, toxicity, and peroxide notes.

    Args:
        solvent_name: Solvent name, abbreviation, or CAS number.
        operating_temp_c: Optional process/heating temperature in Celsius.
        include_pubchem: If True, enrich the local profile with PubChem GHS,
            toxicity, flash point, vapor pressure, and autoignition data.

    WHEN TO USE:
    - "Show a safety card for THF"
    - "How should I heat toluene to 110 C?"
    - "Does diethyl ether form peroxides?"
    - "What are the flash point, LD50, vapor pressure, and boiling point for DMSO?"
    """

    solvent_name = str(solvent_name or "").strip()
    if not solvent_name:
        return json_tool_error(
            "No solvent name provided.",
            tool_name="get_solvent_safety_card",
            error_code="missing_solvent",
        )

    profile = build_solvent_safety_profile(
        solvent_name,
        operating_temp_c=_parse_optional_float(operating_temp_c),
        include_pubchem=include_pubchem,
    )
    display = format_solvent_safety_card(profile)
    return json_tool_success(
        display,
        tool_name="get_solvent_safety_card",
        solvent_name=solvent_name,
        operating_temp_c=_parse_optional_float(operating_temp_c),
        include_pubchem=include_pubchem,
        safety_profile=profile,
    )


@safe_tool_wrapper(structured_output=True)
def compare_solvent_safety_cards(
    solvent_names: str,
    operating_temp_c: float | None = None,
    include_pubchem: bool = True,
    limit: int = 6,
) -> str:
    """Compare solvent safety cards for a short solvent list.

    Args:
        solvent_names: Comma-separated solvent names.
        operating_temp_c: Optional shared process/heating temperature in Celsius.
        include_pubchem: If True, enrich each profile with PubChem data.
        limit: Maximum number of solvents to compare.

    WHEN TO USE:
    - "Compare safety cards for heptane, cyclohexane, and toluene at 110 C"
    - "Which of THF, toluene, and DMSO has the worst heating risk?"
    """

    names_text = str(solvent_names or "").replace("N,N-", "N§N-")
    solvents = [item.strip().replace("N§N-", "N,N-") for item in names_text.split(",") if item.strip()]
    if not solvents:
        return json_tool_error(
            "No solvent names provided.",
            tool_name="compare_solvent_safety_cards",
            error_code="missing_solvents",
        )
    solvents = solvents[: max(1, min(int(limit or 6), 10))]
    op_temp = _parse_optional_float(operating_temp_c)
    profiles = [
        build_solvent_safety_profile(name, operating_temp_c=op_temp, include_pubchem=include_pubchem)
        for name in solvents
    ]

    display = format_solvent_safety_comparison(profiles, operating_temp_c=op_temp)

    return json_tool_success(
        display,
        tool_name="compare_solvent_safety_cards",
        solvent_names=solvents,
        operating_temp_c=op_temp,
        include_pubchem=include_pubchem,
        profiles=profiles,
    )
