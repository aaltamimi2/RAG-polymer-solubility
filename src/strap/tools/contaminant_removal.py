"""Agent-facing contaminant-removal screening tools."""

from __future__ import annotations

from typing import Any

from strap.services.contaminant_data_service import (
    list_supported_contaminant_families,
    list_supported_contaminants as _list_supported_contaminants,
)
from strap.services.contaminant_screening_service import (
    compare_contaminant_removal_modes as _compare_modes,
    screen_leaching_candidates as _screen_leaching,
    screen_strap_contaminant_removal_candidates as _screen_strap,
)
from strap.services.tool_response_service import json_tool_error, json_tool_success
from strap.tools._helpers import safe_tool_wrapper


def _screening_markdown(title: str, result: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"**Mode:** {result['mode']}")
    lines.append(f"**Target polymer:** {result['target_polymer']}")
    if result.get("other_polymers"):
        lines.append(f"**Other polymers checked:** {', '.join(result['other_polymers'])}")
    lines.append(f"**Contaminants screened:** {', '.join(result['contaminants'])}")
    if result.get("unsupported_contaminants"):
        lines.append(
            "**Unsupported contaminants:** " + ", ".join(result["unsupported_contaminants"])
        )
    lines.append("")
    lines.append("## Recommended solvents")
    if result.get("recommended_solvents"):
        for solvent in result["recommended_solvents"][:10]:
            lines.append(f"- {solvent}")
    else:
        lines.append("- No passing solvents found under the current screening criteria.")

    lines.append("")
    lines.append("## Top candidates")
    lines.append("| Solvent | Pass | Temp (C) | logD min | Target status |")
    lines.append("|---|---:|---:|---:|---|")
    for candidate in result.get("candidate_solvents", [])[:10]:
        temp = candidate.get("operating_temperature_c")
        logd_min = candidate.get("contaminant_logd_min")
        lines.append(
            f"| {candidate['solvent']} | {'Yes' if candidate.get('passes') else 'No'} | "
            f"{temp if temp is not None else '—'} | {f'{logd_min:.2f}' if logd_min is not None else '—'} | "
            f"{candidate.get('target_polymer_status', '—')} |"
        )

    if result.get("caveats"):
        lines.append("")
        lines.append("## Caveats")
        for caveat in result["caveats"]:
            lines.append(f"- {caveat}")
    return "\n".join(lines)


@safe_tool_wrapper(structured_output=True)
def list_supported_contaminants(contaminant_family: str | None = None) -> str:
    """List supported contaminant families and contaminants from the Zhou workbook."""
    contaminants = _list_supported_contaminants(contaminant_family)
    if contaminant_family and not contaminants:
        return json_tool_error(
            f"Unsupported contaminant family: {contaminant_family}",
            tool_name="list_supported_contaminants",
            error_code="unsupported_contaminant_family",
            supported_families=list_supported_contaminant_families(),
        )
    families = list_supported_contaminant_families()
    display = ["# Supported contaminants", "", f"**Families:** {', '.join(families)}"]
    if contaminant_family:
        display.append("")
        display.append(f"## {contaminant_family}")
    else:
        display.append("")
        display.append("## All supported contaminants")
    for name in contaminants:
        display.append(f"- {name}")
    return json_tool_success(
        "\n".join(display),
        tool_name="list_supported_contaminants",
        contaminant_family=contaminant_family,
        supported_families=families,
        contaminants=contaminants,
        n_contaminants=len(contaminants),
    )


@safe_tool_wrapper(structured_output=True)
def screen_contaminant_leaching(
    target_polymer: str,
    contaminants: str,
    other_polymers: str | None = None,
    solvents: str | None = None,
    max_temperature_c: float | None = None,
) -> str:
    """Screen solvents for contaminant removal by leaching while keeping the polymer intact."""
    try:
        result = _screen_leaching(
            target_polymer=target_polymer,
            contaminants=contaminants,
            other_polymers=other_polymers,
            solvents=solvents,
            max_temperature_c=max_temperature_c,
        )
    except ValueError as exc:
        return json_tool_error(
            str(exc),
            tool_name="screen_contaminant_leaching",
            error_code="screening_failed",
            target_polymer=target_polymer,
            contaminants=contaminants,
        )
    return json_tool_success(
        _screening_markdown("Contaminant Leaching Screen", result),
        tool_name="screen_contaminant_leaching",
        **result,
    )


@safe_tool_wrapper(structured_output=True)
def screen_contaminant_strap_removal(
    target_polymer: str,
    contaminants: str,
    other_polymers: str | None = None,
    solvents: str | None = None,
    max_temperature_c: float | None = None,
) -> str:
    """Screen solvents for temperature-swing STRAP contaminant removal."""
    try:
        result = _screen_strap(
            target_polymer=target_polymer,
            contaminants=contaminants,
            other_polymers=other_polymers,
            solvents=solvents,
            max_temperature_c=max_temperature_c,
        )
    except ValueError as exc:
        return json_tool_error(
            str(exc),
            tool_name="screen_contaminant_strap_removal",
            error_code="screening_failed",
            target_polymer=target_polymer,
            contaminants=contaminants,
        )
    return json_tool_success(
        _screening_markdown("STRAP Contaminant-Removal Screen", result),
        tool_name="screen_contaminant_strap_removal",
        **result,
    )


@safe_tool_wrapper(structured_output=True)
def compare_contaminant_removal_modes(
    target_polymer: str,
    contaminants: str,
    other_polymers: str | None = None,
    solvents: str | None = None,
    max_temperature_c: float | None = None,
) -> str:
    """Compare leaching vs temperature-swing STRAP contaminant-removal screening."""
    try:
        result = _compare_modes(
            target_polymer=target_polymer,
            contaminants=contaminants,
            other_polymers=other_polymers,
            solvents=solvents,
            max_temperature_c=max_temperature_c,
        )
    except ValueError as exc:
        return json_tool_error(
            str(exc),
            tool_name="compare_contaminant_removal_modes",
            error_code="screening_failed",
            target_polymer=target_polymer,
            contaminants=contaminants,
        )
    lines = [
        "# Contaminant-removal mode comparison",
        "",
        f"**Recommended mode:** {result['recommended_mode']}",
        "",
        f"**Leaching recommendations:** {', '.join(result['recommended_solvents']['leaching']) or 'None'}",
        f"**STRAP contaminant-removal recommendations:** {', '.join(result['recommended_solvents']['strap_contaminant_removal']) or 'None'}",
    ]
    if result.get("caveats"):
        lines.extend(["", "## Caveats", *[f"- {item}" for item in result["caveats"]]])
    return json_tool_success(
        "\n".join(lines),
        tool_name="compare_contaminant_removal_modes",
        **result,
    )
