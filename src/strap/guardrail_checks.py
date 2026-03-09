"""Pure validation checks used by subagent guardrails."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from .guardrail_utils import (
    extract_completed_tool_names,
    extract_supported_polymers,
    extract_text_content,
    extract_user_temperature_limit_c,
    mentions_temperature,
    parse_structured_result_payload,
    temperature_pattern,
)
from .handoffs import validate_agent_payload
from .solubility import get_boiling_point

_STRUCTURED_RESULT_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(.*?)\s*</STRUCTURED_RESULT>",
    re.DOTALL,
)
_BOILING_POINT_CAVEAT_MARGIN_C = 10.0
_NEAR_BOILING_MARGIN_C = 5.0
_SELECTIVITY_ONLY_TOOLS = {
    "analyze_selective_solubility_enhanced",
    "rank_solvents_for_separation",
    "calculate_selectivity_detailed",
    "optimize_separation_temperature",
    "find_optimal_separation_conditions",
}
_SEPARATION_ROUTE_VALIDATION_TOOLS = {
    "plan_sequential_separation",
    "plan_multiple_separation_schemes",
    "find_optimal_separation_sequence",
    "analyze_integrated_separation",
    "view_alternative_separation_sequence",
    "check_atmospheric_feasibility",
    "check_multi_polymer_atmospheric_feasibility",
}
_SEPARATION_NON_ANALYSIS_TOOLS = {
    "think",
    "write_todos",
    "list_available_polymers",
    "list_available_solvents",
    "get_supported_polymers_and_solvents",
}
_SELECTIVITY_OVERCLAIM_TERMS = (
    "will selectively dissolve",
    "will dissolve",
    "remains a solid",
    "remains solid",
    "remain solid",
    "undissolved solid",
    "undissolved solids",
    "physically separated from",
    "via filtration",
    "practical route exists",
    "wide, safe operating window",
    "safe and controllable",
)
_SELECTIVITY_QUALIFIER_TERMS = (
    "predicted",
    "prediction",
    "best-ranked candidate",
    "best candidate",
    "best available candidate",
    "selectivity-based",
    "based on selectivity",
    "interpolation",
    "model suggests",
    "model-predicted",
    "should be validated experimentally",
    "experimental validation",
    "should be confirmed experimentally",
    "needs experimental confirmation",
)


def get_structured_result_errors(
    message: AIMessage,
    agent_name: str | None = None,
) -> list[str]:
    payload, errors = parse_structured_result_payload(message)
    if errors:
        return errors

    if agent_name:
        errors = validate_agent_payload(agent_name, payload)
        if errors:
            return errors

    return []


def get_separation_feasibility_errors(
    message: AIMessage,
    agent_name: str | None = None,
) -> list[str]:
    if agent_name != "separation-engineer":
        return []

    payload, errors = parse_structured_result_payload(message)
    if errors or payload is None:
        return []

    findings: list[str] = []
    steps = payload.get("steps")
    if isinstance(steps, list):
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            solvent = str(step.get("solvent", "")).strip()
            if not solvent or solvent in {"-", "N/A", "No data", "None found", "Error"}:
                continue
            temp_value = step.get("temperature_c", step.get("temp"))
            try:
                operating_temp = float(temp_value)
            except (TypeError, ValueError):
                continue
            boiling_point = get_boiling_point(solvent)
            if boiling_point is None:
                continue
            if operating_temp >= boiling_point:
                step_no = step.get("step", index)
                findings.append(
                    f"Step {step_no} uses {solvent} at {operating_temp:.1f}C, "
                    f"at or above its boiling point ({boiling_point:.1f}C) at 1 atm"
                )

    text = extract_text_content(message).lower()
    mentions_infeasible = any(
        phrase in text
        for phrase in (
            "infeasible",
            "not feasible",
            "requires pressure",
            "requires pressurization",
            "above boiling point",
            "would need to operate",
        )
    )
    presents_executable_best = bool(payload.get("best_sequence")) and any(
        phrase in text
        for phrase in (
            "optimal separation sequence",
            "optimal sequence",
            "best sequence",
            "most effective separation sequence",
            "recommended sequence",
        )
    )
    if mentions_infeasible and presents_executable_best:
        findings.append(
            "Do not present a route as the best or optimal executable sequence while also "
            "stating that it is infeasible at atmospheric pressure within the user's constraint"
        )

    return findings


def get_separation_analysis_coverage_errors(
    messages: list,
    message: AIMessage,
    agent_name: str | None = None,
) -> list[str]:
    if agent_name != "separation-engineer":
        return []

    payload, errors = parse_structured_result_payload(message)
    if errors or payload is None:
        return []
    if payload.get("no_data") is True:
        return []

    completed_tool_names = [
        name for name in extract_completed_tool_names(messages)
        if name not in _SEPARATION_NON_ANALYSIS_TOOLS
    ]
    if completed_tool_names:
        return []

    route_like_payload = bool(payload.get("best_sequence")) or bool(payload.get("steps")) or bool(payload.get("top_k_sequences"))
    if not route_like_payload:
        return []

    return [
        "You listed support/coverage information but did not run a substantive separation-analysis tool before recommending a route"
    ]


def get_separation_temperature_bound_errors(
    messages: list,
    message: AIMessage,
    agent_name: str | None = None,
) -> list[str]:
    if agent_name != "separation-engineer":
        return []

    user_max_temp_c = extract_user_temperature_limit_c(messages)
    if user_max_temp_c is None:
        return []

    payload, errors = parse_structured_result_payload(message)
    if errors or payload is None:
        return []

    full_text = extract_text_content(message)
    match = _STRUCTURED_RESULT_RE.search(full_text)
    prose_text = full_text[:match.start()] if match else full_text
    prose_lower = prose_text.lower()

    findings: list[str] = []
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return findings

    for step in steps:
        if not isinstance(step, dict):
            continue

        solvent = str(step.get("solvent", "")).strip()
        if not solvent or solvent in {"-", "N/A", "No data", "None found", "Error"}:
            continue

        temp_value = step.get("temperature_c", step.get("temp"))
        try:
            operating_temp_c = float(temp_value)
        except (TypeError, ValueError):
            continue

        boiling_point_c = get_boiling_point(solvent)
        if boiling_point_c is None:
            continue
        if operating_temp_c >= boiling_point_c:
            continue
        if user_max_temp_c < boiling_point_c - _BOILING_POINT_CAVEAT_MARGIN_C:
            continue

        mentions_actual_temp = mentions_temperature(prose_text, operating_temp_c)
        mentions_user_max = mentions_temperature(prose_text, user_max_temp_c)
        mentions_boiling_constraint = any(
            phrase in prose_lower
            for phrase in (
                "boiling point",
                "stay below",
                "stays below",
                "remain below",
                "remains below",
                "below its bp",
                "below the solvent bp",
            )
        )
        mentions_pressure = any(
            phrase in prose_lower
            for phrase in (
                "at 1 atm",
                "at atmospheric pressure",
                "atmospheric pressure",
            )
        )
        implies_max_operation = bool(
            re.search(
                rf"(?:at|operate at|run at|heated to|heating to)\s*{temperature_pattern(user_max_temp_c)}",
                prose_lower,
            )
        )

        if not (mentions_actual_temp and mentions_boiling_constraint and mentions_pressure):
            findings.append(
                f"User allows up to {user_max_temp_c:.1f}C, but {solvent} boils at "
                f"{boiling_point_c:.1f}C. State explicitly that the recommended operating "
                f"temperature is {operating_temp_c:.1f}C and stays below the solvent boiling "
                "point at 1 atm; do not imply operation at the user's maximum temperature"
            )
            continue

        if mentions_user_max and implies_max_operation and user_max_temp_c > operating_temp_c + 0.5:
            findings.append(
                f"Do not describe the {solvent} step as operating at {user_max_temp_c:.1f}C; "
                f"the recommended operating temperature is {operating_temp_c:.1f}C so the solvent "
                "stays below its boiling point at 1 atm"
            )

        bp_margin_c = boiling_point_c - operating_temp_c
        if bp_margin_c <= _NEAR_BOILING_MARGIN_C:
            mentions_narrow_margin = any(
                phrase in prose_lower
                for phrase in (
                    "narrow temperature margin",
                    "narrow operating margin",
                    "close to its boiling point",
                    "close to the boiling point",
                    "near its boiling point",
                    "near the boiling point",
                    "careful temperature control",
                    "tight temperature control",
                )
            )
            if not mentions_narrow_margin:
                findings.append(
                    f"{solvent} is recommended at {operating_temp_c:.1f}C, only "
                    f"{bp_margin_c:.1f}C below its boiling point ({boiling_point_c:.1f}C). "
                    "State explicitly that this is a narrow atmospheric-pressure operating "
                    "margin that requires careful temperature control"
                )

    return findings


def get_separation_support_scope_errors(
    messages: list,
    message: AIMessage,
    agent_name: str | None = None,
) -> list[str]:
    if agent_name != "separation-engineer":
        return []

    payload, errors = parse_structured_result_payload(message)
    if errors or payload is None:
        return []

    payload_polymers = [
        str(polymer).strip().upper()
        for polymer in payload.get("polymers", [])
        if str(polymer).strip()
    ]
    supported_polymers = extract_supported_polymers(messages, payload_polymers)
    if not supported_polymers:
        return []

    unsupported = sorted(
        polymer for polymer in payload_polymers if polymer not in supported_polymers
    )
    if not unsupported:
        return []

    full_text = extract_text_content(message)
    match = _STRUCTURED_RESULT_RE.search(full_text)
    prose_text = full_text[:match.start()] if match else full_text
    prose_lower = prose_text.lower()

    findings: list[str] = []
    mentions_support_scope = any(
        phrase in prose_lower
        for phrase in (
            "supported subset",
            "not supported",
            "unsupported",
            "outside database coverage",
            "not available in the database",
            "not in the database",
        )
    )
    names_mentioned = all(polymer.lower() in prose_lower for polymer in unsupported)
    if not (mentions_support_scope and names_mentioned):
        findings.append(
            f"The response includes unsupported polymers ({', '.join(unsupported)}) "
            "but does not clearly state that conclusions apply only to the supported subset"
        )

    declared_supported = {
        str(polymer).strip().upper()
        for polymer in payload.get("supported_polymers", [])
        if str(polymer).strip()
    }
    declared_unsupported = {
        str(polymer).strip().upper()
        for polymer in payload.get("unsupported_polymers", [])
        if str(polymer).strip()
    }
    expected_supported = {polymer for polymer in payload_polymers if polymer in supported_polymers}
    expected_unsupported = set(unsupported)
    if declared_supported != expected_supported or declared_unsupported != expected_unsupported:
        findings.append(
            "When unsupported polymers are present, the <STRUCTURED_RESULT> must include "
            "`supported_polymers` and `unsupported_polymers` arrays that match the database coverage"
        )

    purity_terms = (
        "purified",
        "pure",
        "clean separation",
        "cleanly isolated",
        "purified pet",
        "isolated pet",
    )
    purity_caveat = any(
        phrase in prose_lower
        for phrase in (
            "cannot determine purity",
            "cannot conclude purity",
            "purity unknown",
            "may remain in the residue",
            "could remain in the residue",
            "residue may still contain",
            "residue could still contain",
            "could contaminate the residue",
        )
    )
    if any(term in prose_lower for term in purity_terms) and not purity_caveat:
        findings.append(
            f"Do not describe a residue or product as purified while unsupported polymers "
            f"({', '.join(unsupported)}) could still be present"
        )

    phase_terms = (
        "remain solid",
        "remains solid",
        "remain as solids",
        "remains as a solid",
        "dissolve",
        "soluble",
        "insoluble",
        "precipitate",
        "stays in the residue",
        "remains in the residue",
    )
    uncertainty_terms = (
        "unknown",
        "cannot determine",
        "cannot conclude",
        "not supported",
        "unsupported",
        "outside database coverage",
    )
    for polymer in unsupported:
        for sentence in re.split(r"[.\n]+", prose_lower):
            if polymer.lower() not in sentence:
                continue
            if any(term in sentence for term in phase_terms) and not any(
                term in sentence for term in uncertainty_terms
            ):
                findings.append(
                    f"Do not assert phase behavior for unsupported polymer {polymer}; "
                    "state that its behavior is unknown without additional data"
                )
                break

    return findings


def _extract_recent_tool_names(messages: list) -> set[str]:
    names: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", None) or []:
                name = tool_call.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
        else:
            name = getattr(message, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
    return names


def get_selectivity_overclaim_errors(
    messages: list,
    prose_text: str,
    agent_name: str | None = None,
) -> list[str]:
    if agent_name != "separation-engineer":
        return []

    tool_names = _extract_recent_tool_names(messages)
    if not tool_names.intersection(_SELECTIVITY_ONLY_TOOLS):
        return []
    if tool_names.intersection(_SEPARATION_ROUTE_VALIDATION_TOOLS):
        return []

    prose_lower = prose_text.lower()
    if not any(term in prose_lower for term in _SELECTIVITY_OVERCLAIM_TERMS):
        return []
    if any(term in prose_lower for term in _SELECTIVITY_QUALIFIER_TERMS):
        return []

    return [
        "This answer is based on selectivity-ranking analysis only. Present the solvent as the "
        "best predicted/selectivity-based candidate within the current model/data, do not state "
        "with certainty that the comparison polymer remains solid or that the route is fully "
        "practical/validated, and explicitly recommend experimental confirmation or a fuller "
        "feasibility check."
    ]


def get_separation_selectivity_scope_errors(
    messages: list,
    message: AIMessage,
    agent_name: str | None = None,
) -> list[str]:
    if agent_name != "separation-engineer":
        return []

    full_text = extract_text_content(message)
    match = _STRUCTURED_RESULT_RE.search(full_text)
    prose_text = full_text[:match.start()] if match else full_text
    return get_selectivity_overclaim_errors(messages, prose_text, agent_name)
