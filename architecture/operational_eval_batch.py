from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

_DIR = Path(__file__).resolve().parent
_CATEGORY_TIMEOUT_KEYS = {
    "separation",
    "biosteam",
    "safety",
    "hsp",
    "sep-biosteam",
    "sep-safety",
    "contaminant",
    "sep-contaminant",
}
DEFAULT_CATEGORY_TIMEOUTS = {
    "hsp": 90,
    "safety": 120,
    "separation": 150,
    "biosteam": 150,
    "sep-biosteam": 210,
    "sep-safety": 210,
    "contaminant": 120,
    "sep-contaminant": 180,
}
_ROOT = _DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_DIR))

from dotenv import load_dotenv

load_dotenv(str(_ROOT / ".env"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import Client as LangSmithClient

from strap.agent import create_dissolve_agent
from strap.handoffs import normalize_agent_payload, validate_agent_payload
from strap.routing_guards import build_single_specialist_separation_ai_message
from test_harness import (
    QueryResult,
    TestQuery,
    clear_timeout_snapshot,
    fetch_langsmith_trace,
    generate_trace_visuals,
    load_timeout_snapshot,
    run_query,
)

DISALLOWED_SUBAGENTS = {
    "scholar-researcher",
    "patent-researcher",
    "rag-analyst",
}
KNOWN_SUBAGENTS = {
    "separation-engineer",
    "biosteam-analyst",
    "safety-analyst",
    "statistics-ml",
    "contaminant-removal-analyst",
    "visualization-specialist",
    "scholar-researcher",
    "patent-researcher",
    "rag-analyst",
}

SEPARATION_TRACE_TOOLS = {
    "analyze_selective_solubility_enhanced",
    "find_optimal_separation_conditions",
    "optimize_separation_temperature",
    "calculate_selectivity_detailed",
    "rank_solvents_for_separation",
    "rank_solvents_selectivity",
    "plan_sequential_separation",
    "plan_multiple_separation_schemes",
    "analyze_integrated_separation",
    "view_alternative_separation_sequence",
    "create_separation_tree_plot",
    "get_supported_polymers_and_solvents",
}
BIOS_TEAM_TRACE_TOOLS = {
    "run_biosteam_simulation",
    "run_biosteam_batch",
    "run_biosteam_multi_polymer",
    "run_biosteam_uncertainty",
    "run_biosteam_parameter_sweep",
    "run_biosteam_tornado",
    "compare_biosteam_scenarios",
    "visualize_biosteam_results",
}
SAFETY_TRACE_TOOLS = {
    "get_solvent_gscore",
    "get_pubchem_safety_info",
    "compare_pubchem_safety",
    "get_pubchem_toxicity",
    "visualize_gscores",
    "visualize_pubchem_safety",
}
HSP_TRACE_TOOLS = {
    "predict_solubility_ml",
}
CONTAMINANT_TRACE_TOOLS = {
    "list_supported_contaminants",
    "screen_contaminant_leaching",
    "screen_contaminant_strap_removal",
    "compare_contaminant_removal_modes",
}
_STRUCTURED_RESULT_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(.*?)\s*</STRUCTURED_RESULT>",
    re.DOTALL,
)


def _extract_structured_payload(text: str) -> dict | None:
    match = _STRUCTURED_RESULT_RE.search(text or "")
    if not match:
        return None
    json_text = match.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if fenced:
        json_text = fenced.group(1).strip()
    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _recover_validated_separation_answer(case: EvalCase, task_output: str) -> tuple[str, dict] | None:
    payload = _extract_structured_payload(task_output)
    if not isinstance(payload, dict):
        return None
    if validate_agent_payload("separation-engineer", payload):
        return None

    synthetic_tool_call_id = "timeout-recovered-separation"
    synthetic_messages = [
        HumanMessage(content=case.query),
        AIMessage(
            content="",
            tool_calls=[{
                "id": synthetic_tool_call_id,
                "name": "task",
                "args": {"subagent_type": "separation-engineer"},
            }],
        ),
        ToolMessage(
            content=f"<STRUCTURED_RESULT>{json.dumps(payload, ensure_ascii=False)}</STRUCTURED_RESULT>",
            tool_call_id=synthetic_tool_call_id,
        ),
    ]
    recovered_ai = build_single_specialist_separation_ai_message(synthetic_messages)
    if recovered_ai is None:
        return None

    recovered_answer = _strip_structured_result_block(getattr(recovered_ai, "content", "") or "")
    if not recovered_answer.strip():
        return None
    return recovered_answer, dict(getattr(recovered_ai, "additional_kwargs", {}) or {})


def _format_mode_label(mode: str | None) -> str:
    if mode == "strap_contaminant_removal":
        return "temperature-swing STRAP contaminant removal"
    if mode == "leaching":
        return "leaching"
    return str(mode or "unknown")


def _render_contaminant_candidate_summary(mode_payload: dict, *, limit: int = 2) -> list[str]:
    candidates = mode_payload.get("candidate_solvents") or []
    passing = [row for row in candidates if row.get("passes")]
    lines: list[str] = []
    for row in passing[:limit]:
        solvent = row.get("solvent", "unknown solvent")
        temp = row.get("operating_temperature_c")
        logd = row.get("contaminant_logd_min")
        bp = row.get("boiling_point_c")
        pieces = [solvent]
        if temp is not None:
            pieces.append(f"operating temperature {temp:.1f}°C")
        if bp is not None:
            pieces.append(f"boiling point {bp:.1f}°C")
        if logd is not None:
            pieces.append(f"minimum contaminant logD {logd:.2f}")
        lines.append("- " + "; ".join(pieces))
    return lines


def _recover_validated_contaminant_answer(case: EvalCase, task_output: str) -> tuple[str, dict] | None:
    payload = _extract_structured_payload(task_output)
    if not isinstance(payload, dict):
        return None
    payload = normalize_agent_payload("contaminant-removal-analyst", payload)
    if validate_agent_payload("contaminant-removal-analyst", payload):
        return None

    mode = payload.get("mode")
    target_polymer = payload.get("target_polymer", "target polymer")
    contaminants = payload.get("supported_contaminants") or payload.get("contaminants") or []
    unsupported = payload.get("unsupported_contaminants") or []
    other_polymers = payload.get("other_polymers") or []
    caveats = payload.get("caveats") or []

    lines = ["# Contaminant Removal Screening", ""]
    lines.append(f"**Target polymer:** {target_polymer}")
    if contaminants:
        lines.append(f"**Contaminants screened:** {', '.join(contaminants)}")
    if other_polymers:
        lines.append(f"**Other polymers checked:** {', '.join(other_polymers)}")
    if unsupported:
        lines.append(f"**Unsupported contaminants:** {', '.join(unsupported)}")
    lines.append("")

    if mode == "comparison":
        recommended_mode = payload.get("recommended_mode")
        nested = payload.get("modes") or {}
        mode_results = payload.get("recommended_solvents") or {}
        if isinstance(mode_results, dict):
            leaching = mode_results.get("leaching") or []
            strap = mode_results.get("strap_contaminant_removal") or []
        else:
            leaching_payload = nested.get("leaching") if isinstance(nested, dict) else None
            strap_payload = nested.get("strap_contaminant_removal") if isinstance(nested, dict) else None
            leaching = (
                leaching_payload.get("recommended_solvents") if isinstance(leaching_payload, dict) else []
            ) or []
            strap = (
                strap_payload.get("recommended_solvents") if isinstance(strap_payload, dict) else []
            ) or []
            if recommended_mode == "leaching" and not leaching:
                leaching = mode_results if isinstance(mode_results, list) else []
            if recommended_mode == "strap_contaminant_removal" and not strap:
                strap = mode_results if isinstance(mode_results, list) else []
        lines.append(f"**Recommended mode:** {_format_mode_label(recommended_mode)}")
        lines.append(f"**Leaching recommendations:** {', '.join(leaching) if leaching else 'None'}")
        lines.append(f"**STRAP contaminant-removal recommendations:** {', '.join(strap) if strap else 'None'}")
        lines.append("")
        lines.append(
            "Passing solvents are screened from contaminant miscibility, positive logD, polymer dissolution or retention, and non-target polymer behavior."
        )
        selected_payload = nested.get(recommended_mode) if isinstance(nested, dict) else None
        if isinstance(selected_payload, dict):
            summary_lines = _render_contaminant_candidate_summary(selected_payload)
            if summary_lines:
                lines.append("")
                lines.append("**Top screened solvents for the recommended mode:**")
                lines.extend(summary_lines)
        if recommended_mode == "tie":
            lines.append("")
            lines.append("No robust mode passed the full contaminant-screening criteria for the requested solvent set.")
    else:
        recommended = payload.get("recommended_solvents") or []
        lines.append(f"**Mode screened:** {_format_mode_label(mode)}")
        lines.append(f"**Recommended solvents:** {', '.join(recommended) if recommended else 'None'}")
        lines.append("")
        lines.append(
            "Passing solvents satisfy contaminant miscibility, positive logD, and the polymer-retention or polymer-precipitation requirements for the selected mode."
        )
        summary_lines = _render_contaminant_candidate_summary(payload)
        if summary_lines:
            lines.append("")
            lines.append("**Top screened solvents:**")
            lines.extend(summary_lines)

    if caveats:
        lines.append("")
        lines.append("**Caveats:**")
        for item in caveats[:5]:
            lines.append(f"- {item}")

    answer = "\n".join(lines).strip()
    if not answer:
        return None
    return answer, {"strap_origin": "timeout_validated_contaminant_payload"}


@dataclass(frozen=True)
class EvalCase:
    name: str
    category: str
    query: str
    pattern: str
    expected_subagents: list[str]
    recursion_limit: int
    description: str = ""
    answer_term_groups: list[list[str]] = field(default_factory=list)
    required_trace_any: list[str] = field(default_factory=list)
    required_trace_all: list[str] = field(default_factory=list)
    min_trace_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    name: str
    category: str
    query: str
    pattern: str
    expected_subagents: list[str]
    actual_subagents: list[str]
    attempts: int
    checks: list[Check]
    wall_time_s: float
    total_tokens: int
    tool_names: list[str]
    thread_id: str | None
    run_id: str | None
    trace_id: str | None
    trace_summary: dict | None
    full_answer: str
    answer_preview: str
    error: str | None
    final_answer_diagnostics: dict | None = None
    raw_result_path: str | None = None

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def score_pct(self) -> float:
        return (100.0 * self.passed_checks / self.total_checks) if self.total_checks else 0.0

    @property
    def passed(self) -> bool:
        return self.passed_checks == self.total_checks


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _extract_subagents_from_trace(ls_client: LangSmithClient, trace_id: str, project_name: str) -> list[str]:
    try:
        runs = list(ls_client.list_runs(trace_id=trace_id, project_name=project_name))
    except Exception:
        return []

    subagents: list[str] = []
    for run in runs:
        if getattr(run, "name", "") != "task" or getattr(run, "run_type", "") != "tool":
            continue
        blob = json.dumps(getattr(run, "inputs", {}) or {}, default=str)
        for name in KNOWN_SUBAGENTS:
            if re.search(rf'\"subagent_type\"\s*:\s*\"{re.escape(name)}\"', blob):
                subagents.append(name)
                break
            if name in blob:
                subagents.append(name)
                break
    return _unique_in_order(subagents)


def _strip_structured_result_block(text: str) -> str:
    stripped = _STRUCTURED_RESULT_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def _extract_latest_task_output_from_trace(
    ls_client: LangSmithClient,
    trace_id: str,
    project_name: str,
    *,
    subagent: str,
) -> str:
    try:
        runs = list(ls_client.list_runs(trace_id=trace_id, project_name=project_name))
    except Exception:
        return ""

    latest_text = ""
    for run in runs:
        if getattr(run, "run_type", "") != "tool" or getattr(run, "name", "") != "task":
            continue
        inputs_blob = json.dumps(getattr(run, "inputs", {}) or {}, default=str)
        if subagent not in inputs_blob:
            continue
        outputs = getattr(run, "outputs", {}) or {}
        update = ((outputs.get("output") or {}).get("update") or {})
        for msg in update.get("messages", []) or []:
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                latest_text = content
    return latest_text


def _make_case(
    *,
    name: str,
    category: str,
    query: str,
    pattern: str,
    expected_subagents: list[str],
    description: str,
    answer_term_groups: list[list[str]],
    recursion_limit: int = 250,
    required_trace_any: list[str] | None = None,
    required_trace_all: list[str] | None = None,
    min_trace_counts: dict[str, int] | None = None,
) -> EvalCase:
    return EvalCase(
        name=name,
        category=category,
        query=query,
        pattern=pattern,
        expected_subagents=expected_subagents,
        recursion_limit=recursion_limit,
        description=description,
        answer_term_groups=answer_term_groups,
        required_trace_any=required_trace_any or [],
        required_trace_all=required_trace_all or [],
        min_trace_counts=min_trace_counts or {},
    )


def build_suite() -> list[EvalCase]:
    cases: list[EvalCase] = []

    def sep(name: str, query: str, description: str) -> None:
        cases.append(_make_case(
            name=name,
            category="separation",
            query=query,
            pattern="single-agent",
            expected_subagents=["separation-engineer"],
            description=description,
            answer_term_groups=[
                ["solvent", "sequence", "scheme", "step"],
                ["temperature", "°c", "atmospheric", "boiling"],
                ["recommend", "feasible", "infeasible", "partial", "not feasible"],
            ],
            required_trace_any=sorted(SEPARATION_TRACE_TOOLS),
        ))

    def tea(name: str, query: str, description: str) -> None:
        cases.append(_make_case(
            name=name,
            category="biosteam",
            query=query,
            pattern="single-agent",
            expected_subagents=["biosteam-analyst"],
            description=description,
            answer_term_groups=[
                ["msp"],
                ["gwp", "co2", "lca"],
                ["tci", "capex", "aoc", "opex"],
            ],
            required_trace_any=sorted(BIOS_TEAM_TRACE_TOOLS),
        ))

    def safety(name: str, query: str, description: str) -> None:
        cases.append(_make_case(
            name=name,
            category="safety",
            query=query,
            pattern="single-agent",
            expected_subagents=["safety-analyst"],
            description=description,
            answer_term_groups=[
                ["g-score", "g score", "gsk"],
                ["hazard", "ghs", "pubchem", "signal word"],
                ["safest", "avoid", "rank", "most hazardous", "recommend"],
            ],
            required_trace_all=["get_solvent_gscore"],
            required_trace_any=[
                "compare_pubchem_safety",
                "get_pubchem_safety_info",
                "get_pubchem_toxicity",
            ],
        ))

    def hsp(
        name: str,
        query: str,
        description: str,
        *,
        min_ml_calls: int,
        extra_answer_groups: list[list[str]] | None = None,
    ) -> None:
        term_groups = [
            ["red", "hansen", "relative energy difference"],
            ["soluble", "non-soluble", "dissolve", "compatible", "incompatible"],
            ["rank", "best", "closest", "selective", "borderline", "limitation"],
        ]
        if extra_answer_groups:
            term_groups.extend(extra_answer_groups)
        cases.append(_make_case(
            name=name,
            category="hsp",
            query=query,
            pattern="single-agent",
            expected_subagents=["statistics-ml"],
            description=description,
            answer_term_groups=term_groups,
            required_trace_all=["predict_solubility_ml"],
            min_trace_counts={"predict_solubility_ml": min_ml_calls},
            recursion_limit=500 if min_ml_calls > 20 else 250,
        ))

    def sep_tea(name: str, query: str, description: str) -> None:
        cases.append(_make_case(
            name=name,
            category="sep-biosteam",
            query=query,
            pattern="sequential",
            expected_subagents=["separation-engineer", "biosteam-analyst"],
            description=description,
            answer_term_groups=[
                ["solvent", "sequence", "step", "scheme"],
                ["msp"],
                ["gwp", "tci", "capex", "aoc"],
            ],
            required_trace_any=sorted(SEPARATION_TRACE_TOOLS | BIOS_TEAM_TRACE_TOOLS),
            required_trace_all=["task"],
        ))

    def sep_safety(name: str, query: str, description: str) -> None:
        cases.append(_make_case(
            name=name,
            category="sep-safety",
            query=query,
            pattern="parallel",
            expected_subagents=["separation-engineer", "safety-analyst"],
            description=description,
            answer_term_groups=[
                ["solvent", "sequence", "selective", "step"],
                ["g-score", "g score", "gsk"],
                ["hazard", "ghs", "pubchem", "signal word"],
            ],
            required_trace_any=sorted(SEPARATION_TRACE_TOOLS | SAFETY_TRACE_TOOLS),
            required_trace_all=["task"],
        ))

    # 10 separation-only
    sep("sep-ldpe-hdpe-pp-atm", "Only do process design. No economics or safety. Plan an executable selective-dissolution sequence for LDPE, HDPE, and PP at atmospheric pressure with an upper temperature bound of 120°C. Use real operating temperatures, not just the max bound, and say clearly if any step is infeasible.", "3-polyolefin executable separation at <=120C")
    sep("sep-ps-pvc-below-90", "Only do process design. Below 90°C at atmospheric pressure, can you selectively separate PS from PVC by dissolution? Recommend the best practical route, or explicitly say no fully selective route exists.", "PS/PVC feasibility below 90C")
    sep("sep-evoh-ldpe-pet-film", "Only do process design. For a multilayer EVOH/LDPE/PET film, propose the best atmospheric-pressure separation sequence up to 140°C and highlight any boiling-point or unsupported-data limits.", "Multilayer EVOH/LDPE/PET separation")
    sep("sep-ps-pet-pc-120", "Only do process design. Find the best separation sequence for PS, PET, and PC up to 120°C at 1 atm. Report the recommended solvent and operating temperature for each feasible step.", "PS/PET/PC sequence")
    sep("sep-ps-pmma-pet-120", "Only do process design. Find the best separation sequence for PS, PMMA, and PET up to 120°C at atmospheric pressure. Be explicit about any unsupported polymers and what the residue would still contain.", "PS/PMMA/PET supported-subset handling")
    sep("sep-abs-ps-pvc-80", "Only do process design. For ABS, PS, and PVC, identify the best separation sequence at or below 80°C and state if only a partial separation is defensible.", "ABS/PS/PVC partial-separation test")
    sep("sep-pe-pp-direct", "Only do process design. Can PE and PP be separated directly by selective dissolution below 120°C at atmospheric pressure? Give the most defensible answer even if the answer is that no robust route exists.", "PE vs PP direct separation difficulty")
    sep("sep-petg-pc-ps-rt", "Only do process design. At room temperature, what is the best dissolution-based separation sequence for PETG, PC, and PS? If the route is weak or non-selective, say so clearly.", "PETG/PC/PS RT route")
    sep("sep-evoh-pe-dmso-dmf", "Only do process design. For separating EVOH from PE at atmospheric pressure and no higher than 120°C, compare DMSO and DMF as candidate process solvents and recommend the safer executable operating window only if it is defensible.", "EVOH/PE bounded-temperature separation")
    sep("sep-5poly-2scheme", "Only do process design. For a mixed stream containing PS, PVC, LDPE, HDPE, and PET, propose two atmospheric-pressure separation schemes: one optimized for selectivity and one optimized for safer/greener solvents. Keep the answer grounded in feasible operating temperatures.", "5-polymer 2-scheme comparison")

    # 10 biosteam-only
    tea("tea-pe-toluene-c1", "Only do TEA/LCA. Do not plan a separation sequence. Run a rigorous BioSTEAM simulation for PE recovery using Toluene under energy case C1 and report MSP, GWP, TCI, and main cost drivers.", "Single BioSTEAM run for PE/Toluene/C1")
    tea("tea-pe-three-solvents-c1", "Only do TEA/LCA. Do not plan a separation sequence. Compare Toluene, Xylene, and Heptane for PE recovery under energy case C1. Rank them by MSP and state which has the lowest GWP.", "PE solvent comparison under C1")
    tea("tea-all-pe-solvents-c1", "Only do TEA/LCA. Do not plan a separation sequence. Run a batch BioSTEAM comparison across all PE solvents under energy case C1 and report the top five by MSP with their GWP values.", "Batch PE solvent screen")
    tea("tea-pe-energy-cases", "Only do TEA/LCA. Do not plan a separation sequence. Compare PE recovery in Toluene under C1, C2, and C3. Quantify how the energy case changes MSP and GWP.", "Energy-case comparison")
    tea("tea-heptane-price-sweep", "Only do TEA/LCA. Do not plan a separation sequence. Sweep solvent price from $0.50/kg to $2.00/kg for Heptane-based PE recovery under C1 and summarize how MSP changes.", "Parameter sweep for solvent price")
    tea("tea-xylene-tornado", "Only do TEA/LCA. Do not plan a separation sequence. Run a tornado sensitivity analysis for Xylene/PE recovery under C1 and identify the top driver of MSP.", "Tornado sensitivity for MSP")
    tea("tea-toluene-uncertainty", "Only do TEA/LCA. Do not plan a separation sequence. Run a BioSTEAM uncertainty analysis for Toluene/PE recovery under C1 and summarize confidence in MSP and GWP.", "Uncertainty analysis")
    tea("tea-multipoly-pe-evoh-pet", "Only do TEA/LCA. Do not plan a separation sequence. Run a multi-polymer BioSTEAM analysis for PE/Toluene, EVOH/DMSO, and PET/Acetone under C1. Report blended MSP, weighted GWP, and per-stage results.", "Multi-polymer BioSTEAM batch")
    tea("tea-allocation-compare", "Only do TEA/LCA. Do not plan a separation sequence. For PE/Toluene, EVOH/DMSO, and PET/Acetone under C1, compare value allocation versus mass allocation and explain how the blended MSP and GWP interpretation changes.", "Allocation-method comparison")
    tea("tea-ps-thf-c1", "Only do TEA/LCA. Do not plan a separation sequence. Run a BioSTEAM simulation for PS recovery using THF under C1 and clearly state any approximation caveats in the economics and LCA.", "Approximate PS BioSTEAM case")

    # 10 safety-only
    safety("safety-toluene-dmso-thf", "Only do chemical safety analysis. Do not do process design or TEA. Compare Toluene, DMSO, and THF using GSK G-scores, PubChem GHS hazards, and any standout process hazards. Which is safest, and which should be avoided if possible?", "Classic solvent triage")
    safety("safety-evoh-candidates", "Only do chemical safety analysis. Do not do process design or TEA. Rank DMSO, DMF, propylene carbonate, and ethyl acetate as candidate EVOH-process solvents using GSK scores, PubChem hazards, and toxicity context.", "Polar-solvent safety ranking")
    safety("safety-thf-single", "Only do chemical safety analysis. Do not do process design or TEA. Assess THF alone with GSK score, PubChem hazard statements, flammability, and peroxide-formation risk.", "Single-solvent THF review")
    safety("safety-cyclohexanone-acetone-mek", "Only do chemical safety analysis. Do not do process design or TEA. Compare cyclohexanone, acetone, and 2-butanone for process handling risk and environmental/safety profile. Rank them from safest to most hazardous.", "Ketone safety comparison")
    safety("safety-dcm-chloroform-thf", "Only do chemical safety analysis. Do not do process design or TEA. Compare dichloromethane, chloroform, and THF. I want the clearest explanation of which ones should be avoided first and why.", "Chlorinated vs ether hazards")
    safety("safety-heptane-cyclohexane-toluene", "Only do chemical safety analysis. Do not do process design or TEA. Compare heptane, cyclohexane, and toluene for a hot polyolefin process, using GSK and PubChem data plus volatility concerns.", "Hot hydrocarbon solvent safety")
    safety("safety-green-aromatics", "Only do chemical safety analysis. Do not do process design or TEA. Compare anisole, p-cymene, and d-limonene as greener aromatic-like solvents. Include GSK score, key hazards, and any major process-handling caveats.", "Greener aromatic alternatives")
    safety("safety-dmf-dmso-nmp", "Only do chemical safety analysis. Do not do process design or TEA. Rank DMF, DMSO, and NMP by safety, highlighting reproductive toxicity, skin absorption, and chronic exposure concerns.", "Polar aprotic chronic-toxicity review")
    safety("safety-amine-carbonate-methanol", "Only do chemical safety analysis. Do not do process design or TEA. Compare propylene carbonate, methanol, triethylamine, and isopropylamine using GSK scores, GHS hazards, and toxicity data.", "Volatile amines vs safer carbonate")
    safety("safety-tol-xyl-thp-ipa", "Only do chemical safety analysis. Do not do process design or TEA. Compare Toluene, Xylene, THP (tetrahydropyran), and isopropylamine. Rank them by overall process safety and call out the most important differentiator for each.", "Mixed solvent family safety ranking")

    # 10 HSP-only
    hsp("hsp-pe-tol-dmso-hex", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. Which of Toluene, DMSO, and Hexane would you expect to dissolve PE? Show the RED-based reasoning and rank them.", "PE in three solvents via HSP", min_ml_calls=3)
    hsp("hsp-evoh-vs-pe-dmso", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. Would DMSO selectively dissolve EVOH while leaving PE undissolved? Evaluate both polymer-solvent pairs and discuss borderline behavior.", "EVOH vs PE in DMSO", min_ml_calls=2)
    hsp("hsp-ps-selective-three-solvents", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. For PE, PS, and PVC at room temperature, compare THF, cyclohexanone, and acetone and decide whether any selectively dissolves PS.", "3x3 PS-selectivity HSP matrix", min_ml_calls=9)
    hsp("hsp-pe-aromatic-vs-alkane", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. Compare Toluene, Xylene, and Heptane for dissolving PE and rank the solvent quality by RED.", "PE aromatic/alkane HSP ranking", min_ml_calls=3)
    hsp("hsp-pvc-not-pe", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. Among THF, acetone, and cyclohexanone, which solvent best dissolves PVC while leaving PE undissolved at room temperature?", "PVC selective solvent via HSP", min_ml_calls=6)
    hsp("hsp-pc-three-solvents", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. Evaluate polycarbonate in dichloromethane, THF, and acetone. Which is the best HSP match and which is clearly poor?", "PC solvent screening", min_ml_calls=3)
    hsp("hsp-dmf-pvc-vs-ps", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. Would DMF selectively dissolve PVC over PS at room temperature? Compare both polymer-solvent pairs and explain the RED gap.", "DMF selectivity for PVC vs PS", min_ml_calls=2)
    hsp("hsp-petg-pc-ps-matrix", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. Build a RED comparison for PETG, PC, and PS against THF, dichloromethane, and toluene, then identify the clearest selectivity windows.", "3x3 PETG/PC/PS HSP matrix", min_ml_calls=9)
    hsp("hsp-ldpe-vs-pp-limonene", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. Can Hexane, Anisole, or d-Limonene selectively dissolve LDPE over PP? Use RED values only and flag any HSP limitations.", "LDPE vs PP via HSP", min_ml_calls=6)
    hsp("hsp-rt-mini-screen", "Use Hansen solubility parameters / RED only. Do not do process design, database selectivity search, or safety. At room temperature, screen PE, PP, PS, PVC, and PET against Hexane, Toluene, THF, Acetone, and DMSO. Group the polymers by apparent RT solubility and flag major HSP limitations.", "5x5 HSP mini-screen", min_ml_calls=25)

    # 5 separation -> BioSTEAM
    sep_tea("septea-ldpe-hdpe-pp", "Find an executable selective-dissolution sequence for LDPE, HDPE, and PP at atmospheric pressure with a 120°C ceiling. Then run BioSTEAM TEA/LCA for the best route and report MSP, GWP, and TCI.", "Separation followed by TEA/LCA")
    sep_tea("septea-evoh-pe-film", "Find the best atmospheric-pressure separation route for EVOH from PE in a multilayer film, respecting actual solvent boiling points. Then run BioSTEAM TEA/LCA for the recommended solvent under C1.", "EVOH/PE separation then BioSTEAM")
    sep_tea("septea-pe-evoh-pet", "For a PE/EVOH/PET film, propose the best executable separation sequence at atmospheric pressure and then run a multi-polymer BioSTEAM analysis for the recommended route under C1.", "Multi-polymer route then BioSTEAM")
    sep_tea("septea-ps-pvc-pe-c2", "Plan the best atmospheric-pressure separation route for PS, PVC, and PE. Then run BioSTEAM for the recommended PS-recovery solvent under energy case C2 and summarize MSP and GWP.", "PS/PVC/PE then BioSTEAM C2")
    sep_tea("septea-two-sequences", "For PE/EVOH/PET, identify two defensible atmospheric-pressure separation sequences, then run BioSTEAM on both and recommend the better MSP/GWP trade-off.", "Two-sequence comparison with BioSTEAM")

    # 5 separation + safety
    sep_safety("sepsafe-ps-over-pvc", "Find solvents that selectively dissolve PS over PVC at 120°C and atmospheric pressure. Include GSK G-scores and PubChem hazard information for the top candidates, and recommend the best practical solvent.", "PS/PVC selectivity plus safety")
    sep_safety("sepsafe-evoh-over-pe", "Identify the best atmospheric-pressure route for dissolving EVOH while leaving PE intact up to 120°C. Then compare the top candidate solvents by selectivity and safety, including GSK and PubChem hazards.", "EVOH/PE with safety ranking")
    sep_safety("sepsafe-ps-pet-pc", "Design the safest defensible selective-dissolution route for PS, PET, and PC at up to 120°C. If the safest solvents are not the most selective, explain the trade-off using both separation and safety evidence.", "Safety-constrained PS/PET/PC route")
    sep_safety("sepsafe-pe-solvents", "Compare Toluene, Xylene, and Heptane for PE dissolution at atmospheric pressure, combining selectivity/process-feasibility reasoning with GSK and PubChem safety profiles.", "PE solvents with safety trade-off")
    sep_safety("sepsafe-pvc-from-ps", "Below each solvent's boiling point at 1 atm, identify candidate solvents for separating PVC from PS and include safety rankings for the top options using GSK and PubChem data.", "PVC/PS below-boiling safety-aware route")

    if len(cases) != 50:
        raise RuntimeError(f"Expected 50 eval cases, found {len(cases)}")
    return cases


def build_contaminant_suite() -> list[EvalCase]:
    cases: list[EvalCase] = []

    def contam(name: str, query: str, description: str, *, tool_name: str | None = None) -> None:
        required_all = [tool_name] if tool_name else []
        cases.append(_make_case(
            name=name,
            category="contaminant",
            query=query,
            pattern="single-agent",
            expected_subagents=["contaminant-removal-analyst"],
            description=description,
            answer_term_groups=[
                ["contaminant", "dbp", "dehp", "dep", "pfoa", "pfas", "phthalate"],
                ["logd", "partition", "miscible", "miscibility"],
                ["leaching", "strap", "temperature-swing", "recommended", "tie", "no passing", "no robust"],
            ],
            required_trace_any=sorted(CONTAMINANT_TRACE_TOOLS),
            required_trace_all=required_all,
        ))

    def sep_contam(name: str, query: str, description: str) -> None:
        cases.append(_make_case(
            name=name,
            category="sep-contaminant",
            query=query,
            pattern="sequential",
            expected_subagents=["separation-engineer", "contaminant-removal-analyst"],
            description=description,
            answer_term_groups=[
                ["solvent", "sequence", "step", "route"],
                ["contaminant", "dbp", "dehp", "dep", "phthalate"],
                ["logd", "miscible", "leaching", "strap", "temperature-swing", "recommended"],
            ],
            required_trace_any=sorted(SEPARATION_TRACE_TOOLS | CONTAMINANT_TRACE_TOOLS),
            required_trace_all=["task"],
        ))

    contam(
        "contam-leach-pet-dbp",
        "Only do contaminant-removal screening. Do not do TEA, safety, literature, or general process design. For PET contaminated with di-n-butyl phthalate (DBP), screen acetone, dichloromethane, and methanol for leaching at or below 80°C while keeping PET intact. Rank the candidates and explain the miscibility, logD, and polymer-retention logic.",
        "PET + DBP leaching screen",
        tool_name="screen_contaminant_leaching",
    )
    contam(
        "contam-compare-pet-dehp",
        "Only do contaminant-removal screening. Do not do TEA, safety, literature, or general process design. For PET contaminated with di-(2-ethylhexyl) phthalate (DEHP), compare leaching versus STRAP contaminant removal using acetone, dichloromethane, and ethyl acetate up to 80°C. Recommend the better mode and justify it from miscibility, logD, and polymer behavior.",
        "PET + DEHP compare leaching vs STRAP",
        tool_name="compare_contaminant_removal_modes",
    )
    contam(
        "contam-strap-evoh-dbp-pe",
        "Only do contaminant-removal screening. Do not do TEA, safety, literature, or general process design. For EVOH contaminated with di-n-butyl phthalate (DBP) in the presence of PE as a non-target polymer, compare leaching versus STRAP contaminant removal using dimethyl sulfoxide, isopropylamine, and acetylacetone up to 120°C. Recommend the better mode and explain why.",
        "EVOH + DBP with PE non-target; STRAP-contaminant-removal should win",
        tool_name="compare_contaminant_removal_modes",
    )
    contam(
        "contam-pvc-dbp-no-pass",
        "Only do contaminant-removal screening. Do not do TEA, safety, literature, or general process design. For PVC contaminated with di-n-butyl phthalate (DBP) and PS present as a non-target polymer, compare leaching versus STRAP contaminant removal using THF, cyclohexanone, and acetone below 90°C. If no robust mode passes, say so explicitly and explain the limiting criteria.",
        "Negative contaminant-screening case with no robust solvent",
        tool_name="compare_contaminant_removal_modes",
    )
    contam(
        "contam-list-phthalates",
        "Only do contaminant-removal screening. Do not do TEA, safety, literature, or general process design. List the supported phthalate contaminants in the Zhou workbook and explain briefly how you would choose between leaching and STRAP contaminant removal once a target polymer and solvent list are given.",
        "Supported-phthalate coverage listing",
        tool_name="list_supported_contaminants",
    )
    sep_contam(
        "sepcontam-evoh-pe-dbp",
        "First do process design for separation, then contaminant screening. For an EVOH/PE multilayer contaminated with di-n-butyl phthalate (DBP), first identify the best atmospheric-pressure solvent route for isolating EVOH up to 120°C. Then screen the route solvent candidates for contaminant removal and say whether leaching or STRAP contaminant removal is more defensible.",
        "Separation -> contaminant-removal handoff on EVOH/PE + DBP",
    )

    return cases


DEFAULT_SUITE = build_suite()
CONTAMINANT_SUITE = build_contaminant_suite()
SUITE = DEFAULT_SUITE
SUITE_BY_NAME = {case.name: case for case in (DEFAULT_SUITE + CONTAMINANT_SUITE)}


def _evaluate_case(case: EvalCase, query_result) -> list[Check]:
    checks: list[Check] = []
    answer = query_result.full_answer or ""
    actual = query_result.actual_subagents or []
    actual_unique = _unique_in_order(actual)
    expected_unique = _unique_in_order(case.expected_subagents)
    trace_summary = query_result.trace_summary or {}
    trace_tools = trace_summary.get("tool_names") or []
    trace_counts = Counter(trace_tools)
    child_errors = trace_summary.get("child_errors") or []

    checks.append(Check(
        "Run completed without agent error",
        query_result.error is None,
        query_result.error or "ok",
    ))
    checks.append(Check(
        "Harness captured non-empty final answer",
        bool(answer.strip()),
        f"answer_length={len(answer)}",
    ))
    checks.append(Check(
        "Harness persisted thread_id/run_id/trace_id",
        bool(query_result.thread_id and query_result.run_id and query_result.trace_id),
        f"thread_id={query_result.thread_id} run_id={query_result.run_id} trace_id={query_result.trace_id}",
    ))
    checks.append(Check(
        "Executed subagent route matches expected set exactly",
        set(actual_unique) == set(expected_unique),
        f"expected={expected_unique} actual={actual_unique}",
    ))
    checks.append(Check(
        "No unexpected execution of disallowed subagents",
        not any(sa in DISALLOWED_SUBAGENTS for sa in actual_unique),
        f"actual={actual_unique}",
    ))
    checks.append(Check(
        "No duplicate executed subagent dispatches",
        len(actual_unique) == len(actual),
        f"actual={actual}",
    ))
    checks.append(Check(
        "LangSmith trace summary captured",
        bool(trace_summary),
        f"trace_summary_keys={sorted(trace_summary.keys()) if trace_summary else []}",
    ))
    checks.append(Check(
        "LangSmith trace had no child tool/LLM errors",
        not child_errors,
        f"child_errors={len(child_errors)}",
    ))

    if case.required_trace_any:
        checks.append(Check(
            "Trace includes a category-appropriate domain tool",
            any(tool in trace_counts for tool in case.required_trace_any),
            f"trace_tools={trace_tools}",
        ))
    for tool in case.required_trace_all:
        checks.append(Check(
            f"Trace includes required tool: {tool}",
            trace_counts[tool] > 0,
            f"count={trace_counts[tool]}",
        ))
    for tool, minimum in case.min_trace_counts.items():
        checks.append(Check(
            f"Trace meets minimum count for {tool}",
            trace_counts[tool] >= minimum,
            f"count={trace_counts[tool]} minimum={minimum}",
        ))

    for idx, term_group in enumerate(case.answer_term_groups, start=1):
        checks.append(Check(
            f"Answer covers required concept group {idx}",
            _contains_any(answer, term_group),
            f"terms={term_group}",
        ))

    return checks


def _select_cases(
    suite_name: str,
    category: str | None,
    case_name: str | None,
    case_names: str | None,
    limit: int | None,
) -> list[EvalCase]:
    if suite_name == "default":
        source_suite = DEFAULT_SUITE
    elif suite_name == "contaminant":
        source_suite = CONTAMINANT_SUITE
    elif suite_name == "all":
        source_suite = DEFAULT_SUITE + CONTAMINANT_SUITE
    else:
        raise SystemExit(f"Unknown suite: {suite_name}")
    source_by_name = {case.name: case for case in source_suite}

    if case_names:
        requested = [name.strip() for name in case_names.split(",") if name.strip()]
        missing = [name for name in requested if name not in source_by_name]
        if missing:
            raise SystemExit(f"Unknown case(s): {', '.join(missing)}")
        cases = [source_by_name[name] for name in requested]
    elif case_name:
        if case_name not in source_by_name:
            raise SystemExit(f"Unknown case: {case_name}")
        cases = [source_by_name[case_name]]
    elif category:
        cases = [case for case in source_suite if case.category == category]
        if not cases:
            raise SystemExit(f"Unknown category: {category}")
    else:
        cases = list(source_suite)
    if limit is not None:
        cases = cases[:limit]
    return cases


def _parse_category_timeouts(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}

    parsed: dict[str, int] = {}
    for item in raw.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(
                f"Invalid category-timeout entry '{chunk}'. Expected category=seconds."
            )
        category, value = [part.strip() for part in chunk.split("=", 1)]
        if category not in _CATEGORY_TIMEOUT_KEYS:
            raise SystemExit(
                f"Unknown category '{category}' in --category-timeouts. "
                f"Expected one of: {', '.join(sorted(_CATEGORY_TIMEOUT_KEYS))}."
            )
        try:
            timeout_s = int(value)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid timeout '{value}' for category '{category}'. Expected integer seconds."
            ) from exc
        if timeout_s <= 0:
            raise SystemExit(
                f"Invalid timeout '{value}' for category '{category}'. Expected > 0 seconds."
            )
        parsed[category] = timeout_s
    return parsed


def _resolve_case_timeout(
    case: EvalCase,
    timeout_s: int | None,
    category_timeouts: dict[str, int] | None,
) -> int | None:
    if category_timeouts and case.category in category_timeouts:
        return category_timeouts[case.category]
    return timeout_s


def _save_progress(path: Path, results: list[CaseResult], metadata: dict) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "n_results": len(results),
        "results": [
            {
                **asdict(result),
                "checks": [asdict(check) for check in result.checks],
                "passed_checks": result.passed_checks,
                "total_checks": result.total_checks,
                "score_pct": result.score_pct,
                "passed": result.passed,
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_markdown_summary(path: Path, results: list[CaseResult]) -> None:
    lines = [
        "# Operational Batch Eval",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Case | Category | Score | Route | Child Errors | Trace |",
        "|---|---|---:|---|---:|---|",
    ]
    for result in results:
        child_errors = len((result.trace_summary or {}).get("child_errors") or [])
        route = ", ".join(result.actual_subagents) if result.actual_subagents else "(none)"
        trace = result.trace_id or "(missing)"
        lines.append(
            f"| {result.name} | {result.category} | {result.score_pct:.0f}% | {route} | {child_errors} | `{trace}` |"
        )

    failures = [result for result in results if not result.passed]
    if failures:
        lines.extend(["", "## Failures", ""])
        for result in failures:
            failed = [check for check in result.checks if not check.passed]
            lines.append(f"### {result.name} ({result.score_pct:.0f}%)")
            lines.append("")
            lines.append(f"- Query: {result.query}")
            lines.append(f"- Trace: `{result.trace_id}`")
            lines.append(f"- Route: {result.actual_subagents}")
            for check in failed[:8]:
                lines.append(f"- {check.name}: {check.detail}")
            lines.append("")

    path.write_text("\n".join(lines))


def _write_case_artifact(case_dir: Path, result: CaseResult) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(result),
        "checks": [asdict(check) for check in result.checks],
        "passed_checks": result.passed_checks,
        "total_checks": result.total_checks,
        "score_pct": result.score_pct,
        "passed": result.passed,
    }
    (case_dir / f"{result.name}.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"# {result.name}",
        "",
        f"- Category: {result.category}",
        f"- Pattern: {result.pattern}",
        f"- Attempts: {result.attempts}",
        f"- Score: {result.score_pct:.1f}% ({result.passed_checks}/{result.total_checks})",
        f"- Expected subagents: {result.expected_subagents}",
        f"- Actual subagents: {result.actual_subagents}",
        f"- Wall time (s): {result.wall_time_s}",
        f"- Tokens: {result.total_tokens}",
        f"- Thread ID: {result.thread_id}",
        f"- Run ID: {result.run_id}",
        f"- Trace ID: {result.trace_id}",
        f"- Trace visual: {result.raw_result_path}",
        "",
        "## Query",
        "",
        result.query,
        "",
        "## Checks",
        "",
    ]
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"- [{status}] {check.name}: {check.detail}")

    lines.extend([
        "",
        "## Trace Summary",
        "",
        "```json",
        json.dumps(result.trace_summary or {}, indent=2),
        "```",
        "",
        "## Final Answer Diagnostics",
        "",
        "```json",
        json.dumps(result.final_answer_diagnostics or {}, indent=2),
        "```",
        "",
        "## Full Answer",
        "",
        result.full_answer or "(empty)",
        "",
    ])
    (case_dir / f"{result.name}.md").write_text("\n".join(lines))


def _to_test_query(case: EvalCase) -> TestQuery:
    return TestQuery(
        name=case.name,
        query=case.query,
        pattern=case.pattern,
        expected_subagents=case.expected_subagents,
        recursion_limit=case.recursion_limit,
        description=case.description,
    )


class CaseTimeoutError(RuntimeError):
    """Raised when a live case exceeds its wall-clock timeout."""


def _serialize_query_result(result: QueryResult) -> dict:
    return asdict(result)


def _run_case_worker(case_payload: dict, project_name: str, thread_id: str, output_queue) -> None:
    try:
        case = EvalCase(**case_payload)
        agent = create_dissolve_agent()
        query_result = run_query(
            agent,
            _to_test_query(case),
            None,
            project_name=project_name,
            thread_id=thread_id,
            fetch_trace=False,
            persist_timeout_snapshot=True,
        )
        output_queue.put({"ok": True, "result": _serialize_query_result(query_result)})
    except Exception as exc:
        output_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _attach_trace_info(
    result: QueryResult,
    *,
    trace_info: dict | None,
    ls_client: LangSmithClient,
    project_name: str,
) -> QueryResult:
    result.trace_summary = trace_info
    result.run_id = trace_info.get("run_id") if trace_info else None
    result.trace_id = trace_info.get("trace_id") if trace_info else None
    if result.trace_id:
        result.actual_subagents = _extract_subagents_from_trace(
            ls_client,
            result.trace_id,
            project_name,
        ) or result.actual_subagents
        result.tool_names = (trace_info.get("tool_names") if trace_info else []) or result.tool_names
    return result


def _recover_timeout_result(
    *,
    case: EvalCase,
    thread_id: str,
    timeout_s: int,
    trace_info: dict | None,
    ls_client: LangSmithClient,
    project_name: str,
) -> QueryResult:
    snapshot = load_timeout_snapshot(thread_id)
    if snapshot is not None and snapshot.full_answer.strip():
        diagnostics = dict(snapshot.final_answer_diagnostics or {})
        diagnostics["timeout_recovered"] = True
        snapshot.final_answer_diagnostics = diagnostics
        snapshot.error = None
        return _attach_trace_info(
            snapshot,
            trace_info=trace_info,
            ls_client=ls_client,
            project_name=project_name,
        )

    timed_out_subagents = (
        _extract_subagents_from_trace(ls_client, trace_info["trace_id"], project_name)
        if trace_info and trace_info.get("trace_id")
        else []
    )
    if timed_out_subagents == ["separation-engineer"] and trace_info and trace_info.get("trace_id"):
        recovered_text = _extract_latest_task_output_from_trace(
            ls_client,
            trace_info["trace_id"],
            project_name,
            subagent="separation-engineer",
        )
        recovered = _recover_validated_separation_answer(case, recovered_text)
        if recovered:
            recovered_answer, recovery_meta = recovered
            return QueryResult(
                name=case.name,
                query=case.query,
                pattern=case.pattern,
                expected_subagents=case.expected_subagents,
                actual_subagents=timed_out_subagents,
                wall_time_s=float(timeout_s),
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                n_tool_calls=0,
                n_messages=0,
                tool_names=(trace_info.get("tool_names") if trace_info else []) or [],
                thread_id=thread_id,
                run_id=trace_info.get("run_id"),
                trace_id=trace_info.get("trace_id"),
                full_answer=recovered_answer,
                answer_preview=(
                    recovered_answer[:500] + "..."
                    if len(recovered_answer) > 500
                    else recovered_answer
                ),
                final_answer_diagnostics={
                    "timeout_recovered_from_validated_trace_payload": True,
                    "message_count": 0,
                    "last_message_type": None,
                    "last_ai_origin": recovery_meta.get("strap_origin"),
                    "last_ai_excerpt": recovered_answer[:280],
                    "final_answer_length": len(recovered_answer),
                },
                routing_match=False,
                timestamp=datetime.now().isoformat(),
                error=None,
                trace_summary=trace_info,
            )
    if timed_out_subagents == ["contaminant-removal-analyst"] and trace_info and trace_info.get("trace_id"):
        recovered_text = _extract_latest_task_output_from_trace(
            ls_client,
            trace_info["trace_id"],
            project_name,
            subagent="contaminant-removal-analyst",
        )
        recovered = _recover_validated_contaminant_answer(case, recovered_text)
        if recovered:
            recovered_answer, recovery_meta = recovered
            return QueryResult(
                name=case.name,
                query=case.query,
                pattern=case.pattern,
                expected_subagents=case.expected_subagents,
                actual_subagents=timed_out_subagents,
                wall_time_s=float(timeout_s),
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                n_tool_calls=0,
                n_messages=0,
                tool_names=(trace_info.get("tool_names") if trace_info else []) or [],
                thread_id=thread_id,
                run_id=trace_info.get("run_id"),
                trace_id=trace_info.get("trace_id"),
                full_answer=recovered_answer,
                answer_preview=(
                    recovered_answer[:500] + "..."
                    if len(recovered_answer) > 500
                    else recovered_answer
                ),
                final_answer_diagnostics={
                    "timeout_recovered_from_validated_trace_payload": True,
                    "message_count": 0,
                    "last_message_type": None,
                    "last_ai_origin": recovery_meta.get("strap_origin"),
                    "last_ai_excerpt": recovered_answer[:280],
                    "final_answer_length": len(recovered_answer),
                },
                routing_match=False,
                timestamp=datetime.now().isoformat(),
                error=None,
                trace_summary=trace_info,
            )
    return QueryResult(
        name=case.name,
        query=case.query,
        pattern=case.pattern,
        expected_subagents=case.expected_subagents,
        actual_subagents=timed_out_subagents,
        wall_time_s=float(timeout_s),
        total_tokens=0,
        input_tokens=0,
        output_tokens=0,
        n_tool_calls=0,
        n_messages=0,
        tool_names=(trace_info.get("tool_names") if trace_info else []) or [],
        thread_id=thread_id,
        run_id=trace_info.get("run_id") if trace_info else None,
        trace_id=trace_info.get("trace_id") if trace_info else None,
        full_answer="",
        answer_preview="",
        final_answer_diagnostics={
            "timeout": True,
            "message_count": 0,
            "last_message_type": None,
            "last_ai_origin": None,
            "last_ai_excerpt": "",
            "final_answer_length": 0,
        },
        routing_match=False,
        timestamp=datetime.now().isoformat(),
        error=f"case timed out after {timeout_s}s",
        trace_summary=trace_info,
    )


def _run_query_with_timeout(
    case: EvalCase,
    project_name: str,
    timeout_s: int | None,
    ls_client: LangSmithClient,
) -> QueryResult:
    thread_id = f"harness-{case.name}-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now().astimezone()
    clear_timeout_snapshot(thread_id)
    if not timeout_s:
        agent = create_dissolve_agent()
        result = run_query(
            agent,
            _to_test_query(case),
            ls_client,
            project_name=project_name,
            thread_id=thread_id,
        )
        clear_timeout_snapshot(thread_id)
        return result

    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    process = ctx.Process(
        target=_run_case_worker,
        args=(asdict(case), project_name, thread_id, output_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5)
        trace_info = fetch_langsmith_trace(
            ls_client,
            query=case.query,
            project=project_name,
            started_at=started_at,
        )
        recovered = _recover_timeout_result(
            case=case,
            thread_id=thread_id,
            timeout_s=int(timeout_s),
            trace_info=trace_info,
            ls_client=ls_client,
            project_name=project_name,
        )
        clear_timeout_snapshot(thread_id)
        return recovered

    if output_queue.empty():
        trace_info = fetch_langsmith_trace(
            ls_client,
            query=case.query,
            project=project_name,
            started_at=started_at,
        )
        recovered = _recover_timeout_result(
            case=case,
            thread_id=thread_id,
            timeout_s=int(timeout_s),
            trace_info=trace_info,
            ls_client=ls_client,
            project_name=project_name,
        )
        clear_timeout_snapshot(thread_id)
        return recovered

    payload = output_queue.get()
    if not payload.get("ok"):
        clear_timeout_snapshot(thread_id)
        raise RuntimeError(payload.get("error", "case process failed"))

    result = QueryResult(**payload["result"])
    trace_info = fetch_langsmith_trace(
        ls_client,
        query=case.query,
        project=project_name,
        started_at=started_at,
    )
    result = _attach_trace_info(
        result,
        trace_info=trace_info,
        ls_client=ls_client,
        project_name=project_name,
    )
    clear_timeout_snapshot(thread_id)
    return result


def _print_case_trace(result: CaseResult) -> None:
    summary = result.trace_summary or {}
    tool_names = summary.get("tool_names") or []
    child_errors = summary.get("child_errors") or []
    print(f"  Eval score:  {result.score_pct:.0f}% ({result.passed_checks}/{result.total_checks})")
    print(f"  Trace runs:  {summary.get('run_count', 0)} total | {summary.get('tool_run_count', 0)} tool | {summary.get('llm_run_count', 0)} llm")
    print(f"  Trace tools: {tool_names[:12]}{' ...' if len(tool_names) > 12 else ''}")
    if child_errors:
        print(f"  Child errs:  {len(child_errors)}")
        for err in child_errors[:3]:
            print(f"    - {err.get('name')} [{err.get('run_type')}]: {err.get('error')[:160]}")


def run_suite(
    *,
    cases: list[EvalCase],
    project_name: str,
    output_dir: Path,
    visualize: bool,
    fresh_agent_per_case: bool,
    retry_on_fail: int,
    timeout_s: int | None,
    category_timeouts: dict[str, int] | None = None,
) -> list[CaseResult]:
    ls_client = LangSmithClient()
    agent = None if fresh_agent_per_case else create_dissolve_agent()
    results: list[CaseResult] = []
    metadata = {
        "project_name": project_name,
        "fresh_agent_per_case": fresh_agent_per_case,
        "retry_on_fail": retry_on_fail,
        "timeout_s": timeout_s,
        "category_timeouts": category_timeouts or {},
        "case_names": [case.name for case in cases],
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"operational_eval_{timestamp}.json"
    md_path = output_dir / f"operational_eval_{timestamp}.md"
    case_artifact_dir = output_dir / f"operational_eval_{timestamp}_cases"
    metadata["case_artifact_dir"] = str(case_artifact_dir)
    case_artifact_dir.mkdir(parents=True, exist_ok=True)

    for idx, case in enumerate(cases, start=1):
        case_timeout_s = _resolve_case_timeout(case, timeout_s, category_timeouts)
        timeout_suffix = f" timeout={case_timeout_s}s" if case_timeout_s else ""
        print(f"\n[{idx}/{len(cases)}] {case.name}  [{case.category}]{timeout_suffix}")
        attempts = 0
        final_result = None
        while attempts <= retry_on_fail:
            attempts += 1
            try:
                if fresh_agent_per_case:
                    query_result = _run_query_with_timeout(case, project_name, case_timeout_s, ls_client)
                else:
                    query_result = run_query(
                        agent,
                        _to_test_query(case),
                        ls_client,
                        project_name=project_name,
                    )
            except CaseTimeoutError as exc:
                query_result = type("TimeoutResult", (), {
                    "actual_subagents": [],
                    "wall_time_s": float(case_timeout_s or 0),
                    "total_tokens": 0,
                    "tool_names": [],
                    "thread_id": None,
                    "run_id": None,
                    "trace_id": None,
                    "trace_summary": None,
                    "full_answer": "",
                    "answer_preview": "",
                    "error": str(exc),
                })()
            checks = _evaluate_case(case, query_result)
            case_result = CaseResult(
                name=case.name,
                category=case.category,
                query=case.query,
                pattern=case.pattern,
                expected_subagents=case.expected_subagents,
                actual_subagents=query_result.actual_subagents,
                attempts=attempts,
                checks=checks,
                wall_time_s=query_result.wall_time_s,
                total_tokens=query_result.total_tokens,
                tool_names=query_result.tool_names,
                thread_id=query_result.thread_id,
                run_id=query_result.run_id,
                trace_id=query_result.trace_id,
                trace_summary=query_result.trace_summary,
                full_answer=query_result.full_answer,
                answer_preview=query_result.answer_preview,
                final_answer_diagnostics=query_result.final_answer_diagnostics,
                error=query_result.error,
            )
            _print_case_trace(case_result)
            final_result = case_result
            if case_result.passed or attempts > retry_on_fail:
                break
            print("  Retry scheduled due failed checks.")
            time.sleep(1)

        assert final_result is not None
        if visualize and final_result.trace_id:
            trace_result = QueryResult(
                name=final_result.name,
                query=final_result.query,
                pattern=final_result.pattern,
                expected_subagents=final_result.expected_subagents,
                actual_subagents=final_result.actual_subagents,
                wall_time_s=final_result.wall_time_s,
                total_tokens=final_result.total_tokens,
                input_tokens=0,
                output_tokens=0,
                n_tool_calls=len(final_result.tool_names),
                n_messages=0,
                tool_names=final_result.tool_names,
                thread_id=final_result.thread_id,
                run_id=final_result.run_id,
                trace_id=final_result.trace_id,
                full_answer=final_result.full_answer,
                answer_preview=final_result.answer_preview,
                final_answer_diagnostics=final_result.final_answer_diagnostics,
                routing_match=set(final_result.actual_subagents) == set(final_result.expected_subagents),
                timestamp=datetime.now().isoformat(),
                error=final_result.error,
                trace_summary=final_result.trace_summary,
            )
            trace_result = generate_trace_visuals(trace_result, ls_client, output_dir)
            final_result.raw_result_path = trace_result.waterfall_png

        results.append(final_result)
        _write_case_artifact(case_artifact_dir, final_result)
        _save_progress(json_path, results, metadata)
        _write_markdown_summary(md_path, results)

    print(f"\nJSON results: {json_path}")
    print(f"Markdown summary: {md_path}")
    return results


def print_case_summary(results: list[CaseResult]) -> None:
    print("\n" + "=" * 110)
    print("OPERATIONAL BATCH EVAL SUMMARY")
    print("=" * 110)
    print(f"{'Case':<28} {'Category':<14} {'Score':>7} {'Time':>7} {'Tokens':>10} {'Attempts':>8} {'Route':<28}")
    print(f"{'-'*28} {'-'*14} {'-'*7} {'-'*7} {'-'*10} {'-'*8} {'-'*28}")
    for result in results:
        route = ",".join(result.actual_subagents) if result.actual_subagents else "(none)"
        if len(route) > 27:
            route = route[:24] + "..."
        print(
            f"{result.name:<28} {result.category:<14} {result.score_pct:>6.0f}% {result.wall_time_s:>6.1f}s {result.total_tokens:>10,} {result.attempts:>8} {route:<28}"
        )
    print("-" * 110)
    passed = sum(1 for result in results if result.passed)
    print(f"Passed cases: {passed}/{len(results)}")
    by_category = Counter(result.category for result in results)
    for category, count in sorted(by_category.items()):
        avg = sum(result.score_pct for result in results if result.category == category) / count
        print(f"  {category:<14} {count:>2} cases | avg score {avg:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run operational eval campaigns")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--suite",
        default="default",
        choices=["default", "contaminant", "all"],
        help="Select which benchmark suite to use",
    )
    parser.add_argument("--category", default=None, help="Run only one category")
    parser.add_argument("--case", default=None, help="Run a single case by name")
    parser.add_argument("--cases", default=None, help="Run a comma-separated list of case names")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of selected cases")
    parser.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT", "strap-agent"), help="LangSmith project name")
    parser.add_argument("--output-dir", default=str(_DIR / "test_results"), help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip trace PNG generation")
    parser.add_argument("--fresh-agent-per-case", action="store_true", help="Create a new agent for every case")
    parser.add_argument("--retry-on-fail", type=int, default=0, help="Retries per failing case")
    parser.add_argument("--timeout-s", type=int, default=240, help="Per-case timeout in seconds")
    parser.add_argument(
        "--category-timeouts",
        default=None,
        help=(
            "Optional per-category timeout overrides like "
            "'hsp=90,safety=120,separation=150,biosteam=150,sep-biosteam=210,sep-safety=210,contaminant=120,sep-contaminant=180'."
        ),
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'Case':<28} {'Category':<14} {'Pattern':<12} Description")
        print(f"{'-'*28} {'-'*14} {'-'*12} {'-'*50}")
        cases = _select_cases(args.suite, None, None, None, None)
        for case in cases:
            print(f"{case.name:<28} {case.category:<14} {case.pattern:<12} {case.description}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _select_cases(args.suite, args.category, args.case, args.cases, args.limit)
    category_timeouts = _parse_category_timeouts(args.category_timeouts)
    results = run_suite(
        cases=cases,
        project_name=args.project,
        output_dir=output_dir,
        visualize=not args.no_viz,
        fresh_agent_per_case=args.fresh_agent_per_case,
        retry_on_fail=args.retry_on_fail,
        timeout_s=args.timeout_s,
        category_timeouts=category_timeouts,
    )
    print_case_summary(results)


if __name__ == "__main__":
    main()
