"""Planning-only harness for deterministic workflow validation.

This harness never invokes the live agent graph. It exercises only the
initial routing and ordered-plan construction path and hard-blocks any
attempt to construct or call Gemini-backed chat models.

Usage:
    python architecture/plan_only_harness.py
    python architecture/plan_only_harness.py --category mixed-research
    python architecture/plan_only_harness.py --json-out /tmp/plan_only.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ARCH_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _ARCH_DIR.parent
sys.path.insert(0, str(_ROOT_DIR / "src"))

from langchain_core.messages import HumanMessage

from strap.routing import RoutingMiddleware
from strap.routing_message_state import _get_ordered_plan
from strap.testing_utils import block_model_access


@dataclass(frozen=True)
class PlanExpectation:
    subagents: tuple[str, ...]
    dependencies: dict[str, tuple[str, ...]]
    pattern: str
    enforce_order: bool = True


@dataclass(frozen=True)
class PlanOnlyCase:
    name: str
    category: str
    query: str
    expectation: PlanExpectation


@dataclass
class PlanOnlyResult:
    name: str
    category: str
    ok: bool
    query: str
    reason: str
    allowed_subagents: list[str]
    planned_subagents: list[str]
    dependencies: dict[str, tuple[str, ...]]


@dataclass
class PlanOnlySummary:
    total: int
    passed: int
    failed: int
    blocked_model_call_attempts: dict[str, int]
    results: list[PlanOnlyResult]


def _pair_case(
    *,
    name: str,
    category: str,
    query: str,
    subagents: tuple[str, ...],
    dependencies: dict[str, tuple[str, ...]],
    pattern: str,
    enforce_order: bool = True,
) -> PlanOnlyCase:
    return PlanOnlyCase(
        name=name,
        category=category,
        query=query,
        expectation=PlanExpectation(
            subagents=subagents,
            dependencies=dependencies,
            pattern=pattern,
            enforce_order=enforce_order,
        ),
    )


def _build_single_separation_cases() -> list[PlanOnlyCase]:
    polymer_pairs = [
        ("PS", "PVC"),
        ("EVOH", "PE"),
        ("LDPE", "HDPE"),
        ("PET", "PS"),
        ("PP", "EVOH"),
        ("PC", "PET"),
        ("PVC", "PS"),
        ("HDPE", "EVOH"),
        ("LDPE", "PET"),
        ("PP", "PET"),
    ]
    temps = [90, 100, 110, 120, 95, 105, 115, 125, 130, 140]
    cases: list[PlanOnlyCase] = []
    for index, ((polymer_a, polymer_b), temp_c) in enumerate(zip(polymer_pairs, temps, strict=True), start=1):
        cases.append(
            _pair_case(
                name=f"single-separation-{index:02d}",
                category="single-separation",
                query=(
                    "Only do process design. "
                    f"Find an optimal separation sequence for {polymer_a} and {polymer_b} "
                    f"using selective dissolution at atmospheric pressure up to {temp_c}C."
                ),
                subagents=("separation-engineer",),
                dependencies={"separation-engineer": ()},
                pattern="single",
            )
        )
    return cases


def _build_single_safety_cases() -> list[PlanOnlyCase]:
    solvent_pairs = [
        ("toluene", "xylene"),
        ("THF", "acetone"),
        ("DMF", "DMSO"),
        ("cyclohexanone", "MEK"),
        ("ethyl acetate", "heptane"),
        ("anisole", "THF"),
        ("toluene", "ethyl acetate"),
        ("xylene", "cyclohexanone"),
        ("DMSO", "acetone"),
        ("NMP", "toluene"),
    ]
    cases: list[PlanOnlyCase] = []
    for index, (solvent_a, solvent_b) in enumerate(solvent_pairs, start=1):
        cases.append(
            _pair_case(
                name=f"single-safety-{index:02d}",
                category="single-safety",
                query=(
                    "Only do safety analysis. "
                    f"Compare GSK safety scores and PubChem hazards for {solvent_a} and {solvent_b}. "
                    "Do not do process design or TEA."
                ),
                subagents=("safety-analyst",),
                dependencies={"safety-analyst": ()},
                pattern="single",
            )
        )
    return cases


def _build_single_biosteam_cases() -> list[PlanOnlyCase]:
    casespecs = [
        ("LDPE", "toluene", "C1"),
        ("PE", "xylene", "C2"),
        ("EVOH", "DMSO", "C1"),
        ("PET", "ethylene glycol", "C3"),
        ("PS", "toluene", "C1"),
        ("PP", "xylene", "C2"),
        ("PVC", "cyclohexanone", "C1"),
        ("PC", "THF", "C3"),
        ("LDPE", "heptane", "C2"),
        ("EVOH", "water", "C1"),
    ]
    cases: list[PlanOnlyCase] = []
    for index, (polymer, solvent, energy_case) in enumerate(casespecs, start=1):
        cases.append(
            _pair_case(
                name=f"single-biosteam-{index:02d}",
                category="single-biosteam",
                query=(
                    "Only do BioSTEAM TEA/LCA. "
                    f"For {polymer} dissolved in {solvent}, run a techno-economic analysis and "
                    f"life-cycle assessment under energy case {energy_case}. "
                    "Do not do solvent screening or process design."
                ),
                subagents=("biosteam-analyst",),
                dependencies={"biosteam-analyst": ()},
                pattern="single",
            )
        )
    return cases


def _build_single_contaminant_cases() -> list[PlanOnlyCase]:
    casespecs = [
        ("EVOH", "di-n-butyl phthalate (DBP)"),
        ("PET", "diethyl phthalate (DEP)"),
        ("PS", "Phthalates"),
        ("LDPE", "PFAS"),
        ("HDPE", "diisobutyl phthalate (DiBP)"),
        ("PP", "dibutyl phthalate (DBP)"),
        ("PVC", "Phthalates"),
        ("EVOH", "PFAS"),
        ("PET", "benzyl butyl phthalate (BBP)"),
        ("PS", "di(2-ethylhexyl) phthalate (DEHP)"),
    ]
    cases: list[PlanOnlyCase] = []
    for index, (polymer, contaminant) in enumerate(casespecs, start=1):
        cases.append(
            _pair_case(
                name=f"single-contaminant-{index:02d}",
                category="single-contaminant",
                query=(
                    "Only do contaminant-removal screening. "
                    f"For {polymer} contaminated with {contaminant}, compare leaching versus "
                    "STRAP contaminant removal. Do not do TEA, safety, literature, or general process design."
                ),
                subagents=("contaminant-removal-analyst",),
                dependencies={"contaminant-removal-analyst": ()},
                pattern="single",
            )
        )
    return cases


def _build_stats_viz_cases() -> list[PlanOnlyCase]:
    polymer_triples = [
        ("PC", "PET", "PS"),
        ("LDPE", "HDPE", "PP"),
        ("PVC", "PS", "PET"),
        ("EVOH", "PET", "PC"),
        ("PP", "PET", "PC"),
        ("PS", "PMMA", "PET"),
        ("LDPE", "EVOH", "PVC"),
        ("HDPE", "PET", "PS"),
        ("PP", "PVC", "PC"),
        ("EVOH", "PS", "PET"),
    ]
    cases: list[PlanOnlyCase] = []
    for index, (polymer_a, polymer_b, polymer_c) in enumerate(polymer_triples, start=1):
        cases.append(
            _pair_case(
                name=f"stats-viz-{index:02d}",
                category="stats-viz",
                query=(
                    f"Look up the glass transition temperatures for {polymer_a}, {polymer_b}, and {polymer_c}, "
                    "then create a chart comparing the values."
                ),
                subagents=("statistics-ml", "visualization-specialist"),
                dependencies={
                    "statistics-ml": (),
                    "visualization-specialist": ("statistics-ml",),
                },
                pattern="sequential",
            )
        )
    return cases


def _build_seq_sep_viz_cases() -> list[PlanOnlyCase]:
    casespecs = [
        ("PS, PMMA, and PET", "selectivity heatmap", 120),
        ("LDPE and EVOH", "chart of the separation results", 110),
        ("HDPE, LDPE, and PP", "dashboard of the recommended steps", 130),
        ("PET and PS", "heatmap of the separation results", 120),
        ("PP and EVOH", "comparison chart of the route", 115),
        ("PC and PET", "chart of the sequence outcome", 100),
        ("PVC and PS", "chart of the recommended route", 105),
        ("HDPE and EVOH", "selectivity chart for the best route", 125),
        ("LDPE and PET", "chart summarizing the route", 110),
        ("PP and PET", "heatmap of the separation results", 135),
    ]
    cases: list[PlanOnlyCase] = []
    for index, (polymer_text, viz_request, temp_c) in enumerate(casespecs, start=1):
        cases.append(
            _pair_case(
                name=f"seq-sep-viz-{index:02d}",
                category="seq-sep-viz",
                query=(
                    f"Find the optimal separation sequence for {polymer_text} at up to {temp_c}C, "
                    f"then create a {viz_request}."
                ),
                subagents=("separation-engineer", "visualization-specialist"),
                dependencies={
                    "separation-engineer": (),
                    "visualization-specialist": ("separation-engineer",),
                },
                pattern="sequential",
            )
        )
    return cases


def _build_seq_sep_biosteam_cases() -> list[PlanOnlyCase]:
    streams = [
        "LDPE/HDPE/PP",
        "PS/PET/PC",
        "EVOH/PE",
        "LDPE/EVOH/PET",
        "PS/PVC",
        "HDPE/PP/EVOH",
        "PET/PS/PP",
        "LDPE/PET",
        "PC/PET/PS",
        "EVOH/LDPE/HDPE",
    ]
    cases: list[PlanOnlyCase] = []
    for index, stream in enumerate(streams, start=1):
        cases.append(
            _pair_case(
                name=f"seq-sep-biosteam-{index:02d}",
                category="seq-sep-biosteam",
                query=(
                    f"Find an optimal separation sequence for a {stream} mixed waste stream "
                    "using selective dissolution at atmospheric pressure. "
                    "Then run a techno-economic analysis on the solvent recovery for the best option."
                ),
                subagents=("separation-engineer", "biosteam-analyst"),
                dependencies={
                    "separation-engineer": (),
                    "biosteam-analyst": ("separation-engineer",),
                },
                pattern="sequential",
            )
        )
    return cases


def _build_seq_sep_contam_bio_cases() -> list[PlanOnlyCase]:
    casespecs = [
        ("HDPE/EVOH", "phthalate"),
        ("LDPE/EVOH", "PFAS"),
        ("PET/EVOH", "di-n-butyl phthalate (DBP)"),
        ("PS/EVOH", "Phthalates"),
        ("HDPE/PET", "PFAS"),
        ("PP/EVOH", "di-n-butyl phthalate (DBP)"),
        ("LDPE/PET", "Phthalates"),
        ("PS/PET", "PFAS"),
        ("PC/EVOH", "di-n-butyl phthalate (DBP)"),
        ("HDPE/PP", "Phthalates"),
    ]
    cases: list[PlanOnlyCase] = []
    for index, (stream, contaminant) in enumerate(casespecs, start=1):
        cases.append(
            _pair_case(
                name=f"seq-sep-contam-bio-{index:02d}",
                category="seq-sep-contam-bio",
                query=(
                    f"Find an optimal separation sequence for an {stream} mixed waste stream using "
                    "selective dissolution at atmospheric pressure. "
                    f"Propose up to 1 additional wash step for {contaminant} removal. "
                    "Then run a techno-economic analysis on solvent recovery for the best option."
                ),
                subagents=(
                    "separation-engineer",
                    "contaminant-removal-analyst",
                    "biosteam-analyst",
                ),
                dependencies={
                    "separation-engineer": (),
                    "contaminant-removal-analyst": ("separation-engineer",),
                    "biosteam-analyst": ("contaminant-removal-analyst",),
                },
                pattern="sequential",
            )
        )
    return cases


def _build_mixed_research_cases() -> list[PlanOnlyCase]:
    topics = [
        "multilayer PE/EVOH film recycling methods",
        "PET tray delamination workflows",
        "polyolefin dissolution in terpene solvents",
        "EVOH barrier-layer recycling routes",
        "PS/PET multilayer separation methods",
        "PVC-compatible delamination strategies",
        "solvent-based recycling of food-packaging laminates",
        "low-temperature polymer delamination methods",
        "multilayer packaging waste delamination methods",
        "mixed-plastic solvent recovery process design",
    ]
    cases: list[PlanOnlyCase] = []
    for index, topic in enumerate(topics, start=1):
        cases.append(
            _pair_case(
                name=f"mixed-research-{index:02d}",
                category="mixed-research",
                query=(
                    f"Do a literature search and patent search for {topic}. "
                    "Answer the question with RAG, then create a chart visualization of the retrieved findings."
                ),
                subagents=(
                    "scholar-researcher",
                    "patent-researcher",
                    "rag-analyst",
                    "visualization-specialist",
                ),
                dependencies={
                    "scholar-researcher": (),
                    "patent-researcher": (),
                    "rag-analyst": ("scholar-researcher", "patent-researcher"),
                    "visualization-specialist": ("rag-analyst",),
                },
                pattern="mixed",
            )
        )
    return cases


def _build_parallel_scholar_patent_cases() -> list[PlanOnlyCase]:
    topics = [
        "multilayer polymer recycling methods",
        "polyolefin solvent recycling methods",
        "PET delamination methods",
        "EVOH recovery processes",
        "PVC-compatible recycling methods",
        "food-packaging laminate recycling",
        "solvent-based multilayer separation",
        "waste-film delamination processes",
        "polymer barrier-layer recycling",
        "low-energy solvent recovery concepts",
    ]
    cases: list[PlanOnlyCase] = []
    for index, topic in enumerate(topics, start=1):
        cases.append(
            _pair_case(
                name=f"parallel-scholar-patent-{index:02d}",
                category="parallel-scholar-patent",
                query=f"Do a literature search and patent search for {topic}.",
                subagents=("scholar-researcher", "patent-researcher"),
                dependencies={
                    "scholar-researcher": (),
                    "patent-researcher": (),
                },
                pattern="parallel",
                enforce_order=False,
            )
        )
    return cases


def build_plan_only_cases() -> list[PlanOnlyCase]:
    """Return the full deterministic 100-query planning-only suite."""
    cases = [
        *_build_single_separation_cases(),
        *_build_single_safety_cases(),
        *_build_single_biosteam_cases(),
        *_build_single_contaminant_cases(),
        *_build_stats_viz_cases(),
        *_build_seq_sep_viz_cases(),
        *_build_seq_sep_biosteam_cases(),
        *_build_seq_sep_contam_bio_cases(),
        *_build_mixed_research_cases(),
        *_build_parallel_scholar_patent_cases(),
    ]
    if len(cases) != 100:  # pragma: no cover - defensive contract
        raise ValueError(f"Expected 100 plan-only cases, found {len(cases)}")
    return cases


def _dependency_map(plan: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    return {
        step["subagent"]: tuple(step.get("depends_on", ()))
        for step in plan
    }


def _check_pattern(expectation: PlanExpectation, plan: list[dict[str, Any]]) -> str | None:
    dependency_map = _dependency_map(plan)
    roots = [name for name, deps in dependency_map.items() if not deps]
    joined = [name for name, deps in dependency_map.items() if len(deps) > 1]

    if expectation.pattern == "single":
        if len(plan) != 1 or any(dependency_map.values()):
            return "expected a single root-only plan"
    elif expectation.pattern == "parallel":
        if len(plan) < 2 or any(dependency_map.values()):
            return "expected a parallel plan with no dependencies"
    elif expectation.pattern == "sequential":
        if len(plan) < 2:
            return "expected a multi-step sequential plan"
        if len(roots) != 1:
            return "expected exactly one root for a sequential plan"
        for step in plan[1:]:
            if len(dependency_map.get(step["subagent"], ())) != 1:
                return "expected each downstream step to have exactly one dependency"
    elif expectation.pattern == "mixed":
        if len(roots) < 2 or not joined:
            return "expected a mixed DAG with parallel roots and a downstream join"

    return None


def _validate_case(case: PlanOnlyCase) -> PlanOnlyResult:
    messages = [HumanMessage(content=case.query)]
    middleware = RoutingMiddleware(classifier_model=None)
    allowed_rules = middleware._get_allowed_rules(messages)
    ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)

    allowed_subagents = [rule["subagent"] for rule in allowed_rules]
    planned_subagents = [step["subagent"] for step in ordered_plan]
    dependencies = _dependency_map(ordered_plan)

    expected_subagents = list(case.expectation.subagents)
    if set(allowed_subagents) != set(expected_subagents):
        return PlanOnlyResult(
            name=case.name,
            category=case.category,
            ok=False,
            query=case.query,
            reason=f"allowed subagents mismatch: expected {expected_subagents}, got {allowed_subagents}",
            allowed_subagents=allowed_subagents,
            planned_subagents=planned_subagents,
            dependencies=dependencies,
        )

    if set(planned_subagents) != set(expected_subagents):
        return PlanOnlyResult(
            name=case.name,
            category=case.category,
            ok=False,
            query=case.query,
            reason=f"planned subagents mismatch: expected {expected_subagents}, got {planned_subagents}",
            allowed_subagents=allowed_subagents,
            planned_subagents=planned_subagents,
            dependencies=dependencies,
        )

    if case.expectation.enforce_order and planned_subagents != expected_subagents:
        return PlanOnlyResult(
            name=case.name,
            category=case.category,
            ok=False,
            query=case.query,
            reason=f"planned order mismatch: expected {expected_subagents}, got {planned_subagents}",
            allowed_subagents=allowed_subagents,
            planned_subagents=planned_subagents,
            dependencies=dependencies,
        )

    for subagent, expected_deps in case.expectation.dependencies.items():
        actual_deps = dependencies.get(subagent)
        if actual_deps != expected_deps:
            return PlanOnlyResult(
                name=case.name,
                category=case.category,
                ok=False,
                query=case.query,
                reason=(
                    f"dependency mismatch for {subagent}: "
                    f"expected {expected_deps}, got {actual_deps}"
                ),
                allowed_subagents=allowed_subagents,
                planned_subagents=planned_subagents,
                dependencies=dependencies,
            )

    pattern_error = _check_pattern(case.expectation, ordered_plan)
    if pattern_error is not None:
        return PlanOnlyResult(
            name=case.name,
            category=case.category,
            ok=False,
            query=case.query,
            reason=pattern_error,
            allowed_subagents=allowed_subagents,
            planned_subagents=planned_subagents,
            dependencies=dependencies,
        )

    return PlanOnlyResult(
        name=case.name,
        category=case.category,
        ok=True,
        query=case.query,
        reason="ok",
        allowed_subagents=allowed_subagents,
        planned_subagents=planned_subagents,
        dependencies=dependencies,
    )


def run_plan_only_suite(
    cases: list[PlanOnlyCase] | None = None,
    *,
    fail_fast: bool = False,
) -> PlanOnlySummary:
    selected_cases = list(cases) if cases is not None else build_plan_only_cases()
    results: list[PlanOnlyResult] = []
    blocked_model_call_attempts: Counter[str] = Counter()

    with block_model_access(blocked_model_call_attempts):
        for case in selected_cases:
            try:
                result = _validate_case(case)
            except Exception as exc:  # pragma: no cover - defensive path
                result = PlanOnlyResult(
                    name=case.name,
                    category=case.category,
                    ok=False,
                    query=case.query,
                    reason=f"{type(exc).__name__}: {exc}",
                    allowed_subagents=[],
                    planned_subagents=[],
                    dependencies={},
                )
            results.append(result)
            if fail_fast and not result.ok:
                break

    passed = sum(1 for result in results if result.ok)
    failed = len(results) - passed
    return PlanOnlySummary(
        total=len(results),
        passed=passed,
        failed=failed,
        blocked_model_call_attempts=dict(blocked_model_call_attempts),
        results=results,
    )


def _filter_cases(
    cases: list[PlanOnlyCase],
    *,
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[PlanOnlyCase]:
    selected = cases
    if categories:
        allowed = set(categories)
        selected = [case for case in selected if case.category in allowed]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _summary_payload(summary: PlanOnlySummary) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for result in summary.results:
        counts = by_category.setdefault(result.category, {"total": 0, "passed": 0, "failed": 0})
        counts["total"] += 1
        counts["passed"] += int(result.ok)
        counts["failed"] += int(not result.ok)

    return {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "blocked_model_call_attempts": summary.blocked_model_call_attempts,
        "by_category": by_category,
        "results": [asdict(result) for result in summary.results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic planning-only harness.")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Run only the named category. Repeat for multiple categories.",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of cases after filtering.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failure.")
    parser.add_argument("--json-out", type=Path, help="Optional path for a JSON summary.")
    args = parser.parse_args(argv)

    cases = _filter_cases(
        build_plan_only_cases(),
        categories=args.categories,
        limit=args.limit,
    )
    summary = run_plan_only_suite(cases, fail_fast=args.fail_fast)
    payload = _summary_payload(summary)

    print(
        json.dumps(
            {
                "total": payload["total"],
                "passed": payload["passed"],
                "failed": payload["failed"],
                "blocked_model_call_attempts": payload["blocked_model_call_attempts"],
                "by_category": payload["by_category"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if summary.failed or any(summary.blocked_model_call_attempts.values()):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
