#!/usr/bin/env python
"""HSP-based solvent screening agent tests (Q1.1.1 – Q1.1.4).

Launches the DISSOLVE agent against 4 test queries, captures tool calls
and final answers, and validates key checkpoints.

Usage:
    python tests/test_hsp_screening.py              # run all 4 tests
    python tests/test_hsp_screening.py --query 1     # run Q1.1.1 only
    python tests/test_hsp_screening.py --query 1,2   # run Q1.1.1 and Q1.1.2
    python tests/test_hsp_screening.py --parallel     # run all tests in parallel
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("STRAP_DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "strap.db"))


# ── Test Queries ──────────────────────────────────────────────────────────

QUERIES = {
    "1.1.1": (
        "Which of toluene, DMSO, and hexane would you expect to dissolve PE "
        "based on Hansen solubility parameters? Show your work."
    ),
    "1.1.2": (
        "I want to find a solvent that dissolves EVOH but NOT PE. Would DMSO "
        "work as a selective solvent? Evaluate using Hansen solubility parameters."
    ),
    "1.1.3": (
        "I have a 3-polymer waste stream: PE, PS, and PVC. Which of THF, "
        "cyclohexanone, or acetone could selectively dissolve PS while leaving "
        "PE and PVC undissolved at room temperature? Look up the relevant HSP "
        "data and evaluate."
    ),
    "1.1.4": (
        "I operate a mixed-plastics recycling facility and receive waste containing "
        "up to 16 different polymers from 8 families:\n\n"
        "Polyolefins: PE, PP, HDPE\n"
        "Styrenics: PS, ABS\n"
        "Vinyls: PVC, PVDF\n"
        "Polyesters: PET, PETG\n"
        "Acrylics/Carbonates: PMMA, PC\n"
        "Polyamides: PA6, PA66\n"
        "Engineering: PSU (polysulfone)\n"
        "Barrier/Specialty: EVOH, POM (acetal)\n\n"
        "Before running expensive COSMO-RS temperature-dependent simulations, I want "
        "to do an initial room-temperature HSP screening to identify: (a) which polymers "
        "can potentially be dissolved at RT, (b) which solvents are most selective for "
        "each, and (c) which polymer pairs are too similar in HSP space to separate. "
        "Screen against at least 10 solvents spanning different chemical classes "
        "(nonpolar, aromatic, chlorinated, ester, ketone, polar aprotic, polar protic). "
        "Build a selectivity matrix, group the polymers by separability, and propose "
        "a RT separation sequence for the soluble subset. Flag all limitations."
    ),
}


# ── Checkpoint Validators ────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TestResult:
    query_id: str
    query: str
    answer: str
    tool_calls: list[dict]
    elapsed_s: float
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def score_pct(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0


def _count_ml_calls(tool_calls: list[dict]) -> int:
    """Count predict_solubility_ml calls."""
    return sum(1 for tc in tool_calls if tc["name"] == "predict_solubility_ml")


def _extract_ml_pairs(tool_calls: list[dict]) -> list[tuple[str, str]]:
    """Extract (polymer, solvent) pairs from predict_solubility_ml calls."""
    pairs = []
    for tc in tool_calls:
        if tc["name"] == "predict_solubility_ml":
            args = tc.get("args", {})
            pairs.append((
                args.get("polymer_name", "").upper(),
                args.get("solvent_name", "").upper(),
            ))
    return pairs


def _answer_mentions(answer: str, terms: list[str], case_insensitive: bool = True) -> list[str]:
    """Return which terms appear in the answer."""
    text = answer.lower() if case_insensitive else answer
    return [t for t in terms if t.lower() in text]


def _has_ml_evidence(answer: str, tool_calls: list[dict]) -> bool:
    """Check if ML predictions were used (from tool calls or answer content)."""
    if _count_ml_calls(tool_calls) >= 1:
        return True
    # Answer-level evidence: RED/Ra values, "predict" + "solubility", "ML model"
    al = answer.lower()
    return any(term in al for term in [
        "red value", "red =", "red:", "red number", "hansen distance",
        "ra =", "ra:", "ml model", "ml prediction", "machine learning",
        "predict_solubility", "solubility prediction",
    ])


def _has_solvent_evidence(answer: str, tool_calls: list[dict], solvents: list[str]) -> list[str]:
    """Check which solvents have prediction evidence (tool calls or answer)."""
    found = []
    pairs = _extract_ml_pairs(tool_calls)
    al = answer.lower()
    for solv in solvents:
        sl = solv.lower()
        # Check tool calls
        if any(sl in s.lower() for _, s in pairs):
            found.append(solv)
        # Check answer content for RED/Ra values near solvent name
        elif sl in al and any(term in al for term in ["red", "ra ", "solub", "dissolv"]):
            found.append(solv)
    return found


def validate_q111(result: TestResult) -> list[CheckResult]:
    """Q1.1.1: PE + {toluene, DMSO, hexane}."""
    checks = []
    answer_lower = result.answer.lower()

    # Check 1: Used ML prediction tool (direct or via subagent)
    used_ml = _has_ml_evidence(result.answer, result.tool_calls)
    checks.append(CheckResult(
        "Used ML prediction / HSP analysis",
        used_ml,
        "ML/HSP evidence found in answer" if used_ml else "No ML evidence found",
    ))

    # Check 2: Analyzed all 3 solvents
    solvents_found = _has_solvent_evidence(result.answer, result.tool_calls,
                                            ["toluene", "DMSO", "hexane"])
    checks.append(CheckResult(
        "Analyzed all 3 solvents (toluene, DMSO, hexane)",
        len(solvents_found) >= 3,
        f"Found evidence for: {solvents_found}",
    ))

    # Check 3: Correctly identifies RED < 1 as dissolution criterion
    red_mention = _answer_mentions(result.answer, ["RED < 1", "RED<1", "RED value", "relative energy difference"])
    checks.append(CheckResult(
        "Interprets RED < 1 as dissolution criterion",
        len(red_mention) > 0,
        f"Mentioned: {red_mention}",
    ))

    # Check 4: Correctly identifies DMSO as non-solvent for PE
    dmso_nonsol = _answer_mentions(result.answer, ["DMSO"])
    answer_lower = result.answer.lower()
    dmso_bad = ("dmso" in answer_lower and
                any(term in answer_lower for term in
                    ["non-sol", "not dissolve", "incompatible", "non-solvent",
                     "will not", "won't", "poor", "unlikely", "not suitable",
                     "cannot dissolve", "does not dissolve"]))
    checks.append(CheckResult(
        "Identifies DMSO as non-solvent for PE",
        dmso_bad,
        f"DMSO discussed: {len(dmso_nonsol) > 0}, identified as poor: {dmso_bad}",
    ))

    # Check 5: Compares RED values to rank solvents
    comparison = any(term in answer_lower for term in
                     ["lowest red", "best match", "rank", "closest",
                      "better", "worse", "comparison", "compare"])
    if not comparison:
        # Also try regex patterns
        comparison = bool(re.search(r'hexane.*(?:better|best|closest|lowest)', answer_lower)) or \
                     bool(re.search(r'toluene.*(?:good|moderate|partial)', answer_lower))
    checks.append(CheckResult(
        "Compares RED values to rank solvent quality",
        comparison,
        "Found comparative ranking in answer" if comparison else "No ranking comparison found",
    ))

    return checks


def validate_q112(result: TestResult) -> list[CheckResult]:
    """Q1.1.2: EVOH + DMSO selectivity over PE."""
    checks = []
    answer_lower = result.answer.lower()

    # Check 1: Analyzed both EVOH+DMSO and PE+DMSO
    evoh_analyzed = "evoh" in answer_lower and "dmso" in answer_lower
    pe_analyzed = any(term in answer_lower for term in ["pe ", "ldpe", "hdpe", "polyethylene"])
    # Look for RED values for both pairs
    has_evoh_red = bool(re.search(r'evoh.*?(?:red|ra\s*[:=]|1\.0)', answer_lower, re.DOTALL))
    has_pe_red = bool(re.search(r'(?:pe|ldpe|hdpe|polyethylene).*?(?:red|ra\s*[:=]|non.?sol)', answer_lower, re.DOTALL))
    checks.append(CheckResult(
        "Analyzed both EVOH+DMSO and PE+DMSO",
        evoh_analyzed and pe_analyzed and (has_evoh_red or has_pe_red),
        f"EVOH analyzed: {evoh_analyzed}, PE analyzed: {pe_analyzed}, "
        f"EVOH RED: {has_evoh_red}, PE RED: {has_pe_red}",
    ))

    # Check 2: Interprets RED gap as selectivity evidence
    selectivity_discussed = any(term in answer_lower for term in
                                 ["selectiv", "gap", "difference", "non-solvent for pe",
                                  "dissolve evoh", "not dissolve pe", "selective solvent"])
    checks.append(CheckResult(
        "Interprets RED gap as selectivity evidence",
        selectivity_discussed,
        "Selectivity discussed" if selectivity_discussed else "No selectivity discussion found",
    ))

    # Check 3: Notes borderline nature of EVOH+DMSO (RED ≈ 1.03)
    borderline = any(term in answer_lower for term in
                     ["borderline", "near", "≈ 1", "close to 1", "approximately 1",
                      "around 1", "marginal", "1.03", "ambiguous"])
    checks.append(CheckResult(
        "Notes EVOH+DMSO borderline prediction (RED ≈ 1)",
        borderline,
        "Borderline noted" if borderline else "No borderline discussion",
    ))

    # Check 4: Flags HSP limitations, suggests temperature or COSMO-RS
    limitations = any(term in answer_lower for term in
                      ["limitation", "temperature", "cosmo", "experimental",
                       "validation", "hsp cannot", "hsp does not", "static",
                       "temperature-independent", "temperature dependent"])
    checks.append(CheckResult(
        "Flags HSP limitations / suggests COSMO-RS or experimental validation",
        limitations,
        "Limitations flagged" if limitations else "No limitation discussion",
    ))

    return checks


def validate_q113(result: TestResult) -> list[CheckResult]:
    """Q1.1.3: PS selectivity from PE+PVC with THF/cyclohexanone/acetone."""
    checks = []
    pairs = _extract_ml_pairs(result.tool_calls)
    ml_count = _count_ml_calls(result.tool_calls)
    answer_lower = result.answer.lower()

    # Check 1: Analyzed all 3 solvents with all 3 polymers
    solvents_mentioned = _has_solvent_evidence(result.answer, result.tool_calls,
                                                ["THF", "cyclohexanone", "acetone"])
    polymers_mentioned = []
    for poly_term in ["PS", "PE", "PVC"]:
        if poly_term.lower() in answer_lower:
            polymers_mentioned.append(poly_term)
    all_analyzed = len(solvents_mentioned) >= 3 and len(polymers_mentioned) >= 3
    checks.append(CheckResult(
        "Analyzed all 3 solvents × 3 polymers",
        all_analyzed,
        f"Solvents: {solvents_mentioned}, Polymers: {polymers_mentioned}",
    ))

    # Check 2: Presents results in matrix or tabular comparison
    matrix = any(term in answer_lower for term in
                 ["matrix", "table", "|", "comparison", "summary"])
    checks.append(CheckResult(
        "Presents RED comparison (matrix or table)",
        matrix,
        "Matrix/table found" if matrix else "No structured comparison",
    ))

    # Check 3: Recommends cyclohexanone for PS selectivity
    cyclo_rec = "cyclohexanone" in answer_lower and any(
        term in answer_lower for term in
        ["ps", "polystyrene", "dissolve ps", "selective for ps", "best", "recommend"])
    checks.append(CheckResult(
        "Recommends cyclohexanone as PS-selective",
        cyclo_rec,
        "Cyclohexanone recommendation found" if cyclo_rec else "Not recommended",
    ))

    # Check 4: Notes THF dissolves PVC (complicating selectivity)
    thf_pvc = ("thf" in answer_lower and "pvc" in answer_lower and
               any(term in answer_lower for term in
                   ["dissolve", "soluble", "compatible", "low red"]))
    checks.append(CheckResult(
        "Notes THF dissolves PVC (not PS-selective)",
        thf_pvc,
        "THF+PVC noted" if thf_pvc else "Not discussed",
    ))

    # Check 5: Flags HSP limitations or recommends follow-up
    limitations = any(term in answer_lower for term in
                      ["limitation", "experiment", "cosmo", "hsp", "crystallin",
                       "temperature", "caution", "caveat"])
    checks.append(CheckResult(
        "Flags HSP limitations or recommends cross-check",
        limitations,
        "Limitations discussed" if limitations else "No limitations flagged",
    ))

    return checks


def validate_q114(result: TestResult) -> list[CheckResult]:
    """Q1.1.4: 16-polymer × 10+ solvent systematic screening."""
    checks = []
    answer_lower = result.answer.lower()

    # Check 1: Large-scale ML/HSP screening evidence
    has_ml = _has_ml_evidence(result.answer, result.tool_calls)
    checks.append(CheckResult(
        "Uses ML/HSP screening methodology",
        has_ml,
        "ML/HSP evidence found" if has_ml else "No ML evidence",
    ))

    # Check 2: Analyzes multiple polymers (≥8 mentioned with results)
    polymer_targets = ["PE", "PP", "HDPE", "PS", "ABS", "PVC", "PVDF", "PET",
                       "PETG", "PMMA", "PC", "PA6", "PA66", "PSU", "EVOH", "POM"]
    polymers_found = [p for p in polymer_targets if p.lower() in answer_lower]
    checks.append(CheckResult(
        "Analyzes ≥8 polymers",
        len(polymers_found) >= 8,
        f"Found {len(polymers_found)}/16: {polymers_found}",
    ))

    # Check 3: Covers ≥6 solvent classes
    solvent_targets = ["toluene", "hexane", "dcm", "dichloromethane", "chloroform",
                       "thf", "tetrahydrofuran", "acetone", "cyclohexanone",
                       "dmf", "dmso", "nmp", "methanol", "ethyl acetate"]
    solvents_found = [s for s in solvent_targets if s.lower() in answer_lower]
    checks.append(CheckResult(
        "Covers ≥6 solvents spanning chemical classes",
        len(solvents_found) >= 6,
        f"Found {len(solvents_found)} solvents: {solvents_found}",
    ))

    # Check 4: Identifies crystallinity false positive
    crystallinity = any(term in answer_lower for term in
                        ["crystallin", "semi-crystalline", "semicrystalline",
                         "false positive", "amorphous", "crystalline packing",
                         "crystalline barrier", "won't dissolve at rt despite"])
    checks.append(CheckResult(
        "Identifies crystallinity false positive (PE/PP/HDPE)",
        crystallinity,
        "Crystallinity discussed" if crystallinity else "Not discussed",
    ))

    # Check 5: Groups polymers by RT-solubility
    grouping = any(term in answer_lower for term in
                   ["rt-soluble", "room temperature soluble", "soluble subset",
                    "group", "category", "separab", "not soluble at rt",
                    "cannot be dissolved"])
    checks.append(CheckResult(
        "Groups polymers by RT separability",
        grouping,
        "Grouping found" if grouping else "No grouping discussion",
    ))

    # Check 6: Proposes separation sequence
    sequence = any(term in answer_lower for term in
                   ["sequence", "step 1", "step 2", "first", "then",
                    "cascade", "order", "workflow", "pipeline"])
    checks.append(CheckResult(
        "Proposes RT separation sequence",
        sequence,
        "Sequence proposed" if sequence else "No sequence proposed",
    ))

    # Check 7: Recommends COSMO-RS follow-up for non-RT-soluble polymers
    cosmo = any(term in answer_lower for term in
                ["cosmo", "temperature-dependent", "elevated temperature",
                 "further analysis", "follow-up", "experimental"])
    checks.append(CheckResult(
        "Recommends COSMO-RS / experimental follow-up",
        cosmo,
        "Follow-up recommended" if cosmo else "No follow-up recommendation",
    ))

    return checks


VALIDATORS = {
    "1.1.1": validate_q111,
    "1.1.2": validate_q112,
    "1.1.3": validate_q113,
    "1.1.4": validate_q114,
}


# ── Agent Runner ─────────────────────────────────────────────────────────

def _extract_tool_calls(messages: list) -> list[dict]:
    """Extract all tool calls from a message history.

    Includes tool calls made inside subagents: when the orchestrator calls
    task(), the subagent's response is returned as a ToolMessage whose
    content often includes the subagent's own tool call details in prose.
    We parse those by scanning ToolMessage content for predict_solubility_ml
    invocation patterns.
    """
    calls = []
    for msg in messages:
        # Direct tool calls from AI messages
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                })

        # Parse subagent ToolMessage content for ML tool invocations.
        # The statistics-ml subagent calls predict_solubility_ml internally;
        # results appear in the task() ToolMessage in various formats.
        if getattr(msg, "type", "") == "tool" and hasattr(msg, "content"):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Format 1: Raw tool output blocks "**Polymer:** X\n**Solvent:** Y"
            polymer_hits = re.findall(r'\*\*Polymer:\*\*\s*(\S+)', content)
            solvent_hits = re.findall(r'\*\*Solvent:\*\*\s*(.+?)(?:\n|$)', content)
            for p, s in zip(polymer_hits, solvent_hits):
                calls.append({
                    "name": "predict_solubility_ml",
                    "args": {"polymer_name": p.strip(), "solvent_name": s.strip()},
                })

            # Format 2: Synthesized prose mentioning predict_solubility_ml
            for m in re.finditer(
                r'predict_solubility_ml.*?polymer.*?["\'](\w[\w\s]*?)["\'].*?solvent.*?["\'](\w[\w\s]*?)["\']',
                content, re.IGNORECASE | re.DOTALL,
            ):
                calls.append({
                    "name": "predict_solubility_ml",
                    "args": {"polymer_name": m.group(1), "solvent_name": m.group(2)},
                })

            # Format 3: Subagent synthesis — detect solvent-polymer pairs from
            # patterns like "**Toluene:** ... Prediction: SOLUBLE ... Ra: 3.586"
            # or "Toluene: Prediction: SOLUBLE"
            # Look for Prediction/Ra/RED patterns associated with solvent names
            for m in re.finditer(
                r'\*\*([A-Z][\w\s(),-]+?)(?:\s*\([\w\s]+\))?:\*\*\s*.*?'
                r'(?:Prediction|Ra|RED|SOLUBLE|NON-SOLUBLE)',
                content, re.IGNORECASE,
            ):
                solvent_name = m.group(1).strip()
                # Skip false positives (section headers, generic terms)
                if solvent_name.lower() not in ("prediction", "interpretation",
                                                  "conclusion", "explanation",
                                                  "hansen distance", "result",
                                                  "summary", "note", "key"):
                    calls.append({
                        "name": "predict_solubility_ml",
                        "args": {"polymer_name": "INFERRED", "solvent_name": solvent_name},
                    })

            # Format 4: Numbered list items with prediction results
            # e.g. "1. **Toluene:** ... **Prediction:** SOLUBLE"
            for m in re.finditer(
                r'\d+\.\s+\*\*([A-Z][\w\s(),-]+?):\*\*.*?(?:SOLUBLE|NON-SOLUBLE|Ra\s*[:=])',
                content, re.IGNORECASE | re.DOTALL,
            ):
                solvent_name = m.group(1).strip()
                calls.append({
                    "name": "predict_solubility_ml",
                    "args": {"polymer_name": "INFERRED", "solvent_name": solvent_name},
                })
    return calls


def _extract_answer(messages: list) -> str:
    """Extract the final AI answer from message history."""
    for msg in reversed(messages):
        if hasattr(msg, "content") and getattr(msg, "type", "") == "ai" and msg.content:
            content = msg.content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                return "\n".join(parts)
            return str(content)
    return ""


def run_single_test(query_id: str, recursion_limit: int = 250) -> TestResult:
    """Run a single test query through the DISSOLVE agent."""
    from strap.agent import create_dissolve_agent

    query = QUERIES[query_id]
    print(f"\n{'='*70}")
    print(f"  Q{query_id}: {query[:80]}...")
    print(f"{'='*70}")

    # For Q1.1.4 (expert), give more iterations and tool calls
    overrides = None
    if query_id == "1.1.4":
        overrides = {
            "statistics-ml": {"max_tool_calls": 200, "max_iterations": 100},
        }
        recursion_limit = 500

    try:
        agent = create_dissolve_agent(subagent_overrides=overrides)

        t0 = time.time()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"recursion_limit": recursion_limit},
        )
        elapsed = time.time() - t0

        messages = result.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        answer = _extract_answer(messages)

        test_result = TestResult(
            query_id=query_id,
            query=query,
            answer=answer,
            tool_calls=tool_calls,
            elapsed_s=elapsed,
        )

        # Run validation
        validator = VALIDATORS[query_id]
        test_result.checks = validator(test_result)

    except Exception as e:
        test_result = TestResult(
            query_id=query_id,
            query=query,
            answer="",
            tool_calls=[],
            elapsed_s=0,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )

    return test_result


# ── Reporting ────────────────────────────────────────────────────────────

def print_report(results: list[TestResult]):
    """Print a formatted test report."""
    print(f"\n{'='*70}")
    print("  HSP SCREENING TEST REPORT")
    print(f"{'='*70}\n")

    total_passed = 0
    total_checks = 0

    for r in results:
        if r.error:
            print(f"Q{r.query_id}: ERROR")
            print(f"  {r.error[:200]}")
            print()
            continue

        status = "PASS" if r.passed == r.total else "PARTIAL" if r.passed > 0 else "FAIL"
        ml_calls = _count_ml_calls(r.tool_calls)
        print(f"Q{r.query_id}: {status} ({r.passed}/{r.total} checks, "
              f"{r.score_pct:.0f}%) | {ml_calls} ML calls | {r.elapsed_s:.1f}s")

        for c in r.checks:
            mark = "PASS" if c.passed else "FAIL"
            print(f"  [{mark}] {c.name}")
            if c.detail:
                print(f"         {c.detail}")

        total_passed += r.passed
        total_checks += r.total
        print()

    pct = (total_passed / total_checks * 100) if total_checks else 0
    print(f"{'='*70}")
    print(f"  OVERALL: {total_passed}/{total_checks} checks passed ({pct:.0f}%)")
    print(f"{'='*70}")

    # Save detailed results
    output_dir = Path(__file__).resolve().parent.parent / "architecture" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"hsp_screening_{ts}.json"

    report_data = []
    for r in results:
        report_data.append({
            "query_id": r.query_id,
            "query": r.query[:200],
            "ml_calls": _count_ml_calls(r.tool_calls),
            "total_tool_calls": len(r.tool_calls),
            "elapsed_s": r.elapsed_s,
            "passed": r.passed,
            "total": r.total,
            "score_pct": r.score_pct,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks],
            "answer_length": len(r.answer),
            "answer_preview": r.answer[:2000],
            "error": r.error,
        })

    with open(detail_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nDetailed results saved to: {detail_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HSP Screening Agent Tests")
    parser.add_argument("--query", default="all",
                        help="Comma-separated query IDs (e.g. '1,2') or 'all'")
    parser.add_argument("--parallel", action="store_true",
                        help="Run tests in parallel (separate agent per test)")
    args = parser.parse_args()

    if args.query == "all":
        query_ids = list(QUERIES.keys())
    else:
        query_ids = [f"1.1.{q.strip()}" if "." not in q else q for q in args.query.split(",")]

    print(f"Running {len(query_ids)} test(s): {query_ids}")
    print(f"Mode: {'parallel' if args.parallel else 'sequential'}")

    if args.parallel and len(query_ids) > 1:
        results = []
        with ThreadPoolExecutor(max_workers=min(len(query_ids), 4)) as pool:
            futures = {pool.submit(run_single_test, qid): qid for qid in query_ids}
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(TestResult(
                        query_id=qid, query=QUERIES[qid], answer="",
                        tool_calls=[], elapsed_s=0,
                        error=f"Thread error: {e}",
                    ))
        results.sort(key=lambda r: r.query_id)
    else:
        results = [run_single_test(qid) for qid in query_ids]

    print_report(results)


if __name__ == "__main__":
    main()
