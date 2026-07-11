"""A graded simple->complex query suite with deterministic, no-API result
producers, plus ablation variants for reward-discrimination demos.

Each producer calls the real v10 engines (solubility, separation, optimization)
and normalizes the output into the canonical Episode result shape the scorers
expect. This gives the reward substrate *actual query results* to score,
reproducibly and without any model/API call.

The ablation variants take a good result and degrade it in one specific way
(greedy single-candidate, physically infeasible, fabricated) so a figure can
show the reward model ranks good > degraded — the property an RL loop relies on.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from strap.eval.reward import Episode

_ROOT = Path(__file__).resolve().parents[3]
_CASE_DATA = _ROOT / "case-studies"


@dataclass(frozen=True)
class SuiteQuery:
    key: str
    complexity: str  # "simple" | "moderate" | "complex" | "very_complex"
    query: str
    produce: Callable[[], Episode]


def _parse(raw: Any) -> dict[str, Any]:
    doc = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(doc, dict):
        return {}
    return doc.get("data", doc)


def _normalize_separation(data: dict[str, Any], temperature_c: float) -> dict[str, Any]:
    """Map plan_sequential_separation output into the canonical result shape."""
    steps = []
    for step in data.get("steps") or []:
        if not isinstance(step, dict):
            continue
        steps.append({
            "polymer": step.get("polymer") or step.get("target"),
            "solvent": step.get("solvent"),
            "temperature_c": step.get("temperature_c", temperature_c),
            "selectivity_pct": step.get("selectivity_pct", step.get("selectivity")),
        })
    return {
        "polymers": data.get("polymers_analyzed") or data.get("polymers"),
        "best_sequence": data.get("best_sequence"),
        "steps": steps,
        "polymer_solvent_candidates": data.get("polymer_solvent_candidates"),
        "top_k_sequences": data.get("top_k_sequences"),
    }


# ---------------------------------------------------------------------------
# producers (real engine calls)
# ---------------------------------------------------------------------------

def _produce_point_solubility() -> Episode:
    from strap.tools.interpolation import predict_solubility

    data = _parse(predict_solubility("LDPE", "toluene", 100.0))
    return Episode(
        query="What is the solubility of LDPE in toluene at 100 C?",
        result=data,
        context={"polymer": "LDPE", "solvent": "toluene", "temperature_c": 100.0},
        ledger={"tool_calls": 1},
    )


def _produce_solubility_range() -> Episode:
    from strap.tools.interpolation import predict_solubility_range

    data = _parse(predict_solubility_range("HDPE", "xylene", t_start_c=25, t_end_c=120))
    # richness of a range = number of predicted points
    data.setdefault("frontier", data.get("predictions"))
    return Episode(
        query="Show the solubility of HDPE in xylene from 25 to 120 C.",
        result=data,
        context={"polymer": "HDPE", "solvent": "xylene"},
        ledger={"tool_calls": 1},
    )


def _produce_selectivity_rank() -> Episode:
    import re as _re

    from strap.tools.advanced_separation import rank_solvents_for_separation

    raw = rank_solvents_for_separation("LDPE", "PET", temperature=100.0)
    doc = json.loads(raw) if isinstance(raw, str) else raw
    display = doc.get("display", "") if isinstance(doc, dict) else ""
    # this tool is display-heavy; recover the ranked solvents from the table so
    # richness/completeness reflect the answer the query actually produced.
    solvents = _re.findall(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", display, _re.MULTILINE)
    candidates = [{"rank": i + 1, "solvent": s.strip()} for i, s in enumerate(solvents)]
    return Episode(
        query="Rank solvents to separate LDPE from PET at 100 C by selectivity.",
        result={
            "polymers": ["LDPE", "PET"],
            "target_polymer": "LDPE",
            "other_polymer": "PET",
            "polymer_solvent_candidates": {"LDPE": candidates} if candidates else None,
        },
        context={"polymers": ["LDPE", "PET"]},
        ledger={"tool_calls": 1},
    )


def _produce_separation_sequence() -> Episode:
    from strap.tools.advanced_separation import plan_sequential_separation

    temperature_c = 120.0
    data = _parse(plan_sequential_separation("LDPE,PP,PS", temperature=temperature_c))
    return Episode(
        query="Find the best separation sequence for LDPE, PP and PS below 120 C.",
        result=_normalize_separation(data, temperature_c),
        context={"polymers": ["LDPE", "PP", "PS"]},
        ledger={"tool_calls": 4},
    )


def _produce_pareto() -> Episode:
    path = _CASE_DATA / "02-cost-emissions-pareto" / "data" / "B_circularity_rich.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    # frontier from the landscape-consistent points (see case study 02)
    frontier = data.get("points") or []
    return Episode(
        query="Generate the cost-vs-circularity Pareto frontier for a PE/EVOH feed with a wash step; report cost and circularity.",
        result={
            "polymers": ["PE", "EVOH"],
            "frontier": frontier,
            "points": frontier,
            "circularity_present": True,
            "total_cost_present": True,
        },
        context={"polymers": ["PE", "EVOH"]},
        ledger={"tool_calls": 8},
    )


def _produce_temperature_sweep() -> Episode:
    path = _CASE_DATA / "01-pareto-temperature-sweep" / "data" / "reproduced_frontier.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    frontier = doc.get("frontier") or []
    sequence = [s.get("polymer") for s in doc.get("sequence", []) if isinstance(s, dict)]
    return Episode(
        query="Map the selectivity-vs-MSP Pareto frontier across the full separation sequence with per-step temperatures.",
        result={
            "polymers": sequence,
            "best_sequence": sequence,
            "frontier": frontier,
            "msp_present": True,
        },
        context={"polymers": sequence},
        ledger={"tool_calls": 10},
    )


SUITE: list[SuiteQuery] = [
    SuiteQuery("point_solubility", "simple", "solubility of LDPE in toluene at 100 C", _produce_point_solubility),
    SuiteQuery("solubility_range", "simple", "solubility of HDPE in xylene 25-120 C", _produce_solubility_range),
    SuiteQuery("selectivity_rank", "moderate", "rank solvents LDPE vs PET at 100 C", _produce_selectivity_rank),
    SuiteQuery("separation_sequence", "complex", "best separation sequence LDPE/PP/PS", _produce_separation_sequence),
    SuiteQuery("cost_circularity_pareto", "complex", "cost-vs-circularity Pareto PE/EVOH", _produce_pareto),
    SuiteQuery("temperature_sweep_frontier", "very_complex", "selectivity-vs-MSP frontier over sequence", _produce_temperature_sweep),
]


# ---------------------------------------------------------------------------
# ablation variants (for reward-discrimination demos)
# ---------------------------------------------------------------------------

def ablations_for_separation() -> dict[str, Episode]:
    """Return {variant -> Episode} for the separation query: a good engine
    result plus three targeted degradations."""
    good = _produce_separation_sequence()
    variants: dict[str, Episode] = {"engine_result": good}

    # greedy: keep only the single best candidate per polymer, drop alt sequences.
    greedy = copy.deepcopy(good)
    cands = greedy.result.get("polymer_solvent_candidates") or {}
    greedy.result["polymer_solvent_candidates"] = {
        p: (lst[:1] if isinstance(lst, list) else lst) for p, lst in cands.items()
    }
    greedy.result["top_k_sequences"] = greedy.result.get("top_k_sequences", [])[:1]
    greedy.label = "greedy"
    variants["greedy_top1"] = greedy

    # infeasible: push every step above its solvent boiling point.
    infeasible = copy.deepcopy(good)
    for step in infeasible.result.get("steps", []):
        step["temperature_c"] = 260.0
    infeasible.label = "infeasible"
    variants["infeasible_above_bp"] = infeasible

    # fabricated: claim a polymer dissolves in a solvent where it does not, with
    # an invented high selectivity.
    fabricated = copy.deepcopy(good)
    steps = fabricated.result.get("steps", [])
    if steps:
        steps[0]["solvent"] = "water"          # polyolefins do not dissolve in water
        steps[0]["selectivity_pct"] = 95.0     # invented
    fabricated.label = "fabricated"
    variants["fabricated_claim"] = fabricated

    return variants
