"""Concrete reward scorers for STRAP analysis results.

Each scorer reads an :class:`~strap.eval.reward.Episode` and returns one or
more normalized [0,1] :class:`~strap.eval.reward.RewardComponent` values. All
grounding is done against the deterministic v10 engines (no API calls):

- ``PhysicalValidityScorer``  (GATE): atmospheric-pressure feasibility.
- ``GroundingScorer``:               claims recomputed against the solubility engine.
- ``RichnessScorer``:                thoroughness / exploration breadth.
- ``CompletenessScorer``:            coverage of what the query asked for.
- ``EfficiencyScorer``:              cost relative to a task-shaped budget.

The result payload is treated structurally: a scorer that finds no relevant
fields marks its component ``applicable=False`` (excluded from the weighted
mean) rather than penalizing — a simple solubility lookup should not be
punished for lacking a Pareto frontier.
"""

from __future__ import annotations

import re
from typing import Any

from strap.eval.reward import Episode, RewardComponent

# Tolerance for "the recomputed number matches the claimed number".
_REL_TOL = 0.15
_SOLUBLE_THRESHOLD_PCT = 1.0  # a target claimed to dissolve should exceed this


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    steps = result.get("steps")
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _boiling_point(solvent: str) -> float | None:
    from strap.solubility import get_boiling_point

    try:
        return get_boiling_point(solvent)
    except Exception:
        return None


def _solubility(polymer: str, solvent: str, temp_c: float) -> float | None:
    from strap.solubility import get_solubility

    try:
        return get_solubility(polymer, solvent, temp_c, method="auto")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Physical validity — the gate
# ---------------------------------------------------------------------------

class PhysicalValidityScorer:
    """Every recommended dissolution step must run at/below the solvent's
    boiling point at 1 atm. Score = fraction of steps that are atmospherically
    feasible; this is a GATE, so any violation collapses the reward.

    A small tolerance above BP is allowed (some processes run at a gentle
    reflux); beyond that the step is a hard physical violation.
    """

    name = "physical_validity"

    def __init__(self, *, bp_tolerance_c: float = 2.0) -> None:
        self._bp_tol = bp_tolerance_c

    def score(self, episode: Episode) -> RewardComponent:
        steps = _steps(episode.result)
        if not steps:
            return RewardComponent(
                self.name, 1.0, "no temperature-bearing steps to check",
                weight=3.0, applicable=False, is_gate=True,
            )
        violations: list[str] = []
        checked = 0
        for step in steps:
            solvent = str(step.get("solvent") or "").strip()
            temp = _num(step.get("temperature_c"))
            if not solvent or temp is None:
                continue
            bp = _boiling_point(solvent)
            if bp is None:
                continue
            checked += 1
            if temp > bp + self._bp_tol:
                violations.append(f"{step.get('polymer', '?')}/{solvent} @ {temp:.0f}C > BP {bp:.0f}C")
        if checked == 0:
            return RewardComponent(
                self.name, 1.0, "no solvents with known boiling points",
                weight=3.0, applicable=False, is_gate=True,
            )
        score = 1.0 - len(violations) / checked
        detail = "all steps atmospherically feasible" if not violations else \
            f"{len(violations)}/{checked} steps above BP: " + "; ".join(violations[:3])
        return RewardComponent(self.name, score, detail, weight=3.0, is_gate=True)


# ---------------------------------------------------------------------------
# 2. Grounding — claims recomputed against the engine (anti-hallucination)
# ---------------------------------------------------------------------------

class GroundingScorer:
    """Recompute each step's chemistry from the solubility engine and check the
    claim is consistent. Catches fabricated recommendations: a step claiming a
    polymer dissolves in a solvent where the engine says it is insoluble, or a
    positive selectivity that the data contradicts, scores low.
    """

    name = "grounding"

    def score(self, episode: Episode) -> RewardComponent:
        result = episode.result
        steps = _steps(result)
        if not steps:
            # Non-route results (e.g. a point solubility lookup): validate that
            # any claimed solubility value matches the engine.
            return self._score_point_result(episode)

        sequence = [str(p) for p in (result.get("best_sequence") or result.get("polymers") or [])]
        checks = 0
        passed = 0
        problems: list[str] = []
        for index, step in enumerate(steps):
            polymer = str(step.get("polymer") or "").strip()
            solvent = str(step.get("solvent") or "").strip()
            temp = _num(step.get("temperature_c"))
            if not (polymer and solvent and temp is not None):
                continue
            target_sol = _solubility(polymer, solvent, temp)
            if target_sol is None:
                continue
            checks += 1
            # (a) a recovered polymer should actually be soluble at its step.
            if target_sol >= _SOLUBLE_THRESHOLD_PCT:
                passed += 1
            else:
                problems.append(f"{polymer}/{solvent}@{temp:.0f}C: engine solubility {target_sol:.1f}% (claimed recovered)")
            # (b) if a positive selectivity is claimed, the target should be
            # more soluble than the polymers still in the stream.
            claimed_sel = _num(step.get("selectivity_pct"))
            others = sequence[index + 1:] if sequence else []
            if claimed_sel is not None and claimed_sel > 0 and others:
                other_sols = [s for s in (_solubility(o, solvent, temp) for o in others) if s is not None]
                if other_sols and target_sol <= max(other_sols):
                    checks += 1
                    problems.append(f"{polymer}/{solvent}: claims +selectivity but engine shows a co-dissolving polymer")
                elif other_sols:
                    checks += 1
                    passed += 1
        if checks == 0:
            return RewardComponent(self.name, 1.0, "no recomputable claims", weight=2.5, applicable=False)
        score = passed / checks
        detail = "all recomputed claims consistent with the engine" if not problems else \
            f"{checks - passed}/{checks} inconsistent: " + "; ".join(problems[:2])
        return RewardComponent(self.name, score, detail, weight=2.5)

    def _score_point_result(self, episode: Episode) -> RewardComponent:
        result = episode.result
        polymer = str(result.get("polymer_name") or result.get("polymer") or episode.context.get("polymer") or "").strip()
        solvent = str(result.get("solvent_name") or result.get("solvent") or episode.context.get("solvent") or "").strip()
        temp = _num(result.get("temperature_c") or episode.context.get("temperature_c"))
        claimed = _num(result.get("solubility_pct") or result.get("predicted_solubility"))
        if not (polymer and solvent and temp is not None and claimed is not None):
            return RewardComponent(self.name, 1.0, "no recomputable point claim", weight=2.5, applicable=False)
        engine = _solubility(polymer, solvent, temp)
        if engine is None:
            return RewardComponent(self.name, 0.5, "claimed pair has no engine data", weight=2.5)
        denom = max(abs(engine), 1.0)
        rel_err = abs(claimed - engine) / denom
        score = max(0.0, 1.0 - rel_err / _REL_TOL) if rel_err <= _REL_TOL else max(0.0, 1.0 - rel_err)
        detail = f"claimed {claimed:.2f}% vs engine {engine:.2f}% (rel err {rel_err:.0%})"
        return RewardComponent(self.name, min(1.0, score), detail, weight=2.5)


# ---------------------------------------------------------------------------
# 3. Richness — thoroughness / exploration breadth
# ---------------------------------------------------------------------------

class RichnessScorer:
    """Reward breadth of exploration, not just a single argmax answer:
    multiple viable solvent candidates per polymer, alternative sequences, and
    Pareto frontier size. This is the dimension a learner should push to get
    "more thorough / more exploratory" results.
    """

    name = "richness"

    def __init__(self, *, target_candidates_per_polymer: int = 3,
                 target_frontier_points: int = 4) -> None:
        self._target_cands = target_candidates_per_polymer
        self._target_frontier = target_frontier_points

    def score(self, episode: Episode) -> RewardComponent:
        result = episode.result
        signals: list[float] = []
        details: list[str] = []

        candidates = result.get("polymer_solvent_candidates")
        if isinstance(candidates, dict) and candidates:
            per_polymer = []
            for solvents in candidates.values():
                n = len({str((s or {}).get("solvent") if isinstance(s, dict) else s) for s in (solvents or [])})
                per_polymer.append(min(1.0, n / self._target_cands))
            if per_polymer:
                mean = sum(per_polymer) / len(per_polymer)
                signals.append(mean)
                details.append(f"{mean:.0%} of per-polymer candidate depth (target {self._target_cands})")

        topk = result.get("top_k_sequences")
        if isinstance(topk, list) and topk:
            signals.append(min(1.0, len(topk) / 3.0))
            details.append(f"{len(topk)} ranked sequences")

        # Pareto frontier richness (landscape-consistent frontier or points).
        frontier = result.get("frontier") or result.get("points") or result.get("pareto_points")
        if isinstance(frontier, list) and frontier:
            signals.append(min(1.0, len(frontier) / self._target_frontier))
            details.append(f"{len(frontier)} frontier points")

        if not signals:
            return RewardComponent(self.name, 1.0, "richness not applicable to this result", weight=1.5, applicable=False)
        score = sum(signals) / len(signals)
        return RewardComponent(self.name, score, "; ".join(details), weight=1.5)


# ---------------------------------------------------------------------------
# 4. Completeness — coverage of the request
# ---------------------------------------------------------------------------

_METRIC_PATTERNS = {
    "msp": r"\bmsp\b|minimum selling price",
    "gwp": r"\bgwp\b|emissions?|life cycle|lca",
    "cost": r"\bcost\b|capex|opex|tco",
    "selectivity": r"selectivit",
    "circularity": r"circularit",
}


class CompletenessScorer:
    """Did the result cover what the query asked for — every named polymer, and
    every requested metric (MSP, GWP, cost, selectivity, circularity)?"""

    name = "completeness"

    def score(self, episode: Episode) -> RewardComponent:
        query = episode.query.lower()
        result = episode.result
        requested_polymers = episode.context.get("polymers") or self._polymers_in(query)
        if isinstance(requested_polymers, str):
            requested_polymers = [requested_polymers]
        # a point lookup names one polymer in context.polymer
        if not requested_polymers and episode.context.get("polymer"):
            requested_polymers = [str(episode.context["polymer"])]
        covered = self._covered_polymers(result)

        parts: list[float] = []
        details: list[str] = []
        if requested_polymers:
            hit = sum(1 for p in requested_polymers if p.upper() in covered)
            parts.append(hit / len(requested_polymers))
            details.append(f"{hit}/{len(requested_polymers)} polymers covered")

        requested_metrics = [m for m, pat in _METRIC_PATTERNS.items() if re.search(pat, query)]
        if requested_metrics:
            flat = _flatten_keys(result)
            hit = sum(1 for m in requested_metrics if any(m in k for k in flat))
            parts.append(hit / len(requested_metrics))
            details.append(f"{hit}/{len(requested_metrics)} requested metrics present")

        if not parts:
            return RewardComponent(self.name, 1.0, "no explicit coverage requirements", weight=1.5, applicable=False)
        score = sum(parts) / len(parts)
        return RewardComponent(self.name, score, "; ".join(details), weight=1.5)

    @staticmethod
    def _polymers_in(query: str) -> list[str]:
        known = ["LDPE", "HDPE", "LLDPE", "EVOH", "PETG", "PET", "PP", "PS", "PVC", "PC", "PES", "NYLON6", "NYLON66", "PE"]
        found: list[str] = []
        upper = query.upper()
        for p in known:
            if re.search(rf"\b{re.escape(p)}\b", upper) and not any(p in f and p != f for f in found):
                found.append(p)
        return found

    @staticmethod
    def _covered_polymers(result: dict[str, Any], context: dict[str, Any] | None = None) -> set[str]:
        out: set[str] = set()
        for key in ("polymers", "best_sequence", "supported_polymers", "polymers_analyzed"):
            vals = result.get(key)
            if isinstance(vals, list):
                out.update(str(v).upper() for v in vals)
        # singular fields on point/lookup results
        for key in ("polymer_name", "polymer", "target", "target_polymer", "other_polymer"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                out.add(val.upper())
        for step in _steps(result):
            for key in ("polymer", "target"):
                if step.get(key):
                    out.add(str(step[key]).upper())
        cands = result.get("polymer_solvent_candidates")
        if isinstance(cands, dict):
            out.update(str(k).upper() for k in cands)
        return out


def _flatten_keys(obj: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k).lower())
            keys |= _flatten_keys(v, str(k).lower())
    elif isinstance(obj, list):
        for item in obj[:20]:
            keys |= _flatten_keys(item, prefix)
    return keys


# ---------------------------------------------------------------------------
# 5. Efficiency — cost relative to a task-shaped budget
# ---------------------------------------------------------------------------

class EfficiencyScorer:
    """Score cost (tool calls) against a budget that scales with task
    complexity, so a thorough multi-stage answer is not penalized for costing
    more than a lookup. Applicable only when a ledger is present."""

    name = "efficiency"

    def score(self, episode: Episode) -> RewardComponent:
        ledger = episode.ledger or {}
        calls = ledger.get("tool_calls")
        if calls is None:
            return RewardComponent(self.name, 1.0, "no cost ledger", weight=1.0, applicable=False)
        budget = float(episode.context.get("tool_call_budget", 12))
        used = float(calls)
        score = max(0.0, 1.0 - used / (2.0 * budget))  # 2x budget -> 0
        return RewardComponent(self.name, score, f"{used:.0f} tool calls vs budget {budget:.0f}", weight=1.0)


# ---------------------------------------------------------------------------
# 6. Separation quality — how cleanly the route separates
# ---------------------------------------------------------------------------

class SeparationQualityScorer:
    """Score the *quality* of a separation route by its worst step: the
    minimum selectivity across steps, normalized against a target. This is the
    dimension that lets exploration distinguish a clean route (min-sel ~35%)
    from a barely-viable one (min-sel ~5%) — both are physical, one is better.
    """

    name = "separation_quality"

    def __init__(self, *, target_min_selectivity_pct: float = 30.0) -> None:
        self._target = target_min_selectivity_pct

    def score(self, episode: Episode) -> RewardComponent:
        result = episode.result
        min_sel = _num(result.get("min_selectivity"))
        if min_sel is None:
            sels = [
                _num(step.get("selectivity_pct"))
                for step in _steps(result)
                if _num(step.get("selectivity_pct")) is not None
            ]
            min_sel = min(sels) if sels else None
        if min_sel is None:
            return RewardComponent(
                self.name, 1.0, "no selectivity information (not a separation route)",
                weight=2.0, applicable=False,
            )
        score = max(0.0, min(1.0, min_sel / self._target))
        return RewardComponent(
            self.name, score,
            f"min selectivity {min_sel:.1f}% vs target {self._target:.0f}%",
            weight=2.0,
        )


# ---------------------------------------------------------------------------
# 7. Optimization outcome — did the optimization deliver a feasible result?
# ---------------------------------------------------------------------------

class OptimizationOutcomeScorer:
    """Score whether an optimization/Pareto episode actually produced the
    requested deliverable.

    The separation dimensions (physical-validity gate, selectivity) key off
    dissolution *steps*; an optimization result has none, so without this
    scorer an honest-but-infeasible optimization is *ungated and unpenalized*
    and can outscore a real feasible frontier — which would teach any RL loop
    over optimization decisions to prefer infeasibility. This dimension closes
    that gap: it is the optimization analog of separation quality.

    Not a gate. An infeasible result that honestly reports the infeasibility is
    a legitimate (if low-value) outcome, not a physical violation to be
    collapsed — so it scores low, not zero. Frontier *size* is left to
    RichnessScorer; this scorer asks only "was a feasible deliverable produced,
    with the requested economics on its points".
    """

    name = "optimization_outcome"

    def __init__(self, *, infeasible_floor: float = 0.1) -> None:
        self._floor = infeasible_floor

    def score(self, episode: Episode) -> RewardComponent:
        result = episode.result
        analysis = str(result.get("analysis_type") or "").strip().lower()
        frontier = result.get("frontier") or result.get("points") or result.get("pareto_points") or []
        feasible_points = [p for p in frontier if isinstance(p, dict)]
        looks_like_optimization = bool(analysis) or bool(feasible_points) or bool(result.get("infeasible"))
        if not looks_like_optimization:
            return RewardComponent(
                self.name, 1.0, "not an optimization result", weight=2.0, applicable=False,
            )
        if analysis == "infeasible" or result.get("infeasible") or not feasible_points:
            reason = str(result.get("failure_reason") or "no feasible points")
            return RewardComponent(
                self.name, self._floor,
                f"deliverable absent (infeasible: {reason})", weight=2.0,
            )
        has_cost = any(("total_cost" in p) or ("cost" in p) for p in feasible_points)
        has_emissions = any(("emissions" in p) or ("gwp" in p) for p in feasible_points)
        if analysis == "point_optimum" or (has_cost and has_emissions):
            score = 1.0
        elif has_cost or has_emissions:
            score = 0.7
        else:
            score = 0.5
        return RewardComponent(
            self.name, score,
            f"{len(feasible_points)} feasible point(s); cost={has_cost} emissions={has_emissions}",
            weight=2.0,
        )
