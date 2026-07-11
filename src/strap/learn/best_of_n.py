"""Best-of-N exploratory analysis (the cheapest genuinely-exploratory loop).

Instead of one greedy engine run, generate N *diverse* candidate analyses by
varying the harness's decision knobs (temperature, solvent-pool exclusions,
candidate depth), score every candidate with the physically-gated reward
model, and keep the best feasible one. The reward gate guarantees exploration
can never surface an unphysical "winner"; the richness dimension rewards
candidates that expose more of the design space.

Engine calls are deterministic, so candidates are memoized process-wide: a
bandit loop that revisits the same (query, knobs) pays for the engine once.

Generic entry point::

    outcome = best_of_n(specs, produce_fn, reward_model)

Separation-specific explorer::

    outcome = explore_separation(["LDPE", "PP", "PS"], n=6, seed=7)
    outcome.best.episode        # highest-reward feasible candidate
    outcome.solvent_diversity   # unique solvents surfaced across candidates
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from strap.eval.reward import Episode, RewardModel, RewardResult
from strap.eval.reward import default_reward_model as _default_reward_model


# ---------------------------------------------------------------------------
# Generic best-of-N
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    spec: dict[str, Any]
    episode: Episode
    reward: RewardResult


@dataclass(frozen=True)
class ExplorationOutcome:
    candidates: tuple[Candidate, ...]     # reward-ranked, best first
    n_requested: int

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    @property
    def best_feasible(self) -> Candidate | None:
        return next((c for c in self.candidates if c.reward.feasible), None)

    @property
    def solvent_diversity(self) -> int:
        """Unique solvents surfaced across all candidates' steps/candidate lists."""
        solvents: set[str] = set()
        for cand in self.candidates:
            result = cand.episode.result
            for step in result.get("steps") or []:
                if isinstance(step, dict) and step.get("solvent"):
                    solvents.add(str(step["solvent"]).lower())
            pools = result.get("polymer_solvent_candidates") or {}
            if isinstance(pools, dict):
                for entries in pools.values():
                    for entry in entries or []:
                        name = entry.get("solvent") if isinstance(entry, dict) else entry
                        if name:
                            solvents.add(str(name).lower())
        return len(solvents)

    def summary(self) -> dict[str, Any]:
        return {
            "n_requested": self.n_requested,
            "n_scored": len(self.candidates),
            "best_reward": self.best.reward.scalar if self.candidates else None,
            "best_feasible_reward": self.best_feasible.reward.scalar if self.best_feasible else None,
            "rewards": [c.reward.scalar for c in self.candidates],
            "solvent_diversity": self.solvent_diversity,
        }


def best_of_n(
    specs: Sequence[dict[str, Any]],
    produce: Callable[[dict[str, Any]], Episode],
    reward_model: RewardModel | None = None,
) -> ExplorationOutcome:
    """Produce and score one episode per spec; return reward-ranked candidates."""
    model = reward_model or _default_reward_model()
    candidates = []
    for spec in specs:
        episode = produce(spec)
        candidates.append(Candidate(spec=dict(spec), episode=episode, reward=model.score(episode)))
    candidates.sort(key=lambda c: c.reward.scalar, reverse=True)
    return ExplorationOutcome(candidates=tuple(candidates), n_requested=len(specs))


# ---------------------------------------------------------------------------
# Separation-specific exploration
# ---------------------------------------------------------------------------

_ENGINE_CACHE: dict[str, dict[str, Any]] = {}


def _run_separation_engine(polymers: tuple[str, ...], temperature: float,
                           top_k: int, excluded: tuple[str, ...]) -> dict[str, Any]:
    key = json.dumps([sorted(polymers), temperature, top_k, sorted(excluded)])
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key]
    from strap.tools.advanced_separation import plan_sequential_separation

    raw = plan_sequential_separation(
        ",".join(polymers),
        temperature=temperature,
        top_k_solvents=top_k,
        excluded_solvents=",".join(excluded),
    )
    doc = json.loads(raw) if isinstance(raw, str) else raw
    data = doc.get("data", doc) if isinstance(doc, dict) else {}
    _ENGINE_CACHE[key] = data if isinstance(data, dict) else {}
    return _ENGINE_CACHE[key]


def clear_engine_cache() -> None:
    _ENGINE_CACHE.clear()


def _normalize(data: dict[str, Any], temperature: float) -> dict[str, Any]:
    steps = []
    for step in data.get("steps") or []:
        if isinstance(step, dict):
            steps.append({
                "polymer": step.get("polymer") or step.get("target"),
                "solvent": step.get("solvent"),
                "temperature_c": step.get("temperature_c", temperature),
                "selectivity_pct": step.get("selectivity_pct", step.get("selectivity")),
            })
    return {
        "polymers": data.get("polymers_analyzed") or data.get("polymers"),
        "best_sequence": data.get("best_sequence"),
        "steps": steps,
        "polymer_solvent_candidates": data.get("polymer_solvent_candidates"),
        "top_k_sequences": data.get("top_k_sequences"),
        "min_selectivity": data.get("min_selectivity"),
    }


def separation_candidate_specs(
    polymers: Sequence[str],
    *,
    n: int,
    temperature_ceiling_c: float = 140.0,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministic, diverse knob settings for N separation candidates.

    Spec 0 is always the greedy default (ceiling temperature, standard depth,
    no exclusions) so best-of-1 == today's behavior. Later specs sweep
    temperature and progressively exclude the previously-dominant solvents,
    forcing the engine to surface non-obvious alternatives.
    """
    import random

    rng = random.Random(seed)
    temps = [temperature_ceiling_c, temperature_ceiling_c - 20.0,
             temperature_ceiling_c - 40.0, temperature_ceiling_c - 10.0]
    specs: list[dict[str, Any]] = [{
        "temperature": temperature_ceiling_c, "top_k": 5, "excluded": (),
    }]
    if n >= 2:
        # same conditions, deeper candidate pool — pure thoroughness variant
        specs.append({"temperature": temperature_ceiling_c, "top_k": 8, "excluded": ()})
    dominant: list[str] = []
    while len(specs) < n:
        index = len(specs)
        temperature = temps[index % len(temps)]
        # discover dominant solvents from the default run to exclude them
        if not dominant:
            base = _run_separation_engine(tuple(polymers), temperature_ceiling_c, 5, ())
            dominant = [
                str(step.get("solvent"))
                for step in (base.get("steps") or [])
                if isinstance(step, dict) and step.get("solvent")
            ]
        n_excl = min(index // 2 + (1 if index % 2 else 0), len(dominant))
        excluded = tuple(dominant[:n_excl]) if n_excl else ()
        top_k = rng.choice([5, 8]) if index > 1 else 8
        spec = {"temperature": temperature, "top_k": top_k, "excluded": excluded}
        if spec not in specs:
            specs.append(spec)
        else:  # perturb temperature to keep specs unique
            specs.append({**spec, "temperature": temperature - 5.0 * index})
    return specs[:n]


def explore_separation(
    polymers: Sequence[str],
    *,
    n: int,
    temperature_ceiling_c: float = 140.0,
    seed: int | None = None,
    reward_model: RewardModel | None = None,
) -> ExplorationOutcome:
    """Best-of-N over the separation engine for one query. Zero API calls."""
    polymers = [str(p).upper() for p in polymers]
    query = (f"Find the best separation sequence for {', '.join(polymers)} "
             f"below {temperature_ceiling_c:.0f} C.")
    specs = separation_candidate_specs(
        polymers, n=n, temperature_ceiling_c=temperature_ceiling_c, seed=seed)

    def produce(spec: dict[str, Any]) -> Episode:
        data = _run_separation_engine(
            tuple(polymers), float(spec["temperature"]), int(spec["top_k"]),
            tuple(spec["excluded"]))
        return Episode(
            query=query,
            result=_normalize(data, float(spec["temperature"])),
            context={"polymers": list(polymers), "tool_call_budget": 12},
            ledger={"tool_calls": n},  # exploration cost: N engine runs for this answer
            label=f"T={spec['temperature']:.0f} k={spec['top_k']} excl={len(spec['excluded'])}",
        )

    return best_of_n(specs, produce, reward_model)
