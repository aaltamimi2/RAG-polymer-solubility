"""Reward / evaluation substrate for the STRAP analysis harness.

This is the scoring layer that turns a query + its structured result into a
rigorous, decomposed reward. It is designed as *library* code so any learning
loop can import it:

- **Contextual bandit / policy over harness decisions (RL option 2):** the
  loop picks an action (top_k, solver rung, solvent pool, ...), the harness
  produces a result, and ``RewardModel.score(episode).scalar`` is the reward.
- **Search-with-value over the analysis tree (RL option 3):** score partial or
  terminal states the same way; ``RewardResult`` exposes per-component signals
  for value shaping.
- **Best-of-N exploration + offline eval:** score N candidate results and rank.

Design principles:

1. **Physically grounded.** Reward is computed from structured outputs and the
   deterministic v10 engines (solubility, boiling points), not from an LLM's
   self-report. No API calls.
2. **Hard physical gates, not soft rewards.** A route that recommends a solvent
   above its atmospheric boiling point is not "slightly worse" — it is invalid.
   Gate scorers set ``feasible=False`` and collapse the reward, so a learner
   cannot reward-hack by producing rich-but-unphysical answers.
3. **Decomposed and inspectable.** Every component is a normalized [0,1] score
   with a human-readable ``detail`` string, so a reward can be audited (and put
   in a figure) rather than trusted as an opaque scalar.
4. **Composable.** Scorers are small objects; ``RewardModel`` weights and
   combines them. Add or reweight dimensions without touching the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Episode:
    """One scorable unit: a query and the structured result produced for it.

    ``result`` is the structured payload (a subagent ``<STRUCTURED_RESULT>``
    dict, a tool output ``data`` block, or an RL rollout's terminal state).
    ``context`` carries request-derived facts a scorer needs (requested
    polymers, requested metrics/artifacts, feed, etc.). ``ledger`` optionally
    carries cost signals (tool-call count, tokens) for the efficiency scorer.
    """

    query: str
    result: dict[str, Any]
    context: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] | None = None
    label: str | None = None  # optional tag, e.g. "good" / "infeasible" ablation


@dataclass(frozen=True)
class RewardComponent:
    """One normalized reward dimension."""

    name: str
    score: float          # normalized to [0, 1]
    detail: str = ""
    weight: float = 1.0   # relevance weight within the weighted sum
    applicable: bool = True  # False -> excluded from the weighted sum (e.g. richness on a trivial lookup)
    is_gate: bool = False    # gate failure (score < gate_threshold) collapses the reward

    def clamped(self) -> "RewardComponent":
        return RewardComponent(
            self.name, max(0.0, min(1.0, float(self.score))),
            self.detail, self.weight, self.applicable, self.is_gate,
        )


@dataclass(frozen=True)
class RewardResult:
    """Final reward plus its decomposition."""

    scalar: float
    feasible: bool
    components: tuple[RewardComponent, ...]

    def component(self, name: str) -> RewardComponent | None:
        return next((c for c in self.components if c.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scalar": self.scalar,
            "feasible": self.feasible,
            "components": [
                {
                    "name": c.name, "score": c.score, "weight": c.weight,
                    "applicable": c.applicable, "is_gate": c.is_gate, "detail": c.detail,
                }
                for c in self.components
            ],
        }


@runtime_checkable
class Scorer(Protocol):
    """Produces one or more reward components for an episode."""

    name: str

    def score(self, episode: Episode) -> RewardComponent | list[RewardComponent]:
        ...


class RewardModel:
    """Weighted, gated combination of scorers.

    Reward = (gate multiplier) x (weighted mean of applicable, non-gate
    components). Gate components whose score falls below ``gate_threshold``
    set ``feasible=False`` and multiply the reward by ``gate_penalty`` (default
    0.1 — a heavy but non-zero penalty so a learner keeps a gradient toward
    "infeasible but otherwise complete" over "infeasible garbage"; set to 0.0
    for a hard gate).
    """

    def __init__(
        self,
        scorers: list[Scorer],
        *,
        gate_penalty: float = 0.1,
        gate_threshold: float = 1.0,
    ) -> None:
        if not 0.0 <= gate_penalty <= 1.0:
            raise ValueError("gate_penalty must be in [0, 1]")
        self._scorers = list(scorers)
        self._gate_penalty = gate_penalty
        self._gate_threshold = gate_threshold

    def score(self, episode: Episode) -> RewardResult:
        components: list[RewardComponent] = []
        for scorer in self._scorers:
            produced = scorer.score(episode)
            if isinstance(produced, RewardComponent):
                produced = [produced]
            components.extend(c.clamped() for c in produced)

        feasible = True
        gate_multiplier = 1.0
        for comp in components:
            if comp.is_gate and comp.applicable and comp.score < self._gate_threshold:
                feasible = False
                gate_multiplier = min(gate_multiplier, self._gate_penalty)

        weighted = [c for c in components if c.applicable and not c.is_gate]
        total_weight = sum(c.weight for c in weighted)
        base = (
            sum(c.score * c.weight for c in weighted) / total_weight
            if total_weight > 0
            else 0.0
        )
        scalar = round(base * gate_multiplier, 6)
        return RewardResult(scalar=scalar, feasible=feasible, components=tuple(components))

    # --- RL plug-in conveniences -------------------------------------------

    def reward_fn(self):
        """Return a bare ``episode -> float`` callable (the RL reward)."""
        return lambda episode: self.score(episode).scalar

    def scalar_of(self, episode: Episode) -> float:
        return self.score(episode).scalar

    def rank(self, episodes: list[Episode]) -> list[tuple[Episode, RewardResult]]:
        """Score and sort candidates best-first (best-of-N / exploration)."""
        scored = [(ep, self.score(ep)) for ep in episodes]
        scored.sort(key=lambda pair: pair[1].scalar, reverse=True)
        return scored


def default_reward_model(**kwargs: Any) -> RewardModel:
    """The standard STRAP reward model (physical gate + grounding + richness +
    completeness + efficiency). Kwargs pass through to ``RewardModel``."""
    from strap.eval.scorers import (
        CompletenessScorer,
        EfficiencyScorer,
        GroundingScorer,
        OptimizationOutcomeScorer,
        PhysicalValidityScorer,
        RichnessScorer,
        SeparationQualityScorer,
    )

    return RewardModel(
        [
            PhysicalValidityScorer(),
            GroundingScorer(),
            SeparationQualityScorer(),
            OptimizationOutcomeScorer(),
            RichnessScorer(),
            CompletenessScorer(),
            EfficiencyScorer(),
        ],
        **kwargs,
    )
