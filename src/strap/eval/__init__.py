"""STRAP reward / evaluation substrate.

Import surface for scoring analysis results and plugging the reward into
learning loops (contextual bandits, search-with-value, best-of-N).

    from strap.eval import Episode, default_reward_model

    model = default_reward_model()
    reward = model.score(episode)      # RewardResult: .scalar, .feasible, .components
    fn = model.reward_fn()             # bare episode -> float, for an RL loop
"""

from strap.eval.reward import (
    Episode,
    RewardComponent,
    RewardModel,
    RewardResult,
    Scorer,
    default_reward_model,
)
from strap.eval.scorers import (
    CompletenessScorer,
    EfficiencyScorer,
    GroundingScorer,
    OptimizationOutcomeScorer,
    PhysicalValidityScorer,
    RichnessScorer,
    SeparationQualityScorer,
)

__all__ = [
    "Episode",
    "RewardComponent",
    "RewardModel",
    "RewardResult",
    "Scorer",
    "default_reward_model",
    "PhysicalValidityScorer",
    "GroundingScorer",
    "RichnessScorer",
    "CompletenessScorer",
    "EfficiencyScorer",
    "SeparationQualityScorer",
    "OptimizationOutcomeScorer",
]
