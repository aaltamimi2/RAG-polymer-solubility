# Case Study 03 — Reward / Evaluation Substrate

**What this is.** A rigorous, decomposed reward function for STRAP analysis
results (`src/strap/eval/`), plus a publication figure showing it scored across
a simple→complex query spectrum. It is the scoring layer needed for any
learning loop — contextual bandits over harness decisions, search-with-value
over the analysis tree, or best-of-N exploration — and it doubles as an
automatic quality/regression evaluator.

## Why it's built this way

1. **Physically grounded, zero API.** Every score is computed from the
   structured result plus the deterministic v10 engines (solubility, boiling
   points) — never from an LLM's self-report. Reproducible and free to run.
2. **Hard physical gate, not a soft reward.** A route that recommends a solvent
   above its atmospheric boiling point is *invalid*, not merely worse. The
   `PhysicalValidityScorer` is a gate: any violation sets `feasible=False` and
   collapses the reward. This is what stops a learner from reward-hacking its
   way to rich-but-unphysical answers — the central risk of RL in a
   physical-science domain.
3. **Decomposed and inspectable.** The reward is a weighted, gated combination
   of seven normalized dimensions, each with a human-readable reason, so it can
   be audited (and plotted) rather than trusted as an opaque scalar.

## The seven dimensions

| Scorer | Measures | Gate? |
|---|---|---|
| `PhysicalValidityScorer` | every step at/below the solvent's BP at 1 atm | **yes** |
| `GroundingScorer` | claims recomputed against the engine (anti-hallucination) | no |
| `SeparationQualityScorer` | worst-step selectivity vs a target (clean vs barely-viable route) | no |
| `OptimizationOutcomeScorer` | did an optimization/Pareto episode deliver a feasible frontier (vs infeasible) | no |
| `RichnessScorer` | breadth of exploration: candidates/polymer, alt sequences, frontier size | no |
| `CompletenessScorer` | coverage of requested polymers & metrics | no |
| `EfficiencyScorer` | tool-call cost vs a task-shaped budget | no |

`SeparationQualityScorer` and `OptimizationOutcomeScorer` are the two
route-quality dimensions: the first keys off dissolution steps, the second off
an optimization result's feasible points. Each marks itself non-applicable when
the result is the other kind, so a separation answer is never judged on a
frontier it wasn't asked for, and vice versa. `OptimizationOutcomeScorer` was
added after offline RL on real optimization traces showed that, without it, an
honest-but-infeasible optimization was ungated and could outscore a real
feasible frontier (see case study 05).

## The figure

`figures/reward_evaluation.png`
- **Panel A** — reward decomposition across six real, engine-backed queries
  ordered simple→very complex. Bars are the per-dimension scores (hatched/faint
  where a dimension doesn't apply, e.g. no richness on a point lookup); the line
  is the composite reward. Simple lookups score high on correctness and
  completeness; complex analyses earn their reward through richness; efficiency
  tapers as tasks get more expensive.
- **Panel B** — reward *discrimination* on one complex separation query. The
  reward model ranks the engine result (0.80) above a greedy single-candidate
  version (0.69), and the physical-validity gate collapses the
  physically-infeasible (above-BP) and fabricated (polyolefin-in-water)
  variants to 0.07 with `feasible=False`. This ordering — good > thorough-but-
  greedy ≫ invalid — is exactly the signal a learning loop consumes.

## Plugging into RL (options 2 & 3)

```python
from strap.eval import Episode, default_reward_model

model = default_reward_model()
reward = model.score(episode)      # RewardResult: .scalar, .feasible, .components
fn     = model.reward_fn()         # bare episode -> float

# contextual bandit (option 2): action -> produce result -> reward
r = fn(Episode(query, result, context, ledger))

# best-of-N exploration: score N candidates, take the best feasible one
ranked = model.rank([Episode(query, cand, ctx) for cand in candidates])
```

An `Episode` is `{query, result, context, ledger}` — produced by a live agent
run, a direct engine call, or an RL rollout; the scorer treats them uniformly.
Reweight or add dimensions by passing your own scorer list to `RewardModel`.

## Reproduce

```bash
python case-studies/03-reward-evaluation/reproduce.py
```

Scores the graded suite (`src/strap/eval/query_suite.py`) and the ablation
variants, writes `data/scores.json`, and renders the figure. No API calls.
Unit tests: `tests/test_reward_eval.py` (18 tests — gate firing, grounding
catching fabrication, richness rewarding thoroughness, RL adapters, and an
end-to-end check that the engine result outranks every ablation).
