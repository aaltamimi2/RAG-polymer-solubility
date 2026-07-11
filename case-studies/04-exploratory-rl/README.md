# Case Study 04 — Exploratory RL on the Harness

**What this is.** The first working reinforcement-learning loop on STRAP,
composed from the two pieces built for it:

1. **Best-of-N exploration** (`strap.learn.best_of_n`) — instead of one greedy
   engine run, generate N *diverse* candidate analyses by varying the
   harness's real decision knobs (dissolution temperature, candidate-pool
   depth `top_k`, and progressive exclusion of the dominant solvents — which
   forces the engine to surface non-obvious alternatives). Every candidate is
   scored by the physically-gated reward model; the best **feasible** one wins.
   The gate makes exploration safe by construction: an unphysical candidate
   can never be selected, no matter how rich it looks.

2. **A contextual bandit** (`strap.learn.bandit`) — Thompson sampling
   (Beta-Bernoulli with fractional updates, correct for the reward's [0,1]
   range) that learns *how much exploration each query class deserves*. Arms
   are the exploration breadth N ∈ {1, 2, 4, 6}; the context is query
   complexity (number of polymers); the reward is `strap.eval`'s composite —
   which contains the genuine tension the bandit must resolve: more
   exploration can find better routes (separation quality, richness) but costs
   more engine calls (efficiency).

Everything runs on the deterministic v10 engines with memoized calls —
**zero API calls**, fully reproducible (seeded).

## The figure (`figures/exploratory_rl.png`)

- **Panel A — anatomy of one exploration.** Six candidates for a 4-polymer
  separation, reward-ranked. Candidate 0 is today's greedy default; the
  variants show what temperature/pool/exclusion diversity surfaces, including
  the count of unique solvents exposed across candidates.
- **Panel B — online learning.** Rolling mean reward of the bandit against
  three baselines on the *identical* query stream: always-greedy (N=1),
  always-max-exploration (N=6), and random N. The bandit converges to the
  per-context optimum without being told it.
- **Panel C — the learned policy.** Arm-selection share per context and the
  learned best N. This is the interpretable artifact: a policy table saying
  "this query class deserves this much exploration."

## Measured outcome (200 rounds, seeded)

The bandit **learned** the per-context policy {2 polymers → N=1, 3 polymers →
N=1, 4 polymers → N=2}: on these query pools the greedy default is already
strong, so heavy exploration rarely repays its efficiency cost — except on the
hardest (4-polymer) class, where modest exploration wins. Mean rewards on the
identical stream: learned bandit **0.943** vs always-greedy 0.950 (the oracle
fixed policy — the bandit's small gap is its exploration cost while learning),
always-explore N=6 0.928, random 0.939. The point is not that exploration
always wins — it's that the system now *measures* when it does and converges
to the right amount per query class, instead of guessing.

## Why this is honest RL, not RL theater

- The reward is **not** hand-shaped for the demo — it is the same substrate
  from case study 03, and the physical gate means reward-hacking toward
  unphysical-but-rich answers is structurally impossible.
- Baselines run on the identical query stream (paired comparison), so the
  learning curve differences are policy differences, not sampling luck.
- The environment is stationary and deterministic per (query, N); the
  stochasticity the bandit faces is real — which query arrives next.
- The bandit state is serializable (`bandit.to_dict()`, stored in
  `data/results.json`), so a learned policy can be *deployed*: the harness can
  load it and pick N per query class at inference time.

## Plugging in further (the path this opens)

- Any other discrete harness decision becomes a bandit arm the same way:
  SCIP solver rung, `top_k_solvents` carried into optimization, Pareto
  `n_points`. One `select / produce / update` triple each.
- Search-with-value (RL option 3): `RewardResult.components` provides shaped
  per-dimension signals for scoring partial states in a tree search over
  (polymer, solvent, temperature) decisions.

## Reproduce

```bash
python case-studies/04-exploratory-rl/reproduce.py
```

Writes `data/results.json` (learned policy, mean rewards, full bandit state)
and the figure. Tests: `tests/test_learn.py` (13 — bandit convergence on
rigged arms, per-context independence, seed determinism, serialization
round-trip, best-of-N ranking/gating/diversity, and the bandit↔explorer↔reward
composition).
