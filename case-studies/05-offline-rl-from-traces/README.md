# Case Study 05 — Offline RL on Logged Agent Traces

**What this is.** Last night's live multistage stress runs (case in
`architecture/test_results/stress_multistage_*`, traced in LangSmith) are
*logged trajectories from a fixed policy* — the classic **offline / off-policy
RL** setting. The expensive part (the model deciding routes and shortlisting
solvents) is already paid for and sits in the traces; the deterministic v10
engines the reward model and best-of-N run on are free and unlimited. So we
turn those real runs into RL data and learn from them with **zero new API
calls**.

This closes the loop opened by case studies 03 (the physically-gated reward
substrate) and 04 (best-of-N + a contextual bandit): those ran on synthetic
engine replays; this one runs the same machinery on **what the live agent
actually produced**.

## The one network step, then everything is offline

The full structured payloads and token ledgers live only in the LangSmith
traces (the local transcripts are truncated). So we harvest **once** from our
*own* logs — a log read, never model inference — into a committed cache:

```bash
python architecture/harvest_trace_rl_cache.py   # reads our LangSmith logs -> data/trace_rl_cache.json
```

Everything downstream replays from `data/trace_rl_cache.json` with no network,
exactly like the other case studies' cached-replay rule.

## The figure (`figures/offline_rl_from_traces.png`)

- **Panel A — offline policy evaluation of the harness fixes.** Every logged
  run reward-labeled by the physically-gated model. The five runs share one
  query under evolving harness config (the fixes from the stress-test writeup),
  so the curve *measures* whether the fixes raised the reward the harness
  earns. The **optimization-outcome** reward is flat-low (~0.41) while the
  optimization was infeasible (runs 2–4) and jumps to **0.73** the moment the
  candidate-admission (baseline-fallback) fix made it feasible (runs 5–6).
- **Panel B — trajectory-rooted counterfactual, for free.** For the real query,
  the harness's *actual* single greedy separation pass (logged, orange band
  0.65–0.82) sits well below what deterministic best-of-N achieves on the same
  query (green, ~0.94–0.97). That gap is reward left on the table — recoverable
  with **zero** API calls, because the engine exploration is free. Within
  best-of-N, breadth barely moves the ceiling on these 2-polymer queries,
  consistent with case study 04's finding that exploration pays mainly on the
  harder classes.
- **Panel C — offline → online.** The bandit's `N=1` arm seeded from the five
  real logged rewards: the cold uniform `Beta(1,1)` prior a fresh bandit starts
  from vs the warm-started `Beta(4.8, 2.2)` posterior (mean 0.69). Real logged
  experience initializes the policy where data exists.

## A substrate bug the traces caught

Scoring the *real* optimization outputs immediately exposed a gap the synthetic
suite never hit: an optimization result has no dissolution **steps**, so the
physical-validity gate and the selectivity scorer are both non-applicable to
it. An honest-but-**infeasible** optimization was therefore ungated and
unpenalized, and *outscored a real feasible frontier* (0.83 vs 0.60) — which
would teach any RL loop over optimization decisions to prefer infeasibility.

Fix: a new **`OptimizationOutcomeScorer`** (the optimization analog of
separation quality) that scores whether a feasible frontier was actually
produced — infeasible → 0.1, feasible frontier with the requested economics →
1.0. Not a gate: an honest infeasibility is a low-value legitimate outcome, not
a physical violation. Feasible now correctly beats infeasible (see Panel A).
Added to `default_reward_model`; case study 03's decomposition figure was
regenerated.

## A determinism bug the traces caught

Reward-labeling the same cached run gave *different* rewards on different
processes (a run's separation reward flipped between ~0.82 and ~0.60). Root
cause: `resolve_polymer`/`resolve_solvent` returned the first substring match
from a **`set`**, whose iteration order varies across processes under Python
string-hash randomization. The trigger was the umbrella token **`PE`**: the
solubility DB keeps `LDPE` and `HDPE` as separate physical keys, and `PE`
substring-matches *four* chemically distinct polymers — `LDPE`, `HDPE`, `PET`,
`PES` — so it resolved to a different one per process (e.g. polyethersulfone,
insoluble → false grounding failure, on some runs; a real polyethylene,
soluble, on others). Every harness tool that fuzzy-resolves a name shared the
bug.

Fixed rigorously, two parts: (1) an explicit `PE → HDPE` alias (the documented
umbrella representative — `LDPE`/`HDPE` stay independently addressable by their
exact keys), so `PE` never falls to substring; (2) the substring fallback now
resolves **only when exactly one candidate matches** — zero or multiple matches
return `None`, surfacing ambiguity instead of silently conflating distinct
species (four `PE` polymers, or the `o-`/`p-`xylene isomers). A codebase-wide
audit confirmed this was the *only* nondeterminism source — every other
resolver (`solvent_registry`, `ml_assets`, `hsp_registry`) uses ordered
dicts/lists/tuples. Pinned by a cross-hash-seed regression test
(`tests/test_solubility_resolution.py`). Reward-labeling is now bit-reproducible.

Both defects are the kind that only surface when you score *real* outputs — the
point of doing RL on live traces rather than synthetic replays.

## Measured results (committed, reproducible)

| Run | Config milestone | optimization reward | separation reward |
|---|---|---|---|
| 2 | budgets silently dead | 0.42 | 0.76 |
| 3 | guard/handoff/synthesis fixes | 0.40 | 0.82 |
| 4 | workbook-constrained shortlist | 0.42 | 0.77 |
| 5 | baseline-fallback fix → feasible | **0.73** | 0.80 |
| 6 | enriched knee/cheapest answer | **0.73** | 0.65 |

Trajectory-rooted headroom (logged single pass → best-of-N) is **+0.10 to
+0.25** across runs. Warm-started `N=1` posterior mean **0.69** from 5 real
samples.

## How this plugs into the live harness

- **Offline pretraining:** seed `strap.learn` bandit arms from harvested logs
  before any online play — the policy starts where real experience exists.
- **Regression reward-eval in CI:** harvest new traces, reward-label them, and
  gate merges on the reward curve — an automatic, physically-grounded quality
  signal for harness changes, with no inference cost.
- **Free counterfactuals:** any decision the deterministic engine can replay
  (separation top-k, temperature, solvent exclusion, optimization knobs) can be
  explored best-of-N *rooted in a real logged trajectory*, at zero API cost.

## Reproduce

```bash
python case-studies/05-offline-rl-from-traces/reproduce.py
```

Reads the committed cache; writes `data/results.json` and the figure. Fully
offline, deterministic. Tests: `tests/test_trace_rl.py` (14 — ingestion, the
optimization-outcome fix, the three analyses, and a smoke test on the committed
cache) and `tests/test_solubility_resolution.py` (7 — the determinism fix).
