# Case Study 01 — Temperature-Resolved Separation Economics

**Question.** For the STRAP dissolution sequence that recovers eight polymers
one at a time, the dissolution *temperature* at each step trades separation
**selectivity** against **minimum selling price (MSP)**. What is the cheapest
set of per-step temperatures that still guarantees a chosen minimum selectivity
at every step, and where is the economic knee of that trade-off?

**Sequence.** `EVOH → HDPE → LDPE → PP → PS → PVC → Nylon6 → PET`, each step
using the dynamic-programming-optimal solvent (glycol, propylene glycol,
diphenyl ether, dodecane, o-xylene, DMSO, cyclohexanol, DMF).

## What the figures show

`figures/pareto_selectivity_vs_cost.png` (and `.pdf`)
- **Top-left** — per-step selectivity vs dissolution temperature (8 curves).
- **Bottom-left** — per-step MSP vs dissolution temperature.
- **Right** — the global Pareto frontier: for each *guaranteed minimum
  selectivity* floor, the lowest average MSP achievable by choosing the
  cheapest temperature at each step that clears the floor. Markers flag the
  max-selectivity end, the cheapest configuration, and the economic knee.

`figures/per_step_detail.png` — one dual-axis panel per step (selectivity in
colour, MSP in grey) with the DP-optimal temperature marked.

**Headline insight.** MSP is compressed (~$1.07–1.10/kg across the whole
frontier), so guaranteeing high minimum selectivity costs only a few percent
on price — but pushing to the *cheapest* corner drives one step's selectivity
negative (the target dissolves less than a co-dissolving polymer), which is
exactly the trade-off the frontier makes explicit.

## The fundamental fix (what porting from v9 corrected)

The v9 original (`langchain-STRAP-v9-contaminants/architecture/pareto_temperature.py`)
computed selectivity from a bespoke, uncached selectivity routine and hardcoded
an absolute `langchain-STRAP-v8/src` path. The v10 port:

1. **Recomputes selectivity live from v10's validated interpolation engine**
   (`strap.solubility.get_solubility`), the same engine audited in the
   solubility feature review — so the selectivity curves inherit v10's
   clamping/extrapolation rigor instead of a parallel implementation.
2. **Replays the 217 real BioSTEAM simulations** that were already run once in
   v9 (`data/biosteam_sims_cache.json`) instead of re-invoking BioSTEAM — no
   API calls, fully deterministic.
3. **Cross-checks** the reproduced frontier against the v9 reference
   (`data/reference_frontier_v9.json`): the cheapest MSP reproduces exactly
   ($1.066/kg); the frontier is denser because v10's selectivity values differ
   slightly from the v9 cache (the intended improvement).
4. Uses only paths relative to the script and the shared publication style in
   `../_shared/casestudy_style.py`.

## Reproduce

```bash
python case-studies/01-pareto-temperature-sweep/reproduce.py
```

Writes both PNG (300 dpi) and PDF (vector) into `figures/`, plus the recomputed
frontier to `data/reproduced_frontier.json`. No network or API access required.

## Files

```
01-pareto-temperature-sweep/
├── README.md                     this file
├── reproduce.py                  replay data/ → figures/ (no API)
├── data/
│   ├── biosteam_sims_cache.json  217 cached BioSTEAM sims (replay source)
│   ├── reference_frontier_v9.json  v9 computed frontier (cross-check)
│   └── reproduced_frontier.json  v10 output (written by reproduce.py)
└── figures/                      generated PNG + PDF
```
