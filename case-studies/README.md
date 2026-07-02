# STRAP v10 Case Studies

Each case study is a **single flat folder** that reproduces one analysis and
its publication-quality figures. The layout is deliberately shallow:

```
case-studies/
├── _shared/                       shared plotting style (one folder)
│   └── casestudy_style.py
├── NN-<case-name>/
│   ├── README.md                  question, insight, the fundamental fix, how to run
│   ├── reproduce.py               one script: data/ → figures/  (no API calls)
│   ├── data/                      replay inputs (cached sims, reference results)
│   └── figures/                   generated PNG + PDF
└── README.md                      this index
```

**Reproducibility rule.** Every `reproduce.py` runs with **no network or API
access**. Analyses that originally required BioSTEAM or LLM calls replay from
cached results committed under the case's `data/`; deterministic parts
(solubility, selectivity) recompute live from the v10 `strap` engine so the
figures inherit the current, validated numerics.

## Index

| # | Case study | What it shows | Engine / replay source |
|---|---|---|---|
| 01 | [Temperature-resolved separation economics](01-pareto-temperature-sweep/) | Selectivity-vs-MSP Pareto frontier across the 8-polymer dissolution sequence, per-step temperature trade-offs | v10 interpolation (live) + 217 cached BioSTEAM sims |
| 02 | [Cost-vs-emissions / circularity Pareto landscapes](02-cost-emissions-pareto/) | Why single-point "broken" frontiers appear and the fix — landscape + recomputed rich frontier | v10 waste-optimization MINLP (live SCIP, workbook TEA) |

## Legacy case-study material

`case-1/` and `case-2-validation/` hold earlier, more deeply nested
transcript/artifact material from live agent runs. New work follows the flat
`NN-<case-name>/` convention above; the legacy folders will be migrated into it
as each is reproduced.
