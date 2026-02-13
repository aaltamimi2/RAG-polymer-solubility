# BioSTEAM-Based TEA/LCA for STRAP Process

---

## Overview

The reference notebook (`Branch-TEA.ipynb`) implements a rigorous techno-economic analysis (TEA) and life cycle assessment (LCA) of the **Solvent-Targeted Recovery and Precipitation (STRAP)** process using [BioSTEAM](https://biosteam.readthedocs.io/) — an open-source platform for the design, simulation, and evaluation of chemical processes.

The process recovers **polyethylene (PE)** and **ethylene vinyl alcohol (EVOH)** from multilayer plastic waste through selective dissolution and anti-solvent precipitation.

---

## Process Model Architecture

### Unit Operations (46 total)

```
Feedstock ─► Shredder ─► Conveyor ─► Storage
                                        │
                          ┌─────────────┘
                          ▼
              ┌──── PE Separation Line ────┐
              │  Dissolution Tank (U3)     │
              │  Centrifuge (C1)           │
              │  Vacuum Dryer (U4)         │
              │  Microfilter (U5)          │
              │  Precipitation Tank (T5)   │
              │  Centrifuge (C2)           │
              │  Flash Drum (F1)           │
              │  Screw Press (U6)          │
              │  Heat Exchangers (H1, H2)  │
              └────────────────────────────┘
                          │
                     coproduct
                          │
              ┌─── EVOH Separation Line ───┐
              │  (Mirror of PE line)       │
              │  Dissolution (U8)          │
              │  Centrifuge (C3)           │
              │  Precipitation (T10)       │
              │  Flash (F2), Dryer (U9)    │
              └────────────────────────────┘
                          │
              ┌─── Facilities ─────────────┐
              │  Boiler/Turbogenerator (BT)│
              │  Cooling Tower (CT)        │
              │  Chilled Water Package     │
              └────────────────────────────┘
```

### Products

| Stream | Description |
|:-------|:------------|
| `PE_resin` | Recovered polyethylene |
| `EVOH_resin` | Recovered ethylene vinyl alcohol |
| `coproduct` | Remaining PET-rich residue |

---

## Simulation Matrix

The notebook evaluates a **3-case x 4-sequence** matrix (12 scenarios per solvent), each with full TEA and 6-category LCA.

### Three Energy Cases

| Case | Configuration | Facilities | Turbogenerator | Grid Electricity |
|:-----|:--------------|:-----------|:---------------|:-----------------|
| **C1** | On-site CHP | Yes | Yes | No |
| **C2** | Grid + AMCOR | No | No | Yes |
| **C3** | Grid + Boiler | Yes | No | Yes |

### Four Recovery Sequences

| Sequence | Description | Target | Plastic % | Capacity (MT/yr) |
|:---------|:------------|:-------|:----------|:------------------|
| **P1** | PE recovered first | PE | 60% | 20,000 |
| **E1** | EVOH recovered first | EVOH | 10% | 20,000 |
| **E2** | EVOH recovered second | EVOH | 25% | 8,000 |
| **P2** | PE recovered second | PE | 66.7% | 18,000 |

---

## Solvents Evaluated

### PE Dissolution (9 solvents)

| Solvent | Abbrev. | Price ($/kg) | T_diss (C) | GWP (kgCO2e/kg) |
|:--------|:--------|:-------------|:-----------|:-----------------|
| sec-Butyl Acetate | SBA | 1.60 | 110 | 4.98 |
| Isobutyl Acetate | IBA | 1.60 | 114 | 4.81 |
| Tetrachloroethylene | TCE | 1.38 | 120 | 3.85 |
| o-Chlorotoluene | OCT | 2.40 | 120 | 2.74 |
| Methylcyclohexane | MCH | 1.55 | 98 | 2.55 |
| Dodecanol | DDL | 1.50 | 120 | 4.12 |
| Heptane | HEP | 1.42 | 96 | 0.90 |
| **Toluene** | **TOL** | **0.82** | **110** | **1.61** |
| Xylene | XYL | 0.84 | 120 | 1.52 |

### EVOH Dissolution (9 solvents across sequences)

| Solvent | Price ($/kg) | GWP (kgCO2e/kg) |
|:--------|:-------------|:-----------------|
| Ethylene Glycol | 0.53 | 2.70 |
| Pyridazine | 4.95 | 10.70 |
| Butane-1,4-diol | 1.22 | 5.50 |
| Diethanolamine | 1.06 | 3.71 |
| Diethylene Glycol | 0.59 | 3.15 |
| Propylene Glycol | 1.53 | 5.16 |
| gamma-Butyrolactone | 2.58 | 6.54 |

---

## LCA Impact Categories

Six TRACI-aligned categories are evaluated per scenario:

| Category | Unit | Description |
|:---------|:-----|:------------|
| **GWP** | kgCO2e/kg resin | Global Warming Potential |
| **HTC** | CTUh/kg resin | Human Toxicity (cancer) |
| **HTNC** | CTUh/kg resin | Human Toxicity (non-cancer) |
| **ETOX** | CTUe/kg resin | Ecotoxicity |
| **FE** | kg P eq/kg resin | Freshwater Eutrophication |
| **ME** | kg N eq/kg resin | Marine Eutrophication |

### Characterization Factor Sources

- **Solvent**: Per-solvent impact factors + 0.1563 kgCO2e/kg burning correction
- **Natural gas** (C1/C3): GWP = 3.841, HTC = 2.48e-7, HTNC = 1.20e-7, ETOX = 6.68
- **Grid electricity** (C2/C3): GWP = 0.197 kgCO2e/MJ
- **Water makeup**: GWP = 1.27e-4 kgCO2e/kg

---

## Results Extracted Per Scenario

Each simulation produces:

```
TEA                              LCA
─────────────────────────        ──────────────────────────
OPEX        (USD/yr)             GWP    (kgCO2e/kg resin)
CAPEX / TCI (USD)                HTC    (CTUh/kg resin)
MSP         (USD/kg resin)       HTNC   (CTUh/kg resin)
Labor cost  (USD/yr)             ETOX   (CTUe/kg resin)

Water Balance                    Energy
─────────────────────────        ──────────────────────────
CT blowdown      (m3/yr)         Electricity    (MJ/yr)
CT evaporation   (m3/yr)         Heating duty   (MJ/yr)
BT blowdown      (m3/yr)         Cooling duty   (MJ/yr)
Total consumed   (m3/yr)         Total energy   (MJ/kg resin)
Total circulated (m3/yr)

Waste
─────────────────────────
Waste generated  (kg/yr)   ← spent activated carbon
Waste diverted   (kg/yr)   ← recovered PE resin
```

### Validated Output (Toluene, C1, PE first)

| Metric | Value |
|:-------|:------|
| Total Capital Investment | $200.4 M |
| Annual Operating Cost | $21.8 M/yr |
| Minimum Selling Price | $0.128/kg |
| GWP | 0.45 kgCO2e/kg |
| Water consumed | 136,598 m3/yr |
| PE resin produced | 112,129 MT/yr |

---

## Integration with DISSOLVE Agent

### New `tea-lca-analyst` Subagent

The BioSTEAM process model replaces the current hand-rolled TEA/LCA correlations with rigorous simulation. The `tea-lca-analyst` subagent gains access to the full `BaselineSTRAPModel` through tool wrappers.

### Variables the Agent Can Iterate

The agent can programmatically sweep across any combination of:

| Variable | Range | Unit |
|:---------|:------|:-----|
| Solvent identity | 18 solvents (9 PE + 9 EVOH) | — |
| Solvent price | 0.50 - 5.00 | USD/kg |
| Dissolution temperature | T_m + 15 to T_b - 5 | K |
| Precipitation temperature | 265 - T_diss | K |
| Dissolution capacity | 1 - 10 | wt% |
| Processing capacity | 5,000 - 50,000 | MT/yr |
| Target plastic fraction | 2 - 98 | wt% |
| Energy case | C1, C2, C3 | — |
| Recovery sequence | PE-first, EVOH-first | — |
| Solvent loss rate | 0.01 - 0.2 | % |
| Feedstock price | 0.01 - 0.05 | USD/kg |
| IRR (internal rate of return) | 10 - 20 | % |

### Multi-Agent Workflows

The TEA/LCA subagent connects to other DISSOLVE subagents to form end-to-end analysis pipelines:

```
                    ┌─────────────────────────┐
                    │    User Query           │
                    │  "Find the cheapest     │
                    │   separation for this   │
                    │   9-polymer waste"      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     Orchestrator         │
                    │  Routes to subagents     │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
   ┌──────────▼────────┐ ┌─────▼──────┐ ┌───────▼────────┐
   │  Separation       │ │  Safety    │ │  Scholar       │
   │  Engineer         │ │  Analyst   │ │  Researcher    │
   │                   │ │            │ │                │
   │ Selectivity data  │ │ GSK scores │ │ Literature     │
   │ Solvent ranking   │ │ PubChem    │ │ references     │
   │ 3-scheme plans    │ │ hazards    │ │                │
   └────────┬──────────┘ └─────┬──────┘ └───────┬────────┘
            │                  │                 │
            └──────────┬───────┘                 │
                       │                         │
            ┌──────────▼──────────┐              │
            │   TEA/LCA Analyst   │◄─────────────┘
            │                     │
            │  BioSTEAM simulation│
            │  for top solvents   │
            │  from sep-engineer  │
            │                     │
            │  MSP, GWP, water,   │
            │  energy, CAPEX/OPEX │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   Visualization     │
            │   Specialist        │
            │                     │
            │  Publication-ready  │
            │  comparison plots   │
            └─────────────────────┘
```

### Parallel Agent Execution

BioSTEAM simulations can run in parallel across solvents, enabling a **fan-out / filter / fan-in** pattern:

```
                  Orchestrator
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
    │ BioSTEAM│  │ BioSTEAM│  │ BioSTEAM│     Fan-out:
    │ Toluene │  │ Heptane │  │ Xylene  │     Run N solvents
    │ C1, P1  │  │ C1, P1  │  │ C1, P1  │     in parallel
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └────────────┼────────────┘
                      │
              ┌───────▼───────┐
              │    Filter     │                Filter:
              │  Rank by MSP  │                Keep top-K
              │  Keep top 3   │                results
              └───────┬───────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
    │ Safety  │  │ Safety  │  │ Safety  │     Fan-in:
    │ Analyst │  │ Analyst │  │ Analyst │     Deep-dive on
    │ best #1 │  │ best #2 │  │ best #3 │     winners only
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └────────────┼────────────┘
                      │
              ┌───────▼───────┐
              │  Final Report │                Synthesize:
              │  with ranked  │                Combine TEA +
              │  recommendations│              safety scores
              └───────────────┘
```

This pattern avoids running expensive safety analysis on all 18 solvents. Instead:

1. **Fan-out** — Run BioSTEAM TEA/LCA for all candidate solvents in parallel (~1s each)
2. **Filter** — Rank by MSP or GWP, keep only the top 3-5
3. **Fan-in** — Route winners to safety-analyst and scholar-researcher for deep evaluation
4. **Synthesize** — Combine economic, environmental, and safety data into a final recommendation

### Comparison: Before and After

| Aspect | Hand-Rolled (Current) | BioSTEAM (New) |
|:-------|:---------------------|:---------------|
| Equipment costing | Six-tenths rule | Rigorous BioSTEAM TCI |
| Energy balance | Cp + Hvap formula | Full thermodynamic simulation |
| LCA categories | GWP only | 6 categories (GWP, HTC, HTNC, ETOX, FE, ME) |
| Energy scenarios | Single | 3 cases (CHP, Grid, Grid+Boiler) |
| Recovery sequences | Not modeled | PE-first vs EVOH-first |
| Water balance | Not modeled | CT + BT blowdown, evaporation |
| Simulation time | ~0.01s | ~1s per scenario |
| Validated against | Correlations | Published experimental data |
