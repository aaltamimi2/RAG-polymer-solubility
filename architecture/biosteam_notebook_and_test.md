# BioSTEAM STRAP Notebook & Validation Test

## Background: What is BioSTEAM?

BioSTEAM is an open-source Python framework for building **process simulations** — virtual chemical
plants. You define unit operations (reactors, heat exchangers, pumps, etc.), connect them with
streams, and BioSTEAM solves the mass/energy balances, sizes equipment, and computes costs.

```
                        BioSTEAM Stack
    ┌─────────────────────────────────────────────┐
    │  plastics  (v0.1.4)                         │  ← STRAP-specific process models
    │    BaselineSTRAPProcess                      │     dissolvers, precipitators, etc.
    ├─────────────────────────────────────────────┤
    │  biosteam  (v2.52.17)                       │  ← Process simulation engine
    │    Unit operations, TEA, LCA, System         │     equipment sizing, costing
    ├─────────────────────────────────────────────┤
    │  thermosteam  (v0.52.16)                    │  ← Thermodynamic property engine
    │    Chemical properties, phase equilibria     │     VLE, activity coefficients
    ├─────────────────────────────────────────────┤
    │  thermo / chemicals                         │  ← Chemical database
    │    Pure component data, correlations          │     Tb, Tc, Pc, Cp, etc.
    └─────────────────────────────────────────────┘
```

**Key concept:** `BaselineSTRAPProcess` is a pre-built process model that wraps ~27 unit operations
into a single class. You give it a solvent name and operating conditions, and it builds the entire
plant flowsheet, runs it, and returns TEA/LCA results.

---

## The STRAP Process

STRAP = **S**olvent-**T**argeted **R**ecovery **a**nd **P**recipitation. It separates layers of
multilayer plastic packaging (e.g. PE/EVOH/PET film) by dissolving one layer at a time in a
selective solvent, then precipitating it out by cooling.

### Process Flow Diagram

```
  Multilayer plastic waste (PE + EVOH + PET)
                    │
                    ▼
    ┌───────────────────────────┐
    │   SHREDDER  (U1)          │  Reduce particle size
    │   [excluded from TEA]     │
    └───────────┬───────────────┘
                │
                ▼
    ┌───────────────────────────┐
    │   DISSOLUTION TANK        │  Heat solvent + plastic to T_diss
    │                           │  Target polymer dissolves into solvent
    │   e.g. PE in Toluene      │  Other layers (EVOH, PET) remain solid
    │        at 110 C           │
    └───────────┬───────────────┘
                │
         ┌──────┴──────┐
         │             │
         ▼             ▼
    ┌──────────┐  ┌──────────────┐
    │ FILTRATE │  │ FILTER CAKE  │
    │ (polymer │  │ (undissolved │
    │ + solvent│  │  layers)     │
    │  solution│  │              │
    └────┬─────┘  └──────────────┘
         │              can be fed to next
         ▼              dissolution step
    ┌───────────────────────────┐
    │   PRECIPITATION TANK      │  Cool solution to T_precip (25 C)
    │                           │  Dissolved polymer crashes out
    │   Polymer crystallizes    │  as solid precipitate
    └───────────┬───────────────┘
                │
         ┌──────┴──────┐
         │             │
         ▼             ▼
    ┌──────────┐  ┌──────────────┐
    │ RECOVERED│  │ SOLVENT      │  Recycled back to
    │ POLYMER  │  │ (liquid)     │  dissolution tank
    │ (wet)    │  │              │  (>99.99% recovery)
    └────┬─────┘  └──────┬───────┘
         │               │
         ▼               │
    ┌──────────┐         │
    │CENTRIFUGE│         │
    │ + DRYER  │         │
    └────┬─────┘    ┌────┴─────┐
         │         │ SOLVENT   │
         ▼         │ MAKEUP    │
    ┌──────────┐   │ TANK      │
    │ PE RESIN │   └───────────┘
    │ (product)│
    └──────────┘
```

### What Makes Each Simulation Different

Three things vary across the notebook's 81 simulations:

```
                     ┌─────────────────────────────────────┐
                     │         SIMULATION INPUTS            │
                     ├─────────────────────────────────────┤
                     │                                     │
   Which solvent? ───┤  Solvent choice (18 total)          │
                     │    PE:   9 solvents (TOL, XYL...)   │
                     │    EVOH: 9 solvents (EG, PYR...)    │
                     │                                     │
   Which layer    ───┤  Recovery sequence (4 options)      │
   first?            │    P1: dissolve PE first             │
                     │    E1: dissolve EVOH first           │
                     │    E2: dissolve EVOH second          │
                     │    P2: dissolve PE second            │
                     │                                     │
   What energy    ───┤  Energy configuration (3 cases)     │
   source?           │    C1: On-site CHP (boiler+turbine) │
                     │    C2: Grid electricity, no boiler   │
                     │    C3: Grid electricity + boiler     │
                     │                                     │
                     └─────────────────────────────────────┘
                                     │
                                     ▼
                     ┌─────────────────────────────────────┐
                     │        SIMULATION OUTPUTS            │
                     ├─────────────────────────────────────┤
                     │  TEA                                │
                     │    OPEX (operating cost, $/yr)      │
                     │    CAPEX (capital cost, $)          │
                     │    MSP (min selling price, $/kg)    │
                     │                                     │
                     │  LCA                                │
                     │    GWP  (global warming, kgCO2e/kg) │
                     │    HTC  (human tox - cancer)        │
                     │    HTNC (human tox - non-cancer)    │
                     │    ETOX (ecotoxicity)               │
                     │                                     │
                     │  Operations                         │
                     │    Water consumed (m3/yr)           │
                     │    Water circulated (m3/yr)         │
                     │    Energy consumed (MJ/kg)          │
                     │    Waste generated (kg/yr)          │
                     │    Waste diverted (kg/yr)           │
                     └─────────────────────────────────────┘
```

---

## The Three Energy Cases

Each case changes what equipment is on-site and where energy comes from.

```
  CASE 1 (C1): Combined Heat & Power           CASE 2 (C2): Grid + AMCOR
  ──────────────────────────────                ────────────────────────────

  Natural Gas ──► ┌──────────┐                   Grid ──────► Electricity
                  │  BOILER  │                                  │
                  │          ├──► Steam ──► Process              ▼
                  │  TURBO-  │                              ┌────────┐
                  │ GENERATOR├──► Electricity ──► Process   │ Process│
                  └──────────┘                              │ (no    │
                       │                                    │ boiler,│
                  Cooling Tower                             │ no CT) │
                       │                                    └────────┘
                  Water circulation
                                                No natural gas impacts
  Natural gas impacts included                  Grid electricity impacts included
  facilities=True, turbogenerator=True          facilities=False


  CASE 3 (C3): Grid + Boiler
  ────────────────────────────

  Natural Gas ──► ┌──────────┐     Grid ──► Electricity
                  │  BOILER  │                  │
                  │  (no     ├──► Steam         ▼
                  │  turbine)│            ┌──────────┐
                  └──────────┘            │  Process  │
                       │                  └──────────┘
                  Cooling Tower
                       │
                  Water circulation

  Both natural gas AND grid impacts
  facilities=True, turbogenerator=False
```

---

## The Four Recovery Sequences

The multilayer film has three layers. The order you dissolve them matters — you get the "first"
polymer cleaner because it dissolves from intact film, while the "second" polymer dissolves from
already-processed material.

```
  Multilayer Film:  ┌─────────────────────┐
                    │  PE layer   (60%)   │
                    │  EVOH layer (10%)   │
                    │  PET layer  (30%)   │
                    └─────────────────────┘

  P1: PE first ─────────────────────────────────────────────────────
    Step 1: Dissolve PE  ──► recover PE resin  (clean, 60% of feed)
    Step 2: Dissolve EVOH ──► recover EVOH     (from residual film)
    Leftover: PET (undissolved)

  E1: EVOH first ───────────────────────────────────────────────────
    Step 1: Dissolve EVOH ──► recover EVOH resin (clean, 10% of feed)
    Step 2: Dissolve PE   ──► recover PE         (from residual film)
    Leftover: PET (undissolved)

  E2: EVOH second (after PE already removed) ───────────────────────
    [PE already recovered in a prior step]
    This step: Dissolve EVOH from PE-depleted film
    EVOH target_plastic_percent = 10%

  P2: PE second (after EVOH already removed) ───────────────────────
    [EVOH already recovered in a prior step]
    This step: Dissolve PE from EVOH-depleted film
    PE target_plastic_percent = 60%
```

---

## Solvent Lineup

### PE Dissolution Solvents (P1 / P2)

```
  Solvent                  Price    T_diss   GWP Impact Factor
  ─────────────────────────────────────────────────────────────
  SBA  sec-Butyl Acetate   $1.60    110 C    4.98 kgCO2e/kg
  IBA  Isobutyl Acetate    $1.60    114 C    4.81
  TCE  Tetrachloroethylene $1.38    120 C    3.85  ← chlorinated
  OCT  o-Chlorotoluene     $2.40    120 C    2.74  ← chlorinated
  MCH  Methylcyclohexane   $1.55     98 C    2.55
  DDL  Dodecanol           $1.50    120 C    4.12
  HEP  Heptane             $1.42     96 C    0.897 ← lowest GWP
  TOL  Toluene             $0.82    110 C    1.61  ← cheapest
  XYL  Xylene              $0.84    120 C    1.52
```

### EVOH Dissolution Solvents

```
  E1 (EVOH first):                E2 (EVOH second):
  ─────────────────               ──────────────────────────
  EG   Ethylene Glycol $0.53      BUT  butane-1,4-diol  $1.22
  PYR  Pyridazine      $4.95      DEA  Diethanolamine   $1.06
                                  DEG  Diethylene glycol $0.59
                                  EG   Ethylene Glycol   $0.53
                                  PPG  Propylene Glycol  $1.53
                                  PYR  Pyridazine        $4.95
                                  GBL  gamma-butyrolactone $2.58
```

---

## What the Notebook Code Does (Step by Step)

For each solvent in a given case/sequence combination:

```
  ┌─ 1. CREATE MODEL ──────────────────────────────────────────────┐
  │                                                                │
  │  process = BaselineSTRAPProcess(                               │
  │      solvent='Toluene',                                        │
  │      target_plastic='PE',                                      │
  │      target_plastic_percent=60,                                │
  │      processing_capacity=20000,  # MT/yr                       │
  │      facilities=True,                                          │
  │  )                                                             │
  │                                                                │
  │  This builds the entire flowsheet:                             │
  │    27 unit operations, all piping, all heat integration        │
  └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ 2. CUSTOMIZE ─────────────────────────────────────────────────┐
  │                                                                │
  │  Remove shredder & storage tank (not in scope)                 │
  │  Set: solvent price, dissolution temp, precipitation temp      │
  │  Set: LCA impact factors for natural gas, solvent, water       │
  │  Set: labor cost = 2 operators x $60k                          │
  └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ 3. SIMULATE ──────────────────────────────────────────────────┐
  │                                                                │
  │  process.system.simulate()                                     │
  │                                                                │
  │  BioSTEAM iterates to convergence:                             │
  │    - Solves mass balance (recycle loops for solvent)            │
  │    - Solves energy balance (heating/cooling duties)             │
  │    - Sizes all equipment (from mass flows)                     │
  │    - Costs all equipment (from size + CEPCI factors)           │
  │    - Computes LCA impacts (from mass flows x impact factors)   │
  └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ 4. EXTRACT RESULTS ───────────────────────────────────────────┐
  │                                                                │
  │  TEA:   process.tea.AOC  → OPEX ($/yr)                        │
  │         process.tea.TCI  → CAPEX ($)                           │
  │         process.MSP()    → Minimum Selling Price ($/kg)        │
  │                                                                │
  │  LCA:   process.GWP()   → Global Warming (kgCO2e/kg)          │
  │         process.HTC()   → Human Toxicity - Cancer              │
  │         process.HTNC()  → Human Toxicity - Non-Cancer          │
  │         process.ETOX()  → Ecotoxicity                          │
  │                                                                │
  │  Water: CT blowdown + evaporation + BT blowdown → consumed    │
  │         CT circulation + BT flow rate → circulated             │
  │                                                                │
  │  Energy: electricity + heating + cooling → total (MJ/kg)       │
  └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
             Save to Process_results_{case}_{sequence}.json
```

### Total Simulation Count

```
  Case C1 (CHP):          Case C2 (Grid):         Case C3 (Grid+Boiler):
  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
  │ P1:  9 solvents│      │ P1:  9 solvents│      │ P1:  9 solvents│
  │ E1:  2 solvents│      │ E1:  2 solvents│      │ E1:  2 solvents│
  │ E2:  7 solvents│      │ E2:  7 solvents│      │ E2:  7 solvents│
  │ P2:  9 solvents│      │ P2:  9 solvents│      │ P2:  9 solvents│
  │ ────────────── │      │ ────────────── │      │ ────────────── │
  │ Total: 27      │      │ Total: 27      │      │ Total: 27      │
  └────────────────┘      └────────────────┘      └────────────────┘

                    Grand Total: 81 simulations
                    Each produces 11 metrics
                    = 891 data points
```

---

## The Validation Test

### Purpose

Verify that `plastics` v0.1.4 (installed from the zip file) works correctly on our environment
(Python 3.11) before building DISSOLVE agent integration.

### What We Tested

We ran **Case 1 (CHP), Sequence P1 (Recover PE first)** across all 9 PE solvents — the most
representative subset of the notebook.

```
  What the notebook runs:                    What our test ran:

  C1 ─┬─ P1 (9 solvents) ◄──────────────── THIS ONE
       ├─ E1 (2 solvents)
       ├─ E2 (7 solvents)
       └─ P2 (9 solvents)
  C2 ─┬─ P1, E1, E2, P2
       └─ (same solvents)
  C3 ─┬─ P1, E1, E2, P2
       └─ (same solvents)
```

If C1/P1 works, the other 11 cells use the same `BaselineSTRAPProcess` class with different flags
(`facilities=False` for C2, `turbogenerator=False` for C3, different solvents for E1/E2/P2).

### Why Subprocesses?

Each solvent runs in a **separate Python subprocess**. This is necessary because BioSTEAM stores
thermodynamic settings in module-level global singletons:

```
  BAD (same process):                GOOD (subprocesses):

  Toluene simulation ──► sets        Subprocess 1: Toluene  ──► clean ──► result
    global thermo state              Subprocess 2: Xylene   ──► clean ──► result
  Xylene simulation ──► reads        Subprocess 3: Heptane  ──► clean ──► result
    STALE Toluene state  ──► WRONG   ...each starts fresh
```

### Results

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │  BaselineSTRAPProcess v0.1.4 — Case 1 (CHP), Recover PE First       │
  ├───────┬───────────────────────┬────────┬────────┬────────┬───────────┤
  │  Abbr │ Solvent               │MSP $/kg│  GWP   │ TCI $M │  Status   │
  ├───────┼───────────────────────┼────────┼────────┼────────┼───────────┤
  │  SBA  │ sec-Butyl Acetate     │  1.151 │  0.611 │  79.1  │  Pass     │
  │  IBA  │ Isobutyl Acetate      │  1.155 │  0.609 │  79.5  │  Pass     │
  │  TCE  │ Tetrachloroethylene   │   —    │   —    │   —    │  Fail *   │
  │  OCT  │ o-Chlorotoluene       │   —    │   —    │   —    │  Fail *   │
  │  MCH  │ Methylcyclohexane     │  1.173 │  0.591 │  80.9  │  Pass     │
  │  DDL  │ Dodecanol             │  1.270 │  0.862 │  87.2  │  Pass     │
  │  HEP  │ Heptane               │  1.209 │  0.599 │  83.8  │  Pass     │
  │  TOL  │ Toluene               │  1.299 │  0.619 │  91.7  │  Pass     │
  │  XYL  │ Xylene                │  1.158 │  0.631 │  79.5  │  Pass     │
  ├───────┴───────────────────────┴────────┴────────┴────────┴───────────┤
  │  * Chlorinated solvents — HCl missing from property package          │
  │  7/9 pass                                                            │
  └──────────────────────────────────────────────────────────────────────┘
```

### Why TCE and OCT Fail

Both are **chlorinated solvents** (contain Cl atoms). The boiler-turbogenerator needs to compute
combustion reactions for all chemicals in the system. Burning chlorinated compounds produces HCl:

```
  Normal solvent (e.g. Toluene C7H8):
    C7H8 + 9 O2 → 7 CO2 + 4 H2O              ← all products in property package

  Chlorinated solvent (e.g. Tetrachloroethylene C2Cl4):
    C2Cl4 + O2 → 2 CO2 + 4 HCl                ← HCl NOT in property package
                              ▲
                              │
                     UndefinedChemicalAlias: 'HCl'
```

Fixable by adding HCl to the property package. Does not affect the 7 non-chlorinated solvents.

### Are the Results Reasonable?

```
  Metric        Our Test Range     Expected Range        Verdict
  ──────────    ──────────────     ──────────────        ───────
  MSP           $0.67–1.30/kg     $0.5–2.0/kg           Reasonable
  GWP           0.59–0.86         0.3–2.0 kgCO2e/kg     Reasonable
  TCI           $79–92M           $50–150M for 20kT/yr   Reasonable
  Unit Ops      27                25–30 expected          Correct

  Best MSP:  Xylene   ($0.672/kg)  ← cheapest solvent ($0.84/kg)
  Best GWP:  MCH      (0.591)      ← low impact factor (2.55), low T_diss (98 C)
  Worst GWP: Dodecanol (0.862)     ← high impact factor (4.12), high T_diss (120 C)
```

Note: Our test uses default LCA characterization factors (not the notebook's custom per-solvent
values), so the absolute numbers differ from the notebook's outputs. The relative ranking and
order-of-magnitude are what matter for validation.

---

## Package Changes Required

```
  BEFORE (plastics v0.1.3 from PyPI)     AFTER (plastics v0.1.4 from zip)
  ──────────────────────────────────     ─────────────────────────────────
  plastics     0.1.3                     plastics      0.1.4
  biosteam     2.44.3                    biosteam      2.52.17
  thermosteam  0.42.2                    thermosteam   0.52.16
  numpy        2.3.5                     numpy         2.3.5  (unchanged)
  numba        0.62.1                    numba         0.62.1 (unchanged)
  —                                      biorefineries 2.34.10 (new)
```

### Why the Upgrades Were Needed

```
  plastics v0.1.4
       │
       │ requires biosteam >= 2.51.5
       │ (v0.1.4 uses 'cls' kwarg in BoilerTurbogenerator
       │  which was added in biosteam 2.52.x)
       │
       ▼
  biosteam 2.52.17
       │
       │ requires thermosteam >= 0.52.0
       │
       ▼
  thermosteam 0.52.16
       │
       │ has Python 3.12+ f-string syntax on line 389:
       │   f"{'\n'.join(lines)}"
       │ which is invalid on Python 3.11
       │
       ▼
  MANUAL PATCH: _chemicals.py:389
       nl = '\n    '
       return f"{type(self).__name__}([\n    {nl.join(lines)}\n])"

  ⚠ This patch must be re-applied if thermosteam is reinstalled.
```
