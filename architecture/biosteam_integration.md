# BioSTEAM Process Simulation Integration

## Overview

The DISSOLVE agent uses a unified **biosteam-analyst** subagent for all TEA/LCA
work. It runs rigorous STRAP process simulations via BioSTEAM (plastics v0.1.4)
with full flowsheet simulations: 27 unit operations, mass/energy balances, and
equipment sizing. The previous correlation-based `tea-lca-analyst` has been
merged into this subagent.

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────┐
│  Orchestrator (routing middleware)   │
│  "biosteam" / "process simulation"  │
│  → delegates to biosteam-analyst    │
└──────────────┬──────────────────────┘
               │  task()
               ▼
┌─────────────────────────────────────┐
│  biosteam-analyst  (7 tools)        │
│  6 domain + 1 reflection (think)    │
└──────────────┬──────────────────────┘
               │  tool call
               ▼
┌─────────────────────────────────────┐
│  Tool Layer  (biosteam_tea_lca.py)  │
│  Builds config dict, calls runner   │
└──────────────┬──────────────────────┘
               │  function call
               ▼
┌─────────────────────────────────────┐
│  Runner      (biosteam_runner.py)   │
│  Manages subprocess lifecycle,      │
│  parallelism, result formatting     │
└──────────────┬──────────────────────┘
               │  subprocess.run()
               ▼
┌─────────────────────────────────────┐
│  Worker      (biosteam_worker.py)   │
│  Isolated Python process per sim    │
│  BioSTEAM BaselineSTRAPProcess      │
│  27 unit ops, full mass/energy      │
└─────────────────────────────────────┘
```

Each simulation runs in an **isolated subprocess** to prevent BioSTEAM's global
state from contaminating subsequent runs.

## Agent Capabilities

### Tools

| Tool | Purpose | Typical use |
|------|---------|-------------|
| `run_biosteam_simulation` | Single solvent/energy case | "Run BioSTEAM for Toluene under C1" |
| `run_biosteam_batch` | Multi-solvent screening | "Compare all PE solvents", "Rank by MSP" |
| `compare_biosteam_scenarios` | Side-by-side custom scenarios | "Toluene 20k MT vs Xylene 50k MT" |
| `get_biosteam_solvents` | List supported configurations | "What solvents can BioSTEAM simulate?" |
| `visualize_biosteam_results` | Cost/GWP/scenario charts | "Plot cost breakdown from results" |
| `run_biosteam_multi_polymer` | Sequential multi-polymer recovery | "PE + EVOH blended MSP" |

### Metrics Returned

**TEA**: MSP ($/kg), TCI ($), AOC ($/yr)

**LCA**: GWP (kg CO2e/kg), HTC (CTUh/kg), HTNC (CTUh/kg), ETOX (CTUe/kg)

**Operations**: Water consumed/circulated (m3/yr), electricity/heating/cooling
duty (MJ/kg), waste generated/diverted (kg/yr), unit operation count

### Energy Configurations

| Case | Description | Facilities | Turbogenerator |
|------|-------------|:----------:|:--------------:|
| C1 | Combined Heat & Power (CHP) | Yes | Yes |
| C2 | Grid + AMCOR (no on-site utilities) | No | No |
| C3 | Grid + natural gas boiler | Yes | No |

### Supported Solvents

**PE (7 working)**:
sec-Butyl Acetate, Isobutyl Acetate, Methylcyclohexane, Dodecanol, Heptane,
Toluene, Xylene

**EVOH E1 (2)**: Ethylene Glycol, Pyridazine

**EVOH E2 (7)**: butane-1,4-diol, Diethanolamine, Diethylene glycol,
Ethylene Glycol, Propylene Glycol, Pyridazine, gamma-butyrolactone

**Excluded**: Tetrachloroethylene, o-Chlorotoluene (HCl not in property package)

### Batch Shorthand Keywords

- `all_pe` — all 7 PE solvents
- `all_evoh` — E1 EVOH solvents
- `all_evoh_e2` — E2 EVOH solvents
- `all` (energy_cases) — C1, C2, C3

## Routing

The biosteam-analyst (priority 3) handles all TEA/LCA queries:
- **Phrases** (strong match): "biosteam", "process simulation", "energy case",
  "techno-economic", "life cycle", "operating cost", "capital cost", "strap process"
- **High stems**: "msp", "ghg", "payback", "biosteam", "capex", "opex"
- **Low stems**: "tea", "lca", "emission", "cost", "gwp"

Multi-agent coordination:
- **Parallel** with `safety-analyst` (e.g., "BioSTEAM MSP + safety comparison")
- **Sequential** after `separation-engineer` (scheme first, then simulate)

## Guardrails

| Parameter | Value |
|-----------|-------|
| Max tool calls | 12 |
| Token budget | 200,000 |
| Synthesis triggers | All 6 domain tools |
| Tool result truncation | 2,000 chars |
| Free tools | `think` |

## Validation Results

| Solvent | MSP ($/kg) | GWP (kg CO2e/kg) | TCI ($M) | Runtime |
|---------|-----------|-------------------|----------|---------|
| Toluene | 1.14 | 0.53 | 83.3 | 14.3s |
| Heptane | 1.03 | 0.50 | 75.4 | 11.6s |

Batch mode (2 solvents, parallel): both succeed, results correctly ranked and
formatted as markdown tables.

## Dependencies

- `plastics==0.1.4`, `biosteam==2.52.17`, `thermosteam==0.52.16`
- thermosteam patched for Python 3.11 (`_chemicals.py:389` f-string fix)
- `biorefineries==2.34.10` (required by plastics)

## Files

| File | Lines | Layer |
|------|-------|-------|
| `vendor/biosteam_worker.py` | 352 | Subprocess worker |
| `vendor/biosteam_runner.py` | 408 | API / lifecycle manager |
| `tools/biosteam_tea_lca.py` | 560 | LLM tool wrappers |
| `subagents.yaml` | +47 | Subagent definition |
| `routing.py` | +18 | Routing rule + pairs |
| `agent.py` | +2 | Registry entry |
| `tools/__init__.py` | +8 | Tool group getter |
