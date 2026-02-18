# New Polymer Parameter Specification

How the DISSOLVE agent resolves each parameter needed to define a new polymer/solvent pair for BioSTEAM simulation. Parameters are grouped by source: auto-resolved, user-provided, or agent-estimated with confirmation.

## Parameter summary

| # | Parameter | Source | Default | User override? |
|---|-----------|--------|---------|:--------------:|
| 1 | Repeat-unit formula | Auto-lookup from polymer name | — | Yes |
| 2 | Density (kg/m3) | Auto-lookup from polymer name | — | Yes |
| 3 | Heat capacity (J/g K) | Auto-lookup from polymer name | — | Yes |
| 4 | Melting point (K) | Auto-lookup from polymer name | — | Yes |
| 5 | Dissolution temperature (K) | COSMO-RS prediction from separation-engineer | — | Yes |
| 6 | Dissolution time (hr) | Hardcoded default | 0.5 hr | Yes |
| 7 | Dissolution capacity (wt/vol) | Hardcoded default | 0.03 | Yes |
| 8 | Centrifuge solvent content | Hardcoded default | 0.5 | Yes |
| 9 | Precipitation temperature (K) | Hardcoded default | 308.15 K (35 C) | No |
| 10 | Residual solubility (wt/wt) | Hardcoded default | 0 | Yes |
| 11 | Precipitate moisture (centrifuge) | Hardcoded default | 0.8 | Yes |
| 12 | Precipitate moisture (screw press) | Hardcoded default | 0.4 | Yes |
| 13 | Dissolved-form proxy molecule | LLM suggestion | — | User confirms |
| 14 | Solvent price ($/kg) | Scholar/web lookup | — | Yes |
| 15 | Solvent LCA CFs (GWP, HTC, HTNC, ETOX) | Scholar lookup | — | User confirms |

---

## Detailed parameter resolution

### 1-4. Basic physical properties

**Source:** Auto-lookup from polymer name with user override.

The agent maintains a built-in table of common polymers (PE, PP, PET, PVC, PS, PC, EVOH, Nylon-6, Nylon-66, PMMA, ABS, etc.) mapping name to formula, density, Cp, and Tm. For unknown polymers, the agent queries the scholar-researcher or web search.

```
User: "Run BioSTEAM for polycarbonate in THF"
Agent: Resolved PC → formula=C16H14O3, rho=1200 kg/m3, Cp=1.17 J/gK, Tm=540 K
       Using these values. Override any? [proceed / edit]
```

**What the user sees:** A confirmation of resolved values before simulation starts. They can override any value by providing their own.

**Implementation:** Static dict in `biosteam_worker.py` keyed by polymer name. Falls back to orchestrator asking user if polymer is not in the table.

### 5. Dissolution temperature

**Source:** COSMO-RS prediction from separation-engineer, with user override.

The separation-engineer already predicts dissolution temperatures for polymer/solvent pairs using COSMO-RS sigma profiles. When the biosteam-analyst needs to simulate a new pair, the orchestrator first routes to separation-engineer to get the predicted dissolution temperature, then passes it to biosteam-analyst.

```
Orchestrator flow:
  1. separation-engineer: predict_dissolution_temp(polymer="PC", solvent="THF")
     → returns 336 K (63 C)
  2. biosteam-analyst: run_biosteam_simulation(solvent="THF", dissolution_temp_c=63)
```

**What the user sees:** "Predicted dissolution temperature for PC/THF: 63 C (from COSMO-RS). Override? [proceed / edit]"

**Risk:** COSMO-RS predictions can be off by 10-20 C. The override option is critical for users with experimental data.

### 6. Dissolution time

**Source:** Hardcoded default of 0.5 hr, with user override.

All current polymers except EVOH/DMSOWater (4 hr) use 0.5 hr. This is a reasonable starting point for most dissolution processes. Users with kinetics data can override.

**Implementation:** Default in tool parameter definition. No lookup needed.

### 7-8. Dissolution capacity and centrifuge solvent content

**Source:** Hardcoded defaults with user override.

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Dissolution capacity | 0.03 wt/vol | 0.03-0.05 | Viscosity-limited; nearly constant |
| Centrifuge solvent content | 0.5 | 0.4-0.5 | Equipment-dependent, not polymer-dependent |

These are equipment parameters, not polymer properties. They vary little across the current polymer set. Advanced users with pilot-plant data may want to tune them.

**Implementation:** Default values in tool parameter definitions with optional override arguments.

### 9. Precipitation temperature

**Source:** Hardcoded at 308.15 K (35 C). No override.

All current polymers precipitate at 35 C. This is the standard STRAP cooling target and does not change with polymer type.

**Implementation:** Hardcoded constant in the worker. Not exposed as a user-facing parameter.

### 10. Residual solubility at precipitation temperature

**Source:** Default 0 (complete precipitation), with user override.

Most polymer/solvent pairs precipitate completely when cooled. The only exception in the current data is PEPP/Xylene (0.001 wt/wt). Users with experimental solubility curves can override.

**Implementation:** Default 0 in tool parameter definition. Optional override argument.

### 11-12. Precipitate moisture content

**Source:** Hardcoded defaults with user override.

| Parameter | Default | Notes |
|-----------|---------|-------|
| After centrifuge | 0.8 | 80% solvent by mass in wet cake |
| After screw press | 0.4 | 40% solvent by mass after mechanical drying |

These are equipment parameters. Same across all current polymers. Override available for users with specific equipment data.

### 13. Dissolved-form proxy molecule

**Source:** LLM suggestion with user confirmation.

The dissolved polymer form needs a small-molecule proxy for thermosteam to model its thermodynamic behavior in solution (vapor pressure, heat of mixing, etc.). Current mappings:

| Polymer | Proxy molecule | Rationale |
|---------|---------------|-----------|
| PE | 1-Hexene | Short-chain olefin, same C=C backbone |
| PC | 1-Heptene | Slightly longer chain for heavier repeat unit |
| EVOH | 3-buten-2-ol | Vinyl alcohol monomer analog |
| PEPP | 1-Hexene | Same as PE (polyolefin family) |

**Agent behavior:** Given a new polymer, the LLM suggests a proxy based on the repeat unit structure. For example:

```
User: "Add polystyrene"
Agent: PS repeat unit is C8H8 (styrene). Suggested proxy: Ethylbenzene
       (styrene monomer analog, similar MW and aromatic character).
       Use this proxy? [confirm / suggest alternative]
```

**Implementation:** LLM call within the biosteam-analyst's tool logic. The suggestion is presented to the user before the simulation proceeds. No simulation runs until the proxy is confirmed.

### 14. Solvent price

**Source:** Scholar/web lookup with user override.

For solvents not in the hardcoded price table (16 solvents currently), the agent uses the scholar-researcher to search for bulk industrial pricing from chemical suppliers, ICIS, or published TEA literature.

```
Agent: Solvent "gamma-valerolactone" not in price table.
       scholar-researcher found: ~$2.50/kg (bulk, 2024 ICIS estimate).
       Use this price? [confirm / enter your own]
```

**Implementation:** Orchestrator routes to scholar-researcher for a targeted price search. Result passed to biosteam-analyst as `solvent_price` parameter. User can override at any point.

### 15. Solvent LCA characterization factors

**Source:** Scholar lookup with user confirmation.

For new solvents, the agent needs cradle-to-gate impact factors:

| Factor | Unit | Example (Toluene) |
|--------|------|-------------------|
| GWP | kg CO2e/kg | 1.61 |
| HTC | CTUh/kg | 4.43e-07 |
| HTNC | CTUh/kg | 4.40e-07 |
| ETOX | CTUe/kg | 27.2 |

The scholar-researcher searches ecoinvent, published LCA studies, and GREET databases for these values. Results are presented to the user for confirmation before running the simulation.

```
Agent: Found LCA CFs for gamma-valerolactone from [Smith et al. 2023]:
       GWP=3.2, HTC=8.1e-07, HTNC=1.1e-06, ETOX=55.3
       Use these values? [confirm / enter your own]
```

**Implementation:** Scholar-researcher performs a targeted search. If no published CFs exist, the agent warns that LCA results will be incomplete and offers to run TEA-only.

---

## Agent workflow for a new polymer

```
User: "Simulate BioSTEAM for polystyrene dissolved in limonene"

Step 1: Resolve polymer properties
  → Auto-lookup: PS, formula=C8H8, rho=1050, Cp=1.3, Tm=513 K
  → Present to user for confirmation

Step 2: Resolve dissolution conditions
  → Route to separation-engineer: COSMO-RS predicts T_diss = 383 K for PS/limonene
  → Default time = 0.5 hr
  → Present to user for confirmation

Step 3: Resolve proxy molecule
  → LLM suggests: Ethylbenzene (styrene monomer analog)
  → User confirms

Step 4: Resolve solvent economics
  → scholar-researcher: limonene bulk price ~$4.50/kg
  → scholar-researcher: limonene GWP ~1.8 kg CO2e/kg (bio-based)
  → Present to user for confirmation

Step 5: Apply defaults
  → capacity=0.03, centrifuge_solvent=0.5
  → precip_T=308.15 K, solubility=0, precip_moisture=0.8/0.4

Step 6: Run simulation
  → biosteam-analyst calls run_biosteam_simulation with all resolved parameters
  → Returns MSP, TCI, AOC, GWP, HTC, HTNC, ETOX
```

---

## What the user must provide (minimum)

For a completely new polymer/solvent pair with no prior data:

1. **Polymer name** (everything else auto-resolves or defaults)
2. **Solvent name** (everything else auto-resolves or defaults)
3. **Confirm** LLM-suggested proxy molecule
4. **Confirm** scholar-found price and LCA factors

For a known polymer with a new solvent, only items 2-4 are needed.
For a known polymer/solvent pair already in the table, nothing — just run.
