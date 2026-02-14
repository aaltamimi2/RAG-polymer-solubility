# BioSTEAM Gap Analysis: Agent vs plastics-master-3

## Current biosteam-analyst (6 tools)

| Tool | Purpose |
|------|---------|
| `run_biosteam_simulation` | Single solvent/plastic/energy-case simulation |
| `run_biosteam_batch` | Multi-solvent comparison (up to 3 parallel subprocesses) |
| `compare_biosteam_scenarios` | Side-by-side custom scenario comparison |
| `get_biosteam_solvents` | List supported solvents and configurations |
| `visualize_biosteam_results` | Cost/GWP breakdown charts |
| `run_biosteam_multi_polymer` | Sequential multi-polymer recovery with blended metrics |

**Polymers**: PE (7 solvents), EVOH E1 (2 solvents), EVOH E2 (7 solvents)

**Metrics returned**: MSP, TCI, AOC, GWP, HTC, HTNC, ETOX

**Energy cases**: C1 (CHP), C2 (grid-only), C3 (grid + boiler)

---

## Capabilities in plastics-master-3 not yet exposed

### 1. Monte Carlo uncertainty quantification

`simulation.py` provides `run_monte_carlo()` and `sobol_analysis()` with stochastic price distributions (`data/price_distributions_2022.py`, `data/price_distributions_2023.py`). This answers the most common TEA question: "how certain are we about the MSP?"

Currently no tool lets the agent run N stochastic simulations and report confidence intervals.

### 2. Sensitivity / tornado analysis

Parameter sweeps with Spearman rank correlation to identify which input (feedstock price, solvent loss, dissolution temperature, etc.) drives MSP/GWP the most. The reference has 14 controllable parameters:

- `set_polymer_mass_fraction` (0.6-0.95)
- `set_dissolution_capacity` (5-10 wt%)
- `set_solvent_loss` (0.01-1.0)
- `set_precipitation_temperature` (313-323 K)
- `set_dissolution_temperature` (363-393 K)
- `set_centrifuged_plastic_solvent_content` (25-75%)
- `set_feedstock_distance` (50-1000 km)
- `set_feedstock_price` (0-0.10 $/kg)
- `set_solvent_price` (1.04-2.17 $/kg)
- `set_IRR` (0.10-0.25)

The agent cannot answer "which parameter matters most for MSP?"

### 3. Additional polymers

plastics-master-3 defines dissolution protocols for polymers the agent does not support:

| Polymer | Solvent | Temp (K) | Time (hr) |
|---------|---------|----------|-----------|
| PE+PP (PEPP) | Xylene | 403 | 0.5 |
| PC | THF | 336 | 0.5 |
| PET | (multiple protocols) | varies | varies |

PEPP = polyethylene + polypropylene blend, dissolved together since they have similar solubility in Xylene at 130 C.

### 4. Missing LCA indicators

plastics-master-3 tracks 8 impact categories. The agent only returns 4:

| Indicator | In agent? | Unit |
|-----------|:---------:|------|
| GWP (global warming) | Yes | kg CO2e/kg |
| HTC (human tox, cancer) | Yes | CTUh/kg |
| HTNC (human tox, non-cancer) | Yes | CTUh/kg |
| ETOX (ecotoxicity) | Yes | CTUe/kg |
| FFC (fossil fuel consumption) | No | MJ/kg |
| WU (water usage) | No | kg/L |
| ACD (acidification) | No | mol H+ eq/kg |
| OZD (ozone depletion) | No | kg CFC11 eq/kg |

### 5. MSW preprocessing simulation

`create_STRAPMSW_system` models MRF residue processing: hand sorting, eddy current detection, shredding, magnetic separation, crumbling. This is the upstream step before dissolution and is not accessible through the agent.

### 6. Ethanol co-production (biorefinery integration)

The biogenic fraction of MSW (cellulose/hemicellulose) can be converted to ethanol via cofermentation. `STRAPMSWProcess` integrates plastic recycling + biochemical conversion into a single flowsheet with combined economics. Not exposed.

### 7. Industrial scenario presets

plastics-master-3 includes ready-to-run case studies:

| Scenario | File | Description |
|----------|------|-------------|
| Amcor | `amcor.py` | Film packaging recycling (industrial partner) |
| Cytiva | `cytiva_tea_lca.py` | Chromatography column waste |
| Pilot plant | `pilot_plant.py` | 200 MT/yr bench scale |
| Supersacks | `supersacks.py` | Bulk container recycling |
| Pallet wrap | `pallet_wrap.py` | Wrapping film recovery |
| Exxon RFP | `EXXON_RFP.py` | Request for proposal analysis |

None of these are loadable as presets by the agent.

### 8. Detailed TEA tables

The reference generates itemized VOC/FOC breakdown, CAPEX equipment lists, cash flow projections, and NPV analysis. The agent only returns summary metrics (MSP, TCI, AOC).

### 9. Process flow / Sankey diagrams

Mass and energy balance Sankey diagrams (MRF flow, carbon flow) are available in the reference notebooks but not as agent tools.

---

## Recommended integration priorities

### High impact, moderate effort

1. **Monte Carlo tool** -- wrap `run_monte_carlo()` in a subprocess tool. Run N simulations (default 200), return MSP and GWP with 5th/50th/95th percentile confidence intervals. Uses existing price distributions.

2. **Sensitivity analysis tool** -- single-parameter sweeps across the 14 controllable parameters, return Spearman rank correlation coefficients and a tornado chart PNG.

3. **Full LCA indicators** -- small change in `biosteam_worker.py` to extract and return all 8 impact categories instead of 4. No new tool needed, just expand the result dict.

### Medium impact, moderate effort

4. **PEPP and PC support** -- add dissolution/precipitation protocols from plastics-master-3 to the worker. Expands polymer coverage from 2 to 4 types.

5. **Scenario presets tool** -- `load_biosteam_preset(name)` that configures and runs a known case study (Amcor, pilot plant, etc.) with published parameters.

### Lower priority

6. **MSW preprocessing** -- adds upstream modeling but most user queries focus on dissolution economics.

7. **Ethanol co-production** -- niche use case, complex integration.

8. **Detailed TEA tables** -- useful for reports but adds significant output size to tool results.

---

## File reference

| plastics-master-3 module | Purpose |
|--------------------------|---------|
| `plastics/strap/process_model.py` | `BaselineSTRAPProcess`, `STRAPMSWProcess` |
| `plastics/strap/systems.py` | System factory functions |
| `plastics/strap/dissolution_steps.py` | `DissolutionStep`, polymer protocols |
| `plastics/strap/precipitation_steps.py` | `PrecipitationStep`, recovery equilibria |
| `plastics/strap/units.py` | Custom BioSTEAM unit operations |
| `plastics/strap/tea.py` | `STRAPTEA` financial model |
| `plastics/strap/simulation.py` | Monte Carlo, Sobol, sensitivity |
| `plastics/strap/process_settings.py` | CFs, utilities, CEPCI |
| `plastics/strap/data/` | Price distributions, LCA CFs |
| `plastics/strap/tables.py` | Report generation |
