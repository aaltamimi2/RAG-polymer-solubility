# Contaminant-Removal Agent Plan

## Goal
Add a new specialist that screens solvents for contaminant removal in two modes:

1. `leaching`
2. `strap_contaminant_removal`

This agent should work closely with `separation-engineer`, reuse existing polymer-solvent and atmospheric-feasibility logic where possible, and return structured results that can be handed back into the separation workflow.

## What Is In The Zhou Workbook
Source file: `data/zhou_contamintant_removal_SI_Data.xlsx`

Workbook sheets:
- `PFAS_Miscibility`
- `PFAS_log_D`
- `Phthalates_Miscibility`
- `Phthalates_log_D`

### PFAS sheets
- Wide solvent x contaminant matrices
- `Miscibility` sheet contains `Yes/No` solvent-contaminant miscibility data
- `log_D` sheet contains solvent-contaminant partition coefficients
- Solvent names are in a `Solvents` column
- Contaminants are PFAS species spanning carboxylic, sulfonic, and fluorinated ether groups

### Phthalate sheets
- `Phthalates_Miscibility` includes:
  - solvent name
  - boiling point
  - `T higher`
  - contaminant miscibility at `RT` and `T higher`
- `Phthalates_log_D` includes solvent-contaminant logD values

## Screening Modes To Implement

### 1. Leaching mode
Selection criteria:
- Target contaminants must be miscible in the solvent
- Contaminants should have `logP/logD > 0`; higher is better
- Solvent should swell but not dissolve the polymer

Interpretation for the agent:
- This is a polymer-retention process
- Polymer stays as a solid matrix
- Solvent is used to extract contaminants out of the polymer
- Polymer dissolution is a failure, not a success

### 2. STRAP contaminant-removal mode
Selection criteria:
- Solvent dissolves target polymer and target contaminants
- Solvent does not dissolve non-target compounds or non-target polymers
- Target polymer can be precipitated by cooling alone
- Contaminants should have positive partition coefficients; higher is better
- Contaminants must remain miscible in the solvent under the precipitation condition where polymer drops out
- Precipitation feasibility threshold: polymer solubility `< 1 wt%`

Interpretation for the agent:
- This is a polymer-dissolution / contaminant-retention process
- Polymer is dissolved first, then precipitated by cooling
- Contaminants should remain in the solvent phase while polymer precipitates
- This is temperature-swing only in v1, not antisolvent-driven contaminant removal

## Recommended Architecture

### A. Data layer
Add a dedicated parser/service instead of changing the global database loader.

Recommended new module:
- `src/strap/services/contaminant_data_service.py`

Responsibilities:
- read the Zhou workbook directly
- normalize the wide sheets into long-format records
- cache parsed records in memory
- normalize solvent names against the existing solvent registry when possible
- expose a stable API for contaminant families, contaminants, miscibility, and logD

Recommended normalized records:
- `contaminant_family`
- `contaminant_name`
- `solvent_name_raw`
- `solvent_name_normalized`
- `metric_type` = `miscibility` or `logd`
- `temperature_regime` = `rt`, `t_higher`, or `unspecified`
- `value`

Recommended service functions:
- `list_supported_contaminant_families()`
- `list_supported_contaminants(family: str | None = None)`
- `get_contaminant_miscibility(solvent, contaminants, regime)`
- `get_contaminant_logd(solvent, contaminants)`
- `get_supported_contaminant_solvents(contaminants)`

### B. Screening logic layer
Add a deterministic domain service.

Recommended new module:
- `src/strap/services/contaminant_screening_service.py`

Responsibilities:
- implement the two screening modes
- combine Zhou contaminant data with existing STRAP polymer-solvent data
- score and explain solvent candidates
- emit machine-readable candidate tables and explicit failure reasons

Recommended service entrypoints:
- `screen_leaching_candidates(...)`
- `screen_strap_contaminant_removal_candidates(...)`
- `compare_contaminant_removal_modes(...)`

## How To Combine Zhou Data With Existing STRAP Data

### Existing data we should reuse
- polymer dissolution / selectivity tools from `separation-engineer`
- solvent properties from `solvent_data` / solvent registry
- boiling point / atmospheric feasibility logic already used by `separation-engineer`
- precipitation feasibility tools and the existing `1 wt%` precipitation threshold

### Mode-specific integration

#### Leaching mode
For each solvent candidate:
1. contaminant miscibility must pass
2. contaminant logD should be positive
3. target polymer must *not* dissolve under the recommended operating condition
4. solvent should ideally swell the polymer

Important gap:
- we do **not** currently have a first-class polymer swelling dataset

Recommended v1 handling:
- treat swelling as a proxy criterion, not a hard measured fact
- rank candidate solvents by:
  - contaminant miscibility pass
  - contaminant logD
  - polymer remains non-dissolved
  - polymer is near the dissolution boundary or has borderline compatibility, which may indicate swelling potential
- explicitly label swelling as `proxy_inferred`, not experimentally validated

#### STRAP contaminant-removal mode
For each solvent candidate:
1. target polymer dissolves at the screening temperature
2. non-target polymers stay undissolved
3. contaminants are miscible in the solvent
4. contaminant logD is positive
5. target polymer precipitates below `1 wt%` on cooling
6. contaminants remain miscible under the cooled precipitation condition
7. operation stays below solvent boiling point at `1 atm`

## Recommended Tool Surface
Keep the tool layer small and deterministic.

Recommended new tools:
- `list_supported_contaminants(contaminant_family: str | None = None)`
- `screen_contaminant_leaching(target_polymer, contaminants, solvents=None, max_temperature_c=None)`
- `screen_contaminant_strap_removal(target_polymer, contaminants, other_polymers=None, solvents=None, max_temperature_c=None)`
- `compare_contaminant_removal_modes(target_polymer, contaminants, other_polymers=None, solvents=None, max_temperature_c=None)`

Tool output contract:
- use the standard JSON envelope already used elsewhere:
  - `display`
  - `data.success`
  - `data.tool_name`

## New Subagent
Recommended name:
- `contaminant-removal-analyst`

Why this name:
- explicit domain ownership
- distinct from `safety-analyst`
- close enough to `separation-engineer` to support sequential routing

Recommended responsibilities:
- solvent screening for contaminant removal
- contaminant-family support lookup
- explanation of why solvents pass/fail the leaching or STRAP-removal criteria
- handoff back to `separation-engineer` when a full route needs to be revised

Recommended routing triggers:
- `contaminant removal`
- `decontamination`
- `leaching`
- `remove PFAS`
- `remove phthalates`
- `extract contaminants`
- `retain contaminants in solvent`
- `clean polymer from contaminants`

## Recommended Structured Result Schema

```json
{
  "agent": "contaminant-removal-analyst",
  "schema_version": "1.0",
  "mode": "leaching",
  "target_polymer": "PET",
  "other_polymers": ["PE"],
  "contaminants": ["di-n-butyl phthalate (DBP)", "diethyl phthalate (DEP)"],
  "supported_contaminants": ["di-n-butyl phthalate (DBP)", "diethyl phthalate (DEP)"],
  "unsupported_contaminants": [],
  "candidate_solvents": [
    {
      "solvent": "acetone",
      "passes": true,
      "contaminant_miscibility": "pass",
      "contaminant_logd_min": 0.69,
      "contaminant_logd_avg": 0.77,
      "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
      "other_polymer_status": {"PE": "undissolved"},
      "polymer_precipitation_pass": null,
      "contaminants_retained_after_precipitation": null,
      "operating_temperature_c": 55.0,
      "boiling_point_c": 56.0,
      "caveats": ["swelling inferred from proxy, not direct swelling data"]
    }
  ],
  "recommended_solvents": ["acetone"],
  "decision_basis": [
    "all requested contaminants miscible",
    "positive logD for all requested contaminants",
    "target polymer not predicted to dissolve"
  ],
  "caveats": [
    "polymer swelling is proxy-inferred in v1",
    "experimental validation recommended"
  ]
}
```

## Handoff Plan
This agent should work closely with `separation-engineer`, so add typed handoffs in both directions.

### 1. `separation-engineer -> contaminant-removal-analyst`
Use when the user asks for:
- a separation plan with decontamination
- solvent screening that must preserve polymer purity and route feasibility
- contaminant-removal evaluation for a candidate separation solvent

Recommended handoff payload:
- target polymer
- non-target polymers
- candidate solvents from the separation plan
- operating temperature bounds
- precipitation temperatures if known
- supported / unsupported polymers
- route objective and user constraints

### 2. `contaminant-removal-analyst -> separation-engineer`
Use when the contaminant agent has screened solvents and the route needs to be revised.

Recommended handoff payload:
- ranked solvent candidates by mode
- disqualified solvents and reasons
- contaminant miscibility summary
- logD summary
- cooling-precipitation constraints
- atmospheric-pressure caveats

## Execution-Pair Changes
Recommended additions:
- sequential:
  - `separation-engineer -> contaminant-removal-analyst`
  - `contaminant-removal-analyst -> separation-engineer`
- optional later:
  - `contaminant-removal-analyst -> safety-analyst`
  - `contaminant-removal-analyst -> biosteam-analyst`

I would **not** add the safety/TEA pairings in v1 until the core contaminant workflow is stable.

## Implementation Phases

### Phase 1: Data ingestion
Files:
- `src/strap/services/contaminant_data_service.py`
- tests for workbook parsing and normalization

Deliverables:
- workbook parser
- solvent-name normalization
- contaminant-family listing
- miscibility/logD query API

### Phase 2: Deterministic screening service
Files:
- `src/strap/services/contaminant_screening_service.py`
- tests for both screening modes

Deliverables:
- leaching screening logic
- STRAP contaminant-removal screening logic
- explanation and ranking logic

### Phase 3: Agent-facing tools
Files:
- new tool module, likely `src/strap/tools/contaminant_removal.py`
- tests for standard tool envelopes and failure modes

Deliverables:
- supported contaminant listing
- leaching screen
- STRAP-removal screen
- mode comparison tool

### Phase 4: New specialist
Files:
- `src/strap/config/subagents/09_contaminant-removal-analyst.yaml`
- `src/strap/subagents.yaml`
- `src/strap/config/execution_pairs.yaml`

Deliverables:
- routing config
- specialist prompt
- structured result schema
- stop conditions

### Phase 5: Handoffs and routing integration
Files:
- `src/strap/handoff_adapters.py`
- `src/strap/handoff_store.py` or `src/strap/handoffs.py` only if new validation entries are needed
- routing tests

Deliverables:
- typed bidirectional handoffs with `separation-engineer`
- route enforcement
- fallback behavior for unsupported contaminants

### Phase 6: Benchmarking and live eval
Deliverables:
- focused contaminant-removal benchmark set
- route checks
- structured-result validation
- handoff validation with `separation-engineer`

## Tests Required

### Data-layer tests
- workbook sheets parse consistently
- phthalate RT / T-higher columns normalize correctly
- PFAS miscibility and PFAS logD records align by solvent/contaminant
- solvent-name normalization resolves expected aliases

### Logic tests
- leaching rejects solvents that dissolve the target polymer
- STRAP-removal rejects solvents that dissolve non-target polymers
- STRAP-removal rejects solvents where polymer does not precipitate below `1 wt%`
- STRAP-removal rejects solvents where contaminants fail miscibility at precipitation condition
- unsupported contaminant names produce structured failures, not silent drops

### Orchestration tests
- routing to `contaminant-removal-analyst`
- `separation -> contaminant` typed handoff
- `contaminant -> separation` typed handoff
- repeated handoffs append without overwrite

## Main Technical Risk
The only major data gap for a rigorous v1 is **polymer swelling**.

We have:
- contaminant miscibility
- contaminant logD
- polymer dissolution / precipitation data
- solvent boiling point and process constraints

We do **not** have:
- direct polymer swelling measurements for the candidate solvents

Recommended v1 approach:
- implement leaching with an explicit swelling proxy
- mark that proxy clearly in both tool outputs and specialist synthesis
- never present swelling as experimentally confirmed unless a direct dataset is added later

## Decisions I Want Your Signoff On
1. New subagent name: `contaminant-removal-analyst`
2. v1 scope limited to contaminant families present in the Zhou workbook:
   - PFAS
   - phthalates
3. `STRAP contaminant removal` means **cooling-induced polymer precipitation only** in v1, no antisolvent branch
4. Leaching mode will use a **swelling proxy** until we add a direct swelling dataset
5. Initial orchestration pairings will be only with `separation-engineer`, not safety/TEA yet

## Recommended Build Order
1. Data parser and normalization
2. Deterministic screening service
3. Agent-facing tools
4. New subagent config
5. Typed handoffs with `separation-engineer`
6. Focused contaminant-removal eval suite
