# Solvent Expansion v8: 16 → 43 BioSTEAM Solvents

**Date:** 2026-02-14 | **Branch:** v8

## Summary

Expanded the BioSTEAM TEA/LCA simulation layer from 16 validated solvents to 43 total by adding all solvents from `COMMON-SOLVENTS-DATABASE.csv` that are registered in thermosteam's chemical database. This enables the separation-engineer → biosteam-analyst pipeline to simulate any solvent it recommends, closing the gap exposed in Trace 12 (LDPE/PET/EVOH pipeline) where DMSO and Dodecane were recommended but unavailable.

## Files Modified

| File | Changes |
|------|---------|
| `src/strap/vendor/biosteam_runner.py` | Added 27 entries to `_SOLVENT_DEFAULTS` and `_SOLVENT_LCA_IFS`; split `_PE_SOLVENTS` into `_CORE` + `_EXTENDED`; expanded EVOH E2 (7→13), PET (2→7), chlorinated blocklist (2→4) |
| `src/strap/tools/biosteam_tea_lca.py` | Mirrored solvent lists; added `_SOLVENT_ALIASES` dict for comma-in-name handling (DMF, DMSO, MEK, THF, IPA, DCM); updated `_expand_solvents()` with alias resolution; updated `get_biosteam_solvents()` display |
| `src/strap/routing.py` | Added EVOH phrase patterns to biosteam-analyst routing rule |

## Solvent Coverage by Polymer

| Polymer | Core (Branch-TEA) | Extended (new) | Total | Batch Keyword |
|---------|-------------------|----------------|-------|---------------|
| PE | 7 | 23 | 30 | `all_pe` |
| LDPE | 7 | 23 | 30 | `all_ldpe` |
| EVOH (E1) | 2 | — | 2 | `all_evoh` |
| EVOH (E2) | 7 | 6 | 13 | `all_evoh_e2` |
| PET | 2 | 5 | 7 | `all_pet` |

Chlorinated solvents (Tetrachloroethylene, o-Chlorotoluene, Dichloromethane, Chloroform) are in `_SOLVENT_DEFAULTS` for explicit use but excluded from batch keyword expansions.

## Data Quality Tiers

### Tier 1 — Validated (16 solvents)
Branch-TEA ecoinvent-derived price, GWP, HTC, HTNC, ETOX. No action needed.

### Tier 2 — Price/GWP from web research (7 solvents)
Acetone, 2-Butanone, THF, Ethyl acetate, DMF, DMSO, DCM. Prices from ChemAnalyst/ECHEMI. GWP from literature estimates. **HTC/HTNC/ETOX are class-average scaled — need ecoinvent validation.**

### Tier 3 — All estimated (20 solvents)
1-Propanol, 2,3-Dihydropyran, Acetylacetone, Benzene, Chloroform, Cyclohexane, Cyclohexanol, Diphenyl ether, Dodecane, Ethanol, Hexane, Isopropanol, Isopropylamine, Methanol, Methyl acetate, o-Xylene, p-Xylene, Tetrahydropyran, tert-Butanol, Triethylamine.

**All parameters (price, GWP, HTC, HTNC, ETOX) are estimates.** Prices based on commodity market knowledge; GWP from chemical-class averages; HTC/HTNC/ETOX scaled from validated solvents in the same class.

## Data Gaps to Fill

| Data Type | Solvents Affected | Source Needed |
|-----------|-------------------|---------------|
| **Price ($/kg)** | 20 Tier-3 solvents | ChemAnalyst, ECHEMI, or ICIS market reports |
| **GWP (kg CO2e/kg)** | 20 Tier-3 solvents | ecoinvent 3.x or GaBi LCA database |
| **HTC (CTUh/kg)** | All 27 new solvents | ecoinvent TRACI 2.1 characterization factors |
| **HTNC (CTUh/kg)** | All 27 new solvents | ecoinvent TRACI 2.1 characterization factors |
| **ETOX (CTUe/kg)** | All 27 new solvents | ecoinvent TRACI 2.1 characterization factors |

### Highest-Priority Gaps (solvents recommended in Trace 12)

| Solvent | Current Price | Current GWP | Issue |
|---------|-------------|-------------|-------|
| **Dodecane** | $1.80 (est) | 1.2 (est) | All parameters estimated |
| **DMSO** | $1.50 (lookup) | 2.8 (lookup) | HTC/HTNC/ETOX estimated |
| **Cyclohexane** | $0.90 (est) | 1.2 (est) | All parameters estimated |

## Audit Results

13/13 consistency checks passed:
- All solvents in every polymer list have matching `_SOLVENT_DEFAULTS` and `_SOLVENT_LCA_IFS` entries
- Chlorinated blocklists match between runner and tool layer
- Overlapping price/GWP values consistent between `biosteam_runner.py` and `solvent_lookup.py`
- No duplicate entries in any list
- Alias resolution handles DMF comma-in-name and common abbreviations
- EVOH, LDPE, PET all route correctly to biosteam-analyst
