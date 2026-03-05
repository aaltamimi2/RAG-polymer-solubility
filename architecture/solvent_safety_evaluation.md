# Solvent Safety Evaluation for Selectivity Spread Analysis

## Overview
This document evaluates the safety profile of 14 solvents used in polymer separation selectivity studies, examining whether lower LogP (octanol-water partition coefficient) correlates with improved safety. Data sourced from PubChem GHS classifications (February 2026).

## Summary Table (Sorted by LogP)

| Solvent | LogP | Key Hazards | GHS Signal | Environmental | Safety Rating |
|---------|------|-------------|------------|---------------|---------------|
| DMSO (dimethylsulfoxide) | -1.48 | Skin/eye irritation, may damage organs | Warning | Low | Medium |
| Ethylene glycol | -1.10 | Harmful if swallowed (H302), acute toxicity | Warning | Low | Poor |
| DMF (dimethylformamide) | -0.66 | Reproductive toxicity, harmful skin contact | Danger | Low | Poor |
| Propylene glycol | -0.41 | Organ damage (H370, H372), narcotic effects | Danger | Low | Poor |
| GVL (gamma-valerolactone) | -0.12 | Not classified (97% reports) | None | Very Low | Excellent |
| THF (tetrahydrofuran) | 0.87 | Highly flammable (H225), carcinogen | Danger | Moderate | Poor |
| 2,3-dihydropyran | 1.60 | Highly flammable, skin allergen, aquatic harm | Danger | Moderate | Medium |
| Cyclohexanol | 1.47 | Harmful if swallowed (H302), irritant | Warning | Low | Medium |
| Chloroform | 2.17 | Carcinogen, organ damage, acute toxicity | Danger | Low | Very Poor |
| Toluene | 2.53 | Flammable, reproductive toxin, narcotic | Danger | Moderate | Poor |
| o-Xylene | 2.94 | Flammable (H225), harmful if swallowed/inhaled | Danger | Moderate | Poor |
| Cyclohexane | 3.15 | Highly flammable, aspiration hazard, aquatic harm | Danger | Moderate | Poor |
| Diphenyl ether | 3.68 | Eye irritation, very toxic to aquatic life | Warning | High | Medium |
| Dodecane | 7.07 | Aspiration hazard (H304), flammable | Danger | Moderate | Medium |

## Analysis

### LogP vs. Safety Correlation

**The data reveals NO consistent correlation between lower LogP and better safety.**

Key findings:

1. **Best safety profile**: GVL (LogP = -0.12) - 97% of manufacturers report it does not meet GHS hazard criteria
2. **Worst safety profiles occur across the LogP spectrum**:
   - Low LogP: DMF (-0.66) = reproductive toxin; Ethylene glycol (-1.10) = acute oral toxicity
   - Mid LogP: Chloroform (2.17) = carcinogen + organ damage; Toluene (2.53) = reproductive toxin
   - High LogP: Cyclohexane (3.15) = aspiration hazard

### Critical Exceptions to LogP-Safety Hypothesis

1. **Ethylene glycol (LogP = -1.10)**: Despite very low LogP, it's harmful if swallowed with acute oral toxicity (H302). This is the classic antifreeze poisoning hazard.

2. **Propylene glycol (LogP = -0.41)**: Surprisingly rated Danger with organ damage warnings (H370, H372), despite being in many consumer products at low concentrations. This rating likely reflects concentrated industrial use.

3. **DMSO (LogP = -1.48)**: Low LogP but carries warnings for potential organ damage and skin permeation enhancement (can carry other toxins through skin).

4. **DMF (LogP = -0.66)**: Low LogP but classified as reproductive toxin and hepatotoxin.

5. **Chloroform (LogP = 2.17)**: Mid-range LogP but one of the worst overall due to carcinogenicity and multi-organ toxicity.

### Hazard Categories by Frequency

**Flammability** (9/14 solvents): THF, toluene, o-xylene, cyclohexane, dodecane, 2,3-dihydropyran, chloroform, cyclohexanol, DMF

**Acute toxicity** (6/14): Ethylene glycol, chloroform, cyclohexanol, propylene glycol, DMF, dodecane

**Reproductive/developmental toxicity** (2/14): DMF, toluene

**Carcinogenicity** (2/14): Chloroform, THF

**Aquatic toxicity** (4/14): Diphenyl ether, 2,3-dihydropyran, cyclohexane, dodecane

### Environmental Concerns

Solvents with high LogP (>3) show greater bioaccumulation potential:
- **Diphenyl ether (3.68)**: Very toxic to aquatic life with long-lasting effects (H410)
- **Dodecane (7.07)**: Moderate aquatic toxicity, high bioaccumulation risk
- **Cyclohexane (3.15)**: Aquatic harm from spills

Paradoxically, low LogP solvents (ethylene glycol, propylene glycol) have lower environmental persistence but higher acute aquatic toxicity per mass released.

## Recommendations

1. **Safest choice**: GVL (gamma-valerolactone) - excellent safety profile with minimal hazards
2. **Avoid**: Chloroform (carcinogen), DMF (reproductive toxin), THF (carcinogen + extremely flammable)
3. **Use with caution**: DMSO (skin permeation), ethylene glycol (oral toxicity), toluene (CNS effects)
4. **LogP is NOT a reliable safety predictor** - chemical structure, reactivity, and biological activity are far more important

## Data Source
All GHS classifications from PubChem REST API (https://pubchem.ncbi.nlm.nih.gov), aggregated from ECHA C&L Inventory and NITE-CMC databases. Retrieved February 2026.

**Note**: Safety ratings reflect industrial/laboratory concentrated use. Some solvents (e.g., propylene glycol) are safe in dilute consumer products but hazardous in pure form.
