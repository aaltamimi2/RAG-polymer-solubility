# MSP Allocation Methods in Multi-Polymer BioSTEAM Simulations

## What is MSP Allocation?

When a STRAP process recovers **multiple polymers** from the same feedstock, the shared costs (equipment, energy, labor) must be split across polymers. The **allocation method** determines each polymer's share of the total cost, which directly affects the per-polymer and blended **Minimum Selling Price (MSP)**.

## Two Methods

### Value-Weighted Allocation

Each polymer's cost share is proportional to its **market value** ($/kg):

| Polymer | Market Value | Weight |
|---------|-------------|--------|
| PE | $1.10/kg | 1.10 |
| EVOH | $4.50/kg | 4.50 |
| PET | $1.05/kg | 1.05 |

The blended MSP is:

```
blended_msp = sum(polymer_msp_i * weight_i) / sum(weight_i)
```

**Effect:** EVOH absorbs ~4x more cost per kg than PE/PET. This makes EVOH's per-polymer MSP appear lower (costs spread toward it) and inflates the blended MSP because the high-weight polymer dominates.

### Mass Allocation

Every polymer has **weight = 1.0** regardless of market value:

| Polymer | Market Value | Weight |
|---------|-------------|--------|
| PE | $1.10/kg | 1.0 |
| EVOH | $4.50/kg | 1.0 |
| PET | $1.05/kg | 1.0 |

**Effect:** Equal cost share per kg recovered. No premium for specialty polymers.

## Example: PE/EVOH/PET Literature Process (Toluene + DMSO + Ethylene Glycol)

### At 20,000 MT/yr

| Metric | Value Alloc (Trace 13) | Mass Alloc (Trace 14) |
|--------|----------------------|---------------------|
| PE MSP | $2.00/kg | $1.14/kg |
| EVOH MSP | $2.37/kg | $0.95/kg |
| PET MSP | $3.23/kg | $0.97/kg |
| **Blended MSP** | **$2.45/kg** | **$1.02/kg** |
| Blended GWP | 1.48 | 1.17 |
| Combined TCI | $65.3M | $221.9M |

### At 3,000 MT/yr (paper-matched capacity)

| Metric | Mass Alloc (Trace 15) |
|--------|---------------------|
| PE MSP | $2.26/kg |
| EVOH MSP | $1.98/kg |
| PET MSP | $2.01/kg |
| **Blended MSP** | **$2.08/kg** |
| Blended GWP | 1.20 |
| Combined TCI | $68.0M |

## Paper Comparison

| Source | Capacity | Allocation | MSP ($/kg) |
|--------|----------|-----------|------------|
| Green solvents paper | ~3k MT/yr | — | $1.44–1.62 |
| STRAP patent (STRAP-C) | 3k t/yr | — | $2.18 |
| **Trace 15 (ours)** | **3k MT/yr** | **mass** | **$2.08** |
| Trace 14 (ours) | 20k MT/yr | mass | $1.02 |
| Trace 13 (ours) | 20k MT/yr | value | $2.45 |

**Mass allocation at paper-matched capacity ($2.08) is within 5% of the STRAP patent ($2.18)** which includes extruders. The gap vs. the green solvents paper ($1.44–1.62) reflects differences in process assumptions and BioSTEAM model version.

## Which to Use?

- **Mass allocation** — closer to paper-reported values; appropriate when reporting a single blended MSP for all recovered output
- **Value-weighted** — better for business case analysis where specialty polymers (EVOH) command premium prices and should absorb proportionally more cost
- Papers typically report a single MSP for the whole process, which is conceptually closer to mass allocation

## Safety-Optimal Advantage (consistent across all methods)

| Trace | Capacity | Alloc | Lit MSP | Safe MSP | GWP Delta |
|-------|----------|-------|---------|----------|-----------|
| 13 | 20k | value | $2.45 | $2.40 | -24% |
| 14 | 20k | mass | $1.02 | $0.99 | -19% |
| 15 | 3k | mass | $2.08 | $2.03 | -18% |

The EVOH->PET->PE sequence (Methanol + DCM + Toluene) consistently outperforms the literature PE->EVOH->PET sequence (Toluene + DMSO + Ethylene Glycol) regardless of allocation method or capacity.
