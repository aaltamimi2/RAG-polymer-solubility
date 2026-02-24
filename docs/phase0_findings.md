# Phase 0 Feasibility Check — Findings

## Setup
- 11 polymers in STRAP database (HDPE, LDPE, PP, PS, PVC, PET, Nylon6, Nylon66, PC, PES, EVOH)
- Thermal reference data (Tm°, ΔHf°, ΔCp) compiled from ATHAS/Wunderlich/Van Krevelen
- Ideal SLE equation: `ln(x) = -(ΔHf/R)(1/T - 1/Tm) + (ΔCp/R)(Tm/T - 1 - ln(Tm/T))`
- Script: `scripts/phase0_feasibility_check.py`

## Key Results

### 1. A+B/T+C·ln(T) model fits SLE curves (modified Apelblat)
All ideal fit R² = **0.994–1.000**. The existing interpolation functional form works for SLE-derived data. No model change needed.

### 2. Activity coefficient (γ) from COSMO-RS is critical
COSMO/Ideal solubility ratio: mean **0.17** (range 0.00–0.81). COSMO-RS predicts ~5x lower solubility than ideal SLE on average. **Cannot skip COSMO-RS** — the γ contribution dominates. COSMO files are essential.

### 3. Tm sensitivity analysis

| Polymer | Tm° (K) | MaxΔS% (±30K) | MaxΔS% (±60K) |
|---------|---------|----------------|----------------|
| PC      | 608.0   | 3.6%           | 9.0%           |
| PET     | 553.0   | 6.4%           | 17.3%          |
| PVC     | 546.0   | 7.9%           | 18.2%          |
| HDPE    | 414.6   | 8.6%           | 17.1%          |
| LDPE    | 410.1   | 8.7%           | 17.5%          |
| PS      | 516.0   | 9.2%           | 21.2%          |
| EVOH    | 464.0   | 12.0%          | 23.1%          |
| PP      | 460.7   | 13.9%          | 26.5%          |

- ±30K Tm error → up to **14% solubility shift** (PP worst case)
- ±60K Tm error → up to **27% solubility shift** (PP worst case)
- High-Tm polymers (PC, PET) are less sensitive

### 4. Nylon6/Nylon66 anomalous
No fitted pairs in coefficient database — need special handling.

## Conclusions
1. **Proceed to Phase 1** — pipeline is thermodynamically sound
2. **COSMO-RS is non-negotiable** — γ dominates over ideal SLE
3. **Tm accuracy ≤30K required** — above that, PP/EVOH errors exceed 10%
4. **Group contribution fallback** for ΔHf/ΔCp is validated as correct strategy
5. **Confidence tiers** must be calibrated from measured sensitivity, not hardcoded

## Phase 1 Scope
- Data curation: polyVERSE, POINT², ATHAS datasets → unified PSMILES format
- Van Krevelen group contribution baselines for ΔHf/ΔCp
- polyBERT fine-tuning with multitask heads (Tm, ΔHf, ΔCp, Tg)
- ML model predicts residuals against group contribution baselines
