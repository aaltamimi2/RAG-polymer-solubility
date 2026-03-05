# BioSTEAM Polymer Property Package

Source: `reference-scripts/plastics-master-3/plastics/strap/property_package.py`

All polymers route through the PE dissolution/precipitation flowsheet via `_TARGET_PLASTIC_MAP` in `biosteam_worker.py`.

## How Properties Affect TEA

All polymers currently route through the PE flowsheet, so the simulation uses **PE's properties** for the process model. The dedicated definitions below will matter when polymer-specific flowsheets are built.

### rho (Density) → CAPEX (strongest impact)
Vessel sizing uses volumetric flow: `F_vol = F_mass / rho`. Lower density → larger vessels → higher equipment purchase cost → higher TCI → higher MSP.

### Cp (Heat Capacity) → OPEX
Heating duty is `ΔH = ∫Cp·dT` when heating the polymer+solvent mixture to dissolution temperature. Higher Cp → more steam → higher utility costs → higher MSP.

### LHV (Lower Heating Value) → Energy Credits
Only matters in C1/C3 energy cases where leftover plastic is burned. Higher LHV → more combustion energy recovered → less natural gas → lower fuel costs → lower MSP. Also used for LCA energy allocation.

### Tm (Melting Temperature) → Constraint Only
Not directly used in cost calculations. Sets the minimum dissolution temperature to keep solvent liquid (`Tmin = solvent.Tm + 5K`). The actual dissolution temperature is an independent parameter from `DissolutionStep`.

### MW / formula → LCA
Carbon count determines CO₂ emissions if plastic is combusted. Minimal direct MSP impact unless carbon pricing is applied.

## Tunable Parameters Summary

| Parameter | Symbol | Units | TEA Impact | Mechanism |
|-----------|--------|-------|------------|-----------|
| **Density** | rho | kg/m3 | CAPEX | Vessel sizing → equipment cost → TCI |
| **Heat capacity** | Cp | J/(g·K) | OPEX | Heating duty → utility cost → AOC |
| **Lower heating value** | LHV | J/mol | OPEX | Energy recovery → fuel savings |
| **Melting temperature** | Tm | K | Constraint | Bounds feasible dissolution T |
| **Repeat unit formula** | formula | — | LCA | CO₂ emissions from combustion |

## Polymer Definitions

| Polymer | formula | rho | Cp | Tm (K) | Tm (°C) | LHV | Oligomer Proxy | Proxy CAS |
|---------|---------|-----|-----|--------|---------|-----|----------------|-----------|
| PE | C2H4 | 920 | 1.87 | 398 | 125 | set | 1-Hexene | 592-41-6 |
| PC | C16H14O3 | 1200 | 1.2 | 573 | 300 | — | 1-Heptene | 592-76-7 |
| PET | C10H8O4 | 1380 | 1.0 | 523 | 250 | set | — | — |
| EVOH | C2H4OC2H4 | 1130 | 2.4 | 451 | 177 | set | 3-buten-2-ol | 598-32-3 |
| Nylon6 | C6H11NO | 1130 | 1.7 | 533 | 260 | — | Caprolactam | 105-60-2 |
| Nylon66 | C12H22N2O2 | 1140 | 1.7 | 574 | 301 | — | Hexamethylenediamine | 124-09-4 |
| PS | C8H8 | 1050 | 1.3 | 516 | 243 | — | Styrene | 100-42-5 |
| PVC | C2H3Cl | 1380 | 0.9 | 546 | 273 | — | 1,2-Dichloroethane | 107-06-2 |
| PP | C3H6 | 905 | 1.9 | 461 | 188 | — | 4-Methyl-1-pentene | 691-37-2 |
| PES | C12H8O3S | 1370 | 1.37 | 613 | 340 | — | Diphenyl sulfone | 127-63-9 |

## Polymers Without Dedicated Definitions

LDPE and HDPE use PE's chemical properties directly (acceptable since they are polyethylene).

## Routing Map (_TARGET_PLASTIC_MAP)

All polymers → PE flowsheet:

```
PE, LDPE, HDPE, PET, PC, EVOH, PS, PP, PVC,
Nylon6, Nylon66, PA6, PA66, PES
```

## Missing LHV

LHV only affects C1/C3 energy cases (CHP). Currently set for PE, PET, EVOH only. Missing for: PC, PP, Nylon6, Nylon66, PS, PVC, PES.
