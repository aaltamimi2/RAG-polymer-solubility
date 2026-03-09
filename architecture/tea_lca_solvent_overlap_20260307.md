# TEA/LCA Solvent Dataset Overlap Summary

## Scope

Compared:

- [60_common_solvents-TEA-LCA.csv](/tmp/strap-orchestration-redesign/data/60_common_solvents-TEA-LCA.csv)
- [solvent-econ-lca-summary.csv](/tmp/strap-orchestration-redesign/data/solvent-econ-lca-summary.csv)

Current BioSTEAM LCA sources checked:

- hardcoded solvent-specific IF table in [biosteam_runner.py](/tmp/strap-orchestration-redesign/src/strap/vendor/biosteam_runner.py)
- curated CSV fallback in [solvent-econ-lca-summary.csv](/tmp/strap-orchestration-redesign/data/solvent-econ-lca-summary.csv)

Comparison bases:

1. raw names
2. normalized names via [solvent_registry.py](/tmp/strap-orchestration-redesign/src/strap/solvent_registry.py)
3. CAS numbers

## Overlap Counts

- `60_common_solvents-TEA-LCA.csv` rows: `61`
- `solvent-econ-lca-summary.csv` rows: `100`
- raw-name overlap: `0`
- normalized-name overlap: `14`
- CAS overlap: `38`

The raw-name overlap is zero because the two files use different naming conventions.
CAS overlap is the meaningful chemical-identity comparison.

## Normalized-Name Overlap

These are the `14` solvents that match directly under current name normalization:

- Benzene
- Cyclohexane
- Cyclohexanol
- Diphenyl ether
- Dodecane
- Ethanol
- Ethylene Glycol
- Heptane
- Hexane
- Methanol
- Methyl acetate
- Toluene
- Triethylamine
- tert-Butanol

## Missing Solvent-Specific LCA Coverage

Important: under the current actual BioSTEAM LCA sources, the missing set is **23 solvents**, not 22.

That missing set is defined as:

- not present in the hardcoded solvent-specific LCA table in [biosteam_runner.py](/tmp/strap-orchestration-redesign/src/strap/vendor/biosteam_runner.py)
- and not present in [solvent-econ-lca-summary.csv](/tmp/strap-orchestration-redesign/data/solvent-econ-lca-summary.csv) by CAS

These solvents will currently fall through to chemical-class-average LCA estimates rather than solvent-specific compiled factors.

## 23 Solvents Missing Solvent-Specific LCA Parameters

| solvent_id | name_cosmobase | name_biosteam | CAS |
|---|---|---|---|
| 32 | ccl4 | ccl4 | 56-23-5 |
| 33 | cs2 | Carbon_disulfide | 75-15-0 |
| 35 | resorcinol | resorcinol | 108-46-3 |
| 36 | acetaldehyde | acetaldehyde | 75-07-0 |
| 38 | pyridine | pyridine | 110-86-1 |
| 39 | diethylether | diethylether | 60-29-7 |
| 40 | methyl-t-butylether | methyl-t-butylether | 1634-04-4 |
| 41 | diglyme | diglyme | 111-96-6 |
| 42 | chlorobenzene | chlorobenzene | 108-90-7 |
| 43 | 1,2-dichloroethane | 1,2-dichloroethane | 107-06-2 |
| 44 | nitromethane | nitromethane | 75-52-5 |
| 46 | ethylacetoacetate | ethyl_acetoacetate | 141-97-9 |
| 47 | camphene | camphene | 79-92-5 |
| 48 | dipentene | dipentene | 138-86-3 |
| 50 | hexamethylphosphoramide | hexamethylphosphoramide | 680-31-9 |
| 51 | styrene | styrene | 100-42-5 |
| 52 | pyrrole | pyrrole | 109-97-7 |
| 53 | naphthalene | naphthalene | 91-20-3 |
| 56 | aceticanhydride | acetic_anhydride | 108-24-7 |
| 57 | benzonitrile | benzonitrile | 100-47-0 |
| 58 | benzaldehyde | benzaldehyde | 100-52-7 |
| 59 | isophorone | isophorone | 78-59-1 |
| 60 | aceticacid | acetic_acid | 64-19-7 |

## Interpretation

- The new `60_common_solvents-TEA-LCA.csv` is now a good source for:
  - solvent selection
  - default `th`
  - default `price`
- It is **not** yet a replacement for solvent-specific LCA factors.
- For the `23` solvents above, current LCA results still rely on class-average fallback logic unless we add solvent-specific compiled IFs.

## Recommendation

If you want full solvent-specific LCA coverage for the new 60-solvent TEA/LCA set, the next data task is to add solvent-specific GWP / HTC / HTNC / ETOX parameters for the `23` solvents listed above.
