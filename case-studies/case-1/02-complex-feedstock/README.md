# Part 2: Complex Co-Mingled Feedstock

Status: placeholder for later extension

Feedstock:

```text
HDPE / LDPE / PP / PVC / PA6 / PA66 / PC / EVOH / PET / PS
```

This part should be attempted after the LDPE/EVOH/PET multi-turn workflow is stable.

## Intended Purpose

Stress-test DISSOLVE on a feedstock with many polymers, multiple feasible separation branches, and likely incomplete data coverage.

Expected complexity:

- candidate solvent discovery across many polymers
- unsupported or weakly supported polymer-solvent pairs
- larger separation sequence search spaces
- more need for pruning, ranking, and explicit assumptions
- stronger need for structured artifacts and progress reporting

## Planned Query Families

1. Broad solvent discovery by polymer.
2. Selectivity analysis for high-value or high-risk polymer subsets.
3. Feasible sequence generation under temperature constraints.
4. Greenness-oriented sequence ranking.
5. Cost or operating-burden ranking when supported.
6. Explicit unsupported-data report.

## Validation Focus

The agent should:

- clearly report which polymers are supported by available solubility data
- avoid silently dropping polymers
- avoid claiming full sequence coverage when pruning was required
- save structured outputs and figures under this folder
- provide progress updates for longer multi-step runs

