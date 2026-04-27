# Case Study 1: Agentic Solubility Assessment of Mixed Plastic Waste

Status: draft scaffold

Primary goal: evaluate whether DISSOLVE can act as a separation engineer for changing mixed-plastic feedstocks and maintain coherent multi-turn context while producing solubility, selectivity, and sequence-analysis artifacts.

## Scientific Framing

This case study demonstrates how DISSOLVE can support dissolution-based recycling decisions when upstream sorting identifies polymer types and approximate mass fractions. The intended progression is:

1. Determine whether an individual polymer is soluble in a candidate solvent over a temperature range.
2. Compare solubility behavior for multiple polymers in the same candidate solvents.
3. Identify candidate solvents that selectively dissolve one polymer from a mixture.
4. Generate feasible separation sequences under process constraints such as maximum temperature.
5. Compare sequence choices under alternative objectives such as predicted separation efficiency, operating cost, and solvent greenness.

## Study Parts

### Part 1: LDPE/EVOH/PET Solubility And Sequence Analysis

Folder:

```text
01-ldpe-evoh-pet-solubility/
```

This is the first implementation target. The key agent-harness test is a multi-turn conversation where the user incrementally asks about:

- good solvents for dissolving any of LDPE, EVOH, or PET
- solubility plots for LDPE/EVOH/PET in dodecane and o-xylene from 25 to 100 deg C
- a state map of all 3! separation sequences under a 100 deg C max-temperature constraint
- sequence rankings under efficiency, cost, and greenness objectives

The agent should preserve context across turns and route each request to the correct solubility, separation, and visualization tools without confusing upstream prose with structured outputs.

### Part 2: Complex Co-Mingled Feedstock

Folder:

```text
02-complex-feedstock/
```

Later target. Feedstock:

```text
HDPE / LDPE / PP / PVC / PA6 / PA66 / PC / EVOH / PET / PS
```

This part should reuse the same workflow pattern after the simple LDPE/EVOH/PET case is stable.

## Artifact Policy

Each part has:

- `images/`: generated figures
- `json/`: structured tool outputs, handoffs, or typed-runtime ledgers
- `transcripts/`: DISSOLVE CLI or harness transcripts
- `notes/`: manual audit notes, figure captions, and interpretation drafts

Do not overwrite prior outputs during iterative runs. Use timestamped or composition-specific filenames.

## Validation Focus

For Part 1, the agent harness should be able to:

- maintain LDPE/EVOH/PET and 25 to 100 deg C context across turns
- generate a solubility plot for specified polymers, solvents, and temperature range
- generate a DP/separation state map for all feasible sequences
- separately evaluate efficiency, greenness, and cost-oriented sequence choices
- save all plots and structured outputs to this case-study folder
- produce final prose anchored to verified artifacts, not only intermediate text

