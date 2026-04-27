# Query Script: LDPE/EVOH/PET Multi-Turn Solubility Assessment

Use this as the first DISSOLVE harness script for Case Study 1.

Output root:

```text
/home/aaltamimi2/langchain-STRAP-v10-core/case-studies/case-1/01-ldpe-evoh-pet-solubility
```

## Turn 1: Solvent Discovery

```text
For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, identify solvents that are promising for dissolving any one of the components below 100 deg C. Focus on separation-engineering usefulness, not just listing every solvent. Save any structured output to /home/aaltamimi2/langchain-STRAP-v10-core/case-studies/case-1/01-ldpe-evoh-pet-solubility/json.
```

Expected:

- Candidate solvents grouped by target polymer.
- Notes on boiling-point or temperature constraints.
- No optimization or TEA/LCA unless explicitly requested.

## Turn 2: Solubility Plot

```text
Using the same LDPE/EVOH/PET feedstock, plot the predicted solubility or solubility-related response of all three polymers in dodecane and o-xylene from 25 to 100 deg C. Save the figure and structured data under /home/aaltamimi2/langchain-STRAP-v10-core/case-studies/case-1/01-ldpe-evoh-pet-solubility.
```

Expected:

- One or more figures in `images/`.
- Structured plot payload in `json/`.
- The plot should include LDPE, EVOH, and PET for both solvents when supported by the data/tools.

## Turn 3: Separation State Map

```text
Now generate a separation state map for all 3! possible LDPE/EVOH/PET separation sequences under a maximum processing temperature of 100 deg C. Save the state map figure and structured sequence-ranking output under the same case-study folder.
```

Expected:

- State-map figure, not a generic solubility curve.
- Structured sequence ranking.
- Clear statement of infeasible or weak sequence steps, if any.

## Turn 4: Efficiency Objective

```text
From that state map, which separation sequence maximizes predicted separation efficiency across each step under the 100 deg C constraint? Use the structured sequence results rather than re-describing the feedstock.
```

Expected:

- Answer anchored to the state-map or sequence-ranking artifact.
- Sequence order and key solvent/temperature choices.

## Turn 5: Greenness Objective

```text
Using the same candidate sequence space and 100 deg C maximum temperature, identify the greenest feasible separation sequence. Save any new comparison figure or structured output under the same case-study folder.
```

Expected:

- Greenness-focused ranking or comparison.
- Explicit tradeoff relative to efficiency-best sequence.

## Turn 6: Cost Or Operating Burden Objective

```text
Using the same LDPE/EVOH/PET feedstock and 100 deg C maximum temperature, identify the lowest-cost or lowest-operating-burden feasible separation sequence based on the currently available cost proxies. Save any new comparison figure or structured output under the same case-study folder.
```

Expected:

- Cost or burden proxy clearly defined.
- If true cost data are unavailable, the response must say so and use a named proxy only.

## Turn 7: Case-Study Summary

```text
Summarize the LDPE/EVOH/PET case-study results for a manuscript figure caption and short results paragraph. Reference the generated solubility plot, state map, and sequence-comparison artifacts by filename.
```

Expected:

- Short manuscript-style summary.
- Figure filenames.
- No unsupported claims beyond generated artifacts.

