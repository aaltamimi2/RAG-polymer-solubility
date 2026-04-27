# Query Script: Complex Co-Mingled Feedstock

This script is intentionally not first priority. Use after the LDPE/EVOH/PET case is stable.

Output root:

```text
/home/aaltamimi2/langchain-STRAP-v10-core/case-studies/case-1/02-complex-feedstock
```

## Draft Turn 1: Data Coverage

```text
For a co-mingled feedstock containing HDPE, LDPE, PP, PVC, PA6, PA66, PC, EVOH, PET, and PS, report which polymers have usable DISSOLVE solubility/separation data and identify promising solvent families below 100 deg C. Save structured output under /home/aaltamimi2/langchain-STRAP-v10-core/case-studies/case-1/02-complex-feedstock/json.
```

## Draft Turn 2: Sequence Planning

```text
Using only the polymers and solvent candidates with usable data, propose feasible separation sequence options under a maximum temperature of 100 deg C. If the full search space is too large, explain the pruning strategy and save structured outputs and figures under the same case-study folder.
```

## Draft Turn 3: Objective Comparison

```text
Compare the top feasible sequence options under separation efficiency, solvent greenness, and cost or operating-burden proxies. Save all generated figures and structured outputs under the same case-study folder.
```

