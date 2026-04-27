# Part 1: LDPE/EVOH/PET Solubility And Sequence Analysis

Status: first active case-study target

Feedstock:

```text
LDPE / EVOH / PET
```

Default temperature window:

```text
25 to 100 deg C
```

Example candidate solvents for first plot:

```text
dodecane
o-xylene
```

## Intended Figure Set

Figure Xa:

- Solubility-related plot for LDPE/EVOH/PET in dodecane and o-xylene from 25 to 100 deg C.

Figure Xb:

- State map for all 3! LDPE/EVOH/PET separation sequences with maximum processing temperature of 100 deg C.

Figure Xc:

- Efficiency-ranked separation sequence comparison.

Figure Xd:

- Greenness-constrained or greenness-ranked sequence comparison.

Figure Xe:

- Cost-constrained or cost-ranked sequence comparison, if the currently available tools expose enough cost proxy data.

## Key Multi-Turn Harness Test

The core test is not one single prompt. It is whether DISSOLVE can keep the same feedstock and constraints active over several turns.

Expected conversation flow:

1. Ask which solvents look promising for dissolving any of LDPE, EVOH, or PET below 100 deg C.
2. Ask for a plot of LDPE/EVOH/PET in dodecane and o-xylene from 25 to 100 deg C.
3. Ask for the 3! separation state map under max 100 deg C.
4. Ask which sequence maximizes predicted separation efficiency.
5. Ask which sequence is best if solvent greenness is prioritized.
6. Ask which sequence is best if cost or operating-temperature burden is prioritized.
7. Ask for a concise summary of the tradeoffs with references to the generated figures.

## Expected Harness Behavior

The harness should:

- retain the polymer set without requiring repetition every turn
- retain the temperature range unless the user changes it
- treat dodecane and o-xylene as the requested solvents for the solubility plot
- use separation-planning tools for the state map, not generic solubility plotters
- use authoritative tool outputs and generated figures in the final synthesis
- save outputs under this folder

## Open Scientific Placeholders

To fill after runs:

- Best efficiency sequence under 100 deg C: `TBD`
- Best greenness sequence under 100 deg C: `TBD`
- Best cost-oriented sequence under 100 deg C: `TBD`
- Any infeasible sequence explanation: `TBD`

