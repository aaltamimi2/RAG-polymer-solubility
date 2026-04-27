# Artifact Checklist: LDPE/EVOH/PET

Use this checklist while running the DISSOLVE harness.

## Required Outputs

- [ ] Solvent discovery structured JSON
- [ ] Solubility plot for LDPE/EVOH/PET in dodecane and o-xylene from 25 to 100 deg C
- [ ] Solubility plot structured payload
- [ ] 3! separation sequence state map
- [ ] State-map structured payload
- [ ] Efficiency-ranked sequence result
- [ ] Greenness-ranked sequence result
- [ ] Cost or operating-burden-ranked sequence result, if supported
- [ ] Final manuscript-style summary
- [ ] Transcript of the multi-turn run

## Harness Checks

- [ ] Turn 2 uses same LDPE/EVOH/PET context from Turn 1
- [ ] Turn 3 calls separation/state-map tooling, not solubility plotting tooling
- [ ] Turn 4 answers from state-map artifacts
- [ ] Turn 5 does not rerun unrelated optimization
- [ ] Turn 6 states proxy limitations if true cost data are absent
- [ ] Final summary cites generated filenames

## Known Failure Modes To Watch

- Solubility plotter selected when user asks for a state map
- Agent loses PET after Turn 1
- Agent changes the 25 to 100 deg C temperature range without user instruction
- Agent invents a cost ranking without a named proxy or structured source
- Final answer cites upstream prose instead of generated artifacts

