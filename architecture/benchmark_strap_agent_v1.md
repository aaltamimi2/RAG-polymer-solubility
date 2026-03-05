# STRAP Agent Benchmark — Solvent Selection & Process Design

## Overview

This benchmark tests whether the STRAP dissolve agent can **reason** about solvent-based polymer recycling — applying the same logic and methodology as the core STRAP literature, not simply recalling facts. Each question requires multi-step reasoning, calculations, or process design decisions.

**Categories covered (v1):**
- Category 1: Solvent Selection & Screening (14 questions)
- Category 2: Process Design & Separation Sequences (13 questions)
- Category 3: Techno-Economic Analysis (5 questions)
- Category 4: Life Cycle Assessment (5 questions)

**Scoring:** Each question has an expected reasoning path and key checkpoints. The agent earns credit for demonstrating correct methodology, not for matching exact numbers.

---

## Category 1: Solvent Selection & Screening

### 1.1 — HSP-Based Solvent Screening

**Q1.1.1** (Difficulty: Basic)
> Which of toluene, DMSO, and hexane would you expect to dissolve PE based on Hansen solubility parameters? Show your work.

**Expected reasoning:**
- Agent should call `predict_solubility_ml()` for each polymer-solvent pair: (PE, Toluene), (PE, DMSO), (PE, Hexane)
- Tool returns Ra, R0, RED (=Ra/R0), and SOLUBLE/NON-SOLUBLE classification for each pair
- Expected results:
  - PE + Toluene: RED ≈ 0.57 → SOLUBLE → **good solvent**
  - PE + DMSO: RED ≈ 2.48 → NON-SOLUBLE → **non-solvent**
  - PE + Hexane: RED ≈ 0.29 → SOLUBLE → **good solvent** (best match, lowest RED)
- Agent should reason: RED < 1 means the solvent falls inside the polymer's Hansen sphere → dissolution expected. Lower RED = better solvent match. Hexane is the closest match, toluene is good, DMSO is far outside.
- **Key checkpoint:** Agent must (1) use `predict_solubility_ml` tool to retrieve RED/Ra values, (2) correctly interpret RED < 1 as the dissolution criterion, (3) compare RED values to rank solvent quality (hexane > toluene >> DMSO)

---

**Q1.1.2** (Difficulty: Intermediate)
> I want to find a solvent that dissolves EVOH but NOT PE. Would DMSO work as a selective solvent? Evaluate using Hansen solubility parameters.

**Expected reasoning:**
- Agent should call `predict_solubility_ml()` for: (EVOH, DMSO) and (PE, DMSO)
- Expected results:
  - EVOH + DMSO: RED ≈ 1.03 → borderline (near the sphere boundary)
  - PE + DMSO: RED ≈ 2.48 → NON-SOLUBLE → clearly outside PE's Hansen sphere
- Agent should reason: DMSO is a strong non-solvent for PE (RED >> 1) but borderline for EVOH (RED ≈ 1). This makes DMSO a potential selective solvent — it may dissolve EVOH while leaving PE intact.
- Agent should note HSP limitation: RED ≈ 1.03 is ambiguous — HSP is temperature-independent, but experimentally EVOH dissolves in DMSO at 95°C (24 wt%). Agent should recommend COSMO-RS or experimental validation for borderline cases.
- **Key checkpoint:** Agent must (1) use the tool to get RED values for both pairs, (2) interpret the large RED gap (2.48 vs 1.03) as evidence of selectivity, (3) flag HSP limitations for borderline predictions and suggest temperature-dependent methods

---

**Q1.1.3** (Difficulty: Advanced)
> I have a 3-polymer waste stream: PE, PS, and PVC. Which of THF, cyclohexanone, or acetone could selectively dissolve PS while leaving PE and PVC undissolved at room temperature? Look up the relevant HSP data and evaluate.

**Expected reasoning:**
- Agent should call `predict_solubility_ml()` for all 9 combinations: 3 solvents × 3 polymers
- Build a RED matrix to compare selectivity:
  |  | PS | PE | PVC |
  |--|----|----|-----|
  | THF | RED ≈ 1.26 (borderline) | RED >> 1 (non-solvent) | RED ≈ 1.54 (non-solvent per HSP, but experimentally dissolves PVC!) |
  | Cyclohexanone | RED ≈ 0.62 (**good**) | RED >> 1 (non-solvent) | check RED |
  | Acetone | RED >> 1 (non-solvent) | RED >> 1 (non-solvent) | RED >> 1 (non-solvent) |
- Agent should reason: Cyclohexanone has the lowest RED for PS (best solvent match) while having high RED for PE. THF is borderline for PS and experimentally dissolves PVC despite HSP predicting non-solubility.
- **Key checkpoint:** Agent must (1) run all 9 predictions, (2) compare RED values in a matrix to evaluate selectivity, (3) flag that THF's HSP prediction for PVC is wrong (experimentally dissolves it — HSP limitation), (4) recommend cyclohexanone as the PS-selective choice and suggest COSMO-RS or experimental cross-check

---

**Q1.1.4** (Difficulty: Expert)
> I operate a mixed-plastics recycling facility and receive waste containing up to 16 different polymers from 8 families:
>
> **Polyolefins:** PE, PP, HDPE
> **Styrenics:** PS, ABS
> **Vinyls:** PVC, PVDF
> **Polyesters:** PET, PETG
> **Acrylics/Carbonates:** PMMA, PC
> **Polyamides:** PA6, PA66
> **Engineering:** PSU (polysulfone)
> **Barrier/Specialty:** EVOH, POM (acetal)
>
> Before running expensive COSMO-RS temperature-dependent simulations, I want to do an initial room-temperature HSP screening to identify: (a) which polymers can potentially be dissolved at RT, (b) which solvents are most selective for each, and (c) which polymer pairs are too similar in HSP space to separate. Screen against at least 10 solvents spanning different chemical classes (nonpolar, aromatic, chlorinated, ester, ketone, polar aprotic, polar protic). Build a selectivity matrix, group the polymers by separability, and propose a RT separation sequence for the soluble subset. Flag all limitations.

**Expected reasoning:**

*Phase 1 — Systematic screening (~160 tool calls):*
- Agent should call `predict_solubility_ml()` for 16 polymers × 10+ solvents. Recommended solvents spanning HSP space:
  - Nonpolar: hexane, toluene
  - Chlorinated: DCM, chloroform
  - Esters/ethers: ethyl acetate, THF
  - Ketones: acetone, cyclohexanone
  - Polar aprotic: DMF, DMSO, NMP
  - Polar protic: methanol

*Phase 2 — Build selectivity matrix and identify patterns:*
- Agent should organize results into a RED matrix (16 rows × 10+ columns) and identify selectivity windows
- Expected patterns the agent should discover:

  | Polymer Family | RT-Soluble? | Best Solvents (RED < 1) | Key Insight |
  |----------------|:-----------:|-------------------------|-------------|
  | PE, PP, HDPE | HSP says yes, reality says **NO** | Hexane, toluene show low RED | Semi-crystalline — crystallinity barrier prevents RT dissolution despite HSP match. **Critical false positive.** |
  | PS | **Yes** (amorphous) | Toluene, THF, cyclohexanone, DCM, chloroform | Dissolves in many solvents at RT due to amorphous structure |
  | ABS | **Yes** (amorphous) | Cyclohexanone, THF, DMF | More polar than PS due to acrylonitrile component — different solvent window |
  | PVC | **Yes** (partially amorphous) | THF, cyclohexanone, DMF | Wide solvent window overlaps with PS/ABS |
  | PVDF | **Partially** | DMF, NMP only (very small R0 ≈ 4.1) | Very restrictive Hansen sphere |
  | PET | **No** at RT | No common solvent gives RED < 1 | Semi-crystalline polyester — needs aggressive solvents (TFA:DCM) or elevated temperature |
  | PETG | **Yes** (amorphous PET variant) | DCM, chloroform, THF | Unlike crystalline PET, PETG dissolves at RT |
  | PMMA | **Yes** (amorphous) | DCM, chloroform, THF, acetone | Wide solvent window |
  | PC | **Yes** (amorphous) | DCM, chloroform, THF | Strong response to chlorinated solvents |
  | PA6, PA66 | **No** at RT via HSP | Formic acid works experimentally but via protonation, not HSP matching | HSP cannot predict acid-catalyzed dissolution |
  | PSU | **Yes** (amorphous) | DCM, chloroform, DMF, NMP | Engineering plastic with specific solvent window |
  | EVOH | **No** at RT | DMSO borderline (RED ≈ 1.0) | Needs elevated temperature (95°C) per COSMO-RS |
  | POM | **No** at RT | Essentially chemically resistant | Only dissolves in hexafluoroisopropanol or similar |

*Phase 3 — Group polymers by separability:*
- **RT-Soluble subset (~8 polymers):** PS, ABS, PVC, PVDF, PETG, PMMA, PC, PSU
- **Not RT-soluble (~8 polymers):** PE, PP, HDPE, PET, PA6, PA66, EVOH, POM → recommend COSMO-RS temperature sweep
- Within the RT-soluble subset, identify **difficult pairs** (overlapping Hansen spheres):
  - PS vs PMMA: both dissolve in THF, DCM — but PMMA also dissolves in acetone while PS doesn't → **acetone is the differentiator**
  - PC vs PETG: both dissolve in DCM, chloroform — may need different solvent or temperature to separate
  - PVC vs ABS: both dissolve in cyclohexanone, THF — check for RED gap to find a selective solvent

*Phase 4 — Propose RT separation sequence for soluble subset:*
- Example sequence (agent may propose differently if justified):
  1. PVDF → NMP or DMF (only PVDF dissolves in these among the non-polar subset; very selective due to tiny R0)
  2. PS → toluene (dissolves PS but not PVC, PMMA, PC at RT in toluene — check RED gaps)
  3. PVC → THF (after removing PS, THF is selective for PVC over remaining polymers)
  4. PMMA → acetone (dissolves PMMA but not PC, PSU, PETG, ABS)
  5. PC → DCM (dissolves PC; check selectivity vs PETG, PSU)
  6. ABS → cyclohexanone (after removing other styrenics)
  7. PETG → chloroform (remaining polyester)
  8. PSU → NMP at RT (if not already dissolved in step 1 with PVDF — agent should check)

- **Key checkpoints:** Agent must:
  1. **Orchestrate large-scale screening** — systematically run 100-160+ `predict_solubility_ml()` calls and organize results
  2. **Identify the crystallinity false positive** — PE/PP/HDPE show RED < 1 with hexane/toluene but won't dissolve at RT due to crystalline packing. This is the single most important HSP limitation to flag
  3. **Recognize chemical mechanism limitations** — PA6/PA66 dissolve in formic acid via protonation, not HSP matching; POM needs specialty solvents
  4. **Group by separability** — distinguish RT-soluble (amorphous) from not-RT-soluble (semi-crystalline) and explain why
  5. **Identify difficult pairs** within the soluble subset where HSP overlap makes separation challenging
  6. **Propose a justified sequence** for the RT-soluble subset with selectivity rationale at each step
  7. **Recommend COSMO-RS follow-up** for the 8 polymers that cannot be separated at RT

---

### 1.2 — COSMO-RS & Temperature-Dependent Solubility

**Q1.2.1** (Difficulty: Basic)
> COSMO-RS predicts that PE solubility in toluene increases from ~0.1 wt% at room temperature to ~23 wt% at 110°C (near boiling point). EVOH solubility in toluene stays below 0.3 wt% even at 110°C. Can I use toluene to selectively dissolve PE from a PE/EVOH mixture? At what temperature should I operate?

**Expected reasoning:**
- Yes, toluene is PE-selective over EVOH across all temperatures
- Operating temperature should be at or near 110°C to maximize PE dissolution
- At 110°C: ~23 wt% PE dissolves vs <0.3 wt% EVOH → selectivity ratio >70:1
- The standard STRAP protocol uses toluene at 110°C for PE, which matches
- **Key checkpoint:** Agent should recommend 110°C and justify based on maximizing PE solubility while maintaining EVOH insolubility

---

**Q1.2.2** (Difficulty: Intermediate)
> I need to dissolve EVOH from a PE/EVOH/PET film. COSMO-RS data shows EVOH solubility in DMSO varies with temperature: ~3 wt% at 30°C, ~12 wt% at 60°C, ~20 wt% at 80°C, ~24 wt% at 95°C. PE and PET both stay below 0.5 wt% in DMSO at all temperatures. What dissolution temperature should I choose, and why? What if I need to precipitate the EVOH afterwards — how could I do it without antisolvent?

**Expected reasoning:**
- Choose 95°C for maximum EVOH dissolution (24 wt%)
- Selectivity: 24 / 0.5 = 48:1 ratio at 95°C
- For precipitation without antisolvent (STRAP-B approach): cool the solution from 95°C to 20-25°C
- At 25°C, EVOH solubility drops to ~3 wt%, so ~21 wt% of the dissolved EVOH will precipitate
- Recovery: (24-3)/24 ≈ 87.5% single-pass recovery
- **Key checkpoint:** Agent must connect temperature-dependent solubility to the STRAP-B concept and calculate approximate recovery

---

**Q1.2.3** (Difficulty: Advanced)
> My COSMO-RS predictions for a new polymer X show 30.8 wt% solubility in DMF at 120°C, but my experiment measured only 0.5 wt%. What are the most likely reasons for this discrepancy? How would you troubleshoot?

**Expected reasoning:**
- COSMO-RS can overpredict by 10-100× for certain polymer-solvent pairs (PET in DMF is a known example from the literature)
- Possible reasons:
  1. Oligomer model too short — may not capture crystalline packing effects
  2. Wrong copolymer structure (block vs alternating vs random gives different predictions)
  3. Missing crystallinity contribution — COSMO-RS computes liquid-phase interactions but doesn't fully account for crystalline melting barrier
  4. Reference solubility calibration point may be wrong
- Troubleshooting steps: (a) try longer oligomer, (b) adjust conformer sampling, (c) use experimental reference at a different temperature, (d) compare with HSP as cross-check
- **Key checkpoint:** Agent must identify crystallinity and oligomer modeling as key error sources, not just dismiss COSMO-RS

---

**Q1.2.4** (Difficulty: Intermediate)
> I want to screen solvents for separating Nylon 6 from a mixture with PE and PP. What solubility database would you use? How many solvents are available, and at what temperatures? What criteria define a "good" selective solvent?

**Expected reasoning:**
- The COSMO-RS database covers 8 polymers × 1007 solvents × 2 temperatures (RT and near-BP) = ~16,000 data points
- Nylon 6 is one of the 8 polymers in the database
- Criteria: Nylon 6 solubility >10-15 wt% at operating temperature, PE and PP solubility <1-2 wt%
- Known selective solvents for Nylon 6: formic acid at 25°C, acetic acid at elevated temperature
- Temperature selection: 1°C below solvent boiling point for upper bound
- **Key checkpoint:** Agent should reference the 1007-solvent database, state the selectivity thresholds, and suggest formic acid as a known N6 solvent

---

### 1.3 — Green Solvent Screening Framework

**Q1.3.1** (Difficulty: Intermediate)
> I've identified 5 solvents that dissolve EVOH selectively. Now I need to evaluate their "greenness." What framework should I use, and what criteria would eliminate unsafe or environmentally harmful solvents?

**Expected reasoning:**
- Apply the 8-step green solvent screening framework:
  1. Benchmark solubility at operating temperature
  2. Selective solubility screening (>15 wt% target, <2 wt% non-target)
  3. Operating temperature and boiling point >25°C
  4. Target polymer solubility >15 wt%
  5. Precipitation feasibility (<2 wt% at RT)
  6. Energy demand assessment (heat capacity × ΔT)
  7. Greenness: LogP < 3 (bioaccumulation), GSK scores, EHS hazard classes
  8. Full LCA & TEA assessment
- Eliminate solvents on REACH restricted list, EPA TRI list
- Prefer LogP < 3 to avoid bioaccumulation
- Check GHS hazard statements (H-codes) for carcinogenicity, mutagenicity, reproductive toxicity
- **Key checkpoint:** Agent should enumerate multiple screening criteria (LogP, regulatory, energy, safety) not just one

---

**Q1.3.2** (Difficulty: Advanced)
> Compare toluene and DMSO as STRAP solvents from a green chemistry perspective. Toluene: BP=111°C, LogP=2.7, GHS H304/H315/H361d/H373. DMSO: BP=189°C, LogP=-1.4, minimal GHS hazards. Both dissolve PE at 110°C (toluene: 23 wt%) and EVOH at 95°C (DMSO: 24 wt%). Which is "greener" and why? What are the trade-offs?

**Expected reasoning:**
- DMSO is greener: lower LogP (no bioaccumulation risk), fewer GHS hazards, not on REACH restricted list
- Toluene concerns: reproductive toxicity (H361d), aspiration hazard (H304), moderate bioaccumulation potential
- Trade-offs:
  - DMSO has higher BP (189°C vs 111°C) → more energy for distillation recovery
  - Energy requirement: DMSO needs ~310 J/g to heat from RT to 95°C vs toluene ~160 J/g to 110°C
  - But DMSO allows STRAP-B (temperature-controlled precipitation) which saves antisolvent
  - Toluene requires acetone antisolvent at 3:1 ratio
- Net assessment: DMSO greener per solvent mass but higher energy; toluene cheaper but safety concerns
- **Key checkpoint:** Agent must weigh BOTH safety/environment AND energy/economics, not just declare one winner

---

**Q1.3.3** (Difficulty: Intermediate)
> A candidate solvent has: boiling point 85°C, LogP=1.2, predicted EVOH solubility of 18 wt% at 84°C, PE solubility 0.3 wt%, PET solubility 0.1 wt%. Energy requirement ~140 J/g. No regulatory restrictions. Should this proceed to pilot testing? What concerns remain?

**Expected reasoning:**
- Passes selectivity: 18 wt% EVOH vs <0.3 wt% PE, <0.1 wt% PET
- Passes greenness: LogP=1.2 < 3, no restrictions
- Energy: 140 J/g is reasonable (<200 J/g threshold)
- **Concern:** BP=85°C is low — operating at 84°C is dangerously close to boiling
  - Solvent may flash/evaporate, causing safety hazards and losses
  - Pressurized vessel would be needed, increasing CapEx
  - Solvent recovery by distillation would be easy (low energy) but process control is harder
- Also check: precipitation feasibility at RT (<2 wt% EVOH at 25°C?)
- **Key checkpoint:** Agent MUST flag the low boiling point as a concern — operating 1°C below BP is the protocol but at 85°C there are practical safety issues

---

### 1.4 — Oligomer Modeling & Computational Parameters

**Q1.4.1** (Difficulty: Advanced)
> I'm setting up COSMO-RS predictions for polypropylene (PP). What oligomer chain length and how many conformers should I use? How do I validate my model?

**Expected reasoning:**
- Based on systematic studies: start with 6-mer PP (isotactic)
- Conformer sampling: 20-30 conformers via MD simulation
  - Use Rg (radius of gyration) vs SASA (solvent-accessible surface area) scatter plot to select representative conformers
  - Cluster conformers and pick representatives from each cluster
- Validation: need at least one experimental reference solubility
  - PP in toluene at 110°C: ~31.2 wt% (known reference point)
  - Calibrate solid-liquid equilibrium using this reference
- Cross-check: predict solubility in 2-3 other known solvents and compare to experiment
- Computational cost: DFT at BP86/TZVP/DGA1 level with CPCM, ~2 hours for 524 solvents after initial setup
- **Key checkpoint:** Agent should specify oligomer size (6-mer), conformer count (20-30), MD-based sampling method, and need for experimental reference

---

**Q1.4.2** (Difficulty: Advanced)
> Why does EVOH require more conformers (20-30) than PE (10) for accurate COSMO-RS predictions? And why does the block copolymer structure give better predictions than alternating or random for EVOH?

**Expected reasoning:**
- EVOH is a copolymer (ethylene + vinyl alcohol units) — more conformational diversity than homopolymer PE
- The vinyl alcohol -OH groups create intramolecular hydrogen bonds that vary with conformation
- More conformers needed to adequately sample the distribution of H-bond patterns
- Block structure: ethylene blocks and vinyl alcohol blocks segregate, which better represents real EVOH phase behavior
  - Block copolymer EVOH in DMSO: predicted 24 wt% (matches experiment)
  - Alternating: underpredicts at ~21 wt%
  - Random: underpredicts at ~20 wt%
- Real EVOH (32 mol% ethylene) has some blockiness in its microstructure
- **Key checkpoint:** Agent must connect conformational diversity to H-bonding and explain why block structure matters for solubility prediction accuracy

---

## Category 2: Process Design & Separation Sequences

### 2.1 — Separation Sequence Design

**Q2.1.1** (Difficulty: Basic)
> I have a multilayer film with three polymers: PE (outer layer), EVOH (barrier), and PET (inner layer). Design a STRAP separation sequence to recover all three polymers. Specify the solvent, temperature, and antisolvent for each step.

**Expected reasoning:**
- Step 1: Dissolve PE in toluene at 110°C (4 hours), filter out undissolved EVOH+PET. Precipitate PE with acetone (3:1 acetone:toluene ratio)
- Step 2: Dissolve EVOH in DMSO at 95°C (30 min), filter out PET. Precipitate EVOH with acetone (2:1 ratio)
- Step 3: Dissolve PET in TFA:DCM (1:1) at 25°C (30 min). Precipitate PET with methanol
- Logic: PE first because toluene is selective at high temp; EVOH second because DMSO is selective; PET last with aggressive solvent
- Expected recovery: PE ~98%, EVOH ~95%, PET ~97%
- **Key checkpoint:** Agent must specify correct solvent-polymer pairings, temperatures, AND antisolvents. Order matters — PE must come before EVOH (toluene at 110°C would damage EVOH if EVOH dissolved first)

---

**Q2.1.2** (Difficulty: Intermediate)
> For a 4-polymer waste stream (LDPE 60%, EVOH 5%, PET 25%, Nylon 6 10%), how many possible separation sequences exist? Which sequence would you recommend based on economics?

**Expected reasoning:**
- For N polymers sequentially separated, number of sequences = N! / 2 (for binary tree structures) or up to N! orderings
- For 4 polymers: 4! = 24 orderings, but many collapse to equivalent binary trees → ~6-12 unique sequences depending on tree structure
- Recommended sequence based on economics:
  - LDPE first (60% of feed — largest fraction, high-value recovery)
  - Then EVOH (small fraction but high value $2.23/kg)
  - Then Nylon 6 or PET last
- Economic logic: recover the largest mass fraction first to maximize revenue early; recover high-value polymers next
- From literature: LDPE-first sequence gives lowest MSP (~$0.48/kg) vs PET-first (~$0.78/kg)
- **Key checkpoint:** Agent should consider both mass fraction AND polymer value in determining optimal order

---

**Q2.1.3** (Difficulty: Advanced)
> I want to separate 7 polymers: LDPE, PET, PP, PC, EVOH, PVC, Nylon 6. How would you approach designing the separation sequence computationally? What algorithm would you use?

**Expected reasoning:**
- With 7 polymers, brute-force enumeration of all sequences is feasible but large (7! = 5040 orderings)
- Use a greedy algorithm or dynamic programming (bitmask DP):
  1. For each remaining mixture, find the polymer-solvent pair with highest selectivity
  2. Remove that polymer, update the remaining set
  3. Repeat until all polymers separated
- For each step, screen solvents using the COSMO-RS 1007-solvent database
- Selectivity metric: max(target_solubility - max_other_solubility) at optimal temperature
- Then evaluate top sequences with BioSTEAM for TEA/LCA
- The DP approach maximizes minimum selectivity across all steps
- Literature found 36 feasible sequences for a similar 7-polymer case
- **Key checkpoint:** Agent should describe the algorithmic approach (DP or greedy), mention computational solvent screening, and note that not all orderings are feasible

---

**Q2.1.4** (Difficulty: Intermediate)
> Why is the order of polymer dissolution important? What happens if I try to dissolve EVOH before PE from a PE/EVOH multilayer film using the standard STRAP solvents?

**Expected reasoning:**
- EVOH dissolution requires DMSO at 95°C
- At 95°C, PE is NOT soluble in DMSO (<0.5 wt%) — so PE layer remains as physical barrier
- In a laminated film, the PE outer layer physically encapsulates the EVOH barrier layer
- If EVOH is sandwiched between PE layers, DMSO cannot access it without first removing PE
- Correct order: dissolve PE first (toluene, 110°C) to expose EVOH, then dissolve EVOH (DMSO, 95°C)
- Exception: if film is shredded to small pieces, solvent can access EVOH from cut edges — but dissolution is slower
- **Key checkpoint:** Agent must mention physical accessibility (layer structure) as the primary reason for ordering, not just selectivity

---

### 2.2 — Antisolvent Selection & Precipitation

**Q2.2.1** (Difficulty: Basic)
> PE is dissolved in toluene at 110°C (solution concentration ~10 wt%). I want to precipitate the PE. What antisolvent should I use, at what ratio, and what recovery do I expect?

**Expected reasoning:**
- Antisolvent: Acetone (most common for PE precipitation from toluene)
- Ratio: 3:1 acetone:toluene by volume
- Process: Cool solution to ~60°C, then add acetone
- PE solubility drops to near 0 in toluene/acetone mixture → ~97-98% recovery
- Alternative: cool to 25°C without antisolvent (STRAP-B) — PE solubility in toluene drops to ~0.1 wt% at RT, so ~99% precipitates
- **Key checkpoint:** Agent should state acetone at 3:1 ratio OR identify cooling as an alternative

---

**Q2.2.2** (Difficulty: Intermediate)
> I dissolved EVOH in DMSO at 95°C. Now I have two options for precipitation: (A) add acetone as antisolvent at 2:1 ratio, or (B) cool to 20°C and add a small amount of water. Compare the two approaches in terms of recovery, cost, and process complexity.

**Expected reasoning:**
- **Option A (STRAP-A):** Add acetone 2:1 → near-complete EVOH precipitation, ~95% recovery, high purity. Cost: large volume of acetone needed, must recover acetone+DMSO by distillation (energy-intensive separation of two high-BP solvents)
- **Option B (STRAP-B):** Cool to 20°C → EVOH solubility drops from ~24 wt% to ~3 wt%, recovery ~87%. Add small amount of water to further reduce solubility. Cost: only energy for cooling, 90% less antisolvent needed. Simpler distillation (just DMSO+water)
- Trade-off: STRAP-A gives higher single-pass recovery but higher cost; STRAP-B gives lower recovery but 90% antisolvent reduction
- STRAP-B is preferred when: (1) lower CapEx needed, (2) antisolvent cost is dominant, (3) multi-pass operation acceptable
- **Key checkpoint:** Agent must quantify the ~90% antisolvent reduction and the recovery trade-off

---

**Q2.2.3** (Difficulty: Advanced)
> For precipitating PET, the standard approach uses methanol as antisolvent after dissolving in TFA:DCM (1:1). Why can't you use temperature-controlled precipitation (STRAP-B) for PET? What limits STRAP-B to certain polymer-solvent pairs?

**Expected reasoning:**
- STRAP-B requires strong temperature-dependent solubility — large Δ(solubility) between Thot and Tcold
- PET in TFA:DCM: dissolution occurs at 25°C (room temperature) — there's no hot-to-cold gradient to exploit
- PET doesn't show significant temperature-dependent solubility in most solvents within the practical range
- STRAP-B works best when:
  1. Dissolution requires elevated temperature (>60°C)
  2. Solubility drops dramatically upon cooling (>10× decrease)
  3. Solvent system allows internal antisolvent effect (e.g., water in DMSO)
- Examples where STRAP-B works: PE/toluene (0.1→23 wt%), EVOH/DMSO (3→24 wt%)
- Examples where it doesn't: PS/cyclohexanone (dissolves at 25°C), PET/TFA:DCM (dissolves at 25°C)
- **Key checkpoint:** Agent must identify that STRAP-B requires a large temperature-solubility gradient and explain why room-temperature dissolution precludes it

---

### 2.3 — Dissolution Kinetics & Process Optimization

**Q2.3.1** (Difficulty: Intermediate)
> I'm dissolving PS waste in cyclohexanone at 25°C. The dissolution follows first-order kinetics. What process parameters can I adjust to speed up dissolution, and which has the biggest impact?

**Expected reasoning:**
- Key parameters (ranked by impact):
  1. **Particle size reduction** — 10× smaller diameter ≈ 10× faster dissolution (surface area effect). Shredding to <3mm is critical
  2. **Stirring speed** — increase from 200 to 400 RPM gives significant improvement; diminishing returns above 400 RPM
  3. **Temperature** — dissolution rate roughly doubles every 15°C increase (Arrhenius). But PS in cyclohexanone dissolves at 25°C, so heating is optional
  4. **Impeller type** — pitched blade turbine (axial flow) more efficient than Rushton turbine (radial flow) for dissolution
  5. **Solvent-to-polymer ratio** — higher ratio maintains concentration gradient driving force
- For industrial scale: baffled reactor design improves mixing efficiency
- **Key checkpoint:** Agent should rank particle size as most impactful and mention the diminishing returns of stirring speed

---

**Q2.3.2** (Difficulty: Intermediate)
> PE dissolution in toluene at 110°C takes 4 hours with whole film pieces but only 8 minutes with 3mm shredded pieces. Explain the 30× speedup and estimate what dissolution time you'd expect for 1mm pieces.

**Expected reasoning:**
- Dissolution is surface-area limited — solvent must penetrate from the outside
- Whole film: large pieces, low surface-area-to-volume ratio, long diffusion path
- 3mm pieces: ~30× more surface area per unit mass
- For 1mm pieces: surface area scales as 1/d (diameter), so 3× more surface area than 3mm
- If first-order kinetics: time ∝ d² (Fick's law, diffusion-limited)
  - From 3mm to 1mm: (1/3)² = 1/9 of the time
  - Estimated: 8 min / 9 ≈ 0.9 minutes, so <1 minute
- But practical limit: very fine particles may create filtration problems, gel formation
- Optimal: 1-3mm particle size balances dissolution speed with downstream processing
- **Key checkpoint:** Agent should apply diffusion/surface-area scaling logic, not just say "smaller is faster"

---

**Q2.3.3** (Difficulty: Basic)
> What is a typical solvent-to-polymer ratio for STRAP dissolution, and why can't you just use minimal solvent?

**Expected reasoning:**
- Typical ratio: 7:1 to 60:1 mL solvent per gram polymer (varies by paper/application)
- Lab scale: often 60:1 for complete dissolution
- Pilot/industrial: 7:1 to 20:1 for economic reasons
- Why not minimal solvent:
  1. **Viscosity** — polymer solutions above ~20 wt% become extremely viscous (gel-like), especially PE. Hard to stir and filter
  2. **Incomplete dissolution** — not enough solvent to dissolve all polymer
  3. **Mass transfer** — need concentration gradient to drive dissolution
  4. **Selectivity** — higher polymer concentration may cause co-dissolution of non-target polymers
- Entanglement concentration for PE: ~14 wt% — above this, viscosity increases dramatically
- **Key checkpoint:** Agent must mention viscosity/gel formation as the primary constraint

---

### 2.4 — Multi-Polymer Case Studies

**Q2.4.1** (Difficulty: Advanced)
> Design a STRAP process to recover PP from disposable face masks. The masks contain ~81 wt% PP fibers, plus elastic bands (polyurethane), metal nose clips, adhesives, and dyes. Specify: pretreatment, dissolution conditions, purification, and expected product quality.

**Expected reasoning:**
- Pretreatment: (1) Remove metal nose clips manually or magnetically, (2) Cut/shred masks to <5mm pieces, (3) Remove elastic ear loops (different polymer)
- Dissolution: Toluene at 110°C, 20:1 solvent ratio, 30 minutes, continuous stirring
- Filtration: Hot filtration to remove undissolved PU, adhesives, other contaminants
- Precipitation: Add acetone (3:1 ratio) at room temperature
- Color removal (critical step): Re-dissolve PP in DMAc at 165°C with activated carbon → removes dyes
- Expected results: >95% PP recovery, >99% purity, minimal Mw degradation
- Post-processing: Dry precipitated PP, melt-process into pellets
- **Key checkpoint:** Agent MUST include decolorization step — colored PP has limited market value

---

**Q2.4.2** (Difficulty: Advanced)
> A printed multilayer film has PE/ink/PET structure (reverse-printed). When I dissolve PE in toluene at 110°C, the Yellow 12 diarylide pigment partially dissolves and co-precipitates with PE, giving it a yellow color. How do I solve this?

**Expected reasoning:**
- Problem: diarylide pigments partially decompose at STRAP temperatures, becoming soluble in hot alkane solvents
- Two-pronged approach:
  1. **Mechanical (filtration optimization):** Use fine filter paper (25 μm), apply compression filtration to squeeze out pigment-laden solvent from PE precipitate. This reduces solvent retention (Vsolv/mPE from 16 to 3.5 mL/g)
  2. **Chemical (activated carbon adsorption):** Add activated carbon pellets to the hot PE/toluene solution before precipitation. AC adsorbs dissolved/decomposed pigments. Filter out AC at temperature before cooling/precipitation
- Combined approach achieves near-virgin PE color (Yellowness Index ≈ -5.2 vs virgin -5.2)
- Quantitative: piston compression + AC reduces colorant to 0.4 ppm (vs virgin 0.3 ppm)
- **Key checkpoint:** Agent must identify BOTH mechanical and chemical decolorization pathways and explain why pigments become soluble at STRAP temperatures

---

**Q2.4.3** (Difficulty: Intermediate)
> I'm processing a 5-layer film: PE/PETG/EVOH/PET/EVA. This is more complex than the standard 3-layer PE/EVOH/PET. What additional challenges does this create, and how would you modify the standard STRAP-A sequence?

**Expected reasoning:**
- Additional challenges:
  1. PETG (amorphous PET) has different solubility than crystalline PET — need separate removal step
  2. EVA (ethylene vinyl acetate) is similar to PE — toluene may dissolve both
  3. More steps = more solvent, more processing time, higher cost
- Modified sequence (STRAP-C approach):
  1. PETG: Dissolve in 60% DMSO / 40% water at 60°C (PETG dissolves, EVOH and PET do not)
  2. PE: Toluene at 110°C (standard)
  3. EVOH + PET: DMSO-water at 90°C dissolves both, then cool → EVOH precipitates first (at 35°C) while PET stays dissolved
  4. PET: Recover from remaining solution
  5. EVA: Toluene at 110°C, similar to PE
- Temperature-controlled fractionation of EVOH/PET is a key innovation
- **Key checkpoint:** Agent should use DMSO-water cosolvent for PETG and identify the temperature-based EVOH/PET fractionation

---

**Q2.4.4** (Difficulty: Intermediate)
> For the 10-polymer separation system (LDPE, HDPE, PS, PVC, EVOH, PET, PP, PA6, PA66, PA66/6), why are some polymers dissolved at room temperature (PS in cyclohexanone, PVC in THF, PET in TFA:DCM) while others require elevated temperature (PE in toluene at 110°C, HDPE in decalin at 180°C)?

**Expected reasoning:**
- Fundamental reason: polymer crystallinity and solubility parameter matching
- **Room-temp dissolution (amorphous polymers):**
  - PS: amorphous, low Tg (~100°C), easily penetrated by solvents
  - PVC: partially amorphous, polar — THF is an excellent match
  - PET in TFA:DCM: strong acid breaks hydrogen bonds; DCM swells the polymer
  - PA6 in formic acid: acid protonates amide groups, disrupting H-bond network
- **High-temp dissolution (semi-crystalline polymers):**
  - PE: highly crystalline (>50%), solvent must overcome crystal lattice energy. Temperature above ~90°C needed to begin dissolving crystalline regions
  - HDPE: even higher crystallinity than LDPE, needs 180°C with decalin
  - PP: isotactic PP is highly crystalline, needs 110°C with toluene
- **General rule:** crystallinity requires thermal energy to disrupt crystal packing before solvent can dissolve
- EVOH in DMSO at 95°C: crystalline segments from vinyl alcohol H-bonding
- **Key checkpoint:** Agent must connect crystallinity to dissolution temperature requirements

---

### 2.5 — Process Economics & Scale-Up

**Q2.5.1** (Difficulty: Intermediate)
> A STRAP plant processes PE/EVOH/PET multilayer film at 3,000 ton/yr. Capital cost is $22.4 million. The key cost drivers are: solvent purchase, energy for heating/distillation, and antisolvent. Which of these is most impactful, and how would switching from STRAP-A to STRAP-B change the economics?

**Expected reasoning:**
- Major cost drivers (in order):
  1. **Solvent/antisolvent** — even at >99% recovery, makeup costs are significant at scale
  2. **Energy** — heating solvents to 95-110°C, distilling solvent-antisolvent mixtures
  3. **Capital depreciation** — distillation columns, reactors, filtration equipment
- Antisolvent is particularly expensive because:
  - 2:1 to 3:1 ratio means 2-3× more solvent volume to process
  - Must separate antisolvent from process solvent by distillation
- STRAP-B impact:
  - 90% reduction in antisolvent → major savings in (a) antisolvent purchase, (b) distillation energy, (c) smaller distillation column (CapEx)
  - Trade-off: need more robust heating/cooling system, slightly lower single-pass recovery
- Sensitivity: ±30% change in solvent/polymer ratio impacts MSP by ±$0.15/kg
- **Key checkpoint:** Agent should identify antisolvent as the dominant cost and quantify the STRAP-B advantage

---

**Q2.5.2** (Difficulty: Advanced)
> Solvent recovery rate is >99.9% in the STRAP pilot plant. Why is this critical? If recovery drops to 99%, how does that affect economics for a 3,000 ton/yr plant using toluene at 7:1 solvent ratio?

**Expected reasoning:**
- At 7:1 ratio: 3,000 ton polymer × 7 = 21,000 ton solvent circulated per year
- At 99.9% recovery: 21,000 × 0.001 = 21 ton/yr solvent loss
- At 99.0% recovery: 21,000 × 0.01 = 210 ton/yr solvent loss → 10× more
- Toluene cost ~$800-1000/ton → loss increases from ~$17K to ~$170K/yr
- Also: 210 ton/yr toluene emissions → environmental and safety regulatory issues
- Solvent loss also means: 1g plastic traps ~1mL solvent → better filtration/drying reduces losses
- Going from 99.9% to 99% roughly doubles operating cost contribution from solvent
- **Key checkpoint:** Agent must do the arithmetic showing 10× increase in solvent loss and translate to dollar impact

---

**Q2.5.3** (Difficulty: Basic)
> What is the minimum plant scale for STRAP to be economically viable? What determines the break-even point?

**Expected reasoning:**
- Break-even scale: ~2,500 ton/yr (from pilot data)
- At 2,500 ton/yr: MSP ≈ $750/ton, which equals virgin PE price (~$750/ton)
- Below this scale: MSP > virgin price → not competitive
- Above this scale: economies of scale reduce MSP
- Key factors determining break-even:
  1. Capital cost amortization — fixed equipment costs spread over more product
  2. Solvent recovery efficiency — scales better at larger volumes
  3. Labor — similar crew size for 1,000 vs 5,000 ton/yr
  4. Feedstock cost — waste plastic often has negative cost (tipping fees) at scale
  5. Product value — recycled PE at virgin-equivalent quality commands premium price
- Comparison: mechanical recycling is cheaper but produces lower-quality product
- **Key checkpoint:** Agent should cite ~2,500 ton/yr and explain why fixed costs drive the break-even

---

## Category 3: Techno-Economic Analysis (TEA)

### 3.1 — BioSTEAM Process Simulation & MSP

**Q3.1.1** (Difficulty: Basic)
> I want to evaluate the cost of recycling PE from a multilayer film using toluene as the dissolution solvent at 110°C. Run a BioSTEAM simulation and report the minimum selling price (MSP) of the recycled PE. What are the main cost contributors?

**Expected reasoning:**
- Agent should run BioSTEAM simulation with: polymer=PE, solvent=toluene, temperature=110°C, energy_case=C1
- Expected MSP: ~$1.10/kg (varies by simulation parameters)
- Cost contributors (from TEA literature):
  1. Solvent/antisolvent recovery (distillation): ~30-35% of variable OpEx
  2. Capital depreciation (equipment): ~25% — dominated by extruders (55% of equipment cost) and distillation columns
  3. Energy (heating to 110°C, distillation): ~20-25%
  4. Fixed operating costs (labor, maintenance): ~15%
- Compare to virgin PE price: ~$0.75-0.90/kg
- **Key checkpoint:** Agent must run the simulation, report MSP, and identify distillation/solvent recovery as the dominant cost driver

---

**Q3.1.2** (Difficulty: Intermediate)
> Compare the MSP of recycling PE using three different solvents: toluene (BP=111°C), dodecane (BP=216°C), and cyclohexane (BP=81°C) — all at their optimal dissolution temperatures. Which gives the lowest MSP, and why does solvent boiling point matter for economics?

**Expected reasoning:**
- Agent should run 3 BioSTEAM simulations with different solvents
- Solvent boiling point affects economics through:
  1. **Distillation energy**: Higher BP solvents require more energy for recovery via distillation
  2. **Dissolution temperature**: Must operate below BP; higher BP allows higher operating temp → faster dissolution
  3. **Solvent loss**: Lower BP solvents evaporate more easily → higher losses
  4. **Equipment sizing**: Distillation column reflux ratios change with BP
- Key insight from literature: **MSP doesn't change dramatically** with solvent choice when >99.9% recovery is enforced — differences are typically <10% ($1.34-$1.46/kg across 7 solvent pairs tested)
- However, the **energy case** (C1 vs C3) has a much larger effect on economics than solvent choice
- **Key checkpoint:** Agent must recognize that at high recovery rates, solvent choice has surprisingly small impact on MSP — the recovery system dominates costs regardless of solvent

---

**Q3.1.3** (Difficulty: Advanced)
> For a PE/EVOH/PET multilayer film at 3,000 ton/yr, compare three STRAP process variants: (A) antisolvent precipitation (standard), (B) temperature-controlled precipitation, (C) optimized with PETG recovery. Report the MSP, CapEx, and key economic differences for each variant.

**Expected reasoning:**
- From TEA literature:
  - **STRAP-A**: MSP = $2.05/kg, CapEx = $25.65M
  - **STRAP-B**: MSP = $1.62/kg, CapEx = $22.42M (21% lower MSP, 13% lower CapEx)
  - **STRAP-C**: MSP = $2.18/kg, CapEx = $31.78M (recovers 4 polymers instead of 3)
- STRAP-B is cheaper because:
  - 60% reduction in antisolvent usage (from 5 kg/kg to 2 kg/kg polymer)
  - 25% reduction in total energy (from 10.2 to 7.65 MJ/kg)
  - Smaller distillation columns (less antisolvent to separate)
  - 15% reduction in operating cost ($850 → $723/ton)
- STRAP-C is more expensive per kg but recovers more value:
  - Recovers PETG (20.83 wt% of film) in addition to PE, EVOH, PET
  - Higher CapEx due to additional separation step
  - May be economically preferred if PETG has market value
- **Key checkpoint:** Agent must explain WHY STRAP-B is cheaper (antisolvent elimination) and identify the MSP/value trade-off of STRAP-C

---

### 3.2 — Economies of Scale & Break-Even

**Q3.2.1** (Difficulty: Intermediate)
> A STRAP plant processing PE/EVOH/PET film has the following MSP at different scales: 1,000 ton/yr → $2.80/kg, 3,000 ton/yr → $1.62/kg, 7,000 ton/yr → $1.00/kg, 15,000 ton/yr → $0.78/kg, 30,000 ton/yr → $0.75/kg. At what scale does the plant become competitive with virgin PE ($0.75-0.90/kg)? Why does MSP flatten at larger scales?

**Expected reasoning:**
- Competitive with virgin PE at ~15,000 ton/yr (MSP = $0.78/kg ≈ virgin PE price)
- However, at current recycled plastic market prices (>$1.50/kg), the plant is profitable at just 2,500-3,000 ton/yr
- MSP flattens because:
  1. **Fixed costs amortization**: Equipment cost is spread over more product, but equipment scales sub-linearly (power law scaling ~0.6-0.7 exponent)
  2. **Variable costs dominate at scale**: Solvent, energy, and labor are proportional to throughput
  3. **Diminishing returns**: Beyond 15,000 ton/yr, variable costs floor sets the minimum MSP
  4. **Equipment sizing limits**: Very large distillation columns may need parallelization, reducing economies
- IRR analysis: At 3,000 ton/yr with polymer selling price $3.00/kg → IRR = 16.6%. At 9,000 ton/yr → IRR much higher
- **Key checkpoint:** Agent must distinguish fixed vs variable cost behavior and explain why MSP has an asymptotic floor

---

**Q3.2.2** (Difficulty: Advanced)
> A sensitivity analysis shows these impacts on MSP (±30% parameter change): dissolution time (±11%), solvent/polymer ratio (±10%), steam price (±8%), filter equipment cost (±7%), distillation column cost (±5%), extruder cost (±5%), project lifetime (±4%). Which parameter should I prioritize for process optimization to reduce costs? Are there any interactions between these parameters?

**Expected reasoning:**
- **Prioritize dissolution time reduction**: Highest sensitivity (±11%), and achievable through:
  - Particle size reduction (shredding to <3mm → 30× faster)
  - Increased stirring speed (up to diminishing returns at ~400 RPM)
  - Optimized temperature (near but below solvent BP)
  - This reduces reactor residence time → smaller equipment → lower CapEx AND OpEx
- **Second priority: solvent/polymer ratio reduction** (±10%):
  - Lower ratio means less solvent to heat and recover
  - Limited by viscosity — below ~7:1 ratio, solutions become too viscous (PE entanglement at ~14 wt%)
  - Affects both energy (less solvent to heat/distill) and capital (smaller tanks/columns)
- **Interactions:**
  - Dissolution time × solvent ratio: faster dissolution allows lower ratio (less time at high viscosity)
  - Steam price × distillation: correlated — both reflect energy cost
  - Equipment costs are additive, not multiplicative
- **Extruder cost is notable**: Despite only ±5% MSP sensitivity, extruders account for 55% of total equipment purchase cost — any cost reduction in extruders has disproportionate capital impact
- **Key checkpoint:** Agent should rank dissolution time first, explain why, and identify the viscosity-limited floor for solvent/polymer ratio

---

### 3.3 — Sequence Economics & Multi-Polymer Optimization

**Q3.3.1** (Difficulty: Expert)
> For a 4-polymer film (LDPE 60%, EVOH 5%, PET 25%, Nylon 6 10%), two separation sequences were evaluated: Sequence S₁ (LDPE first → EVOH+PET together → PET → N6) gave MSP=$0.48/kg. Sequence S₄ (EVOH+N6 in first step → rest later) gave MSP=$0.78/kg. Why is S₁ so much cheaper? If my objective is lowest GWP instead of lowest MSP, would I choose the same sequence?

**Expected reasoning:**
- S₁ is cheaper because:
  1. **LDPE first** recovers the largest mass fraction (60%) in the first step — maximum revenue from a single separation
  2. LDPE has high recycling rate (99.96%) and low separation cost (toluene is cheap, well-characterized)
  3. Purity = 99% (tie layer contributes only 1%) → high market value
  4. Early revenue stream reduces capital recovery pressure
- S₄ is more expensive because:
  1. EVOH + N6 are only 15% of feed — small mass recovery in first step
  2. More complex solvents needed (DMSO, formic acid) → higher cost
  3. Multiple recovery steps before reaching the high-value LDPE fraction
- **MSP vs GWP trade-off:**
  - S₁: MSP = $0.48/kg, GWP (CCI) = 0.49 kg CO₂-eq/kg
  - S₂ (different sequence): MSP = $0.77/kg, GWP = 0.41 kg CO₂-eq/kg → **lower GWP but higher cost**
  - So **no, the lowest MSP sequence is NOT the lowest GWP sequence**
  - The trade-off exists because: energy-intensive steps that reduce MSP (e.g., large-scale toluene distillation) may increase GWP
  - A Pareto analysis shows multiple non-dominated solutions
- From literature: cost and emissions can vary by **24-84%** depending on sequence
- **Key checkpoint:** Agent must (1) explain why recovering the largest fraction first minimizes MSP, (2) recognize that MSP-optimal ≠ GWP-optimal, (3) suggest Pareto analysis for multi-objective optimization

---

## Category 4: Life Cycle Assessment (LCA)

### 4.1 — GWP & Energy Scenarios

**Q4.1.1** (Difficulty: Basic)
> Run a BioSTEAM LCA for recycling PE from a multilayer film using toluene at 110°C. Compare the GWP under two energy scenarios: C1 (grid electricity + natural gas heat) and C3 (renewable electricity + natural gas heat). How does the energy source affect the environmental footprint?

**Expected reasoning:**
- Agent should run BioSTEAM simulations with energy_case=C1 and energy_case=C3
- Expected results:
  - C1: GWP ≈ 1.7 kg CO₂-eq/kg polymer
  - C3: GWP ≈ 0.4 kg CO₂-eq/kg polymer
  - **76% reduction** by switching to renewable electricity
- Why such a large difference:
  - Electricity is ~45% of STRAP energy demand (heating dissolution)
  - US grid average: 0.45 kg CO₂-eq/kWh
  - Renewable (wind/solar): ~0.05 kg CO₂-eq/kWh (90% lower)
  - Natural gas heat stays the same in both scenarios
- Compare to virgin PE production: 1.9 kg CO₂-eq/kg
  - C1: 11% lower than virgin (modest improvement)
  - C3: 79% lower than virgin (dramatic improvement)
- **Key checkpoint:** Agent must run both scenarios, report the 76% reduction, and explain that electricity source is the dominant GWP driver

---

**Q4.1.2** (Difficulty: Intermediate)
> The STRAP process energy breakdown is: heating (dissolution) 45%, distillation (solvent recovery) 35%, cooling 12%, mechanical (pumps/filters) 8%. Total energy: 8.5 MJ/kg polymer. If I switch from STRAP-A (antisolvent precipitation) to STRAP-B (temperature-controlled), energy drops 25% to 6.4 MJ/kg. Which energy component is most reduced, and why?

**Expected reasoning:**
- **Distillation energy is most reduced** (35% → ~15% of total):
  - STRAP-A requires distilling large volumes of antisolvent-solvent mixture (e.g., acetone-toluene at 2:1-3:1 ratio)
  - STRAP-B eliminates or dramatically reduces antisolvent use (60% reduction)
  - Less antisolvent = less distillation = less energy
  - Distillation column can be smaller → also reduces CapEx
- Energy redistribution in STRAP-B:
  - Heating increases slightly (need more robust heating/cooling cycle)
  - Cooling increases (temperature-controlled precipitation requires active cooling from 95°C to 20°C)
  - Mechanical stays roughly the same
  - Net: total energy drops from 8.5 to 6.4 MJ/kg (25% reduction)
- GWP impact: STRAP-B GWP ≈ 1.50 kg CO₂-eq/kg vs STRAP-A GWP ≈ 1.70 kg CO₂-eq/kg (12% reduction)
- Note: GWP reduction (12%) is less than energy reduction (25%) because the residual energy (heating, cooling) has similar carbon intensity
- **Key checkpoint:** Agent must identify distillation as the component most affected by the antisolvent elimination and connect energy reduction to GWP reduction

---

### 4.2 — Solvent Loss & Environmental Impact

**Q4.2.1** (Difficulty: Intermediate)
> The STRAP pilot plant achieves 99.5% solvent recovery. The sensitivity analysis shows: 1% solvent loss adds 0.15 kg CO₂-eq/kg to GWP, 5% loss adds 0.75 kg CO₂-eq/kg, and 10% loss adds 1.5 kg CO₂-eq/kg. At what solvent loss rate does STRAP become worse than virgin PE production (GWP = 1.9 kg CO₂-eq/kg)? What does this mean for plant design?

**Expected reasoning:**
- Base GWP (STRAP C1) = 1.7 kg CO₂-eq/kg at 0.5% loss
- Relationship is approximately linear: each 1% loss adds ~0.15 kg CO₂-eq/kg
- At 0.5% loss: 1.7 kg CO₂-eq/kg (base case)
- At 1.5% loss: 1.7 + 0.15 = 1.85 kg CO₂-eq/kg
- At ~1.8% loss: 1.7 + (1.8 × 0.15) = 1.97 → **exceeds virgin PE at ~2% solvent loss**
- More precisely: (1.9 - 1.7) / 0.15 = 1.33% additional loss → total 1.83% loss is the break-even
- At 5% loss: GWP = 2.45 kg CO₂-eq/kg → **44% worse than base case**, worse than virgin PE
- At 7% loss: STRAP becomes environmentally worse than virgin production across all metrics
- **Plant design implications:**
  - Must achieve >98% solvent recovery minimum (for environmental break-even with virgin)
  - Target >99.5% for meaningful environmental benefit
  - Solvent recovery system is the most critical unit operation for both economics AND environment
  - Sources of loss: solvent trapped in polymer (1g plastic traps ~1mL solvent), evaporation, incomplete distillation
  - Mitigation: improved filtration, drying steps, closed-loop distillation, solvent quality monitoring
- **Key checkpoint:** Agent must calculate the ~2% loss threshold, and connect solvent recovery to both economic and environmental viability

---

### 4.3 — System Boundaries & Allocation

**Q4.3.1** (Difficulty: Advanced)
> A STRAP plant recovers PE, EVOH, and PET from a multilayer film. Using economic allocation, EVOH receives a disproportionately high share of environmental burden despite being only 5% of the feed mass. Why? If I switch to mass allocation, how does the GWP per kg change for each polymer? Which allocation method should I use?

**Expected reasoning:**
- **Economic allocation formula:**
  - Allocation_i = (Mass_i × Price_i) / Σ(Mass_j × Price_j)
  - PE: 72.2% mass × $1,200/ton = $866.4
  - EVOH: 5% mass × $2,800/ton = $140.0
  - PET: 20.8% mass × $900/ton = $187.2
  - Total value = $1,193.6
  - PE allocation: $866.4/$1193.6 = 72.6%
  - EVOH allocation: $140.0/$1193.6 = 11.7% (vs 5% by mass → 2.3× overweighted)
  - PET allocation: $187.2/$1193.6 = 15.7%
- **Effect on GWP:** EVOH gets 11.7% of total process GWP despite being 5% of mass → GWP per kg of EVOH is ~2.3× higher than it would be under mass allocation
- **Mass allocation:** Each polymer receives GWP proportional to its mass fraction
  - PE: 72.2% of total GWP → lower per-kg burden (most mass)
  - EVOH: 5% of total GWP → much lower per-kg burden
  - PET: 20.8% of total GWP
- **Which method to use:**
  - ISO 14044 recommends avoiding allocation if possible (system expansion preferred)
  - Economic allocation is standard when co-products have different market values
  - Mass allocation is simpler but ignores that high-value products drive process investment
  - System expansion (substitution): credit each polymer with avoided virgin production
    - PE credit: -1.9, PET credit: -2.5, EVOH credit: -3.8 kg CO₂-eq/kg
    - This often gives negative net GWP (carbon-negative recycling)
  - **Recommendation:** Report all three methods for transparency; use economic allocation for decision-making
- **Key checkpoint:** Agent must calculate the allocation factors, show EVOH is overweighted by 2.3× under economic allocation, and discuss the ISO 14044 hierarchy (avoid → physical → economic)

---

### 4.4 — Multi-Objective & Comparative Assessment

**Q4.4.1** (Difficulty: Expert)
> Seven different green solvent pairs were evaluated for STRAP recycling of PE/EVOH/PET film. Their MSP and climate change impact (CCI) are: Process I (Toluene/DMSO): $1.45/kg, 1.0 CCI. Process V (Cyclohexanol/Acetic acid): $1.46/kg, 0.85 CCI. Process VI (α-chloro methyl acrylate/Bromoacetic acid): $1.34/kg, 4.3 CCI. Process VII (Toluene/DMF): $1.34/kg, 0.95 CCI. Which process would you recommend? What additional metrics beyond MSP and GWP should be considered?

**Expected reasoning:**
- **Pareto analysis of MSP vs CCI:**
  - Process VI and VII have lowest MSP ($1.34/kg) but very different CCI
  - Process VI has lowest MSP but **worst CCI (4.3)** — due to high GWP of solvent production
  - Process VII has lowest MSP AND good CCI (0.95) — Pareto-dominant over Process VI
  - Process V has best CCI (0.85) but slightly higher MSP ($1.46/kg)
  - Process I is baseline reference (standard solvents)
- **Recommendation:** Process VII (Toluene/DMF) — Pareto-optimal (lowest MSP at reasonable CCI)
  - But if environmental impact is priority: Process V (Cyclohexanol/Acetic acid)
  - Process VI is dominated: never choose it (same MSP as VII but 4.5× worse CCI)
- **Additional metrics to consider (green solvent screening framework):**
  1. **Human toxicity (cancer):** HTC ranges from 4.2×10⁻¹⁰ to 5.8×10⁻¹⁰ CTUh/kg — Process VI significantly worse
  2. **Human toxicity (non-cancer):** HTN ranges from 9.0×10⁻⁸ to 9.5×10⁻⁸ CTUh/kg
  3. **Freshwater ecotoxicity:** FE ranges from 0.019 to 0.024 CTUe/kg — notable variation
  4. **LogP (bioaccumulation):** Prefer LogP < 3; toluene LogP = 2.7 (borderline), DMSO LogP = -1.4 (excellent)
  5. **GHS hazard statements:** α-chloro methyl acrylate has severe hazard classifications
  6. **REACH/regulatory status:** Some solvents restricted or candidates for restriction
  7. **Total Solvent Energy (TSE):** Energy to heat + recover solvent; varies by boiling point
  8. **Solvent availability and price stability:** Industrial-scale supply chain considerations
- **Key insight from literature:** No single solvent pair is optimal across ALL metrics — this is a multi-criteria decision problem requiring trade-off analysis (radar chart or weighted scoring)
- **Key checkpoint:** Agent must (1) identify Process VII as Pareto-optimal on MSP/CCI, (2) explain why Process VI is dominated despite lowest MSP, (3) enumerate at least 4 additional dimensions beyond MSP and GWP for holistic evaluation

---

## Category 5: Process Safety

### 5.1 — Solvent Hazard Identification

**Q5.1.1** (Difficulty: Basic)
> I'm considering toluene, DMSO, and THF as solvents for a STRAP recycling process. Look up their GSK safety scores and GHS hazard classifications. Which is the safest option? Which should I avoid if possible?

**Expected reasoning:**
- Agent should call `get_solvent_gscore()` for each solvent and `compare_pubchem_safety()` for the set:
  - **Toluene**: G-score ≈ 5.7 (Problematic). GHS: Flammable (H225), Skin/eye irritant, Reproductive toxicity (H361d), STOT-RE (H372). Signal word: Danger.
  - **DMSO**: G-score ≈ 6.6 (Good). GHS: minimal hazard — Irritant only (H315, H319). Signal word: Warning. Non-flammable (BP 189°C), low toxicity.
  - **THF**: G-score ≈ 5.3 (Problematic). GHS: Flammable (H225), May form explosive peroxides, Eye/respiratory irritant, Suspected carcinogen (H351). Signal word: Danger.
- **Safest:** DMSO — highest G-score, fewest GHS hazards, non-flammable, no reproductive or carcinogenic concerns
- **Avoid if possible:** THF — peroxide formation hazard requires stabilizer (BHT) and regular testing; suspected carcinogen; highly flammable (flash point −14°C)
- Toluene is intermediate — flammable and has chronic toxicity but well-understood industrially with established handling protocols
- **Key checkpoint:** Agent must retrieve actual G-scores (not fabricate them), correctly classify each solvent's safety tier, and identify THF peroxide formation as a unique process hazard

---

**Q5.1.2** (Difficulty: Intermediate)
> For a PE/EVOH/PET multilayer film recycling sequence, the candidate solvents are: toluene (PE dissolution, 110°C), DMSO (EVOH dissolution, 135°C), ethyl acetate (PET dissolution, 80°C), DMF (EVOH alternative), and propylene carbonate (EVOH alternative). Compare their safety profiles including G-scores, GHS hazards, and toxicity. Rank them from safest to most hazardous and explain the key differentiators.

**Expected reasoning:**
- Agent should call `get_solvent_gscore()` for all 5 solvents and `get_pubchem_toxicity()` for detailed data:
  - **Propylene carbonate**: G-score ≈ 7.0+ (Good/Excellent). Non-toxic, non-flammable (BP 242°C), biodegradable. Minimal GHS hazards.
  - **DMSO**: G-score ≈ 6.6 (Good). Very low toxicity (LD50 >14,500 mg/kg oral rat), non-flammable, but skin penetration enhancer — can carry dissolved contaminants through skin.
  - **Ethyl acetate**: G-score ≈ 6.0 (Good). Flammable (flash point −4°C) but low chronic toxicity. Common industrial solvent with good handling protocols.
  - **Toluene**: G-score ≈ 5.7 (Problematic). Reproductive toxin (Category 2), STOT-RE. Flammable. OEL: 20 ppm TWA (EU).
  - **DMF**: G-score ≈ 4.5 (Problematic). Reproductive toxin (Category 1B — H360D), liver toxicity, skin absorption. REACH Substance of Very High Concern (SVHC) candidate. OEL: 5 ppm TWA.
- **Ranking (safest → most hazardous):** Propylene carbonate > DMSO > Ethyl acetate > Toluene > DMF
- **Key differentiators:**
  1. Reproductive toxicity: DMF (Cat 1B) and toluene (Cat 2) vs none for DMSO/PC/EtOAc
  2. Flammability: THF/EtOAc (flash point < 0°C) vs PC/DMSO (non-flammable)
  3. Regulatory pressure: DMF is SVHC candidate under REACH → possible phase-out
  4. Skin absorption: DMSO enhances dermal penetration (requires glove protocol)
- **Process implication:** Replacing DMF with propylene carbonate for EVOH dissolution would eliminate the most hazardous solvent in the sequence, but requires verifying dissolution performance
- **Key checkpoint:** Agent must rank all 5, identify DMF as the most hazardous (reproductive toxin + SVHC), and note that propylene carbonate is the safest alternative for EVOH dissolution

---

### 5.2 — Safer Alternatives & Substitution

**Q5.2.1** (Difficulty: Intermediate)
> Toluene is the standard solvent for PE dissolution in STRAP, but its reproductive toxicity (GHS H361d) is a concern. Find safer alternatives from the same chemical family (aromatic hydrocarbons) and from other families. For each alternative, assess whether it can still dissolve PE effectively by checking HSP compatibility.

**Expected reasoning:**
- Agent should call `get_family_alternatives("toluene", min_gscore=6.0)` for aromatic alternatives and check HSP compatibility with `predict_solubility_ml()`:
- **Aromatic alternatives:**
  - p-Cymene: G-score ≈ 6.5, BP 177°C, lower acute toxicity. RED for PE: should be < 1.0 (similar δD to toluene). Viable candidate.
  - Anisole: G-score ≈ 6.2, BP 154°C, lower reproductive concern. Slightly higher δP than toluene — check RED.
  - Mesitylene (1,3,5-trimethylbenzene): G-score ~5.5, BP 165°C. Similar HSP profile to toluene but higher BP means more distillation energy.
- **Non-aromatic alternatives:**
  - d-Limonene: G-score ≈ 6.8, BP 176°C, terpene (renewable). Known PE solvent. RED should be near toluene range.
  - Cyclopentyl methyl ether (CPME): G-score ≈ 6.5, BP 106°C, ether class. Low peroxide formation tendency vs THF.
  - 2-MeTHF: G-score ≈ 6.0, BP 80°C, renewable (from furfural). Good dissolving power but lower BP limits operating temperature.
- **Trade-off matrix:**
  - Safety improvement: DMF replacement gives biggest safety gain
  - Performance: d-limonene and p-cymene closest to toluene's PE dissolution ability
  - Economics: Higher BP alternatives cost more to recover (distillation energy)
  - Availability: Toluene is commodity-scale ($800/ton); alternatives are typically 2-5× more expensive
- **Key checkpoint:** Agent must use both `get_family_alternatives()` and `predict_solubility_ml()` to jointly evaluate safety AND selectivity. Must not recommend an alternative without checking HSP compatibility.

---

**Q5.2.2** (Difficulty: Advanced)
> Design the safest possible 3-step STRAP sequence for PE/EVOH/PET separation. For each step, choose a solvent that: (1) has a G-score ≥ 6.0, (2) achieves RED < 1.0 for the target polymer, and (3) achieves RED > 1.0 for the non-target polymers (selectivity). If no solvent meets all three criteria for a given step, identify the trade-off and recommend whether to relax the safety or selectivity constraint.

**Expected reasoning:**
- This requires cross-referencing safety and selectivity tools for each step:
- **Step 1 — PE dissolution:**
  - Need: dissolves PE (RED < 1) but NOT PET or EVOH (RED > 1)
  - Safe options (G-score ≥ 6): d-limonene, p-cymene, anisole, cyclohexane
  - d-Limonene (G ≈ 6.8): should dissolve PE at elevated temperature, non-polar enough to reject PET/EVOH
  - Cyclohexane (G ≈ 5.8): just below threshold — may need to relax to G ≥ 5.5
  - **Likely choice:** d-limonene or accept toluene with engineering controls
- **Step 2 — EVOH dissolution:**
  - Need: dissolves EVOH (RED < 1) but NOT PET (RED > 1)
  - Safe options: DMSO (G ≈ 6.6), propylene carbonate (G ≈ 7.0+), glycerol (G ≈ 7.5+)
  - DMSO: known EVOH solvent, excellent safety profile
  - Propylene carbonate: safest option — check if RED < 1.0 for EVOH
  - **Likely choice:** DMSO or propylene carbonate
- **Step 3 — PET dissolution:**
  - Need: dissolves PET (RED < 1)
  - Safe options: GVL/γ-valerolactone (G ≈ 7.0+), ethyl acetate (G ≈ 6.0)
  - PET is hard to dissolve — may need aggressive solvents (phenol, cresols — low G-scores)
  - Ethyl acetate at 80°C may work per STRAP data
  - **Trade-off:** PET dissolution may require relaxing G-score threshold to ≥ 5.0 or accepting a narrow solvent set
- **Overall assessment:**
  - Fully safe sequence (all G ≥ 6.0) is achievable for PE and EVOH steps
  - PET step is the bottleneck — limited safe solvent options with sufficient dissolving power
  - If G ≥ 5.5 is acceptable: toluene/DMSO/ethyl acetate sequence works (current STRAP standard)
  - If G ≥ 6.0 is hard requirement: d-limonene/DMSO/ethyl acetate but PE step needs temperature optimization
- **Key checkpoint:** Agent must attempt all three steps, identify where the safety-selectivity trade-off is binding (PET), and make an explicit recommendation about which constraint to relax

---

### 5.3 — Process Hazard Assessment

**Q5.3.1** (Difficulty: Advanced)
> A STRAP pilot plant uses toluene at 110°C for PE dissolution, DMSO at 135°C for EVOH dissolution, and ethyl acetate at 80°C for PET. For each step, evaluate: (a) proximity to the solvent boiling point and vapor pressure hazard, (b) flammability risk at the operating temperature, (c) worker exposure pathway (inhalation vs dermal), and (d) what engineering controls are needed. Which step poses the greatest process safety risk?

**Expected reasoning:**
- Agent should call `get_solvent_properties()` and `get_pubchem_safety_info()` for context:
- **Step 1 — Toluene at 110°C:**
  - BP = 111°C → operating at 99% of BP → **high vapor pressure, near-boiling operation**
  - Flash point = 4°C → well above flash point → flammable vapors present at all operating temperatures
  - At 110°C: vapor pressure near 1 atm → significant toluene vapor generation
  - Exposure: inhalation is primary route (OEL 20 ppm EU TWA); dermal secondary
  - Controls needed: closed system with inert gas blanket (N₂), explosion-proof electrical, LEL monitoring, scrubbed ventilation, SCBA for maintenance
  - **Risk rating: HIGH** — flammable solvent at near-BP temperature
- **Step 2 — DMSO at 135°C:**
  - BP = 189°C → operating at 71% of BP → moderate vapor pressure
  - Flash point = 95°C → operating above flash point → but DMSO is difficult to ignite (auto-ignition 270°C)
  - Non-flammable in normal conditions; exothermic decomposition above 200°C is the real hazard
  - Exposure: dermal is primary route — DMSO enhances skin absorption of dissolved contaminants
  - Controls needed: impervious gloves (butyl rubber, not latex), skin monitoring protocol, temperature alarm at 180°C (decomposition onset)
  - **Risk rating: LOW-MEDIUM** — thermal decomposition and skin penetration are main concerns
- **Step 3 — Ethyl acetate at 80°C:**
  - BP = 77°C → **operating above BP** → must be under pressure or reflux!
  - Flash point = −4°C → extremely flammable at all temperatures
  - At 80°C: would be boiling at atmospheric pressure → pressurized system or reflux condenser required
  - Exposure: inhalation (sweet odor, poor warning properties at chronic exposure levels)
  - Controls needed: pressure-rated vessels, rupture discs, reflux condensers, grounding/bonding for static, explosion-proof zone classification
  - **Risk rating: HIGH** — operation above BP with flammable liquid requires pressure containment
- **Greatest risk:** Step 3 (ethyl acetate) — operating above boiling point with an extremely flammable solvent creates the most severe hazard scenario (pressurized flammable liquid). Step 1 (toluene near-BP) is second.
- **Key checkpoint:** Agent must recognize that ethyl acetate at 80°C is above its BP (77°C) — this is the critical insight. Must also identify toluene near-BP operation as high risk. DMSO should be correctly identified as the lowest risk step despite the highest temperature.

---

**Q5.3.2** (Difficulty: Expert)
> A regulatory review requires a comprehensive safety dossier for a STRAP plant processing 3,000 ton/yr of PE/EVOH/PET film. The plant uses toluene, DMSO, and DMF. For each solvent, assess: (1) REACH registration status and any SVHC listing, (2) EPA Toxics Release Inventory (TRI) reporting obligations, (3) workplace exposure limits (OEL/PEL) and biological exposure indices, (4) environmental fate if released (aquatic toxicity, biodegradability, bioaccumulation), and (5) transport and storage classification. Then provide an overall risk ranking and identify which solvent substitution would most improve the plant's regulatory profile.

**Expected reasoning:**
- Agent should use `get_pubchem_safety_info()`, `get_pubchem_toxicity()`, `get_solvent_gscore()`, and knowledge of regulatory frameworks:
- **Toluene:**
  1. REACH: Registered (>1000 ton/yr). Not SVHC but on Community Rolling Action Plan (CoRAP) for evaluation. Restriction under REACH Annex XVII (concentration limits in consumer products)
  2. TRI: Listed (CAS 108-88-3). Threshold: 25,000 lb/yr manufacture/process, 10,000 lb/yr otherwise → 3,000 ton/yr plant will exceed threshold → **mandatory TRI reporting**
  3. OEL: EU 20 ppm TWA (Directive 2017/164); OSHA PEL 200 ppm; ACGIH TLV 20 ppm. BEI: o-cresol in urine <0.5 mg/L, toluene in blood <0.02 mg/L
  4. Environmental: LogP 2.7, moderate bioaccumulation potential. Readily biodegradable (aerobic). LC50 fish 5.8 mg/L (acute). Volatile — primarily atmospheric fate.
  5. Transport: UN 1294, Class 3 (Flammable liquid), Packing Group II
- **DMSO:**
  1. REACH: Registered. Not SVHC, not on CoRAP. No restrictions.
  2. TRI: Not listed — no reporting obligation
  3. OEL: No established EU OEL. ACGIH proposal discussed but not finalized. Generally handled as nuisance dust/mist.
  4. Environmental: LogP −1.35, no bioaccumulation. Readily biodegradable. Low aquatic toxicity (LC50 >40,000 mg/L fish). Miscible in water — would disperse.
  5. Transport: Not classified as dangerous goods for transport
- **DMF (N,N-Dimethylformamide):**
  1. REACH: Registered. **SVHC candidate** (Reproductive toxin Cat 1B). Included in REACH Annex XIV Authorization List candidate → may require authorization for continued use. Under restriction proposal in EU.
  2. TRI: Listed (CAS 68-12-2). Threshold applies → **mandatory TRI reporting**
  3. OEL: EU 5 ppm TWA (SCOEL); OSHA PEL 10 ppm; ACGIH TLV 10 ppm. BEI: N-methylformamide in urine <15 mg/L. **Skin notation** — significant dermal absorption.
  4. Environmental: LogP −1.01, no bioaccumulation. Biodegradable but slower than DMSO. Moderate aquatic toxicity.
  5. Transport: UN 2265, Class 6.1 (Toxic), Packing Group II — **toxic substance classification**
- **Overall risk ranking:** DMF >> Toluene >> DMSO
- **Priority substitution:** Replace DMF with a non-SVHC alternative:
  - DMSO (already in sequence) — but check if it dissolves EVOH as well as DMF
  - Propylene carbonate — G-score ~7.0, no regulatory flags
  - NMP (N-methyl-2-pyrrolidone) — caution: also SVHC candidate (reproductive toxin)
  - Cyrene (dihydrolevoglucosenone) — bio-based, G-score high, but limited availability
- Replacing DMF eliminates: SVHC authorization burden, Class 6.1 transport, skin absorption risk, reproductive toxicity liability
- **Key checkpoint:** Agent must (1) correctly identify DMF as SVHC candidate, (2) note both toluene and DMF trigger TRI reporting, (3) highlight DMF skin absorption and reproductive toxicity as the most actionable regulatory risk, (4) recommend specific safer alternatives with evidence

---

## Answer Key Format

Each question should be scored on:
1. **Methodology** (0-3): Did the agent apply the correct approach/framework?
2. **Calculations** (0-3): Were numerical calculations correct or reasonable?
3. **Reasoning** (0-3): Did the agent explain the WHY, not just the WHAT?
4. **Completeness** (0-1): Were critical caveats, trade-offs, or limitations mentioned?

**Total per question: 10 points**
**Category 1 total: 140 points (14 questions)**
**Category 2 total: 130 points (13 questions)**
**Category 3 total: 50 points (5 questions)**
**Category 4 total: 50 points (5 questions)**
**Category 5 total: 60 points (6 questions)**
**Benchmark total: 430 points**

---

## Tool Capabilities Tested

| Question | Solubility Lookup | HSP Calc | COSMO-RS | BioSTEAM TEA | Sequence Design | Pareto | GSK Safety | PubChem Safety |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Q1.1.1 | | X | | | | | | |
| Q1.1.2 | | X | X | | | | | |
| Q1.1.3 | | X | | | | | | |
| Q1.1.4 | X | X | | | X | | | |
| Q1.2.1 | X | | X | | | | | |
| Q1.2.2 | X | | X | | | | | |
| Q1.2.3 | | | X | | | | | |
| Q1.2.4 | X | | X | | | | | |
| Q1.3.1 | | | | | | | | |
| Q1.3.2 | X | | | X | | | | |
| Q1.3.3 | X | | | | | | | |
| Q1.4.1 | | | X | | | | | |
| Q1.4.2 | | | X | | | | | |
| Q2.1.1 | X | | | | X | | | |
| Q2.1.2 | | | | X | X | | | |
| Q2.1.3 | X | | X | X | X | | | |
| Q2.1.4 | X | | | | X | | | |
| Q2.2.1 | X | | | | | | | |
| Q2.2.2 | X | | | X | | | | |
| Q2.2.3 | X | | X | | | | | |
| Q2.3.1 | | | | | | | | |
| Q2.3.2 | | | | | | | | |
| Q2.3.3 | | | | | | | | |
| Q2.4.1 | X | | | | X | | | |
| Q2.4.2 | | | | | | | | |
| Q2.4.3 | X | | | | X | | | |
| Q2.4.4 | X | | X | | | | | |
| Q2.5.1 | | | | X | | | | |
| Q2.5.2 | | | | X | | | | |
| Q2.5.3 | | | | X | | | | |
| Q3.1.1 | | | | X | | | | |
| Q3.1.2 | | | | X | | | | |
| Q3.1.3 | | | | X | | | | |
| Q3.2.1 | | | | X | | | | |
| Q3.2.2 | | | | X | | | | |
| Q3.3.1 | | | | X | X | X | | |
| Q4.1.1 | | | | X | | | | |
| Q4.1.2 | | | | X | | | | |
| Q4.2.1 | | | | X | | | | |
| Q4.3.1 | | | | X | | | | |
| Q4.4.1 | X | | | X | | X | | |
| Q5.1.1 | | | | | | | X | X |
| Q5.1.2 | | | | | | | X | X |
| Q5.2.1 | | X | | | | | X | |
| Q5.2.2 | | X | | | X | | X | |
| Q5.3.1 | | | | | | | | X |
| Q5.3.2 | | | | | | | X | X |
