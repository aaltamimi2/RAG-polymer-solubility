# DISSOLVE Agent Test Queries

A gradient set of queries for testing the DISSOLVE agent, progressing from simple single-tool calls to complex multi-step analyses requiring 10-20+ tool invocations.

**Note:** Literature search API calls have been replaced with RAG-based queries that retrieve thresholds, process parameters, and validation data from the STRAP-CORE knowledge base.

---

## Level 1: Single Tool Queries (1-2 tool calls)

### 1.1 Basic Solvent Screening
```
What solvents dissolve LDPE at 110°C? Show the top 5 ranked by solubility percentage.
```

### 1.2 Basic Safety Check
```
What is the G-score safety rating for dodecane, heptane, and toluene?
```

### 1.3 Basic TEA
```
Run a TEA analysis for heptane solvent recovery at 100 kg/hr throughput.
```

---

## Level 2: Two-Tool Queries (2-3 tool calls)

### 2.1 Solvent + Safety
```
Find solvents that dissolve LDPE at 110°C with G-score above 5. Rank by solubility.
```

### 2.2 Dissolution + Properties
```
What solvents dissolve PE at 110°C? For the top 3, show their boiling points, cost, and LogP toxicity values.
```

### 2.3 TEA + Visualization
```
Run TEA for dodecane recovery at 500 kg/hr and generate a cost breakdown visualization.
```

---

## Level 3: Multi-Tool Queries (3-5 tool calls)

### 3.1 Multilayer Separation Strategy
```
Design a separation strategy for a PE/EVOH multilayer film. Find selective solvents for each layer at 110°C, prioritizing safety (G-score > 5).
```

### 3.2 TEA + LCA Comparison
```
Compare TEA and LCA for heptane vs dodecane solvent recovery at 1000 kg/hr. Which has lower operating cost and carbon footprint?
```

### 3.3 Solvent Screening with Full Analysis
```
For LDPE dissolution at 110°C, find the best solvent considering: (1) solubility > 80%, (2) G-score safety, (3) boiling point > 100°C, and (4) cost per kg. Show comparison table.
```

---

## Level 4: Integrated Analysis Queries (5-8 tool calls)

### 4.1 Full STRAP Process Analysis
```
Run a full STRAP analysis for PE recovery from biocontainer film waste at 5000 kg/hr using heptane as solvent. Include TEA, LCA, and generate visualizations.
```

### 4.2 Two-Stage Separation with Economics
```
Design a two-stage STRAP process for PE/EVOH separation: Stage 1 dissolves PE selectively, Stage 2 recovers EVOH. Run TEA at 1000 kg/hr for each stage and compare total costs. Show which solvents are optimal for each stage.
```

### 4.3 Scenario Comparison
```
Compare three recycling scenarios for PE/EVOH film at 5000 kg/hr:
- Scenario 1: PE recovery only using heptane
- Scenario 2: PE recovery with EVOH sold as residue
- Scenario 3: Sequential PE and EVOH recovery with two solvents

Run TEA and LCA for each scenario and generate comparison visualizations.
```

---

## Level 5: Complex Integrated Queries (8-12 tool calls)

### 5.1 Complete Process Design with RAG Knowledge
```
I need to design a solvent-based recycling process for multilayer biocontainer films (PE/EVOH). First, search the RAG knowledge base for PE dissolution temperatures and optimal process conditions from published STRAP studies. Then find optimal solvents for selective PE dissolution at 110°C ranked by safety. Run TEA and LCA at 5000 kg/hr scale, and generate visualizations comparing virgin vs recycled polymer environmental impact.
```

### 5.2 Minimum Selling Price Analysis
```
For a STRAP process recovering PE from multilayer film using dodecane at 110°C:
1. Find the solvent properties and safety data
2. Run TEA at three scales: 1000, 5000, and 10000 kg/hr
3. Calculate minimum selling price at each scale
4. Run LCA to determine GHG reduction vs virgin PE
5. Generate scale economics visualization
```

---

## Level 6: Expert Queries (10-15 tool calls)

These queries require the agent to plan a multi-step analysis, execute numerous tools in sequence, and synthesize findings.

### 6.1 Comprehensive Solvent Selection Pipeline
```
I'm designing a PE recycling process and need a comprehensive solvent analysis:

1. List all solvents that dissolve LDPE above 80% at temperatures between 100-120°C
2. For each candidate, retrieve: G-score, LogP, boiling point, cost per kg
3. Get PubChem GHS hazard data for the top 5 safest options
4. Run TEA comparison at 2000 kg/hr for these 5 solvents
5. Generate a radar chart comparing safety, cost, and efficiency
6. Recommend the optimal solvent with full justification

Show your reasoning at each step.
```

### 6.2 Multilayer Film Recycling Feasibility Study
```
Evaluate the feasibility of recycling a 3-layer packaging film (LDPE/EVOH/PET):

1. For each polymer layer, find solvents that dissolve it above 70% at feasible temperatures
2. Identify temperature windows where each layer dissolves selectively
3. Propose an optimal separation sequence (which layer to dissolve first, second, third)
4. For each separation step, select the safest solvent (G-score ranking)
5. Run TEA for the complete 3-stage process at 5000 kg/hr
6. Run LCA comparing to virgin production of all three polymers
7. Generate process flow visualization and cost breakdown charts
8. Summarize with go/no-go recommendation
```

### 6.3 Scale-Up Decision Analysis
```
Help me decide the optimal scale for a PE/EVOH STRAP recycling facility:

1. Screen solvents for selective PE dissolution at 110°C (must not dissolve EVOH)
2. Select top solvent based on safety and selectivity
3. Run TEA at 5 different scales: 500, 1000, 2000, 5000, 10000 kg/hr
4. Calculate minimum selling price (MSP) at each scale
5. Run LCA at each scale to track environmental impact
6. Generate scale economics curves showing cost vs throughput
7. Identify the minimum viable scale for profitability
8. Compare GHG savings vs virgin PE at the recommended scale
```

---

## Level 7: Research-Grade Queries (15-20+ tool calls)

These represent the most challenging queries, requiring extensive tool orchestration, iterative refinement, and comprehensive synthesis.

### 7.1 Full STRAP Process Optimization Study
```
Conduct a complete STRAP process optimization for biocontainer film recycling (90% PE, 8% EVOH, 2% other):

PHASE 1 - Solvent Screening:
- Find all solvents dissolving PE above 85% at 100-120°C
- Find all solvents dissolving EVOH above 70% at 100-140°C
- Identify solvents that are selective (dissolve PE but NOT EVOH)
- Rank selective solvents by: G-score, cost, boiling point, LogP
- Get PubChem safety data for top 3 candidates

PHASE 2 - Process Economics:
- Run TEA for PE-only recovery at 5000 kg/hr with best solvent
- Run TEA for sequential PE+EVOH recovery at 5000 kg/hr
- Compare capital costs, operating costs, and payback periods
- Calculate MSP for recovered PE and EVOH in each scenario

PHASE 3 - Environmental Assessment:
- Run LCA for both scenarios
- Compare GWP against virgin PE and EVOH production
- Identify the largest contributors to carbon footprint
- Generate Sankey diagram of environmental flows

PHASE 4 - RAG Knowledge Validation:
- Search RAG for experimental PE dissolution validation data and recovery yields
- Search RAG for EVOH barrier layer separation methods and conditions

PHASE 5 - Synthesis:
- Generate comparison visualizations for all scenarios
- Provide final recommendation with economic and environmental justification
```

### 7.2 Comprehensive Polymer Recycling Decision Support
```
I have mixed plastic waste containing HDPE, LDPE, PP, PS, and PET. Design an optimal solvent-based separation process:

STEP 1 - Polymer Characterization:
- List all 5 polymers with their dissolution temperature ranges
- For each polymer, identify which solvents achieve >80% dissolution

STEP 2 - Selectivity Analysis:
- For each polymer pair (10 combinations), determine selectivity windows
- Find temperature/solvent combinations where ONLY ONE polymer dissolves
- Rank separation difficulty for each pair

STEP 3 - Sequence Optimization:
- Propose optimal separation order (easiest separations first)
- For each step, select the safest effective solvent
- Document temperature and expected yield at each stage

STEP 4 - Safety Profiling:
- Get G-scores for all solvents in the proposed process
- Get PubChem GHS data for any solvent with G-score < 5
- Flag any regulatory concerns

STEP 5 - Economics:
- Run TEA for the complete separation train at 5000 kg/hr
- Calculate capital investment and operating costs
- Determine MSP for each recovered polymer
- Compare against virgin polymer prices

STEP 6 - Environmental Impact:
- Run LCA for the complete process
- Compare GWP savings vs virgin production for each polymer
- Identify which recovery streams provide best environmental ROI

STEP 7 - Visualization & Reporting:
- Generate process flow diagram
- Create cost breakdown visualization
- Create environmental comparison charts
- Generate selectivity heatmap for all polymer pairs

STEP 8 - Executive Summary:
- Recommend which polymers to recover (based on economics + environment)
- Identify any polymers not worth recovering via this method
- Provide sensitivity analysis on key variables
```

### 7.3 Publication-Ready STRAP Analysis
```
Generate a publication-ready analysis of STRAP technology for PE/EVOH biocontainer recycling. Structure the analysis as follows:

INTRODUCTION DATA:
- Search RAG for recent STRAP studies on polyolefin solvent recycling conditions and yields
- Search RAG for EVOH barrier film separation methods and process parameters

MATERIALS CHARACTERIZATION:
- Document dissolution behavior of PE at 100, 110, 120°C
- Document dissolution behavior of EVOH at 100, 110, 120, 130, 140°C
- Identify selective dissolution windows

SOLVENT SELECTION:
- Screen all database solvents for PE dissolution >85% at 110°C
- Filter for G-score > 5 (green chemistry compliance)
- Rank by: selectivity over EVOH, boiling point, cost
- Get full property profiles for top 5 candidates
- Get PubChem safety/toxicity data for top 5

PROCESS DESIGN:
- Select optimal solvent for PE recovery stage
- Select optimal solvent for EVOH recovery stage (if sequential)
- Document process conditions for each stage

TECHNO-ECONOMIC ANALYSIS:
- Run TEA for Scenario A: PE recovery only (5 kton/yr)
- Run TEA for Scenario B: PE + EVOH recovery (5 kton/yr)
- Calculate TCI, OpEx, MSP for each scenario
- Generate cost breakdown visualizations
- Analyze scale effects from 1-10 kton/yr

LIFE CYCLE ASSESSMENT:
- Run LCA for both scenarios
- Compare against virgin PE production (GWP)
- Compare against virgin EVOH production (GWP)
- Quantify GHG reduction percentages
- Generate comparative bar charts

SENSITIVITY ANALYSIS:
- Test sensitivity to solvent price (+/- 20%)
- Test sensitivity to energy cost (+/- 20%)
- Test sensitivity to recovered polymer price (+/- 20%)

CONCLUSIONS:
- Summarize key findings with specific numbers
- Recommend optimal scenario with justification
- Identify limitations and future work
```

---

## Scoring Rubric

| Level | Tool Calls | Complexity | Pass Criteria |
|-------|------------|------------|---------------|
| 1 | 1-2 | Basic | Correct tool selection and output |
| 2 | 2-3 | Simple chaining | Proper sequencing of related tools |
| 3 | 3-5 | Multi-tool | Integration of results across tools |
| 4 | 5-8 | Integrated | Coherent analysis with visualizations |
| 5 | 8-12 | Complex | Multi-phase analysis with synthesis |
| 6 | 10-15 | Expert | Comprehensive decision support |
| 7 | 15-20+ | Research | Publication-quality integrated analysis |

---

## Testing Notes

1. **Observe tool selection**: Does the agent choose appropriate tools for each subtask?
2. **Check sequencing**: Are tools called in logical order with proper dependencies?
3. **Evaluate synthesis**: Does the agent integrate findings across tool outputs?
4. **Assess error handling**: How does the agent recover from failed tool calls?
5. **Measure completeness**: Does the agent address all parts of complex queries?
6. **Review reasoning**: Does the agent explain its approach and justify recommendations?
