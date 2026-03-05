# STRAP Life Cycle Assessment Data Extraction - README

## Overview

This directory contains a comprehensive extraction of Life Cycle Assessment (LCA) data from 8+ academic papers related to STRAP (Solvent-Targeted Recovery and Precipitation) - a solvent-based recycling technology for multilayer plastic films. The extraction was performed on March 5, 2026, specifically to support benchmark question development for testing AI agent LCA reasoning capabilities.

---

## File Structure

### 1. **STRAP-LCA-Comprehensive-Extraction.csv**
**Size:** 31 KB | **Lines:** 170 data rows

**Purpose:** Structured database of all quantitative LCA metrics extracted from papers

**Contents:**
- 170+ individual data points across 12 key columns
- All papers from 2018-2023 covered
- Metrics include: GWP, economics, yields, energy, water, impact categories, etc.

**Column Structure:**
```
Paper, Metric_Category, Specific_Metric, Value, Unit, Notes,
System_Boundary, Functional_Unit, Impact_Category, Allocation_Method,
Energy_Scenario, Reference_Location
```

**Example Row:**
```csv
Quantifying environmental benefits (2020),GWP,STRAP Process C1,1.7,kg CO2-eq/kg polymer,
Grid average electricity + natural gas heat,Cradle-to-gate,1 kg recycled polymer,
Climate Change,Economic allocation,C1 - Grid avg electricity + NG heat,Figure 3
```

**Use Cases:**
- Quick lookup of specific metrics
- Comparative analysis across papers
- Data validation for AI agent responses
- Creating structured benchmark question datasets

---

### 2. **STRAP-LCA-Methodology-Details.md**
**Size:** 30 KB | **Lines:** 700

**Purpose:** Deep dive into LCA methodology, assumptions, and advanced analysis

**Contents:**
1. **LCA Methodologies** (TRACI 2.1, CML, ReCiPe)
2. **System Boundaries** (cradle-to-gate, cradle-to-grave, gate-to-gate)
3. **Functional Units** across different studies
4. **Allocation Methods** (economic, substitution, mass)
5. **Energy Scenarios** (C1/C2/C3 detailed breakdown)
6. **Sensitivity Analysis Results** (parameter influence ranking)
7. **Comparison Benchmarks** (STRAP vs virgin, mechanical, other technologies)
8. **Water and Impact Categories** beyond GWP
9. **BioSTEAM LCA Model Details** (assumptions, process flow, calculations)
10. **Process Variants** (STRAP-A/B/C, green solvents)
11. **Benchmark Question Framework** (7 levels of difficulty)
12. **Assumptions and Limitations** (data quality, uncertainty)

**Key Features:**
- **7-Level Question Framework:**
  - Level 1: Direct factual recall
  - Level 2: Comparative analysis
  - Level 3: Sensitivity & trade-off analysis
  - Level 4: System boundary & allocation reasoning
  - Level 5: Multi-dimensional optimization
  - Level 6: Uncertainty & data quality assessment
  - Level 7: Consequential LCA & market effects

- **Energy Scenario Details:**
  - C1: Grid electricity (0.45 kg CO₂-eq/kWh) + NG heat → GWP 1.7
  - C2: All natural gas CHP → GWP 1.2 (29% reduction)
  - C3: Renewable electricity + NG heat → GWP 0.4 (76% reduction)

- **Sensitivity Rankings:**
  1. Electricity grid carbon intensity (highest impact)
  2. Natural gas carbon intensity
  3. Solvent production GWP
  4. Solvent recovery efficiency
  5. Transportation distance (lowest impact)

**Use Cases:**
- Understanding LCA calculation methodology
- Creating complex reasoning questions
- Evaluating AI agent understanding of methodology
- Teaching LCA concepts
- Uncertainty quantification

---

### 3. **STRAP-LCA-Tables-Figures-Reference.md**
**Size:** 24 KB | **Lines:** 627

**Purpose:** Exact reproduction of tables and figures with all numerical values

**Contents:**
- **9 papers** fully referenced (2018-2023)
- **15+ tables** with exact values transcribed
- **25+ figures** with data points extracted
- **Cross-paper benchmark questions** with verified answers
- **Summary statistics** across all papers

**Structure by Paper:**

**Paper 1: Quantifying Environmental Benefits (2020)**
- Figure 3: GWP Comparison (8 scenarios with exact values)
- Table 2: Impact Categories (5 categories × 4 materials)
- Figure 5: Sensitivity Analysis (tornado diagram data)
- Energy breakdown percentages
- Water consumption data

**Paper 2: Reducing Antisolvent Use (2021)**
- Table 2: Polymer yields for STRAP-A/B/C
- Figure 9: MSP breakdown (3 variants)
- Process improvement metrics (energy -25%, antisolvent -60%)

**Paper 3: MRF LCA (2021)**
- Figure 4: GWP by waste composition (3 scenarios)
- Table 2: Material recovery rates
- Table 3: Economic data

**Paper 4: Green Solvents (2022)**
- Figure 2: 8-step solvent selection framework
- Table 1: 5 solvent pairs with MSP, CCI, toxicity
- Figure 11: Impact categories for DMI system

**Paper 5: Computational Framework (2022)**
- Table 1: 4 case studies (polymer mixtures, MSP, CCI, comp time)
- Figure 9: Separation difficulty network (6 polymer pairs)
- Pareto frontier analysis

**Paper 6: Food Packaging (2023)**
- Figure 5: Coffee packaging GWP (3 EOL scenarios)
- Figure 7: Material circularity index (0.72 vs 0.15)
- Cost breakdown

**Paper 7: Original STRAP Science (2018)**
- Table 1: Solubilities in select solvents
- Figure 3: Polymer recovery yields and purities
- Figure 4: TEA for 10,000 t/yr plant

**Paper 8: Pilot Scale (2020)**
- Figure 37.11: Dissolution kinetics (time vs % dissolved)
- Figure 37.16: MSP vs plant scale (5 capacity levels)
- Solvent recovery efficiency

**Paper 9: Review Paper (2023)**
- Table 2: 5 commercial/pilot dissolution plants
- Table 3: Environmental and economic comparison
- Figure 3: GWP reduction potential

**Summary Statistics:**
- **GWP Range:** 0.4 - 1.85 kg CO₂-eq/kg (mean 1.48, median 1.52)
- **MSP Range:** $1,380 - $2,400/tonne (mean $1,720, median $1,640)
- **Yields:** PE 91-96.5%, EVOH 86-94.2%, PET 89-94%

**Use Cases:**
- Creating factual recall questions with exact answers
- Verifying AI agent responses against source data
- Identifying specific figures/tables for citations
- Quick reference for specific numerical values
- Cross-paper comparative questions

---

## Papers Analyzed

### Core STRAP Papers (8 papers)

1. **Quantifying the environmental benefits of a solvent-based separation process for multilayer plastic films** (2020)
   - Primary LCA study with 3 energy scenarios
   - 15 pages analyzed

2. **Reducing Antisolvent Use in the STRAP Process by Enabling a Temperature-Controlled Polymer Dissolution and Precipitation** (2021)
   - STRAP-A/B/C variants
   - 13 pages analyzed

3. **Techno-Economic and life cycle assessment of standalone Single-Stream material recovery facilities in the United states** (2021)
   - MRF context for plastic recycling
   - 9 pages analyzed

4. **Screening green solvents for multilayer plastic film recycling processes** (2022)
   - Green solvent alternatives (DMI, Cyrene)
   - 17 pages analyzed

5. **A fast computational framework for the design of solvent-based plastic recycling processes** (2022)
   - Multi-polymer optimization
   - 12 pages analyzed

6. **Optimal Design of Food Packaging Considering Waste** (2023)
   - Circular economy integration
   - 9 pages analyzed

7. **Recycling of multilayer plastic packaging materials by solvent-targeted recovery and precipitation** (2018)
   - Original Science Advances paper
   - 9 pages analyzed

8. **A Novel Solvent-Based Recycling Technology** (2020)
   - Pilot plant scale-up
   - 18 pages analyzed

9. **Solvent Based Plastic Recycling Review Published Paper** (2023)
   - Comparative analysis across technologies
   - 17 pages analyzed

**Total:** 119 pages analyzed across 9 papers

---

## Key Findings Summary

### Environmental Performance

**GWP Comparison (kg CO₂-eq/kg polymer):**
- Virgin LDPE: **1.9**
- Virgin PET: **2.5**
- Virgin EVOH: **3.8**
- STRAP C1 (grid + NG): **1.7** (45% reduction vs avg virgin)
- STRAP C2 (all NG/CHP): **1.2** (63% reduction)
- STRAP C3 (renewable + NG): **0.4** (86% reduction)
- Mechanical recycling: **0.5** (but can't process multilayer films)

**Trade-off:** STRAP C3 approaches mechanical recycling GWP while handling complex multilayer waste that mechanical recycling cannot process.

### Economic Performance

**MSP by Scale:**
- Pilot (100 kg/hr): **$2,150/tonne**
- Small commercial (10,000 t/yr): **$1,920/tonne**
- Large commercial (50,000 t/yr): **$1,380/tonne**

**Cost drivers:**
- Energy (heat + electricity): **35%** of MSP
- Capital recovery: **25%**
- Solvents (makeup): **15%**
- Labor: **12%**
- Other operating: **13%**

**Competitiveness:** At current virgin polymer prices ($850-1,200/tonne), STRAP requires policy support (carbon tax, EPR credits) or technology improvements to achieve cost parity.

### Process Performance

**Polymer Yields (STRAP-C optimized):**
- PE: **95%**
- EVOH: **93%**
- PET: **94%**

**Purities:**
- All polymers: **>99%** (comparable to virgin quality)

**Solvent Recovery:**
- **98.5-99.5%** (critical for environmental and economic performance)
- 1% loss increases GWP by **9%**
- 5% loss increases GWP by **44%**

**Energy Breakdown:**
- Heating (dissolution): **45%**
- Distillation (solvent recovery): **35%**
- Cooling: **12%**
- Mechanical: **8%**

### Technology Comparison

**STRAP vs Other Solvent Technologies:**
- **STRAP:** GWP 1.4, MSP $1,520/t, Recovery 98.5%
- **PolyStyrene Loop:** GWP 1.5, MSP $1,600/t, Recovery 96%
- **CreaSolv:** GWP 1.8, MSP $2,200/t, Recovery 95%
- **APK Newcycling:** GWP 2.1, MSP $1,900/t, Recovery 97%

**STRAP advantages:**
- Highest solvent recovery (98.5%)
- Lowest MSP ($1,520/t)
- Lowest GWP (1.4 kg CO₂-eq/kg)
- Handles complex multilayer films (PE/EVOH/PET)

---

## Using This Data for AI Agent Benchmarks

### Benchmark Question Categories

**1. Factual Recall (Level 1)**
- Direct lookup from tables/figures
- Example: "What is the GWP of STRAP C1?"
- Answer source: CSV line 2 or Figure 3 reference

**2. Comparative Analysis (Level 2)**
- Compare across scenarios or technologies
- Example: "Which has lower GWP: STRAP C2 or mechanical recycling?"
- Answer source: Multiple CSV rows + methodology notes

**3. Calculation & Reasoning (Level 3)**
- Requires numerical manipulation
- Example: "If solvent loss increases to 5%, what is the new GWP?"
- Answer source: Sensitivity analysis section + base data

**4. Methodology Understanding (Level 4)**
- Tests LCA concept knowledge
- Example: "Why does economic allocation give EVOH higher burden?"
- Answer source: Methodology document allocation section

**5. Multi-Objective Optimization (Level 5)**
- Trade-off analysis
- Example: "Recommend solvent pair minimizing both cost and GWP"
- Answer source: Green solvents table + multi-criteria analysis

**6. Uncertainty Quantification (Level 6)**
- Data quality and error propagation
- Example: "What is the uncertainty in STRAP GWP given ±20% electricity CI uncertainty?"
- Answer source: Sensitivity analysis + methodology assumptions

**7. Policy & Market Analysis (Level 7)**
- Real-world decision support
- Example: "What carbon price makes STRAP competitive with virgin PE?"
- Answer source: Economics data + GWP comparison + market prices

### Sample Benchmark Question Template

```yaml
question:
  text: "What is the GWP of STRAP process under energy scenario C1?"
  difficulty: Level 1 (Factual Recall)

answer:
  value: 1.7
  unit: "kg CO2-eq/kg polymer"

source:
  file: "STRAP-LCA-Comprehensive-Extraction.csv"
  row: 2
  paper: "Quantifying environmental benefits (2020)"
  reference: "Figure 3"

validation:
  exact_match: true
  tolerance: ±0.05 kg CO2-eq/kg

metadata:
  category: "GWP"
  system_boundary: "Cradle-to-gate"
  functional_unit: "1 kg recycled polymer"
  energy_scenario: "C1 - Grid avg electricity + NG heat"
```

### Verification Strategy

For each AI agent response:
1. **Check exact value** against CSV database
2. **Verify units** match functional unit
3. **Confirm context** (system boundary, energy scenario, allocation)
4. **Cross-reference** with methodology document for reasoning
5. **Validate source** using Tables-Figures reference

---

## Data Quality Assessment

### High Quality Data (Confidence: 90%+)
- GWP values from published LCA studies
- Polymer yields from experimental validation
- Energy scenarios (C1/C2/C3) from BioSTEAM modeling
- Solvent properties (boiling point, HSP) from databases
- Economics for 10,000 t/yr scale (well-studied)

### Medium Quality Data (Confidence: 60-90%)
- Virgin polymer GWP (varies by source, geography)
- Solvent production GWP (limited public data for DMSO, DMI)
- Pilot-to-commercial scale-up factors
- Green solvent alternatives (limited validation)
- Transportation assumptions

### Low Quality Data (Confidence: <60%)
- EVOH production GWP (proxy data used)
- Novel green solvent LCA (Cyrene, bio-based alternatives)
- Very large scale economics (>50,000 t/yr, extrapolated)
- Geographic variations (EU/Asia vs US)
- Future projections (2030+ grid decarbonization)

### Data Gaps Identified
1. **Missing impact categories:** Land use, biodiversity, human health (particulate matter)
2. **Limited temporal analysis:** No time-series or dynamic LCA
3. **Incomplete toxicity data:** USEtox factors for green solvents
4. **No social LCA:** Job creation, community impacts, safety
5. **Regional variations:** Most data US-centric, limited EU/Asia data

---

## Future Enhancements

### Recommended Additions
1. **Consequential LCA data** (market-mediated effects)
2. **Uncertainty distributions** (Monte Carlo simulation results)
3. **Time-series projections** (2025-2050 grid decarbonization)
4. **Regional variants** (EU, China, India energy grids)
5. **Social LCA metrics** (employment, safety, equity)
6. **Comparative pyrolysis data** (chemical recycling alternatives)
7. **Hybrid scenarios** (STRAP + mechanical pre-sorting)

### Additional Papers to Analyze
- Large-scale computational polymer solubility predictions (not yet read)
- Recent 2024-2025 STRAP publications (if available)
- Industrial validation studies (if publicly available)
- Comparative studies with CreaSolv, APK, Purecycle (detailed LCA)

---

## Citation

If using this data extraction in publications or research:

```
STRAP LCA Data Extraction (2026)
Comprehensive extraction of life cycle assessment data from 9 academic papers (2018-2023)
covering STRAP (Solvent-Targeted Recovery and Precipitation) technology for multilayer
plastic film recycling. Includes 170+ quantitative metrics, methodology details, and
benchmark question framework.

Primary sources: See individual paper citations in Tables-Figures reference document.
```

---

## Contact & Updates

**Created:** March 5, 2026
**Version:** 1.0
**Data Status:** Static extraction (will not auto-update with new papers)

For the most current STRAP research, search:
- Google Scholar: "STRAP solvent targeted recovery precipitation"
- Keywords: "solvent-based recycling", "multilayer plastic", "dissolution recycling", "BioSTEAM LCA"

---

## Quick Start Guide

### For Creating Benchmark Questions:
1. **Start with:** Tables-Figures reference for exact values
2. **Add context:** Methodology document for system boundaries
3. **Verify answer:** CSV database for quick lookup

### For Understanding STRAP LCA:
1. **Start with:** Methodology document overview (Section 1-3)
2. **Deep dive:** Energy scenarios and sensitivity analysis
3. **Cross-reference:** Tables-Figures for specific papers

### For Validating AI Responses:
1. **Check value:** CSV database (quick lookup)
2. **Check units:** Functional unit column
3. **Check context:** System boundary, allocation method
4. **Check source:** Reference location column

---

## File Dependencies

```
STRAP-LCA-Extraction-README.md (this file)
├── References: All 3 files below
│
├── STRAP-LCA-Comprehensive-Extraction.csv
│   ├── Primary data source
│   └── 170 rows × 12 columns
│
├── STRAP-LCA-Methodology-Details.md
│   ├── LCA framework and assumptions
│   ├── Benchmark question templates (7 levels)
│   └── Sensitivity and uncertainty analysis
│
└── STRAP-LCA-Tables-Figures-Reference.md
    ├── Paper-by-paper table/figure extraction
    ├── Cross-paper comparisons
    └── Summary statistics
```

---

## Appendix: Key Abbreviations

**LCA Terms:**
- **GWP:** Global Warming Potential (kg CO₂-eq)
- **AP:** Acidification Potential (kg SO₂-eq)
- **EP:** Eutrophication Potential (kg N-eq or P-eq)
- **ODP:** Ozone Depletion Potential (kg CFC-11-eq)
- **POCP:** Photochemical Oxidation Creation Potential (kg C₂H₄-eq)
- **HTC:** Human Toxicity Cancer (CTUh)
- **HTNC:** Human Toxicity Non-Cancer (CTUh)
- **FE:** Freshwater Ecotoxicity (CTUe)
- **CCI:** Climate Change Impact (synonym for GWP in some papers)

**Process Terms:**
- **STRAP:** Solvent-Targeted Recovery and Precipitation
- **MSP:** Minimum Selling Price (USD/tonne)
- **TCI:** Total Capital Investment (USD)
- **TEA:** Techno-Economic Analysis
- **MRF:** Material Recovery Facility
- **PE:** Polyethylene
- **PET:** Polyethylene terephthalate
- **EVOH:** Ethylene vinyl alcohol copolymer
- **PA:** Polyamide (nylon)
- **PS:** Polystyrene

**Solvents:**
- **DMSO:** Dimethyl sulfoxide
- **DMI:** 1,3-Dimethyl-2-imidazolidinone
- **DMF:** N,N-Dimethylformamide
- **NMP:** N-Methyl-2-pyrrolidone
- **Cyrene:** Dihydrolevoglucosenone (bio-based green solvent)

**Energy Scenarios:**
- **C1:** Grid average electricity + natural gas heat
- **C2:** Combined heat and power (CHP) from natural gas
- **C3:** Renewable electricity + natural gas heat

**Methods:**
- **TRACI:** Tool for Reduction and Assessment of Chemicals and Other Environmental Impacts
- **CML:** Center of Environmental Science, Leiden University method
- **ReCiPe:** Recipe for impact assessment (combination of CML and Eco-indicator)
- **IPCC:** Intergovernmental Panel on Climate Change (GWP characterization)
- **USEtox:** Consensus toxicity model
- **COSMO-RS:** Conductor-like Screening Model for Real Solvents
- **HSP:** Hansen Solubility Parameters

---

**End of README**
