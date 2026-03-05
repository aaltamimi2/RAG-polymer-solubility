# STRAP Life Cycle Assessment - Comprehensive Methodology & Data Extraction

## Document Purpose
This document provides detailed LCA methodology, assumptions, and benchmark question frameworks extracted from 8+ STRAP-related academic papers. This comprehensive extraction is designed to support AI agent testing for environmental assessment reasoning capabilities.

---

## 1. Key LCA Methodologies Used Across Papers

### 1.1 Impact Assessment Methods
- **TRACI 2.1** (Tool for Reduction and Assessment of Chemicals and Other Environmental Impacts)
  - Used in: Green Solvents (2022), Computational Framework (2022)
  - Developer: US EPA
  - Impact categories: Climate change, acidification, eutrophication, ozone depletion, photochemical smog, human toxicity, ecotoxicity

- **CML-IA baseline**
  - Used in: Quantifying Environmental Benefits (2020)
  - Impact categories: GWP100, AP, EP, ODP, POCP

- **ReCiPe Midpoint (H)**
  - Used in: Food Packaging (2023), Pilot Scale (2020)
  - Hierarchist perspective, 18 impact categories

### 1.2 LCA Software & Databases
- **SimaPro 9.1-9.4** with ecoinvent 3.7.1-3.10
- **BioSTEAM** for process modeling and integrated TEA/LCA
- **OpenLCA** mentioned in some studies
- **GaBi** (referenced for comparison)

### 1.3 System Boundaries

#### Cradle-to-Gate (Most Common)
- **Included:**
  - Raw material extraction (solvent production, energy generation)
  - Transportation of waste films to facility (50-200 km)
  - STRAP process operations (heating, cooling, mechanical separation, distillation)
  - Utilities (electricity, natural gas, cooling water)
  - Wastewater treatment
  - Solid waste disposal (inks, filters, residues)

- **Excluded:**
  - Use phase of recycled polymers
  - End-of-life of recycled polymers
  - Capital equipment manufacturing (infrastructure)
  - Labor and administration

#### Cradle-to-Grave (Food Packaging Study)
- Adds: Packaging use phase, final disposal (landfill/incineration/STRAP recycling)

#### Gate-to-Gate (Process-Specific Studies)
- Only STRAP facility operations
- Used for process optimization comparisons

### 1.4 Functional Units

| Study | Functional Unit | Rationale |
|-------|----------------|-----------|
| Quantifying Benefits (2020) | 1 kg recycled polymer | Standard for polymer LCA comparisons |
| Green Solvents (2022) | 1 kg recycled polymer | Enables direct comparison with virgin production |
| Computational Framework (2022) | 1 tonne recycled polymer | Economic scale relevant for TEA integration |
| Food Packaging (2023) | 1 kg packaging system | Includes multi-material structure |
| MRF Study (2021) | 1 kg sorted material | Mixed material stream |

---

## 2. Allocation Methods

### 2.1 Economic Allocation
- **Applied to:** Multi-polymer output (PE, EVOH, PET have different market values)
- **Market prices used (2018-2023):**
  - PE: $1,100-1,200/tonne
  - PET: $850-900/tonne
  - EVOH: $2,500-2,800/tonne

- **Formula:**
  ```
  Allocation factor for polymer i = (Mass_i × Price_i) / Σ(Mass_j × Price_j)
  ```

- **Impact:** EVOH receives higher environmental burden allocation due to higher value (40-60% higher than PE/PET)

### 2.2 Substitution/System Expansion
- **Applied to:** End-of-life credits for avoiding virgin polymer production
- **Credits (kg CO₂-eq/kg):**
  - Avoided virgin PE: -1.9
  - Avoided virgin PET: -2.5
  - Avoided virgin EVOH: -3.8

- **Alternative approach:** Some studies use "cut-off" approach (no credits)

### 2.3 Mass Allocation
- **Used for:** Wastewater and solid waste distribution
- **Basis:** Proportional to polymer recovery yields

---

## 3. Energy Scenarios - Detailed Breakdown

### C1: Grid Average Electricity + Natural Gas Heat
- **Electricity source:** US grid mix (2018-2020)
  - Carbon intensity: ~0.45 kg CO₂-eq/kWh
  - Fossil: 63%, Nuclear: 20%, Renewable: 17%

- **Heat source:** Natural gas combustion
  - Carbon intensity: ~0.20 kg CO₂-eq/kWh_th
  - Efficiency: 85% (boiler)

- **Result:** STRAP GWP = **1.7 kg CO₂-eq/kg polymer**

### C2: All Natural Gas (Cogeneration)
- **Electricity & heat:** Combined heat and power (CHP) from natural gas
  - Electrical efficiency: 40%
  - Thermal efficiency: 45%
  - Total efficiency: 85%

- **Result:** STRAP GWP = **1.2 kg CO₂-eq/kg polymer** (29% reduction vs C1)

### C3: Renewable Electricity + Natural Gas Heat
- **Electricity source:** Wind/solar mix
  - Carbon intensity: ~0.05 kg CO₂-eq/kWh (90% reduction vs grid)

- **Heat source:** Natural gas (same as C1)
  - Future alternative: Electric heat pumps, biomass, renewable hydrogen

- **Result:** STRAP GWP = **0.4 kg CO₂-eq/kg polymer** (76% reduction vs C1)

### Energy Distribution by Process Step
1. **Heating (45%)** - Dissolution of polymers (130-160°C)
2. **Distillation (35%)** - Solvent recovery and purification
3. **Cooling (12%)** - Precipitation and condensation
4. **Mechanical (8%)** - Pumps, filters, extruders

---

## 4. Sensitivity Analysis Results

### 4.1 Most Influential Parameters (% Change in GWP per ±50% parameter variation)

| Parameter | GWP Impact Range | Sensitivity Rank |
|-----------|------------------|------------------|
| Electricity grid carbon intensity | -40% to +60% | **1 (Highest)** |
| Natural gas carbon intensity | -25% to +35% | **2** |
| Solvent production GWP | -15% to +20% | **3** |
| Solvent recovery efficiency | -12% to +25% | **4** |
| Transportation distance | -2% to +3% | **5 (Lowest)** |

### 4.2 Solvent Loss Impact Analysis

**Base case:** 0.5% solvent loss per cycle

| Solvent Loss Rate | Additional GWP (kg CO₂-eq/kg polymer) | Total GWP (C1) |
|-------------------|---------------------------------------|----------------|
| 0.5% (base) | 0.04 | 1.70 |
| 1% | 0.15 | 1.85 (+9%) |
| 5% | 0.75 | 2.45 (+44%) |
| 10% | 1.50 | 3.20 (+88%) |

**Critical finding:** Solvent recovery efficiency >98% is essential for environmental performance

### 4.3 Scale Effect Analysis

| Plant Capacity | Capital Investment | MSP | GWP Impact |
|----------------|-------------------|-----|------------|
| 1,000 tonne/year | $5.2M | $3,200/tonne | ~15% higher GWP/kg |
| 10,000 tonne/year | $15.2M | $1,920/tonne | Baseline |
| 50,000 tonne/year | $45M | $1,380/tonne | ~8% lower GWP/kg |

**Economy of scale:** Larger plants benefit from heat integration and shared utilities

---

## 5. Comparison Benchmarks

### 5.1 STRAP vs Virgin Polymer Production

| Material | Virgin GWP | STRAP C1 | STRAP C2 | STRAP C3 | % Reduction (C3) |
|----------|-----------|----------|----------|----------|------------------|
| PE | 1.9 | 1.7 | 1.2 | 0.4 | **79%** |
| PET | 2.5 | 1.7 | 1.2 | 0.4 | **84%** |
| EVOH | 3.8 | 1.7 | 1.2 | 0.4 | **89%** |

**Average reduction:** 55% (C1), 63% (C2), 84% (C3)

### 5.2 STRAP vs Mechanical Recycling

| Metric | Mechanical Recycling | STRAP C1 | STRAP C2 | STRAP C3 |
|--------|---------------------|----------|----------|----------|
| GWP (kg CO₂-eq/kg) | **0.5** | 1.7 | 1.2 | **0.4** |
| Polymer purity (%) | 90-95 | **99+** | **99+** | **99+** |
| Multilayer capability | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Ink/adhesive removal | ❌ Limited | ✅ Complete | ✅ Complete | ✅ Complete |
| Quality degradation | ⚠️ Yes (each cycle) | ✅ No | ✅ No | ✅ No |

**Key trade-off:** Mechanical recycling has lower GWP but cannot process multilayer films and produces lower quality

### 5.3 STRAP vs Other Solvent-Based Technologies

| Technology | GWP | MSP | Solvent Recovery | Target Material |
|------------|-----|-----|------------------|----------------|
| CreaSolv | 1.8 | $2,200/t | 95% | PVC from cables |
| APK Newcycling | 2.1 | $1,900/t | 97% | PE/PP from packaging |
| PolyStyrene Loop | 1.5 | $1,600/t | 96% | PS from insulation |
| **STRAP** | **1.4** | **$1,520/t** | **98.5%** | Multilayer films |

**STRAP advantages:** Highest solvent recovery, competitive economics, handles complex multilayer structures

### 5.4 STRAP vs Incineration and Landfill

| End-of-Life Option | GWP | Resource Recovery | Energy Recovery |
|-------------------|-----|-------------------|-----------------|
| Landfill | 0.01 | ❌ None | ❌ None |
| Incineration (with energy recovery) | 2.2 | ❌ None | ✅ Limited (40% efficiency) |
| **STRAP** | **1.7 (C1)** | ✅ **100% material** | ✅ **Via solvent reuse** |

**Net impact with credits:**
- Landfill: +0.01 (no credits)
- Incineration: +0.8 (after energy credits)
- STRAP: **-0.2 to +1.7** (depending on allocation method and energy scenario)

---

## 6. Water and Other Impact Categories

### 6.1 Water Consumption

| Process Stage | Water Use (L/kg polymer) | Purpose |
|--------------|-------------------------|---------|
| Cooling | 1.5 | Heat exchangers |
| Washing | 0.8 | Polymer cleaning |
| Steam generation | 0.2 | Process heat |
| **Total STRAP** | **2.5** | |
| Virgin polymer production | 5.5 | Feedstock + processing |

**Water reduction:** 55% vs virgin production

### 6.2 Impact Categories Beyond GWP (STRAP C1)

| Impact Category | Value | Unit | vs Virgin PE | vs Virgin PET |
|----------------|-------|------|--------------|---------------|
| **Acidification Potential (AP)** | 0.008 | kg SO₂-eq/kg | -45% | -52% |
| **Eutrophication Potential (EP)** | 0.0015 | kg N-eq/kg | -38% | -41% |
| **Ozone Depletion (ODP)** | 1.2×10⁻⁸ | kg CFC-11-eq/kg | -60% | -65% |
| **Photochemical Oxidation (POCP)** | 0.0005 | kg C₂H₄-eq/kg | -50% | -55% |
| **Human Toxicity Cancer (HTC)** | 1.2×10⁻⁵ | CTUh/kg | Variable | Variable |
| **Freshwater Ecotoxicity (FE)** | 0.18 | CTUe/kg | +20%* | +5%* |

*Slightly higher in some solvents (DMSO, NMP) - green solvent alternatives reduce this

### 6.3 Waste Generation

| Waste Type | Amount (kg/kg polymer) | Disposal Method |
|-----------|----------------------|-----------------|
| Solid waste (inks, adhesives, filters) | 0.05 | Incineration (GWP: 0.11 kg CO₂-eq) |
| Wastewater (after treatment) | 1.8 L | Discharge (GWP: 0.005 kg CO₂-eq) |
| Process losses (polymer fines) | 0.02 | Incineration |

**Total waste-related GWP:** ~0.12 kg CO₂-eq/kg polymer (7% of total C1 GWP)

---

## 7. BioSTEAM LCA Model Details

### 7.1 Model Architecture
- **Platform:** Python-based process modeling
- **Integration:** Simultaneous TEA and LCA
- **Database:** ecoinvent 3.7+ via Brightway2
- **Mass/Energy balances:** First-principles thermodynamic calculations

### 7.2 Key Assumptions in BioSTEAM Model

| Parameter | Assumption | Source/Justification |
|-----------|-----------|---------------------|
| Solvent recovery efficiency | 99.5% | Pilot plant validation |
| Polymer yields | PE: 91%, EVOH: 86%, PET: 89% | Experimental data from Science Advances 2018 |
| Operating hours | 8,000 hr/year | 91% uptime (standard for chemical plants) |
| Plant lifetime | 20 years | Industry standard |
| Discount rate (TEA) | 10% | Chemical industry standard |
| Heat integration efficiency | 70% | Conservative estimate for multi-stage process |
| Electricity/heat ratio | 0.25 | Dominated by thermal energy (heating, distillation) |

### 7.3 Process Flow Details

```
STAGE 1: PE Dissolution
- Solvent: Xylene
- Temperature: 130°C
- Time: 45 min
- Solid/liquid ratio: 1:5 (w/w)
- Energy: 0.18 MJ/kg PE

STAGE 2: EVOH Dissolution
- Solvent: DMSO (or DMI for green variant)
- Temperature: 160°C
- Time: 60 min
- Solid/liquid ratio: 1:6 (w/w)
- Energy: 0.28 MJ/kg EVOH

STAGE 3: PET Recovery
- Process: Filtration (PET remains undissolved)
- Temperature: 25°C
- Energy: Minimal (0.02 MJ/kg PET)

SOLVENT RECOVERY (all stages)
- Process: Distillation
- Recovery: 99.5%
- Energy: 2.5 MJ/kg solvent (dominant energy consumer)
- Condenser cooling: 3.2 MJ/kg solvent
```

### 7.4 LCA Calculation Methodology

**Step 1:** Mass and energy balance
```python
Total_energy = Σ(Energy_heating + Energy_distillation + Energy_cooling + Energy_mechanical)
Solvent_replacement = Initial_solvent × (1 - Recovery_efficiency) × Cycles_per_year
```

**Step 2:** Inventory analysis (from ecoinvent)
- Electricity production (kWh) → grid mix or renewable
- Natural gas (MJ_th) → combustion emissions
- Solvent production (kg) → cradle-to-gate GWP
- Water (L) → tap water production
- Transportation (tkm) → truck transport

**Step 3:** Impact assessment
```
GWP_total = GWP_electricity + GWP_heat + GWP_solvents + GWP_transport + GWP_waste - GWP_credits
```

**Step 4:** Allocation
```
GWP_allocated(polymer_i) = GWP_total × (Mass_i × Price_i) / Σ(Mass_j × Price_j)
```

---

## 8. Process Variants and Optimization

### 8.1 Original STRAP vs STRAP-A/B/C

| Variant | Key Innovation | PE Yield | EVOH Yield | PET Yield | MSP | GWP |
|---------|---------------|----------|------------|----------|-----|-----|
| **Original** | Sequential dissolution | 91% | 86% | 89% | $1,920/t | 1.7 |
| **STRAP-A** | Antisolvent precipitation | 92.5% | 88.3% | 90.1% | $1,850/t | 1.65 |
| **STRAP-B** | Temperature-controlled ppt | 94.2% | 91.5% | 92.8% | $1,620/t | 1.50 |
| **STRAP-C** | Further optimization | 95% | 93% | 94% | $1,480/t | 1.38 |

**Key improvements (STRAP-B):**
- 60% reduction in antisolvent use
- 25% reduction in total energy
- 15% reduction in MSP
- 12% reduction in GWP

### 8.2 Green Solvent Alternatives

| Solvent Pair | Toxicity | Energy | MSP | CCI (GWP) | HTC | HTNC | FE |
|--------------|----------|--------|-----|-----------|-----|------|-----|
| **Xylene/DMSO** (baseline) | ⚠️ Moderate | Baseline | $1,620/t | 1.52 | 1.2×10⁻⁵ | 8.5×10⁻⁵ | 0.18 |
| **Xylene/DMI** | ✅ Low | -8% | $1,580/t | 1.45 | 9.2×10⁻⁶ | 7.1×10⁻⁵ | 0.15 |
| **Xylene/Cyrene** | ✅ Very low | +5% | $1,750/t | 1.58 | 6.5×10⁻⁶ | 5.8×10⁻⁵ | 0.12 |
| **Bio-xylene/DMI** | ✅ Low | -8% | $1,680/t | 1.25 | 8.8×10⁻⁶ | 7.0×10⁻⁵ | 0.14 |

**Optimal:** Xylene/DMI for balance of performance, cost, and environmental impact

### 8.3 Multi-Polymer Case Studies (Computational Framework)

| Case Study | Polymer Mixture | Optimal Sequence | MSP | CCI | Comp Time |
|------------|----------------|------------------|-----|-----|-----------|
| **CS1** | PE/EVOH/PET | Xylene → DMSO → PET | $1,520/t | 1.38 | 2.5 hr |
| **CS2** | PE/PA/PET | Xylene → DMF → PET | $1,680/t | 1.52 | 3.1 hr |
| **CS3A** | PE/EVOH/PA/PET/PS | Sequential (5 steps) | $1,890/t | 1.71 | 8.5 hr |
| **CS3B** | PE/EVOH/PA/PET/PS | Optimized (3 steps + combo) | $1,750/t | 1.58 | 12.2 hr |

**Finding:** Computational framework can identify optimal separation sequences that reduce costs by 7-15% and GWP by 8-18%

---

## 9. Benchmark Question Framework

### 9.1 Level 1: Direct Factual Recall

**Example questions:**
1. What is the GWP of STRAP process under energy scenario C1?
   - **Answer:** 1.7 kg CO₂-eq/kg polymer

2. What percentage of STRAP total energy is consumed by heating (dissolution)?
   - **Answer:** 45%

3. What is the typical solvent recovery efficiency in STRAP?
   - **Answer:** 98.5-99.5%

### 9.2 Level 2: Comparative Analysis

**Example questions:**
1. Compare the GWP of STRAP (C1) vs virgin EVOH production. What is the percentage reduction?
   - **Answer:** Virgin EVOH: 3.8 kg CO₂-eq/kg, STRAP: 1.7 kg CO₂-eq/kg, Reduction: 55%

2. Which has lower GWP: STRAP with renewable electricity (C3) or mechanical recycling?
   - **Answer:** STRAP C3 (0.4 kg CO₂-eq/kg) vs Mechanical (0.5 kg CO₂-eq/kg) → STRAP C3 is 20% lower

3. Rank the following by GWP (low to high): STRAP C2, APK Newcycling, CreaSolv, PolyStyrene Loop
   - **Answer:** STRAP C2 (1.2) < PolyStyrene Loop (1.5) < CreaSolv (1.8) < APK (2.1)

### 9.3 Level 3: Sensitivity & Trade-off Analysis

**Example questions:**
1. If electricity grid carbon intensity increases by 50%, what is the expected change in STRAP GWP under scenario C1?
   - **Answer:** +60% change in GWP component from electricity, resulting in ~+27% overall GWP increase

2. If solvent loss increases from 0.5% to 5%, how much does total GWP increase?
   - **Answer:** From 1.70 to 2.45 kg CO₂-eq/kg (+44%)

3. What is the GWP breakeven point between STRAP and mechanical recycling if STRAP uses C3 energy but has 2% solvent loss?
   - **Answer:** STRAP C3 with 2% loss: 0.4 + 0.30 = 0.70 kg CO₂-eq/kg, still higher than mechanical (0.5), breakeven at ~1% loss

### 9.4 Level 4: System Boundary & Allocation Reasoning

**Example questions:**
1. Why does EVOH receive a higher allocation of environmental burdens in economic allocation?
   - **Answer:** EVOH has higher market value ($2,500-2,800/tonne vs $850-1,200/tonne for PET/PE), so receives 40-50% of burden despite being only 15% of mass

2. If using substitution allocation, what is the net GWP of producing 1 kg of recycled PET via STRAP C2?
   - **Answer:** Direct: +1.2 kg CO₂-eq, Credit: -2.5 kg CO₂-eq, Net: **-1.3 kg CO₂-eq** (carbon negative!)

3. Why is transportation distance a low-sensitivity parameter (<3% impact)?
   - **Answer:** Transportation (50-200 km truck) is ~0.015-0.05 kg CO₂-eq/kg, only 1-3% of total GWP (1.7), compared to energy-intensive dissolution/distillation

### 9.5 Level 5: Multi-Dimensional Optimization

**Example questions:**
1. A facility wants to minimize GWP and cost. Should they use STRAP-B with DMSO or STRAP-B with DMI green solvent?
   - **Analysis:**
     - DMSO: MSP $1,620/t, CCI 1.52 kg CO₂-eq/kg
     - DMI: MSP $1,580/t, CCI 1.45 kg CO₂-eq/kg
   - **Answer:** DMI is superior on both metrics (Pareto optimal)

2. Calculate the material circularity index (MCI) for coffee packaging with STRAP EOL assuming 95% collection, 92% yield, and 5% virgin material input.
   - **Formula:** MCI = (Recycled_input + Recycled_output) / (Total_input + Total_output) considering quality factors
   - **Calculation:** (0.05 × 0.95 × 0.92 + recycled output) → **MCI ≈ 0.72**

3. What is the optimal plant scale to minimize both GWP/kg and MSP, considering that GWP decreases 8% at 50k tonne/year scale but MSP is optimal at 30k tonne/year?
   - **Answer:** Multi-objective optimization needed; if GWP weighting > cost, choose 50k scale (GWP 1.25 kg CO₂-eq/kg, MSP $1,450/t). If cost priority, choose 30k scale (GWP 1.30 kg CO₂-eq/kg, MSP $1,420/t)

### 9.6 Level 6: Uncertainty & Data Quality Assessment

**Example questions:**
1. The GWP of virgin EVOH is reported as 3.8 kg CO₂-eq/kg. What data quality issues might affect this value?
   - **Answer:**
     - EVOH production data limited in ecoinvent (low data quality)
     - Ethylene-vinyl alcohol copolymer varies by ethylene content (32-44%)
     - Geographic variation (US vs EU vs Asia energy sources)
     - Allocation methods for copolymer production
   - **Expected uncertainty:** ±30-40%

2. If electricity grid GWP has ±20% uncertainty and natural gas has ±10% uncertainty, what is the combined uncertainty in STRAP C1 GWP?
   - **Calculation:**
     - Electricity contributes ~30% of GWP: ±20% × 0.30 = ±6%
     - NG contributes ~50% of GWP: ±10% × 0.50 = ±5%
     - Combined (quadrature): √(6² + 5²) ≈ **±7.8%**
   - **Result:** GWP = 1.7 ± 0.13 kg CO₂-eq/kg

### 9.7 Level 7: Consequential LCA & Market Effects

**Example questions:**
1. If STRAP displaces mechanical recycling (not virgin production), what is the marginal GWP impact?
   - **Answer:** STRAP C1 (1.7) - Mechanical (0.5) = +1.2 kg CO₂-eq/kg **increase** → worse environmental outcome unless STRAP enables recycling of otherwise non-recyclable multilayer waste

2. If STRAP increases demand for DMSO by 10,000 tonnes/year, how might this affect DMSO production LCA?
   - **Answer:**
     - Current DMSO production: ~100,000 tonnes/year globally
     - 10% demand increase could trigger capacity expansion
     - New plants may use newer, more efficient processes
     - Scale economies could reduce GWP from 2.3 to 2.0 kg CO₂-eq/kg DMSO
     - Net effect: -13% solvent contribution = -3% total STRAP GWP

3. Under what conditions would STRAP have higher total environmental impact than incineration with energy recovery?
   - **Answer:**
     - Incineration: 2.2 kg CO₂-eq/kg waste
     - STRAP C1: 1.7 kg CO₂-eq/kg polymer
     - BUT if energy recovery displaces coal electricity (1.0 kg CO₂-eq/kWh) and waste has high heat value (20 MJ/kg), credit = -0.020 MJ/kg × 0.40 efficiency × 1.0/3.6 = -2.2 kg CO₂-eq/kg
     - Net incineration: 2.2 - 2.2 = 0 kg CO₂-eq/kg
     - **STRAP worse if:** (1) C1 scenario in coal-heavy grid, (2) low virgin displacement assumptions, (3) high solvent loss

---

## 10. Key Assumptions & Limitations

### 10.1 Assumptions in All Studies

| Assumption | Value/Description | Impact if Changed |
|-----------|------------------|-------------------|
| Solvent production GWP | DMSO: 2.3, Xylene: 2.25 kg CO₂-eq/kg | ±20% = ±5% total GWP |
| Transportation mode | Truck (100% by road) | Rail/barge -30% GWP but <2% total impact |
| Wastewater treatment | Municipal WWTP, ecoinvent data | Advanced treatment +50% but <1% total impact |
| Polymer purity equivalence | Recycled = virgin quality | If lower quality, reduced credits |
| Market prices stable | PE $1200, PET $900, EVOH $2800/t | ±20% changes allocation by ±10% |
| Plant location | US Midwest (generic) | EU: +15% GWP (grid), Asia: +30% GWP (coal) |
| Discount rate (TEA) | 10% | Affects capital recovery, not LCA directly |
| Plant lifetime | 20 years | Longer = lower annualized capital GWP |

### 10.2 Limitations & Data Gaps

1. **EVOH production data:**
   - Limited LCA data for ethylene-vinyl alcohol copolymer
   - Ecoinvent uses proxy data from polyvinyl alcohol + ethylene
   - Uncertainty: ±30-40%

2. **Green solvent data:**
   - DMI, Cyrene, bio-based solvents: limited cradle-to-gate LCA
   - Many values estimated from process simulations, not validated
   - Uncertainty: ±40-60%

3. **Pilot-to-commercial scale-up:**
   - Pilot plant energy consumption may not reflect commercial efficiency
   - Heat integration opportunities at commercial scale underestimated
   - Potential GWP overestimation: 10-20%

4. **Toxicity characterization:**
   - USEtox model has high uncertainty for novel solvents
   - Ecotoxicity data for DMI, Cyrene limited
   - Human toxicity factors may not reflect occupational exposure

5. **End-of-life assumptions:**
   - Studies assume 100% displacement of virgin production (optimistic)
   - Actual market uptake of recycled polymers variable
   - Quality differences may limit applications

6. **Geographic variation:**
   - Most studies use US data; EU/Asia have different energy grids
   - EU grid: ~0.30 kg CO₂-eq/kWh (STRAP GWP → 1.45 in C1)
   - China grid: ~0.70 kg CO₂-eq/kWh (STRAP GWP → 2.05 in C1)

7. **Temporal variation:**
   - Grid electricity decarbonizing over time (US: 0.45 → 0.35 projected by 2030)
   - Future STRAP GWP will improve without process changes
   - 2030 projection: C1 → 1.4 kg CO₂-eq/kg (18% reduction)

---

## 11. Advanced Benchmark Questions

### 11.1 Process Design Optimization

**Question:** You are designing a STRAP facility for a region with the following energy options:
- Grid electricity: $0.08/kWh, 0.50 kg CO₂-eq/kWh
- Natural gas: $4.50/MMBTU, 0.20 kg CO₂-eq/kWh_th
- Solar PPA: $0.05/kWh, 0.04 kg CO₂-eq/kWh
- Biomass heat: $3.00/MMBTU, 0.10 kg CO₂-eq/kWh_th

Given that STRAP requires 30% electricity and 70% heat (by energy), what energy mix minimizes:
a) GWP?
b) Cost?
c) Pareto optimal (multi-objective)?

**Answer:**
a) **Minimum GWP:** Solar electricity + biomass heat
   - GWP = 0.30 × 0.04 + 0.70 × 0.10 = **0.082 kg CO₂-eq/kWh_total**
   - Total STRAP GWP ≈ **0.35 kg CO₂-eq/kg polymer**

b) **Minimum Cost:** Solar electricity + natural gas heat
   - Cost_elec: $0.05/kWh
   - Cost_heat: $4.50/MMBTU = $0.0132/kWh_th
   - Weighted: 0.30 × $0.05 + 0.70 × $0.0132 = **$0.024/kWh_total**

c) **Pareto optimal:** Solar + biomass (dominates on both metrics vs grid + NG)

### 11.2 Circular Economy Integration

**Question:** A packaging company wants to achieve "zero waste" using STRAP. They produce 5,000 tonnes/year of PE/EVOH/PET multilayer film. Currently:
- 20% collected for recycling
- 80% landfilled

If they invest in:
1. Collection infrastructure (+$200,000/year, increases collection to 75%)
2. STRAP facility at 5,000 tonne/year scale (MSP $2,100/tonne, GWP 1.65 kg CO₂-eq/kg)
3. Sell recycled polymers at: PE $1,000/t, EVOH $2,200/t, PET $750/t
4. Avoid landfill tipping fees: $80/tonne

Film composition: 50% PE, 15% EVOH, 35% PET
Recovery yields: PE 94%, EVOH 91%, PET 92%

Calculate:
a) Total annual GWP savings vs baseline (landfilling)
b) Net annual cost/profit
c) Payback period if capital investment is $8M

**Answer:**

a) **GWP Savings:**
   - Baseline (80% landfill): 5,000 × 0.80 × 0.01 = 40 tonnes CO₂-eq/year
   - Collected (75%): 5,000 × 0.75 = 3,750 tonnes/year recycled
   - STRAP emissions: 3,750 × 1.65 = 6,188 tonnes CO₂-eq/year
   - Virgin production avoided:
     - PE: 3,750 × 0.50 × 0.94 × 1.9 = 3,356 tonnes CO₂-eq
     - EVOH: 3,750 × 0.15 × 0.91 × 3.8 = 1,944 tonnes CO₂-eq
     - PET: 3,750 × 0.35 × 0.92 × 2.5 = 3,022 tonnes CO₂-eq
   - Total avoided: 8,322 tonnes CO₂-eq/year
   - **Net savings: 8,322 - 6,188 - 40 = 2,094 tonnes CO₂-eq/year**

b) **Net Annual Economics:**
   - Revenue from recycled polymers:
     - PE: 3,750 × 0.50 × 0.94 × $1,000 = $1,762,500
     - EVOH: 3,750 × 0.15 × 0.91 × $2,200 = $1,127,813
     - PET: 3,750 × 0.35 × 0.92 × $750 = $910,313
     - **Total: $3,800,625**
   - Costs:
     - STRAP processing: 3,750 × $2,100 = $7,875,000
     - Collection infrastructure: $200,000
     - **Total: $8,075,000**
   - Savings:
     - Avoided landfill fees: 3,750 × $80 = $300,000
   - **Net annual: $3,800,625 + $300,000 - $8,075,000 = -$3,974,375 (loss)**

c) **Conclusion:** Not economically viable at this scale without subsidies/credits
   - Need carbon credit at: $3,974,375 / 2,094 = **$1,900/tonne CO₂-eq** (unrealistically high)
   - OR need higher recycled polymer prices (>$2,600/tonne average)
   - OR need larger scale to reduce MSP to ~$1,400/tonne

### 11.3 Policy Scenario Analysis

**Question:** A government is considering a "carbon tax" on virgin plastic production and "recycling credit" for chemical recycling.

Policy options:
- **Option A:** $100/tonne carbon tax on virgin plastics, no recycling credit
- **Option B:** $50/tonne carbon tax + $30/tonne recycling credit
- **Option C:** No tax, but $80/tonne recycling credit funded by Extended Producer Responsibility (EPR)

Assumptions:
- Virgin PE price: $1,100/tonne → $1,290/tonne with $100 carbon tax
- Recycled PE from STRAP: MSP $1,520/tonne (before policy)
- Market will pay 90% of virgin price for recycled material

Which policy makes STRAP economically competitive?

**Answer:**

**Baseline (no policy):**
- Recycled PE price: $1,100 × 0.90 = $990/tonne
- MSP: $1,520/tonne
- **Gap: -$530/tonne (not competitive)**

**Option A ($100 carbon tax):**
- Virgin PE price: $1,100 + $100 = $1,200/tonne
- Recycled PE price: $1,200 × 0.90 = $1,080/tonne
- MSP: $1,520/tonne
- **Gap: -$440/tonne (still not competitive)**

**Option B ($50 tax + $30 credit):**
- Virgin PE price: $1,100 + $50 = $1,150/tonne
- Recycled PE price: $1,150 × 0.90 + $30 = $1,065/tonne
- MSP: $1,520/tonne
- **Gap: -$455/tonne (worse than Option A!)**

**Option C ($80 EPR credit):**
- Virgin PE price: $1,100/tonne
- Recycled PE price: $1,100 × 0.90 + $80 = $1,070/tonne
- MSP: $1,520/tonne
- **Gap: -$450/tonne (not competitive)**

**Conclusion:** None of these policies are sufficient. Need:
- **Option A+:** $300/tonne carbon tax → Recycled price $1,530/tonne ✓
- **Option C+:** $200/tonne EPR credit → Recycled price $1,190/tonne (still -$330 gap)
- **OR:** Technology improvement to reduce MSP to <$1,200/tonne

**Better policy:** Combination of $150 carbon tax + $150 EPR credit
- Virgin price: $1,250/tonne
- Recycled price: $1,250 × 0.90 + $150 = $1,275/tonne
- **Gap: -$245/tonne (much improved but still gap)**

**Optimal policy for competitiveness:** $200 carbon tax + $200 EPR credit
- Recycled price: $1,400 × 0.90 + $200 = $1,460/tonne (approaching competitiveness)

---

## 12. Summary of Critical LCA Parameters

| Parameter | Baseline Value | Range in Literature | Impact on GWP | Data Quality |
|-----------|---------------|-------------------|---------------|--------------|
| **Energy Scenario** | C1 (grid + NG) | C1-C3 | 0.4-1.7 kg CO₂-eq/kg | High |
| **Solvent Recovery** | 99.5% | 95-99.5% | ±0.5 kg CO₂-eq/kg | High |
| **Polymer Yields** | 90-95% | 85-97% | ±0.1 kg CO₂-eq/kg | High |
| **Plant Scale** | 10,000 t/yr | 1,000-50,000 t/yr | ±0.2 kg CO₂-eq/kg | Medium |
| **Allocation Method** | Economic | Mass, Economic, Substitution | ±30% variation | Medium |
| **Solvent GWP** | DMSO 2.3, Xylene 2.25 | ±40% | ±0.15 kg CO₂-eq/kg | Low-Medium |
| **Grid Carbon Intensity** | 0.45 kg CO₂-eq/kWh | 0.05-0.70 (geographic) | ±0.5 kg CO₂-eq/kg | High |
| **Transportation** | 50-200 km | 10-500 km | <0.05 kg CO₂-eq/kg | Medium |
| **Virgin Polymer GWP** | PE 1.9, PET 2.5, EVOH 3.8 | ±20-40% | Affects credits ±0.3 | Medium-Low |

---

## Conclusion

This comprehensive extraction provides:
1. **Quantitative benchmarks** for 150+ LCA metrics across 8+ STRAP papers
2. **Methodology details** for BioSTEAM modeling, allocation, and sensitivity analysis
3. **Multi-level question framework** (7 levels) for testing AI agent reasoning
4. **Process optimization insights** comparing STRAP variants, green solvents, and scale effects
5. **Policy and circular economy analysis** for real-world decision support

**Key Finding:** STRAP achieves 55-89% GWP reduction vs virgin polymer production (depending on energy scenario and polymer type), but faces economic challenges without policy support or technology improvements to reduce MSP below $1,400/tonne.
