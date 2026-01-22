# DISSOLVE Agent - Complete Tools Reference

This document provides a comprehensive guide to all **54 tools** available to the DISSOLVE Agent, organized by how the agent uses them to solve polymer-solvent separation problems.

---

## How the Agent Uses Tools

The agent operates using the **ReAct pattern** (Reasoning + Acting) to determine which tools to call:

```
User Query → Agent Reasoning → Tool Selection → Execution → Observation → (Repeat or Respond)
```

### Tool Selection Logic

The agent considers:
1. **What data is needed?** → Determines which tools to call
2. **Are queries independent?** → If yes, call tools in parallel
3. **Do I have enough information?** → If no, call more tools
4. **Can I answer the user now?** → If yes, synthesize and respond

### Parallel vs Sequential Tool Calls

```
PARALLEL (same iteration):
  Agent needs: separation data + cost data + safety data
  These are INDEPENDENT → Call all 3 tools at once
  Result: 1 iteration, ~2 seconds

SEQUENTIAL (multiple iterations):
  Agent needs: first find solvents, THEN get their properties
  Step 2 DEPENDS on Step 1 → Must wait
  Result: 2 iterations, ~4 seconds
```

---

## Table of Contents

1. [Core Database Tools (6)](#1-core-database-tools)
2. [Adaptive Analysis Tools (8)](#2-adaptive-analysis-tools)
3. [Solvent Property Tools (4)](#3-solvent-property-tools)
4. [Statistical Analysis Tools (4)](#4-statistical-analysis-tools)
5. [Visualization Tools (6)](#5-visualization-tools)
6. [GSK Safety Tools (4)](#6-gsk-safety-g-score-tools)
7. [Listing Tools (2)](#7-listing-tools)
8. [ML Prediction Tools (1)](#8-ml-prediction-tools)
9. [PubChem External API Tools (4)](#9-pubchem-external-api-tools)
10. [Literature Search Tools (2)](#10-literature-search-tools)
11. [TEA/LCA Analysis Tools (3)](#11-tealca-analysis-tools)
12. [TEA/LCA Visualization Tools (5)](#12-tealca-visualization-tools)
13. [STRAP Process Tools (5)](#13-strap-process-tools)
14. [Query Examples](#query-examples)

---

## 1. Core Database Tools

These tools enable the agent to explore and query the internal DuckDB database.

### 1.1 `list_tables()`

**When agent uses it**: First exploration, understanding available data

**Returns**: Table names, row counts, column schemas

**Agent reasoning**: "The user wants to know what data is available. I'll list all tables."

---

### 1.2 `describe_table(table_name: str)`

**When agent uses it**: Needs detailed stats for a specific table

**Parameters**:
- `table_name`: Name of the table

**Returns**: Column statistics, unique values, sample data

---

### 1.3 `check_column_values(table_name, column_name, limit=50)`

**When agent uses it**: Needs to verify exact spelling of values

**Agent reasoning**: "User asked about 'polyethylene' but I should verify the exact name in the database."

**Returns**: Unique values with frequency counts

---

### 1.4 `query_database(sql_query, export_csv=False)`

**When agent uses it**: Custom queries beyond what other tools provide

**Parameters**:
- `sql_query`: SQL query to execute
- `export_csv`: Create downloadable CSV

**Returns**: Query results (up to 10 rows preview)

---

### 1.5 `verify_data_accuracy(table_name, filters=None)`

**When agent uses it**: Double-checking data exists before reporting

**Agent reasoning**: "Before telling the user we have X rows, I should verify this is accurate."

---

### 1.6 `validate_and_query(table_name, required_columns, filter_column, filter_values, sql_query)`

**When agent uses it**: Ensuring all inputs are valid before expensive queries

**Returns**: Validation report with checkmarks for each input

```
✅ Table 'common_solvents_database' exists
✅ Column 'polymer' exists
✅ Value 'LDPE' found (896 rows)
```

---

## 2. Adaptive Analysis Tools

These tools implement **intelligent threshold adaptation** - a key agentic feature.

### 2.1 `find_optimal_separation_conditions(...)`

**PRIMARY TOOL for pairwise separation**

**When agent uses it**: User asks "Find solvents to separate X from Y"

**Key parameters**:
- `target_polymer`: Polymer to dissolve
- `comparison_polymers`: Polymers to separate from
- `start_temperature`: Starting temperature (default: 25°C)
- `initial_selectivity`: Starting threshold (default: 30%)

**Agentic behavior**: If no results at 30% selectivity, automatically relaxes to 20%, then 15%, etc.

**Returns**:
```
Optimal Conditions:
  - Temperature: 25°C
  - Solvent: toluene
  - Selectivity: 42.5%
  - Confidence: 92%

Alternative Conditions:
  1. chloroform (38.2%)
  2. xylene (35.8%)
```

---

### 2.2 `adaptive_threshold_search(...)`

**When agent uses it**: User wants to see how thresholds affect results

**Agentic behavior**: Shows the "search path" - which thresholds were tried

**Returns**:
```
Search Path: [0.5 → 0.4 → 0.3 ✓]
Found 12 solvents at threshold 0.3
```

---

### 2.3 `analyze_selective_solubility_enhanced(...)`

**When agent uses it**: Comprehensive selectivity analysis with ranking

**Returns**: Top-k solvents ranked by selectivity with detailed metrics

---

### 2.4 `plan_sequential_separation(...)`

**PRIMARY TOOL for multi-polymer separation**

**When agent uses it**: User asks "Separate X, Y, Z" (3+ polymers)

**Key parameters**:
- `polymers`: Comma-separated list (e.g., "LDPE,PET,PP")
- `temperature`: Temperature to analyze
- `create_decision_tree`: Generate flowchart visualization

**Agentic behavior**: Enumerates ALL possible sequences and ranks them

**Returns**:
```
Analyzing all 6 possible sequences...

Rank #1: LDPE → PET → PP (Min Selectivity: 35.2%)
  Step 1: LDPE from {PET, PP} → toluene (42.5%)
  Step 2: PET from {PP} → DMF (35.2%)

Rank #2: PET → LDPE → PP (Min Selectivity: 32.8%)
  ...

[Flowchart visualization generated]
```

---

### 2.5 `view_alternative_separation_sequence(...)`

**When agent uses it**: User asks "Show 2nd best sequence" or "What if we start with PET?"

**Parameters**:
- `sequence_rank`: Which rank to view
- `starting_polymer`: Force specific starting polymer

---

### 2.6 `analyze_integrated_separation(...)`

**When agent uses it**: Multi-polymer separation analysis with optimal temperatures and all solvent properties

---

### 2.7 `analyze_polymer_dissolution(...)`

**When agent uses it**: Single polymer dissolution analysis with properties (BP, cost, safety, LogP)

---

### 2.8 `get_solubility_for_solvents(...)`

**When agent uses it**: Get solubility for SPECIFIC solvents by name

---

## 3. Solvent Property Tools

Integrate practical considerations: cost, toxicity, boiling point.

### 3.1 `list_solvent_properties()`

**When agent uses it**: User wants to see all solvents with properties

**Returns**: Table with BP, LogP, energy cost, heat capacity

---

### 3.2 `get_solvent_properties(solvent_names: str)`

**When agent uses it**: Get properties for specific solvents

**Agent reasoning**: "I found these solvents work. Now I'll get their practical properties."

**Often called in parallel** with separation tools.

---

### 3.3 `rank_solvents_by_property(property_name, ascending, limit, polymer_filter)`

**When agent uses it**: User asks "cheapest solvents" or "least toxic"

**Parameters**:
- `property_name`: `"energy"`, `"logp"`, or `"bp"`
- `ascending`: True for cheapest/safest first
- `polymer_filter`: Only solvents that dissolve this polymer

---

### 3.4 `analyze_separation_with_properties(...)`

**INTEGRATED TOOL** - Combines selectivity with practical properties

**When agent uses it**: "Find cheap solvents to separate X from Y"

**Returns**: Solvents ranked by property WITH selectivity data

---

## 4. Statistical Analysis Tools

### 4.1 `statistical_summary(table_name, column_name, group_by)`

**Returns**: Mean, median, std dev, 95% CI, outliers

---

### 4.2 `correlation_analysis(table_name, columns)`

**Returns**: Correlation matrix with p-values

---

### 4.3 `compare_groups_statistically(table_name, value_column, group_column, group1, group2)`

**Returns**: t-test or Mann-Whitney U results

---

### 4.4 `regression_analysis(table_name, x_column, y_column)`

**Returns**: Linear regression with R², residual plots

---

## 5. Visualization Tools

### 5.1 `plot_solubility_vs_temperature(...)`

**When agent uses it**: User wants static PNG plots

**Returns**: Temperature curves with confidence bands

---

### 5.2 `plot_solubility_vs_temperature_interactive(...)`

**When agent uses it**: User wants interactive exploration

**Returns**: HTML with:
- Range slider for temperature zoom
- Clickable legend to toggle curves
- Hover tooltips with exact values
- Zoom/pan/download tools

---

### 5.3 `plot_selectivity_heatmap(...)`

**When agent uses it**: User wants to see selectivity patterns

**Returns**: Color-coded heatmap (solvents × polymers)

---

### 5.4 `plot_multi_panel_analysis(...)`

**When agent uses it**: Comprehensive overview needed

**Returns**: 4-panel figure with solubility, selectivity, temperature, confidence

---

### 5.5 `plot_comparison_dashboard(...)`

**When agent uses it**: Side-by-side comparison of multiple scenarios

---

### 5.6 `plot_solvent_properties(...)`

**When agent uses it**: Visualize BP, LogP, energy for solvents

---

## 6. GSK Safety (G-Score) Tools

Industry-standard safety scoring from GlaxoSmithKline.

### 6.1 `get_solvent_gscore(solvent_names: str)`

**When agent uses it**: User asks about safety ratings

**Returns**:
```
| Solvent  | G-Score | Classification |
|----------|---------|----------------|
| water    | 1       | Recommended    |
| ethanol  | 2       | Recommended    |
| toluene  | 5       | Problematic    |
| benzene  | 10      | Hazardous      |
```

---

### 6.2 `get_family_alternatives(solvent_name: str)`

**When agent uses it**: User asks for safer alternatives

**Agent reasoning**: "User is using benzene. I should find safer alternatives in the same chemical family."

---

### 6.3 `visualize_gscores(solvent_names: str)`

**When agent uses it**: Visual safety comparison needed

**Returns**: Color-coded bar chart (green=safe, red=hazardous)

---

### 6.4 `plot_solvent_properties_for_polymer(...)`

**When agent uses it**: Multi-step analysis combining solubility with property scatter plots

---

## 7. Listing Tools

Quick overview tools that the agent often calls first.

### 7.1 `list_available_polymers()`

**When agent uses it**: User asks "What polymers are available?"

**Returns**:
```
**Common Solvents Database:** 15 unique polymers
- EVOH, HDPE, LDPE, LLDPE, Nylon6, Nylon66, PC, PES, PET, PMMA, PP, PS, PTFE, PVC, PVDF

**Hansen Parameters Database:** 466 polymers with HSP data
```

---

### 7.2 `list_available_solvents()`

**When agent uses it**: User asks "What solvents are available?"

**Returns**: Counts and examples from each database

---

## 8. ML Prediction Tools

### 8.1 `predict_solubility_ml(polymer: str, solvent: str)`

**When agent uses it**: User asks for ML/Hansen-based prediction

**Algorithm**: Random Forest (99.998% accuracy)

**Returns**:
```
ML Solubility Prediction

Polymer: HDPE
Solvent: Toluene

PREDICTION: ✅ SOLUBLE
Confidence: 97.5%

Hansen Parameters:
  HDPE:    δD=18.0, δP=0.0, δH=2.0 MPa^0.5
  Toluene: δD=18.0, δP=1.4, δH=2.0 MPa^0.5

RED Value: 0.24 (< 1.0 = soluble)

Visualizations Generated:
1. 3D Interactive Sphere (HTML)
2. Radar Plot (PNG)
3. RED Gauge (PNG)
4. HSP Comparison Bars (PNG)
```

**User favorite**: The 3D interactive sphere lets users rotate and explore the Hansen space.

---

## 9. PubChem External API Tools

Live data from PubChem's REST API with timeout protection.

### 9.1 `get_pubchem_safety_info(compound: str)`

**When agent uses it**: User asks for GHS hazard data

**External API**: `https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/`

**Timeout**: 15 seconds

**Returns**:
```
# PubChem Safety Data: Toluene

**GHS Hazard Pictograms:**
- Flammable (GHS02)
- Health Hazard (GHS08)
- Irritant (GHS07)

**Hazard Statements:**
- H225: Highly flammable liquid and vapor
- H304: May be fatal if swallowed and enters airways
- H315: Causes skin irritation

**Molecular Properties:**
| Property | Value |
|----------|-------|
| Molecular Weight | 92.14 g/mol |
| IUPAC Name | methylbenzene |
```

---

### 9.2 `compare_pubchem_safety(compounds: List[str])`

**When agent uses it**: User wants safety comparison across solvents

**Limit**: Maximum 5 compounds per query

**Returns**: Side-by-side comparison of GHS hazards with recommendations

**Agent reasoning**: "User asked to compare 7 solvents. I'll take the first 5 and warn them about the limit."

---

### 9.3 `visualize_pubchem_safety(compounds: List[str])`

**When agent uses it**: Visual safety chart needed

**Returns**: PNG chart showing hazard categories for each compound

---

### 9.4 `get_pubchem_toxicity(compounds: List[str])`

**When agent uses it**: User asks about LD50, biodegradation, aquatic toxicity

**Limit**: Maximum 5 compounds per query

**Timeout**: 20 seconds

**Returns**:
```
# PubChem Toxicity & Environmental Data

### Acetone

**Toxicity Data:**
- LD50 (oral, rat): 5,800 mg/kg
- LC50 (inhalation, rat): 76,000 mg/m³ (4h)

**Biodegradation:**
- Readily biodegradable under aerobic conditions
- BOD5/COD ratio: 0.42

**Aquatic Toxicity:**
- LC50 (fish, 96h): 8,300 mg/L
- EC50 (daphnia, 48h): 12,600 mg/L

**Assessment:** Low acute toxicity, readily biodegradable
```

---

## 10. Literature Search Tools

Search academic databases for peer-reviewed research.

### 10.1 `search_google_scholar(query, max_results, year_low, year_high)`

**When agent uses it**: User asks for academic papers, broad literature search

**External API**: SerpAPI (Google Scholar)

**Limit**: 100 searches/month (beta feature)

**Returns**:
```
# Google Scholar Results: polymer dissolution

**Found:** 10 articles

### 1. [Article Title](link)
**Authors:** Smith, J, Doe, A et al.
**Publication:** Journal of Polymer Science
**Year:** 2024
**Citations:** 45
*Abstract snippet...*
```

---

### 10.2 `search_web_of_science(query, polymer_name, solvent_name, year_low, year_high, max_results)`

**When agent uses it**: User wants peer-reviewed articles with citation metrics

**External API**: Clarivate Web of Science Starter API

**Returns**:
```
# Web of Science Results: polymer solubility

**Found:** 10 peer-reviewed articles

### 1. [Article Title](link)
**Authors:** Meyer, KH, van der Wyk, A
**Journal:** HELVETICA CHIMICA ACTA
**Year:** 2024
**Volume/Pages:** Vol. 107, pp. 123-130
**DOI:** 10.1000/example
**Times Cited:** 39
```

**Key difference from Google Scholar**: Returns peer-reviewed journal articles only, with citation counts and DOI links.

---

## 11. TEA/LCA Analysis Tools

Techno-Economic Analysis and Life Cycle Assessment for process evaluation.

### 11.1 `analyze_solvent_recovery_tea(solvent, polymer_throughput_kg_hr, ...)`

**When agent uses it**: User asks about capital costs, operating costs, payback period

**Returns**:
```
# Techno-Economic Analysis Results

## Capital Costs
| Equipment | Cost (USD) |
|-----------|------------|
| Distillation column | $1,352,175 |
| Heat exchanger | $606,866 |
| **Fixed Capital Investment** | **$2,741,392** |

## Economic Metrics
- Cost per kg polymer: $1.43/kg
- Simple payback: 0.52 years
```

---

### 11.2 `analyze_solvent_recovery_lca(solvent, polymer_throughput_kg_hr, ...)`

**When agent uses it**: User asks about carbon footprint, emissions, environmental impact

**Returns**:
```
# Life Cycle Assessment Results

## Annual Greenhouse Gas Emissions
| Source | Emissions (kg CO2eq/yr) |
|--------|-------------------------|
| Electricity | 107,027 |
| Steam | 149,838 |
| Solvent makeup | 475,200 |
| **Total** | **732,065** |

## Comparison to Baseline
- Emission reduction: 92.3%
```

---

### 11.3 `compare_solvents_tea_lca(solvents: List[str], ...)`

**When agent uses it**: User wants to compare multiple solvents on cost AND environmental metrics

**Returns**:
```
# Solvent Comparison: TEA & LCA

| Solvent | FCI ($) | $/kg | CO2 (t/yr) | Rank |
|---------|---------|------|------------|------|
| toluene | 2.7M | 1.43 | 732 | 1 |
| ethanol | 2.7M | 1.42 | 1126 | 2 |

## Rankings
- Best Overall: toluene
- Lowest Cost: ethanol
- Lowest Emissions: toluene
```

---

## 12. TEA/LCA Visualization Tools

Generate charts and graphs for economic and environmental analysis.

### 12.1 `generate_tea_visualizations(tea_results)`

**Returns**: Cost breakdown pie charts, equipment cost bars

---

### 12.2 `generate_lca_visualizations(lca_results)`

**Returns**: Emissions breakdown, comparison to baseline

---

### 12.3 `generate_solvent_comparison_visualization(comparison_results)`

**Returns**: Multi-solvent comparison charts

---

### 12.4 `plot_tea_sensitivity_tornado(tea_results, parameters)`

**When agent uses it**: User wants to see which parameters most affect costs

**Returns**: Tornado chart showing sensitivity analysis

---

### 12.5 `plot_tea_cashflow(tea_results, years)`

**When agent uses it**: User wants to see projected cash flows

**Returns**: Cash flow diagram over project lifetime

---

## 13. STRAP Process Tools

Solvent-Targeted Recovery and Precipitation process analysis.

### 13.1 `analyze_strap_process(polymers, solvent, ...)`

**When agent uses it**: Full STRAP process analysis for polymer mixture separation

**Returns**: Complete process analysis with mass balances, energy requirements

---

### 13.2 `calculate_strap_msp(process_results)`

**When agent uses it**: Calculate Minimum Selling Price for recovered polymers

**Returns**: MSP breakdown with cost components

---

### 13.3 `plot_strap_scale_economics(process_results, scales)`

**When agent uses it**: User wants to see how costs change with scale

**Returns**: Scale vs. cost curves showing economies of scale

---

### 13.4 `compare_strap_scenarios(scenarios: List[Dict])`

**When agent uses it**: Compare different process configurations

**Returns**: Side-by-side scenario comparison

---

### 13.5 `generate_strap_visualizations(process_results)`

**When agent uses it**: Generate all STRAP-related visualizations

**Returns**: Process flow diagram, mass balance Sankey, cost breakdowns

---

## Query Examples

### Simple Lookups (1-2 iterations)
```
"What tables are available?"
"List all polymers"
"What are the properties of toluene?"
```

### Separation Analysis (2-3 iterations)
```
"Find solvents to separate LDPE from PET at 25°C"
"Separate HDPE from PP, PVC, PS"
```

### Multi-Polymer Sequences (2-4 iterations)
```
"Plan sequential separation for LDPE, PET, PP at 120°C"
"What are all possible sequences to separate EVOH, LDPE, PET?"
```

### Property-Integrated Analysis (2-3 iterations)
```
"Find cheapest solvents to separate LDPE from PET"
"Rank by LogP, least toxic first"
```

### Safety Analysis (2-3 iterations)
```
"Compare safety of benzene, toluene, and xylene using PubChem"
"What's the LD50 of acetone? Is it biodegradable?"
"Find safer alternatives to dichloromethane"
```

### ML Predictions (1-2 iterations)
```
"Predict solubility of HDPE in toluene using machine learning"
"Will Nylon6 dissolve in DMF?"
```

### Literature Search (1-2 iterations)
```
"Search Web of Science for polymer dissolution mechanisms"
"Find Google Scholar articles on Hansen solubility parameters"
"What are recent publications on selective PET separation?"
```

### TEA/LCA Analysis (2-3 iterations)
```
"Run TEA for toluene recovery at 100 kg/hr"
"What's the carbon footprint of using DMF?"
"Compare toluene, acetone, and ethanol on cost and emissions"
```

### STRAP Process Analysis (2-4 iterations)
```
"Analyze STRAP process for LDPE/PET separation"
"Calculate minimum selling price for recovered HDPE"
"How do costs change with scale for STRAP process?"
```

### Complex Multi-Factor Analysis (3-6 iterations)
```
"Analyze LDPE/EVOH/PET separation at 120°C with cost, safety, and ML validation"
```

---

## Best Practices for Queries

### 1. Be Specific About Polymers
```
Good: "LDPE" (exact name)
Bad: "polyethylene" (ambiguous)
```

### 2. Specify Temperature When Relevant
```
Good: "at 120°C"
Default: 25°C
```

### 3. Let the Agent Adapt
```
Good: "Find optimal separation conditions"
Bad: "Find solvents with exactly 50% selectivity" (might find nothing)
```

### 4. Request Visualizations
```
"Create interactive temperature plot"
"Show selectivity heatmap"
"Generate ML 3D sphere"
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| "No data found" | Typo in polymer/solvent name | Use `list_available_polymers()` |
| "No separation found" | Too stringent threshold | Agent will auto-relax |
| "PubChem timeout" | API slow | Retry or try fewer compounds |
| "Tool output too long" | Too many rows | Use `limit` parameter |
| "Literature search limited" | API quotas | Use WoS for more results |

---

## Summary

The DISSOLVE Agent has **54 tools** organized into 13 categories:

| Category | Tools | Key Capability |
|----------|-------|----------------|
| Core Database | 6 | Query, validate, explore |
| Adaptive Analysis | 8 | Intelligent separation with auto-relaxing thresholds |
| Solvent Properties | 4 | Cost, toxicity, BP integration |
| Statistical | 4 | Rigorous statistical analysis |
| Visualization | 6 | Static PNG + interactive HTML |
| GSK Safety | 4 | Industrial safety scoring |
| Listing | 2 | Quick overviews |
| ML Prediction | 1 | Hansen-based prediction with visualizations |
| PubChem External | 4 | Live GHS/toxicity data |
| Literature Search | 2 | Web of Science + Google Scholar |
| TEA/LCA Analysis | 3 | Cost and environmental analysis |
| TEA/LCA Visualization | 5 | Economic and LCA charts |
| STRAP Process | 5 | Process-specific analysis |

The agent autonomously selects tools based on user queries, executes them in parallel when possible, and iterates until it has enough information to provide a comprehensive answer.
