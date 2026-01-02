# DISSOLVE Agent - Complete Tools Reference

This document provides a comprehensive guide to all **34 tools** available in the DISSOLVE Agent system, along with all visualizations that can be generated.

---

## Table of Contents

1. [Core Database Tools (6 tools)](#1-core-database-tools)
2. [Adaptive Analysis Tools (5 tools)](#2-adaptive-analysis-tools)
3. [Solvent Property Tools (4 tools)](#3-solvent-property-tools)
4. [Statistical Analysis Tools (4 tools)](#4-statistical-analysis-tools)
5. [Visualization Tools (6 tools)](#5-visualization-tools)
6. [GSK Safety (G-Score) Tools (3 tools)](#6-gsk-safety-g-score-tools)
7. [Listing Tools (2 tools)](#7-listing-tools)
8. [ML Prediction Tools (1 tool)](#8-ml-prediction-tools)
9. [Visualization Gallery](#visualization-gallery)
10. [Query Examples](#query-examples)

---

## 1. Core Database Tools

### 1.1 `list_tables()`

**Description**: List all available SQL tables with schemas, row counts, and data quality info.

**Use when**: Starting a new analysis or exploring the database.

**Returns**:
- Table names
- Row counts
- Column names and types
- Data quality metrics

**Example query**:
> "What tables are available?"

**Example output**:
```
Available Tables:

1. common_solvents_database (10,613 rows)
   - solvent (VARCHAR)
   - temperature___c_ (DOUBLE)
   - solubility____ (DOUBLE)
   - polymer (VARCHAR)

2. solvent_data (1,007 rows)
   - s_n (BIGINT)
   - solvent_name (VARCHAR)
   - bp__oc_ (DOUBLE)
   - logp (DOUBLE)
   - energy__j_g_ (DOUBLE)
   ...
```

---

### 1.2 `describe_table(table_name: str)`

**Description**: Get detailed information about a specific table including sample data and statistics.

**Parameters**:
- `table_name`: Name of the table to describe

**Use when**: You need detailed stats for a specific table.

**Returns**:
- Column statistics (min, max, avg for numeric columns)
- Unique value counts for categorical columns
- Sample data (5 rows)

**Example query**:
> "Describe the solubility data table"

**Example output**:
```
Table: common_solvents_database

Rows: 10,613

Columns:
  - solvent: VARCHAR [896 unique values]
  - temperature___c_: DOUBLE [min=25.0, max=150.0, avg=78.5]
  - solubility____: DOUBLE [min=0.0, max=100.0, avg=42.3]
  - polymer: VARCHAR [15 unique values]

Sample data (5 rows):
| solvent  | temperature___c_ | solubility____ | polymer |
|----------|------------------|----------------|---------|
| toluene  | 25               | 95.2           | LDPE    |
| ...      | ...              | ...            | ...     |
```

---

### 1.3 `check_column_values(table_name: str, column_name: str, limit: int = 50)`

**Description**: Check what values exist in a specific column with frequency counts.

**Parameters**:
- `table_name`: Name of the table
- `column_name`: Name of the column
- `limit`: Max number of unique values to return (default: 50)

**Use when**: Verifying exact spelling of values (e.g., polymer names, solvent names).

**Returns**: List of unique values with counts

**Example query**:
> "What polymers are in the database?"

**Example output**:
```
Unique values in common_solvents_database.polymer:

| polymer  | count |
|----------|-------|
| LDPE     | 896   |
| HDPE     | 896   |
| PET      | 896   |
| PP       | 896   |
| ...      | ...   |

Total unique values: 15
Total rows in table: 10,613
```

---

### 1.4 `query_database(sql_query: str, export_csv: bool = False)`

**Description**: Execute a SQL query with enhanced validation and error reporting.

**Parameters**:
- `sql_query`: SQL query to execute
- `export_csv`: If True, creates a CSV export of the results

**Use when**: You need custom queries beyond what other tools provide.

**Returns**: Query results as formatted table (up to 10 rows preview)

**Example query**:
> "Query: SELECT * FROM common_solvents_database WHERE polymer='LDPE' AND temperature___c_=25 LIMIT 5"

**Example output**:
```
Query Results

Query: `SELECT * FROM common_solvents_database WHERE polymer='LDPE' AND temperature___c_=25 LIMIT 5`

Rows returned: 5

Data:
| solvent     | temperature___c_ | solubility____ | polymer |
|-------------|------------------|----------------|---------|
| toluene     | 25               | 95.2           | LDPE    |
| chloroform  | 25               | 92.8           | LDPE    |
| ...         | ...              | ...            | ...     |
```

---

### 1.5 `verify_data_accuracy(table_name: str, filters: Optional[str] = None)`

**Description**: Verify data accuracy by checking actual row counts and sample data.

**Parameters**:
- `table_name`: Name of the table
- `filters`: SQL WHERE clause filters (optional)

**Use when**: Double-checking that data exists before reporting results.

**Returns**: Row count and sample data for verification

**Example query**:
> "Verify data for LDPE at 25°C"

**Example output**:
```
Data Verification for common_solvents_database

Filter: polymer='LDPE' AND temperature___c_=25
Total matching rows: 896

Sample data:
| solvent  | temperature___c_ | solubility____ | polymer |
|----------|------------------|----------------|---------|
| ...      | ...              | ...            | ...     |
```

---

### 1.6 `validate_and_query(table_name, required_columns, filter_column, filter_values, sql_query)`

**Description**: Validate inputs BEFORE executing a query to prevent hallucinations.

**Parameters**:
- `table_name`: Name of the table
- `required_columns`: Comma-separated list of columns to validate
- `filter_column`: Column to filter on (optional)
- `filter_values`: Comma-separated values to validate (optional)
- `sql_query`: Query to execute if validation passes (optional)

**Use when**: You want to ensure all inputs are valid before running expensive queries.

**Returns**: Validation report with  (OK)/ (ERROR) for each input

**Example output**:
```
Input Validation Report

Table 'common_solvents_database' exists (10,613 rows)
Column 'polymer' exists
Column 'solvent' exists
Value 'LDPE' found in polymer (896 rows)
Value 'toluene' found in solvent (128 rows)

Query Execution:
Query successful: 15 rows returned
```

---

## 2. Adaptive Analysis Tools

These tools use **intelligent threshold adaptation** to find optimal separation conditions even with sparse data.

### 2.1 `find_optimal_separation_conditions(...)`

**PRIMARY TOOL for pairwise polymer separation**

**Description**: Find optimal conditions to separate a target polymer from comparison polymers.

**Parameters**:
- `table_name`: `"common_solvents_database"`
- `polymer_column`: `"polymer"`
- `solvent_column`: `"solvent"`
- `temperature_column`: `"temperature___c_"`
- `solubility_column`: `"solubility____"`
- `target_polymer`: Polymer you want to dissolve (e.g., `"LDPE"`)
- `comparison_polymers`: Polymers to separate from (e.g., `"PET,PP"`)
- `start_temperature`: Starting temperature (default: 25.0°C)
- `initial_selectivity`: Initial selectivity threshold (default: 30.0%)
- `export_csv`: Create CSV export (default: False)

**Use when**: User asks "Find solvents to separate X from Y"

**Returns**:
- Optimal solvent and temperature
- Selectivity percentage
- Target and competitor solubilities
- Confidence score
- Alternative conditions
- Recommendations

**Example query**:
> "Find solvents to separate LDPE from PET at 25°C"

**Example output**:
```
Adaptive Separation Analysis

Target: Dissolve LDPE
Separate from: PET
Starting conditions: T=25°C, selectivity threshold=30%

Separation IS FEASIBLE

Optimal Conditions:
  - Temperature: 25°C
  - Solvent: toluene
  - Selectivity: 42.5%
  - Target solubility: 95.2%
  - Max other solubility: 52.7%
  - Confidence: 92%

Alternative Conditions:
  1. T=30°C, chloroform (selectivity=38.2%)
  2. T=25°C, xylene (selectivity=35.8%)
  3. T=40°C, benzene (selectivity=33.1%)

Recommendations:
  - Toluene shows excellent selectivity (42.5%) at 25°C
  - Consider chloroform for higher temperature tolerance
  - Monitor temperature carefully for optimal separation
```

---

### 2.2 `adaptive_threshold_search(...)`

**Description**: Search for selective solvents using adaptive thresholds (starts high, relaxes if needed).

**Parameters**:
- `table_name`, `polymer_column`, `solvent_column`, `temperature_column`, `solubility_column`: Same as above
- `target_polymer`: Target polymer
- `comparison_polymers`: Comma-separated competitors (optional - uses all if not specified)
- `temperature`: Temperature to search at (default: 25.0°C)
- `start_threshold`: Starting selectivity threshold (default: 0.5 = 50%)

**Use when**: You want to see how selectivity thresholds affect results.

**Returns**:
- Search path (which thresholds were tried)
- Results at final threshold
- Recommendations

**Example query**:
> "Search for selective solvents for LDPE at 25°C"

**Example output**:
```
Adaptive Threshold Search

Target: LDPE
Comparing against: PET, PP, PS, ...
Temperature: 25°C
Starting threshold: 0.5

Search Path: [0.5 -> 0.4 -> 0.3  (OK)]

Thresholds tried: 3

Found 12 selective solvent(s) at threshold 0.3

Results:
  1. toluene
     Selectivity: 0.425
     LDPE solubility: 0.952
     Max other solubility: 0.527
  2. xylene
     Selectivity: 0.358
     ...
```

---

### 2.3 `analyze_selective_solubility_enhanced(...)`

**Description**: Detailed selectivity analysis with temperature scanning and ranking.

**Parameters**: Similar to above, plus:
- `top_k`: Number of top solvents to return (default: 10)

**Use when**: You want comprehensive selectivity data with multiple solvents ranked.

**Returns**:
- Top-k solvents ranked by selectivity
- Selectivity values and solubilities
- Temperature recommendations

---

### 2.4 `plan_sequential_separation(...)`

**PRIMARY TOOL for multi-polymer separation sequences**

**Description**: Enumerate ALL possible separation sequences for multiple polymers and find the optimal sequence.

**Parameters**:
- `table_name`, `polymer_column`, `solvent_column`, `temperature_column`, `solubility_column`: Same as above
- `polymers`: Comma-separated list of polymers to separate (e.g., `"LDPE,PET,PP"`)
- `temperature`: Temperature to use (default: 25.0°C)
- `top_k_solvents`: Top solvents to consider per step (default: 5)
- `create_decision_tree`: Create decision tree visualization (default: True)

**Use when**: User asks:
- "What are all possible sequences to separate X, Y, Z?"
- "How can I separate A, B, C, D?"
- "Enumerate separation strategies"

**Returns**:
- **Complete enumeration** of all permutations (e.g., 6 sequences for 3 polymers)
- Top sequence with detailed visualization
- Min selectivity for each sequence
- Step-by-step solvent recommendations
- Visual flowchart (vertical, easy to read)
- Decision tree showing all paths

**Example query**:
> "Plan sequential separation for LDPE, PET, PP at 120°C"

**Example output**:
```
Sequential Separation Analysis

Polymers: LDPE, PET, PP
Temperature: 120°C
Top solvents per step: 5

Analyzing all 6 possible sequences...

SEQUENCE RANKINGS (by minimum selectivity):

Rank #1: LDPE -> PET -> PP (Min Selectivity: 35.2%)
  Step 1: LDPE from {PET, PP} -> toluene (42.5%)
  Step 2: PET from {PP} -> DMF (35.2%)
  Overall: BEST sequence

Rank #2: PET -> LDPE -> PP (Min Selectivity: 32.8%)
  ...

Rank #3-6: (lower selectivity sequences)

Visual flowchart created: /plots/sequential_sep_flowchart_20260102_123456.png

Decision tree created: /plots/decision_tree_20260102_123456.png

Recommendations:
  - Use Rank #1 sequence for maximum selectivity
  - Toluene is optimal for Step 1
  - Monitor temperature at 120°C ±5°C
```

**Visualization**: Creates beautiful vertical flowchart showing:
- Starting mixture at top
- Each step with solvent, temperature, selectivity
- Separated polymer + remaining mixture
- Color-coded by selectivity (green=good, yellow=OK, red=poor)

---

### 2.5 `view_alternative_separation_sequence(...)`

**Description**: View alternative separation sequences (2nd best, 3rd best) or sequences starting with a specific polymer.

**Parameters**:
- Same as `plan_sequential_separation`, plus:
- `sequence_rank`: Which rank to view (e.g., `2` for 2nd best)
- `starting_polymer`: Force sequence to start with this polymer (e.g., `"PET"`)

**Use when**: User asks:
- "Show me the 2nd best sequence"
- "What if we start with PET instead?"
- "Show alternatives to the top sequence"

**Returns**: Detailed view of the requested sequence with visualization

**Example query**:
> "Show me PET-first separation sequence"

**Example output**:
```
Alternative Separation Sequence

Sequence: PET -> LDPE -> PP (Rank #2)
Min Selectivity: 32.8%

Step 1: Separate PET from {LDPE, PP}
  Solvent: DMF
  Temperature: 120°C
  Selectivity: 38.5%
  ...

Flowchart created: /plots/alt_sequence_PET_first_20260102_123456.png
```

---

## 3. Solvent Property Tools

Combine separation analysis with practical considerations (cost, toxicity, boiling point).

### 3.1 `list_solvent_properties()`

**Description**: View all solvents with their properties (BP, LogP, Energy, Cp).

**Use when**: User wants to see all available solvents with properties.

**Returns**: Table of solvents with BP, LogP, energy cost, heat capacity

**Example query**:
> "List all solvent properties"

---

### 3.2 `get_solvent_properties(solvent_names: str)`

**Description**: Get properties for specific solvents.

**Parameters**:
- `solvent_names`: Comma-separated list of solvents

**Use when**: User asks "What are the properties of toluene and xylene?"

**Returns**: Table with BP, LogP, energy, Cp for each solvent

**Example output**:
```
Solvent Properties

| Solvent  | BP (°C) | LogP  | Energy (J/g) | Cp (J/g·K) |
|----------|---------|-------|--------------|------------|
| toluene  | 110.6   | 2.73  | 450.2        | 1.72       |
| xylene   | 144.4   | 3.15  | 523.8        | 1.71       |
```

---

### 3.3 `rank_solvents_by_property(...)`

**Description**: Rank solvents by cost (energy), toxicity (LogP), or boiling point.

**Parameters**:
- `property_name`: `"energy"`, `"logp"`, or `"bp"`
- `ascending`: `True` for cheapest/least toxic first, `False` for most expensive/most toxic
- `limit`: Number of solvents to return (default: 20)
- `polymer_filter`: Only show solvents that dissolve this polymer (optional)

**Use when**: User asks:
- "Rank solvents by cost"
- "Find cheapest solvents"
- "Least toxic solvents"

**Returns**: Ranked list with property values

**Example query**:
> "Rank solvents by energy cost for LDPE, cheapest first"

**Example output**:
```
Solvents Ranked by: energy (ascending)

| Rank | Solvent    | Energy (J/g) | BP (°C) | LogP |
|------|------------|--------------|---------|------|
| 1    | acetone    | 285.3        | 56.2    | -0.24|
| 2    | methanol   | 312.8        | 64.7    | -0.77|
| 3    | ethanol    | 348.5        | 78.4    | -0.31|
| ...  | ...        | ...          | ...     | ...  |
```

---

### 3.4 `analyze_separation_with_properties(...)`

**Description**: **INTEGRATED TOOL** - Combine selectivity with cost/toxicity/boiling point for practical recommendations.

**Parameters**:
- All parameters from `find_optimal_separation_conditions`, plus:
- `rank_by`: `"energy"` (cost), `"logp"` (toxicity), or `"bp"` (boiling point)
- `ascending`: `True` for cheaper/safer/lower BP first

**Use when**: User asks:
- "Find cheapest solvents to separate X from Y"
- "Separation with cost considerations"
- "Least toxic solvents for separation"

**Returns**:
- Top solvents ranked by chosen property
- Selectivity, solubility, AND property values
- Integrated recommendations

**Example query**:
> "Find cheap solvents to separate LDPE from PET, ranked by cost"

**Example output**:
```
Separation Analysis with Properties

Target: LDPE
Separate from: PET
Ranking by: energy (ascending)

Top Solvents:

1. acetone
   Selectivity: 28.5%
   Target solubility: 82.3%
   Energy cost: 285.3 J/g CHEAPEST
   BP: 56.2°C
   LogP: -0.24

2. toluene
   Selectivity: 42.5% BEST SELECTIVITY
   Target solubility: 95.2%
   Energy cost: 450.2 J/g
   BP: 110.6°C
   LogP: 2.73

Recommendations:
  - Acetone: Cheapest but lower selectivity (28.5%)
  - Toluene: Higher cost but excellent selectivity (42.5%)
  - Trade-off: Cost vs separation quality
```

---

## 4. Statistical Analysis Tools

### 4.1 `statistical_summary(...)`

**Description**: Comprehensive statistics with confidence intervals, outlier detection, and normality tests.

**Parameters**:
- `table_name`, `column_name`: Table and column to analyze
- `group_by`: Group by this column (optional)

**Returns**: Mean, median, std dev, min, max, 95% CI, outliers

---

### 4.2 `correlation_analysis(...)`

**Description**: Multi-column correlation analysis with significance testing.

**Parameters**:
- `table_name`: Table name
- `columns`: Comma-separated list of columns

**Returns**: Correlation matrix with p-values

---

### 4.3 `compare_groups_statistically(...)`

**Description**: Hypothesis testing to compare two groups (t-test or Mann-Whitney U).

**Parameters**:
- `table_name`, `value_column`, `group_column`: Columns to compare
- `group1`, `group2`: Groups to compare

**Returns**: Statistical test results, p-value, effect size

---

### 4.4 `regression_analysis(...)`

**Description**: Linear regression with diagnostics and prediction.

**Parameters**:
- `table_name`, `x_column`, `y_column`: Columns for regression

**Returns**: Slope, intercept, R², residual plots

---

## 5. Visualization Tools

### 5.1 `plot_solubility_vs_temperature(...)`

**Description**: Create temperature vs solubility curves with confidence bands (static PNG).

**Parameters**:
- `table_name`, `polymer_column`, `solvent_column`, `temperature_column`, `solubility_column`: Standard params
- `polymers`: Comma-separated polymers to plot
- `solvents`: Comma-separated solvents to plot
- `plot_title`: Custom title (optional)
- `temperature_min`, `temperature_max`: Filter temperature range (optional)

**Use when**: User wants static plots for reports/papers.

**Returns**: PNG image with:
- Temperature curves for each polymer-solvent pair
- Confidence bands (shaded regions)
- Legend
- Grid

**Example query**:
> "Plot solubility vs temperature for LDPE in toluene and xylene"

**Visualization**:
- X-axis: Temperature (°C)
- Y-axis: Solubility (%)
- Lines: Different colors for each polymer-solvent pair
- Shaded regions: 95% confidence intervals

---

### 5.2 `plot_solubility_vs_temperature_interactive(...)`

**Description**: Create **INTERACTIVE** temperature vs solubility curves with sliders and toggleable lines (HTML).

**Parameters**: Same as static version above

**Use when**: User wants to explore data interactively.

**Returns**: Interactive HTML with:
- **Range slider** to zoom into temperature ranges
- **Clickable legend** to show/hide curves
- **Hover tooltips** with exact values
- Zoom, pan, reset tools
- Download as PNG capability

**Example query**:
> "Create interactive temperature plot for LDPE in various solvents"

**Features**:
- Drag the range slider below the plot to zoom into specific temperature ranges
- Click legend items to toggle curves on/off
- Hover over lines to see exact temperature and solubility values
- Use toolbar to zoom, pan, or download as image

---

### 5.3 `plot_selectivity_heatmap(...)`

**Description**: Create selectivity heatmap showing which solvents separate which polymers.

**Parameters**:
- `table_name`, `polymer_column`, `solvent_column`, `temperature_column`, `solubility_column`: Standard params
- `temperature`: Temperature to analyze (default: 25°C)
- `target_polymer`: Highlight best solvents for this polymer (optional)
- `plot_title`: Custom title (optional)

**Use when**: User wants to see selectivity patterns across many solvents/polymers.

**Returns**: Heatmap with:
- Rows: Polymers
- Columns: Solvents
- Colors: Selectivity (green=high, red=low)
- Annotations: Selectivity values

**Example query**:
> "Create selectivity heatmap at 120°C"

**Visualization**:
- Easy to spot "hot spots" (good separations)
- Color scale shows selectivity magnitude
- Can highlight target polymer row

---

### 5.4 `plot_multi_panel_analysis(...)`

**Description**: Comprehensive 4-panel separation analysis.

**Parameters**: Same as separation tools

**Returns**: 4-panel figure with:
1. **Solubility comparison** (bar chart)
2. **Selectivity ranking** (horizontal bars)
3. **Temperature dependence** (line plot)
4. **Confidence heatmap** (data quality visualization)

**Use when**: User wants comprehensive overview of separation options.

---

### 5.5 `plot_comparison_dashboard(...)`

**Description**: Multi-polymer comparison dashboard with solubility distributions.

**Parameters**:
- `polymers`: Comma-separated polymers
- `temperature`: Temperature to analyze

**Returns**: Dashboard with:
- Solubility distributions for each polymer
- Box plots
- Violin plots
- Summary statistics

**Use when**: Comparing multiple polymers side-by-side.

---

### 5.6 `plot_solvent_properties(...)`

**Description**: Plot BP, LogP, energy, or Cp for solvents (bar or scatter plots).

**Parameters**:
- `solvent_names`: Comma-separated solvents (optional - uses all if not specified)
- `property_to_plot`: `"bp"`, `"logp"`, `"energy"`, or `"cp"`
- `plot_type`: `"bar"` or `"scatter"` (default: `"bar"`)
- `polymer_filter`: Only show solvents that dissolve this polymer (optional)

**Use when**: User asks to visualize solvent properties.

**Returns**: Bar or scatter plot with property values

**Example query**:
> "Plot boiling points for solvents that dissolve LDPE"

---

## 6. GSK Safety (G-Score) Tools

### 6.1 `get_solvent_gscore(solvent_names: str)`

**Description**: Get GSK safety G-scores for specific solvents.

**Parameters**:
- `solvent_names`: Comma-separated solvents

**Returns**: G-scores (1-10, lower=safer), classifications

**Example output**:
```
GSK Safety Scores

| Solvent  | G-Score | Classification |
|----------|---------|----------------|
| water    | 1       | Recommended    |
| ethanol  | 2       | Recommended    |
| toluene  | 5       | Problematic    |
| benzene  | 10      | Hazardous      |
```

---

### 6.2 `get_family_alternatives(solvent_name: str)`

**Description**: Find safer alternatives within the same chemical family.

**Parameters**:
- `solvent_name`: Solvent to find alternatives for

**Returns**: List of alternatives with G-scores

**Example query**:
> "Find safer alternatives to benzene"

---

### 6.3 `visualize_gscores(...)`

**Description**: Visualize G-scores for multiple solvents with color-coded safety.

**Parameters**:
- `solvent_names`: Comma-separated solvents

**Returns**: Bar chart with color-coded safety levels

---

## 7. Listing Tools

### 7.1 `list_available_polymers()`

**Description**: **QUICK SUMMARY** of all polymers across all databases.

**Use when**: User asks "List all polymers"

**Returns**:
- Count of polymers in each database
- Example polymers
- Total unique polymers

**Example output**:
```
**Available Polymers Summary**

**Common Solvents Database:** 15 unique polymers

**Example Polymers:**
- EVOH
- HDPE
- LDPE
- LLDPE
- Nylon6
- Nylon66
- PC
- PES
- PET
- PMMA
- PP
- PS
- PTFE
- PVC
- PVDF

**Hansen Parameters Database:** 466 polymers with HSP data

**Usage:** Use exact names in queries (case-sensitive!)
```

---

### 7.2 `list_available_solvents()`

**Description**: **QUICK SUMMARY** of all solvents across all databases.

**Use when**: User asks "List all solvents"

**Returns**:
- Count of solvents in each database
- Example solvents
- Cross-database availability

**Example output**:
```
**Available Solvents Summary**

**Common Solvents Database:** 896 unique solvents

**Example Solvents:**
- toluene
- xylene
- acetone
- methanol
- ethanol
- DMF
- DMSO
- ...

**Solvent Properties Database:** 1,007 solvents

**GSK Safety Database:** 154 solvents with G-scores

**Cross-database:** ~800 solvents with both solubility AND property data
```

---

## 8. ML Prediction Tools

### 8.1 `predict_solubility_ml(polymer: str, solvent: str)`

**Description**: **MACHINE LEARNING PREDICTION** using Hansen Solubility Parameters.

**Algorithm**: Random Forest (99.998% accuracy)

**Parameters**:
- `polymer`: Polymer name
- `solvent`: Solvent name

**Use when**: User asks:
- "Predict solubility of X in Y using ML"
- "Will HDPE dissolve in toluene?"
- "Machine learning prediction for X and Y"

**Returns**:
- **Prediction**: Soluble or Not Soluble
- **Confidence**: 0-100%
- **Hansen Parameters** for both polymer and solvent
- **RED Value**: Relative Energy Difference
- **5 Visualizations** (see below)

**Example query**:
> "Predict solubility of HDPE in toluene using machine learning"

**Example output**:
```
ML Solubility Prediction

Polymer: HDPE
Solvent: Toluene

PREDICTION: SOLUBLE
Confidence: 97.5%

Hansen Parameters:
  HDPE:    δD=18.0, δP=0.0, δH=2.0 MPa^0.5
  Toluene: δD=18.0, δP=1.4, δH=2.0 MPa^0.5

RED Value: 0.24 (< 1.0 = soluble)

Visualizations Generated:
1. 3D Interactive Sphere: /plots/ml_3d_sphere_HDPE_Toluene.html
2. Radar Plot: /plots/ml_radar_HDPE_Toluene.png
3. RED Gauge: /plots/ml_gauge_HDPE_Toluene.png
4. HSP Comparison: /plots/ml_bars_HDPE_Toluene.png
5. Text Summary: /plots/ml_summary_HDPE_Toluene.txt

[Click here to view 3D interactive sphere](/plots/ml_3d_sphere_HDPE_Toluene.html)
```

**Visualizations**:

1. **3D Interactive Sphere** (HTML) - **User's favorite!**
   - Interactive 3D plot you can rotate
   - Polymer sphere (blue)
   - Solvent point (red)
   - Distance = RED value
   - Hover for details

2. **Radar Plot** (PNG)
   - 3-axis radar showing δD, δP, δH
   - Overlapping areas = good match
   - Visual similarity assessment

3. **RED Gauge** (PNG)
   - Speedometer-style gauge
   - Shows RED value on 0-2 scale
   - Color-coded zones (green=soluble, red=not)

4. **HSP Comparison Bars** (PNG)
   - Side-by-side bar chart
   - Compares all 3 parameters
   - Easy visual comparison

5. **Text Summary** (TXT)
   - Detailed prediction report
   - Hansen parameters
   - RED calculation
   - Confidence metrics

---

## Visualization Gallery

### Temperature Curves (Static)
![Temperature Plot](example_temp_plot.png)
- Multiple polymer-solvent pairs
- Confidence bands
- Clear legend and labels

### Temperature Curves (Interactive)
![Interactive Plot](example_interactive.png)
- Range slider for zooming
- Clickable legend
- Hover tooltips
- Zoom/pan/reset tools

### Selectivity Heatmap
![Heatmap](example_heatmap.png)
- Color-coded selectivity
- Annotated values
- Easy to spot patterns

### Sequential Separation Flowchart
![Flowchart](example_flowchart.png)
- Vertical layout (top to bottom)
- Each step clearly labeled
- Color-coded by selectivity
- Solvent boxes with details

### Decision Tree
![Decision Tree](example_tree.png)
- All sequences enumerated
- Branch paths shown
- Selectivity at each split

### ML Visualizations
![ML Viz](example_ml.png)
- 3D sphere (interactive)
- Radar plot
- RED gauge
- HSP comparison bars

---

## Query Examples

### Basic Queries
```
"What tables are available?"
"Describe the solubility table"
"List all polymers"
"List all solvents"
"What are the properties of toluene?"
```

### Separation Analysis
```
"Find solvents to separate LDPE from PET at 25°C"
"Separate HDPE from PP, PVC, PS"
"What are all possible sequences to separate LDPE, PET, PP?"
"Plan sequential separation for EVOH, LDPE, PET at 120°C"
"Show me the 2nd best sequence"
"What if we start with PET instead?"
```

### Property-Based Queries
```
"Rank solvents by cost for LDPE"
"Find cheapest solvents to separate LDPE from PET"
"Least toxic solvents"
"What are the properties of DMF and DMSO?"
"Rank by LogP, least toxic first"
```

### Visualization Queries
```
"Plot solubility vs temperature for LDPE in various solvents"
"Create interactive temperature plot"
"Create selectivity heatmap at 120°C"
"Plot boiling points for LDPE solvents"
"Show comparison dashboard for PP, PET, LDPE"
```

### Safety Queries
```
"What is the G-score for benzene?"
"Find safer alternatives to toluene"
"Rank solvents by safety for EVOH separation"
```

### ML Queries
```
"Predict solubility of HDPE in toluene using machine learning"
"Will Nylon6 dissolve in DMF?"
"ML prediction for PET and chloroform"
"Hansen parameters for PP and acetone"
```

### Complex Multi-Step Queries
```
"Find solvents to separate LDPE, EVOH, and PET at 120°C, ranked by cost and safety"
"Analyze separation for three-layer film: LDPE/EVOH/PET at 120°C"
"Perform integrated analysis across selectivity, safety, cost, and boiling point for LDPE and EVOH separation"
```

---

## Best Practices

### 1. Always Verify Inputs First
Before running expensive analyses:
```
1. "List all polymers" -> Get exact names
2. "Describe the table" -> Understand structure
3. Run analysis with verified names
```

### 2. Start Simple, Then Refine
```
1. "Find solvents for LDPE" -> Get basic options
2. "Rank by cost" -> Add cost consideration
3. "Show G-scores" -> Add safety consideration
4. Make informed decision
```

### 3. Use Visualizations
Always request visualizations for:
- Temperature curves: See trends, not just numbers
- Heatmaps: Spot patterns across many options
- Flowcharts: Understand sequential processes
- Interactive plots: Explore data yourself

### 4. Leverage Adaptive Analysis
Don't guess thresholds - let the agent adapt:
```
Bad: "Find solvents with 50% selectivity" (might find nothing)
Good: "Find optimal separation conditions" (agent adapts)
```

### 5. Combine Tools for Integrated Analysis
```
1. Find optimal separation -> Get selectivity
2. Rank by cost -> Add economic consideration
3. Check G-scores -> Add safety consideration
4. Plot properties -> Visualize trade-offs
-> Make holistic decision
```

---

## Troubleshooting

### "No data found"
**Cause**: Typo in polymer/solvent name
**Solution**: Use `list_available_polymers()` or `list_available_solvents()` to get exact names

### "No separation found"
**Cause**: Too stringent threshold or wrong temperature
**Solution**: Let adaptive tools relax threshold, or try different temperatures

### "Tool output too long"
**Cause**: Returning too many rows
**Solution**: Use `limit` parameter or filter more aggressively

### "Visualization not showing"
**Cause**: Browser caching or plot generation failed
**Solution**: Check logs, refresh browser, or clear plots and regenerate

---

## Conclusion

The DISSOLVE Agent provides **34 specialized tools** covering:
- Database exploration and validation
- Adaptive separation analysis
- Practical property integration (cost, safety, BP)
- Statistical rigor
- Beautiful visualizations (static + interactive)
- Machine learning predictions
- Safety assessments

**All tools are designed to work together** for comprehensive polymer-solvent analysis.

For architectural details, see [ARCHITECTURE.md](./ARCHITECTURE.md).
