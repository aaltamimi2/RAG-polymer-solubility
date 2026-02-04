# Multi-Agent System Architecture for Polymer Separation Planning
## Methods Section for Publication

### 1. System Overview

The DISSOLVE (Dissolution-based Intelligent Solvent Selection for Optimized Layered Valorization and Extraction) system implements a multi-agent architecture for automated polymer separation planning. The system integrates domain-specific specialist agents with intelligent query routing to provide comprehensive analysis of polymer dissolution, differential precipitation, and techno-economic feasibility.

### 2. Architecture Design

#### 2.1 Query Routing Framework

The system employs an LLM-as-a-judge routing mechanism using Google's Gemini 2.0 Flash model for semantic query understanding. **No orchestrator agent is involved** - the LLM Router directly classifies queries and dispatches them to the appropriate execution path. User queries are classified into four processing paths based on complexity scoring (1-5 scale):

| Path | Complexity | Description | Target Response Time |
|------|------------|-------------|---------------------|
| **Fast** | 1-2 | Simple database lookups, property queries | 2-4 seconds |
| **Standard** | 3 | Multi-tool analysis, comparisons | 5-15 seconds |
| **Specialist** | 4-5 | Domain-specific expertise (single specialist) | 8-25 seconds |
| **Integrated** | 5 | Cross-domain collaboration (2+ specialists) | 15-25 seconds |

The router performs three functions in a single LLM call:
1. **Path classification**: Determines complexity and selects execution path
2. **Entity extraction**: Identifies polymers, solvents, temperatures, and constraints from the query
3. **Specialist selection**: For specialist/integrated paths, identifies which domain expert(s) to invoke

Results are cached with 5-minute TTL to minimize redundant LLM calls for similar queries. For Fast and Standard paths, the agent executes tools directly. For Specialist and Integrated paths, the Hybrid Workflow Engine coordinates specialist agent execution.

#### 2.2 Hybrid Workflow Engine

For Specialist and Integrated paths (as determined by the LLM Router), the Hybrid Workflow Engine **executes** predefined workflows without additional routing decisions. The workflow selection is rule-based on the router's output:

**Predefined Workflows (selected by trigger conditions):**
1. **TEA-First** (triggered: >3 polymers): Profitability screening → Separation (top candidates) → Economic analysis
2. **Parallel Separation-Literature** (triggered: separation + literature constraints): Concurrent separation planning and literature validation
3. **Standard Separation-TEA** (triggered: separation + economics, default): Sequential separation → quality review → economic analysis
4. **Literature-Only** (triggered: literature search without separation): Direct literature search for research-focused queries

If no predefined workflow matches, LLM-based adaptive planning generates a custom workflow (confidence threshold: 0.7).

**Feedback Loops:** The engine implements iterative refinement where separation results triggering cost thresholds (e.g., processing cost > $2/kg) loop back to the separation agent with additional constraints (max 2 iterations).

#### 2.3 Specialist Agents

##### 2.3.1 Separation Agent

The separation specialist implements multiple polymer separation strategies:

**A. Sequential Selective Dissolution**
- **Greedy Algorithm** (O(n²)): Iteratively selects highest-selectivity polymer separations
- **Dynamic Programming** (O(n² × 2^n)): Optimal sequencing for ≤8 polymers
- **Branch and Bound**: Pruning-based optimization balancing speed and optimality

Selectivity is calculated as:
```
selectivity = target_solubility(T) - max(other_solubilities(T))
```
where T is the operating temperature (default 80°C, ±10°C tolerance).

**B. Differential/Selective Precipitation**

The PrecipitationAnalyzer module identifies temperature windows for sequential polymer precipitation during cooling:

- **Precipitation threshold**: <1% solubility (polymer fully precipitated)
- **Cloud point**: ~10-50% solubility (onset of precipitation)
- **Dissolution threshold**: >50% solubility (polymer in solution)

For polymer pairs (A, B), the system queries:
```sql
SELECT solvent,
       MAX(CASE WHEN solubility < 1% THEN temperature END) as precip_temp
FROM solubility_data
WHERE polymer IN ('A', 'B')
GROUP BY solvent, polymer
HAVING (precip_temp_A - precip_temp_B) >= min_gap
```

**C. Atmospheric Feasibility Analysis**

For industrial applicability, the system validates whether differential precipitation can operate at atmospheric pressure (1 atm):

```
feasibility_margin = solvent_boiling_point - required_dissolution_temperature
```

The tool maintains a database of 91 solvent boiling points and evaluates:
- If margin > 0: Process feasible at atmospheric pressure
- If margin < 0: Requires pressurized equipment (autoclave)

**D. Antisolvent Precipitation**

The system identifies antisolvents (solvents with near-zero polymer solubility at room temperature) for precipitation-based recovery:

1. **find_antisolvents**: Identifies solvents with <1% solubility at 25°C
2. **find_antisolvent_pairs**: Matches good solvents (high solubility at elevated temperature) with antisolvents
3. **analyze_selective_antisolvent_precipitation**: Evaluates differential antisolvent response for multi-polymer separation

##### 2.3.2 TEA/LCA Agent

The techno-economic analysis specialist evaluates process economics:

**Capital Expenditure (CAPEX):**
- Distillation equipment sizing based on throughput
- Vessel, heat exchanger, and instrumentation costs

**Operating Expenditure (OPEX):**
- Energy consumption (heating, cooling, pumping)
- Solvent makeup costs (based on recovery rate, default 95%)
- Labor and maintenance

**Lifecycle Assessment:**
- CO₂ emissions per kg solvent recovered
- Energy intensity (kWh/kg)
- Environmental impact scoring

**Profitability Screening:**
```
ROI_score = (market_value - processing_cost) × scale_factor × (1 - 0.3 × difficulty_penalty)
```

Market values range from $0.55/kg (PVC) to $4.50/kg (PVDF), with separation difficulty scores of 3-7 based on polymer chemistry.

##### 2.3.3 Literature Agent

The literature specialist implements retrieval-augmented generation (RAG) for research synthesis:

1. **Internal Knowledge Base**: Pre-indexed polymer dissolution and recycling literature
2. **External Search**: Google Scholar, Web of Science, Google Patents
3. **Dual Retrieval**: Dense (semantic) + sparse (keyword) embedding search
4. **Cross-Encoder Reranking**: Relevance scoring for retrieved documents

Quality thresholds: minimum 2 papers, confidence > 0.4, solvent overlap > 30%.

#### 2.4 Smart Aggregation

The Smart Aggregator combines specialist outputs into unified recommendations:

1. **Cross-validation**: Match TEA-recommended solvents against separation results and literature verification
2. **Ranking**: Combined score = selectivity × economics × literature_confidence
3. **Risk Assessment**: Identify uncertainties, knowledge gaps, and process limitations

### 3. Tool Organization

The system organizes 76+ tools across 13 categories:

| Category | Tools | Purpose |
|----------|-------|---------|
| database | 6 | Schema queries, data validation |
| dissolution | 7 | Solubility analysis, optimal conditions |
| separation | 4 | Sequential dissolution planning |
| advanced_separation | 22 | Algorithms, precipitation, atmospheric analysis, antisolvents |
| solvent_properties | 4 | Properties, rankings, G-scores |
| visualization | 5 | Heatmaps, precipitation curves, process diagrams |
| safety | 4 | PubChem toxicity, GHS hazards |
| economics | 8 | TEA, LCA, cost comparisons |
| strap | 5 | Minimum selling price, scale economics |
| literature | 4 | Scholar, WoS, patents |
| rag | 20 | Embedding search, document ingestion |
| ml_prediction | 1 | Hansen parameter prediction |

### 4. Data Infrastructure

#### 4.1 Solubility Database

The system utilizes a DuckDB-based solubility database containing:
- **Polymers**: 11 major types (LDPE, HDPE, PP, PS, PET, PVC, PC, Nylon6, Nylon66, EVOH, PES)
- **Solvents**: 32 common industrial solvents
- **Data Points**: ~10,000+ polymer-solvent-temperature combinations
- **Temperature Range**: 25-160°C with 5°C resolution

#### 4.2 Auxiliary Databases

- **GSK Solvent Sustainability Guide**: 154 solvents with safety, health, and environmental scores
- **Solvent Properties**: Boiling points, densities, viscosities, Hansen parameters
- **Market Data**: Polymer values, processing costs, difficulty scores

### 5. State Management and Reproducibility

#### 5.1 Checkpointing

The system supports multiple persistence backends:
- **Memory** (default): Fast, ephemeral
- **PostgreSQL**: Persistent, production-ready
- **Redis**: Distributed, high-throughput

#### 5.2 Execution Telemetry

Complete execution traces capture:
- Routing decisions with confidence scores
- Agent handoff timing and success rates
- Tool invocations and results
- Total processing duration

### 6. Performance Characteristics

| Component | Latency |
|-----------|---------|
| Router (cache hit) | ~1 ms |
| Router (LLM call) | 200-500 ms |
| Database query | 10-50 ms |
| Separation analysis | 2-8 seconds |
| TEA calculation | 1-3 seconds |
| Literature search | 3-10 seconds |
| Visualization generation | 1-2 seconds |

### 7. Key Algorithms

#### 7.1 Differential Precipitation Temperature Gap

For polymers A and B in solvent S:
```
T_gap = |T_precip(A, S) - T_precip(B, S)|
```
where T_precip is the temperature at which solubility drops below 1%.

Minimum recommended gap: 20°C for reliable sequential separation.

#### 7.2 Multi-Polymer Precipitation Sequence

For n polymers, the system:
1. Queries precipitation temperatures for all polymer-solvent combinations
2. Sorts by precipitation temperature (highest first = precipitates first during cooling)
3. Calculates gaps between consecutive precipitations
4. Validates minimum gap constraint for all pairs
5. Checks atmospheric feasibility against solvent boiling point

#### 7.3 Antisolvent Selectivity

For selective antisolvent precipitation of polymers A and B:
```
differential_response = |solubility(A, antisolvent) - solubility(B, antisolvent)|
```
Higher differential enables selective precipitation when one polymer tolerates the antisolvent better than another.

### 8. Implementation Details

- **Framework**: LangGraph for agent orchestration
- **LLM**: Google Gemini 2.0 Flash (routing), Gemini 1.5 Flash (agents)
- **Database**: DuckDB (in-memory, SQL interface)
- **Visualization**: Matplotlib with publication-quality formatting
- **API**: FastAPI-compatible async architecture

### 9. Limitations and Future Work

**Current Limitations:**
- Solubility data limited to 11 polymer types and 32 solvents
- Temperature range capped at 160°C (atmospheric pressure constraint)
- No kinetic dissolution rate modeling
- Binary polymer mixtures only for differential precipitation

**Planned Enhancements:**
- Expanded polymer database (engineering plastics, biopolymers)
- Pressure-dependent solubility modeling
- Dissolution kinetics integration
- Multi-component (3+) differential precipitation optimization
