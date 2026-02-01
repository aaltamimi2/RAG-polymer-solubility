"""
Multi-Agent System for DISSOLVE Polymer Solubility Agent

This module implements a hybrid router architecture with specialized agents
for complex query types while maintaining fast performance for simple queries.

Architecture:
    Query -> Complexity Router -> Fast/Standard/Specialist/Integrated Path -> Response

NEW: Integrated Path enables iterative collaboration between specialists:
    - Separation Agent proposes separation sequences
    - TEA/LCA Agent evaluates economics of each sequence
    - Smart Aggregator combines results into unified recommendation

Performance Targets:
    - Simple queries (1-2): 2-4s (10-20% improvement)
    - Separation (4-5): 8-25s (40-50% improvement)
    - TEA/LCA (4): 5-15s (50% improvement)
    - Literature (3-4): 6-12s (40% improvement)
    - INTEGRATED (5): 15-25s (single query replaces 2+ queries)
"""

import asyncio
import logging
import traceback
import time
import re
import json
from typing import Optional, List, Dict, Literal, Any, Tuple, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import operator

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Import structured schemas for inter-agent communication
from agent_schemas import (
    SeparationResult,
    TEAResult,
    HandoffPayload,
    SharedContext,
    SeparationStep,
    parse_to_separation_result,
    parse_to_tea_result,
    # P2: Task-oriented handoff schemas
    TEATaskRequest,
    SeparationTaskRequest,
    AggregatorTaskRequest,
    # P3: Enhanced tracking schemas
    HandoffMetrics,
    ExecutionTrace,
)
import uuid

logger = logging.getLogger(__name__)

# ============================================================
# COMPLEXITY SCORING AND SPECIALIST ROUTING
# ============================================================

@dataclass
class RoutingDecision:
    """Result of complexity routing."""
    complexity: int  # 1-5 scale
    path: Literal["fast", "standard", "specialist", "integrated"]
    specialist: Optional[str]  # "separation", "tea_lca", "literature", None
    categories: List[str]  # Tool categories needed
    reason: str  # Explanation for logging
    # NEW: For integrated multi-agent collaboration
    collaboration_specialists: List[str] = field(default_factory=list)


def enhanced_complexity_router(query: str) -> RoutingDecision:
    """
    Rule-based complexity scoring with specialist routing.

    Performance: ~1-2ms (no LLM call)

    Returns:
        RoutingDecision with path, specialist, and complexity score
    """
    query_lower = query.lower()

    # ==========================================================
    # INTEGRATED PATH: Cross-domain queries requiring multiple specialists
    # ==========================================================

    # Separation + TEA/LCA integration triggers
    integrated_sep_tea_triggers = [
        "cost-effective separation",
        "cost effective separation",
        "cheapest way to separate",
        "cheapest separation",
        "economical separation",
        "economic separation",
        "lowest cost separation",
        "separation with tea",
        "separation with economics",
        "compare separation costs",
        "separation cost comparison",
        "msp for separation",
        "profitable separation",
        "viable separation",
    ]

    # Check for separation + TEA combination
    has_separation_keyword = any(w in query_lower for w in [
        "separate", "separation", "multilayer", "multi-layer", "sequence"
    ])
    has_cost_keyword = any(w in query_lower for w in [
        "cost", "economic", "tea", "capex", "opex", "payback", "msp", "cheap", "expensive"
    ])

    if any(trigger in query_lower for trigger in integrated_sep_tea_triggers) or \
       (has_separation_keyword and has_cost_keyword):
        return RoutingDecision(
            complexity=5,
            path="integrated",
            specialist=None,
            categories=["separation", "dissolution", "economics", "strap", "visualization", "solvent_properties"],
            reason="Integrated Separation + TEA analysis detected",
            collaboration_specialists=["separation", "tea_lca"]
        )

    # Separation + Literature integration
    if has_separation_keyword and any(w in query_lower for w in ["literature", "research", "papers", "studies"]):
        return RoutingDecision(
            complexity=5,
            path="integrated",
            specialist=None,
            categories=["separation", "dissolution", "literature", "rag"],
            reason="Integrated Separation + Literature research detected",
            collaboration_specialists=["separation", "literature"]
        )

    # ==========================================================
    # SPECIALIST PATH: Complex multi-step queries (complexity 4-5)
    # ==========================================================

    # Separation Planning Specialist
    separation_triggers = [
        "sequential separation", "separation sequence", "separation strategy",
        "multilayer", "multi-layer", "3-polymer", "4-polymer", "5-polymer",
        "decision tree", "all permutations", "separation pathways",
        "isolate each polymer", "separate all", "plan separation",
        "which sequence", "optimal sequence", "best sequence"
    ]
    if any(trigger in query_lower for trigger in separation_triggers):
        return RoutingDecision(
            complexity=5,
            path="specialist",
            specialist="separation",
            categories=["separation", "dissolution", "solvent_properties", "visualization"],
            reason="Multi-polymer separation planning detected"
        )

    # TEA/LCA Specialist
    tea_triggers = [
        "techno-economic", "technoeconomic", "tea analysis", "run tea",
        "lca analysis", "life cycle", "carbon footprint", "co2 emission",
        "payback period", "capital cost", "operating cost", "capex", "opex",
        "economic analysis", "cost analysis", "solvent recovery cost",
        "energy consumption", "gwp", "environmental impact assessment"
    ]
    if any(trigger in query_lower for trigger in tea_triggers):
        return RoutingDecision(
            complexity=4,
            path="specialist",
            specialist="tea_lca",
            categories=["economics", "strap", "visualization", "solvent_properties"],
            reason="Techno-economic or LCA analysis detected"
        )

    # Literature Research Specialist
    literature_triggers = [
        "search literature", "find papers", "research on", "publications about",
        "what does the literature", "scholarly articles", "peer-reviewed",
        "web of science", "google scholar", "search rag", "ask literature",
        "what papers", "recent research", "state of the art"
    ]
    if any(trigger in query_lower for trigger in literature_triggers):
        return RoutingDecision(
            complexity=4,
            path="specialist",
            specialist="literature",
            categories=["literature", "rag"],
            reason="Literature research query detected"
        )

    # ==========================================================
    # STANDARD PATH: Moderate complexity (complexity 3)
    # ==========================================================

    standard_triggers = [
        "compare", "comparison", "rank by", "ranked by",
        "temperature curve", "temperature window", "vs temperature",
        "selective", "selectivity", "separate", "separation",
        "heatmap", "dashboard", "multi-panel",
        "correlation", "regression", "statistical"
    ]
    if any(trigger in query_lower for trigger in standard_triggers):
        return RoutingDecision(
            complexity=3,
            path="standard",
            specialist=None,
            categories=[],  # Will use existing router
            reason="Standard complexity query"
        )

    # ==========================================================
    # FAST PATH: Simple queries (complexity 1-2)
    # ==========================================================

    simple_triggers = [
        "list tables", "what tables", "describe table", "schema",
        "list polymers", "what polymers", "available polymers", "list all polymers",
        "list solvents", "what solvents", "available solvents", "list all solvents",
        "solubility of", "dissolve", "top 10", "top 5", "top 20",
        "boiling point of", "logp of", "properties of",
        "g-score for", "gscore for", "how many polymers", "how many solvents"
    ]
    if any(trigger in query_lower for trigger in simple_triggers):
        return RoutingDecision(
            complexity=2 if "top" in query_lower else 1,
            path="fast",
            specialist=None,
            categories=["database", "dissolution", "solvent_properties"],
            reason="Simple lookup query"
        )

    # ==========================================================
    # DEFAULT: Standard path for unclassified queries
    # ==========================================================
    return RoutingDecision(
        complexity=3,
        path="standard",
        specialist=None,
        categories=[],
        reason="Default routing - unclassified query"
    )


# ============================================================
# TOOL SUBSETS FOR SPECIALISTS
# ============================================================

# These will be populated from the main agent module
FAST_PATH_TOOLS = []  # ~20 tools
SEPARATION_TOOLS = []  # ~15 tools
TEA_LCA_TOOLS = []     # ~15 tools
LITERATURE_TOOLS = []  # ~25 tools


def initialize_tool_subsets(tool_categories: dict, all_tools: list):
    """
    Initialize tool subsets from main agent's TOOL_CATEGORIES.

    Called during module initialization.
    """
    global FAST_PATH_TOOLS, SEPARATION_TOOLS, TEA_LCA_TOOLS, LITERATURE_TOOLS

    # Helper to get tools by category
    def get_category_tools(categories: List[str]) -> list:
        tools_by_name = {}
        for cat in categories:
            if cat in tool_categories:
                for tool in tool_categories[cat]:
                    tools_by_name[tool.name] = tool
        return list(tools_by_name.values())

    # Fast path: database + basic dissolution + properties
    FAST_PATH_TOOLS = get_category_tools([
        "database", "dissolution", "solvent_properties"
    ])

    # Separation specialist
    SEPARATION_TOOLS = get_category_tools([
        "separation", "dissolution", "solvent_properties",
        "visualization", "safety"
    ])

    # TEA/LCA specialist
    TEA_LCA_TOOLS = get_category_tools([
        "economics", "strap", "visualization", "solvent_properties"
    ])

    # Literature specialist
    LITERATURE_TOOLS = get_category_tools([
        "literature", "rag"
    ])

    logger.info(f"Tool subsets initialized:")
    logger.info(f"  Fast path: {len(FAST_PATH_TOOLS)} tools")
    logger.info(f"  Separation: {len(SEPARATION_TOOLS)} tools")
    logger.info(f"  TEA/LCA: {len(TEA_LCA_TOOLS)} tools")
    logger.info(f"  Literature: {len(LITERATURE_TOOLS)} tools")


# ============================================================
# SPECIALIST AGENT PROMPTS
# ============================================================

SEPARATION_PLANNER_PROMPT = """You are a **Polymer Separation Planning Specialist** for the DISSOLVE system.

YOUR ONLY TASK: Plan optimal sequential separation strategies for multilayer polymer films.

## WORKFLOW
1. **ALWAYS** call `plan_sequential_separation()` first with the polymer list and temperature
2. The tool will enumerate ALL permutations and rank them by worst-case selectivity
3. Use the results to explain the best sequence and solvent choices
4. If asked for visualization, use `plot_selectivity_heatmap()` or the decision tree from step 1

## KEY PARAMETERS
- **Temperature**: Use the specified temperature (default 80°C for better selectivity)
- **Polymers**: Extract polymer names from the query (LDPE, HDPE, PET, PP, PS, PVC, PC, Nylon66, EVOH)
- **Top-k solvents**: Default to 5 per step

## RESPONSE FORMAT
1. Best sequence with reasoning
2. Step-by-step solvent recommendations with properties
3. Selectivity scores and warnings
4. Alternative sequences if applicable

DO NOT: Run general database queries, perform statistical analysis, or search literature.
FOCUS: Separation planning only. Be concise and actionable."""


TEA_LCA_ANALYST_PROMPT = """You are a **Techno-Economic Analysis Specialist** for the DISSOLVE system.

YOUR ONLY TASK: Evaluate economics and environmental impact of polymer solvent recovery processes.

## AVAILABLE TOOLS
1. `analyze_solvent_recovery_tea()` - Calculate capital/operating costs
2. `analyze_solvent_recovery_lca()` - Calculate CO2 emissions
3. `compare_solvents_tea_lca()` - Multi-solvent comparison
4. `generate_tea_visualizations()` - Cost breakdown charts
5. `plot_tea_sensitivity_tornado()` - Sensitivity analysis
6. `plot_tea_cashflow()` - NPV visualization

## MANDATORY PARAMETERS
- **Solvent boiling point**: Required for recovery calculations
- **Throughput**: Use specified value or default to 100 kg/hr
- **Recovery rate**: Default to 95% if not specified

## RESPONSE FORMAT
1. Cost summary (CAPEX, OPEX, payback period)
2. Environmental impact (CO2 emissions, energy)
3. Key sensitivities and assumptions
4. Clear recommendation with justification

DO NOT: Query polymer solubility data, plan separations, or search literature.
FOCUS: Economic and environmental analysis only."""


LITERATURE_RESEARCHER_PROMPT = """You are a **Scientific Literature Research Specialist** for the DISSOLVE system.

YOUR ONLY TASK: Find and synthesize relevant research on polymer dissolution and recycling.

## SEARCH STRATEGY
1. **ALWAYS** check internal RAG first with `search_literature_rag()` - faster and more relevant
2. If RAG insufficient, use `search_google_scholar()` or `search_web_of_science()`
3. For specific questions about indexed papers, use `ask_literature()`
4. Offer to save useful papers with `download_pdf_to_rag()` or `save_scholar_results_to_rag()`

## RESPONSE FORMAT
1. Key findings with citations
2. Relevant excerpts from papers
3. Knowledge gaps identified
4. Suggestions for further reading

## CITATION FORMAT
- Include DOI links when available
- Format: Author et al. (Year) - Journal
- Provide direct quotes with page/section references

DO NOT: Run solubility queries, perform TEA analysis, or plan separations.
FOCUS: Literature research and synthesis only."""


# ============================================================
# MULTI-AGENT STATE (Extended for Collaboration)
# ============================================================

class MultiAgentState(MessagesState):
    """Extended state for multi-agent coordination with collaboration support."""
    iteration_count: int = 0
    max_iterations: int = 15

    # Complexity routing
    complexity: int = 0
    path: str = "standard"  # "fast", "standard", "specialist", "integrated"
    specialist: Optional[str] = None
    routing_reason: str = ""

    # Specialist outputs for aggregation
    specialist_outputs: Dict[str, str] = {}

    # Memory Engine fields (inherited)
    user_id: Optional[str] = None
    memory_context: Optional[str] = None
    memory_enabled: bool = True

    # Router fields (inherited)
    selected_categories: Optional[List[str]] = None

    # Multi-agent metadata for frontend
    multi_agent_active: bool = False
    active_specialist: Optional[str] = None
    specialist_start_time: Optional[float] = None

    # =========================================================
    # NEW: Iterative Collaboration Fields (P1: Typed Schemas)
    # =========================================================

    # Collaboration mode: "separation_tea", "separation_literature", etc.
    collaboration_mode: Optional[str] = None

    # List of specialists to run in sequence
    collaboration_specialists: List[str] = []

    # Index of current specialist in collaboration sequence
    current_specialist_index: int = 0

    # Structured results from Separation Agent (P1: Pydantic schema)
    separation_results: Optional[Dict[str, Any]] = None  # Holds SeparationResult.model_dump()

    # Structured results from TEA Agent (P1: Pydantic schema)
    tea_results: Optional[Dict[str, Any]] = None  # Holds TEAResult.model_dump()

    # Structured results from Literature Agent
    literature_results: Optional[Dict[str, Any]] = None

    # Shared context passed between specialists (P1: SharedContext compatible)
    shared_context: Dict[str, Any] = {}

    # Flag indicating aggregation is needed
    aggregation_required: bool = False

    # Original query for reference
    original_query: Optional[str] = None

    # =========================================================
    # NEW: Handoff Tracking (P0: Command-based routing)
    # =========================================================

    # History of handoffs for debugging/tracing
    # Using Annotated with operator.add allows multiple nodes to append
    handoff_history: Annotated[List[Dict[str, Any]], operator.add] = []

    # Pending handoff payload (set by agent, consumed by router)
    pending_handoff: Optional[Dict[str, Any]] = None

    # =========================================================
    # P3: Enhanced Execution Tracking
    # =========================================================

    # Execution trace ID for this session
    trace_id: Optional[str] = None

    # Detailed handoff metrics (P3: Enhanced tracking)
    handoff_metrics: Annotated[List[Dict[str, Any]], operator.add] = []

    # Timing for each agent (P3: Performance tracking)
    agent_timings: Dict[str, float] = {}

    # Finalized execution trace (P3: Set by smart_aggregator)
    execution_trace: Optional[Dict[str, Any]] = None

    # =========================================================
    # Iteration Counters for Handoff Timing (prevents premature handoffs)
    # =========================================================

    # Iteration counter for separation agent (prevents infinite loops)
    sep_iteration_count: int = 0

    # Iteration counter for TEA agent
    tea_iteration_count: int = 0


# ============================================================
# HELPER FUNCTIONS FOR CONTEXT EXTRACTION
# ============================================================

def extract_polymers(query: str) -> List[str]:
    """Extract polymer names from query."""
    # Sort by length (longest first) to match LDPE before PE, Nylon66 before Nylon6
    known_polymers = [
        "Nylon66", "Nylon6", "PA66", "PA6",  # Long names first
        "LDPE", "HDPE", "EVOH", "PMMA", "PBAT",
        "PET", "PVC", "ABS", "PLA", "PBS", "PHA", "PHB", "PVDF",
        "PE", "PP", "PS", "PC",  # Short names last (to avoid substring conflicts)
    ]
    query_upper = query.upper()
    found = []
    found_positions = set()  # Track matched positions to avoid overlaps

    for polymer in known_polymers:
        polymer_upper = polymer.upper()
        # Find all occurrences
        start = 0
        while True:
            pos = query_upper.find(polymer_upper, start)
            if pos == -1:
                break
            end_pos = pos + len(polymer_upper)

            # Check if this position overlaps with already found polymer
            position_range = set(range(pos, end_pos))
            if not position_range & found_positions:
                # Check word boundary (not part of a longer word)
                before_ok = pos == 0 or not query_upper[pos-1].isalnum()
                after_ok = end_pos == len(query_upper) or not query_upper[end_pos].isalnum()

                if before_ok and after_ok:
                    found.append(polymer)
                    found_positions.update(position_range)
                    break  # Only add once per polymer type
            start = pos + 1

    return found if found else ["LDPE", "PET", "EVOH"]  # Default


def extract_temperature(query: str, default: float = 80.0) -> float:
    """Extract temperature from query."""
    query_lower = query.lower()

    # Check for room temperature keywords first
    if any(phrase in query_lower for phrase in ["room temperature", "ambient", "room temp", "25 c", "25c"]):
        return 25.0

    # Look for patterns like "80C", "80°C", "80 C", "at 80 degrees"
    patterns = [
        r'(\d+)\s*°?\s*[Cc](?:elsius)?',
        r'at\s+(\d+)\s*(?:degrees?)?',
        r'(\d+)\s*degrees',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return float(match.group(1))
    return default


def extract_throughput(query: str, default: float = 100.0) -> float:
    """Extract throughput from query."""
    # Look for patterns like "100 kg/hr", "500kg/hr"
    patterns = [
        r'(\d+)\s*kg/h',
        r'(\d+)\s*kg per hour',
        r'throughput[:\s]+(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return float(match.group(1))
    return default


def parse_separation_results(response_text: str) -> Dict[str, Any]:
    """
    Parse separation agent response to extract structured results.

    Returns dict with sequences, solvents, selectivities for TEA evaluation.
    """
    results = {
        "sequences": [],
        "solvents": [],  # Primary key for solvents
        "selectivities": [],
        "algorithm_used": None,
        "best_sequence": None,
        "raw_response": response_text
    }

    response_lower = response_text.lower()

    # Detect algorithm used (greedy vs exhaustive)
    # Look for specific patterns in the tool output
    if "greedy separation" in response_lower or "algorithm:** greedy" in response_lower:
        results["algorithm_used"] = "greedy"
    elif "greedy" in response_lower and ("o(n²)" in response_lower or "o(n^2)" in response_lower):
        results["algorithm_used"] = "greedy"
    elif "exhaustive" in response_lower or "all permutations" in response_lower:
        results["algorithm_used"] = "exhaustive"
    elif "n!" in response_text or "permutation" in response_lower:
        results["algorithm_used"] = "exhaustive"

    # Pattern 0: Full sequence with arrows (PS → PVC → PP → EVOH → LDPE → PET → HDPE → PA6 → PA66)
    # This is more specific for greedy output
    full_seq_pattern = r'\*\*(?:Optimized\s+)?Sequence:\*\*\s*([A-Z0-9]+(?:\s*→\s*[A-Z0-9]+)+)'
    full_seq_match = re.search(full_seq_pattern, response_text)
    if full_seq_match:
        seq_str = full_seq_match.group(1)
        seq = [p.strip() for p in re.split(r'\s*→\s*', seq_str)]
        if len(seq) > 1:
            results["best_sequence"] = seq
            results["sequences"].append(seq)

    # Pattern 1: Arrow sequences (PE -> EVOH -> PET)
    arrow_pattern = r'([A-Z]{2,6})\s*[-→>]+\s*([A-Z]{2,6})(?:\s*[-→>]+\s*([A-Z]{2,6}))?'
    arrow_matches = re.findall(arrow_pattern, response_text.upper())

    for match in arrow_matches:
        seq = [m for m in match if m]  # Filter empty groups
        if seq and seq not in results["sequences"]:
            results["sequences"].append(seq)

    # Pattern 2: Extract solvents - expanded list with various forms
    common_solvents = [
        # Hydrocarbons
        "cyclohexane", "hexane", "heptane", "pentane", "octane",
        "toluene", "xylene", "benzene", "decalin", "tetralin",
        # Polar aprotic
        "dmso", "dmf", "nmp", "thf", "mek", "dmac", "dma", "sulfolane",
        # Alcohols
        "acetone", "ethanol", "methanol", "isopropanol", "butanol", "propanol",
        # Halogenated
        "dcm", "chloroform", "dichloromethane", "trichloroethylene",
        "dichloroacetic acid", "trifluoroacetic acid",
        # Esters & ethers
        "ethyl acetate", "diethyl ether", "dioxane",
        # Glycols
        "ethylene glycol", "propylene glycol", "glycerol",
        # Others
        "water", "limonene", "gamma-valerolactone", "gvl", "cyrene"
    ]

    found_solvents = set()
    for solvent in common_solvents:
        if solvent in response_lower:
            found_solvents.add(solvent)

    results["solvents"] = list(found_solvents)

    # Pattern 3: Extract selectivity values (various formats)
    selectivity_patterns = [
        r'selectivity[:\s]+(\d+\.?\d*)%?',
        r'(\d+\.?\d*)\s*%?\s*selectiv',
        r'\(selectivity[:\s]*(\d+\.?\d*)',
        r'selectivity\s*[=:]\s*(\d+\.?\d*)',
    ]
    selectivities = set()
    for pattern in selectivity_patterns:
        matches = re.findall(pattern, response_lower)
        for m in matches:
            try:
                val = float(m)
                if 0 < val <= 1:
                    selectivities.add(val)
                elif 1 < val <= 100:
                    selectivities.add(val / 100)  # Convert percentage
            except ValueError:
                pass

    results["selectivities"] = sorted(list(selectivities), reverse=True)

    return results


def parse_tea_results(response_text: str) -> Dict[str, Any]:
    """Parse TEA agent response to extract structured cost data."""
    results = {
        "msp_values": {},
        "best_solvent": None,
        "cost_breakdown": {},
        "total_capex": None,
        "total_opex": None,
        "payback_years": None,
        "cost_per_kg": None,
        "raw_response": response_text
    }

    response_lower = response_text.lower()

    # Pattern for cost extraction from TEA tool output
    # Multiple patterns to catch different formats
    cost_patterns = [
        r'cost\s+per\s+kg(?:\s+polymer)?[:\s]+\$?(\d+\.?\d*)\s*/?\s*kg',  # Cost per kg polymer: $1.39/kg
        r'cost\s+of\s+\$?(\d+\.?\d*)\s*/\s*kg',  # cost of $1.39/kg
        r'\$(\d+\.?\d*)\s*/\s*kg',  # $1.39/kg
        r'(\d+\.?\d*)\s*\$/kg',  # 1.39 $/kg
        r'msp[:\s]+\$?(\d+\.?\d*)',  # MSP: $1.39
    ]
    for pattern in cost_patterns:
        match = re.search(pattern, response_lower)
        if match:
            try:
                cost = float(match.group(1))
                if 0 < cost < 1000:  # Sanity check
                    results["cost_per_kg"] = cost
                    break
            except ValueError:
                pass

    # Extract MSP values - look for patterns like "MSP: $2.45/kg" or "Xylene: 2.45"
    msp_patterns = [
        r'msp[:\s]+\$?(\d+\.?\d*)\s*/?\s*kg',
        r'minimum selling price[:\s]+\$?(\d+\.?\d*)',
        r'\$(\d+\.?\d*)\s*/\s*kg',
    ]
    for pattern in msp_patterns:
        match = re.search(pattern, response_lower)
        if match:
            results["cost_per_kg"] = float(match.group(1))
            break

    # Extract solvent-specific costs from tables or lists
    # Pattern: "| Xylene | 2.45 |" or "Xylene: $2.45/kg"
    common_solvents = [
        "xylene", "toluene", "cyclohexane", "hexane", "dmso", "dmf",
        "nmp", "thf", "mek", "acetone", "ethanol", "dcm", "chloroform",
        "limonene", "gvl", "decalin", "dichloroacetic acid"
    ]

    for solvent in common_solvents:
        # Look for "solvent | price" or "solvent: $price"
        patterns = [
            rf'{solvent}\s*\|\s*\$?(\d+\.?\d*)',
            rf'{solvent}[:\s]+\$?(\d+\.?\d*)\s*/?\s*kg',
            rf'\|\s*{solvent}\s*\|\s*\$?(\d+\.?\d*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                results["msp_values"][solvent] = float(match.group(1))
                break

    # Find best/recommended solvent
    best_patterns = [
        r'(?:best|optimal|recommend(?:ed)?|lowest)[:\s]+(\w+)',
        r'(\w+)(?:\s+|-)?based process (?:has |offers |provides )(?:lowest|best)',
        r'recommend(?:ation)?[:\s]+(?:use\s+)?(\w+)',
    ]
    for pattern in best_patterns:
        match = re.search(pattern, response_lower)
        if match:
            candidate = match.group(1).lower()
            if candidate in common_solvents:
                results["best_solvent"] = candidate
                break

    # Extract CAPEX
    capex_patterns = [
        r'capex[:\s]+\$?(\d[\d,]*)',
        r'capital\s+cost[:\s]+\$?(\d[\d,]*)',
    ]
    for pattern in capex_patterns:
        match = re.search(pattern, response_lower)
        if match:
            results["total_capex"] = float(match.group(1).replace(",", ""))
            break

    # Extract OPEX
    opex_patterns = [
        r'opex[:\s]+\$?(\d[\d,]*)',
        r'operating\s+cost[:\s]+\$?(\d[\d,]*)',
    ]
    for pattern in opex_patterns:
        match = re.search(pattern, response_lower)
        if match:
            results["total_opex"] = float(match.group(1).replace(",", ""))
            break

    # Extract payback
    payback_pattern = r'payback[:\s]+(\d+\.?\d*)\s*years?'
    payback_match = re.search(payback_pattern, response_lower)
    if payback_match:
        results["payback_years"] = float(payback_match.group(1))

    # Build cost breakdown if we have components
    if results["total_capex"] or results["total_opex"]:
        results["cost_breakdown"] = {
            "capex": results["total_capex"],
            "opex": results["total_opex"],
        }

    return results


# ============================================================
# P3: EXECUTION TRACE AND HANDOFF METRICS HELPERS
# ============================================================

def create_handoff_metrics(
    from_agent: str,
    to_agent: str,
    start_time: float,
    tools_called: List[str] = None,
    success: bool = True,
    error_message: str = None,
    task_type: str = None,
    context_size: int = None,
) -> Dict[str, Any]:
    """
    Create enhanced handoff metrics for P3 tracking.

    Args:
        from_agent: Source agent name
        to_agent: Target agent name
        start_time: When the source agent started (time.time())
        tools_called: List of tools invoked by source agent
        success: Whether handoff was successful
        error_message: Error details if not successful
        task_type: Type of task (separation, tea, literature)
        context_size: Approximate size of context passed

    Returns:
        Dict with HandoffMetrics-compatible structure
    """
    duration_ms = (time.time() - start_time) * 1000 if start_time else None

    return HandoffMetrics(
        handoff_id=str(uuid.uuid4())[:8],
        from_agent=from_agent,
        to_agent=to_agent,
        timestamp=datetime.now(),
        duration_ms=duration_ms,
        tools_called=tools_called or [],
        success=success,
        error_message=error_message,
        task_type=task_type,
        context_size_bytes=context_size,
    ).model_dump()


def create_execution_trace(
    query: str,
    complexity: int,
    path: str,
) -> Dict[str, Any]:
    """
    Create a new execution trace for tracking.

    Args:
        query: Original user query
        complexity: Complexity score (1-5)
        path: Routing path (fast, standard, specialist, integrated)

    Returns:
        Dict with ExecutionTrace-compatible structure
    """
    return ExecutionTrace(
        trace_id=str(uuid.uuid4())[:12],
        query=query,
        complexity=complexity,
        path=path,
        start_time=datetime.now(),
        agents_visited=[],
        handoffs=[],
    ).model_dump()


# ============================================================
# INTEGRATED ORCHESTRATOR NODE
# ============================================================
# NOTE: Supervisor node was removed as it's not integrated into the graph.
# Dynamic routing is handled by validation gates in separation agent
# (e.g., skip TEA if no solvents found). SupervisorDecision schema is kept
# in agent_schemas.py for future use if needed.

async def integrated_orchestrator_node(state: MultiAgentState) -> dict:
    """
    Orchestrates sequential specialist execution with context passing.

    Flow:
    1. Parse user intent to determine specialist order
    2. Initialize shared context with extracted parameters
    3. Set up collaboration state for sequential execution
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "path": "standard",
            "routing_reason": "No messages for integrated orchestrator",
            "collaboration_mode": None
        }

    # Get the original query
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    # Determine collaboration specialists from routing decision
    # (Already set by router, but we can validate here)
    collaboration_specialists = state.get("collaboration_specialists", ["separation", "tea_lca"])

    # Extract shared context from query
    shared_context = {
        "original_query": query,
        "polymers": extract_polymers(query),
        "temperature": extract_temperature(query, default=80.0),
        "throughput_kg_hr": extract_throughput(query, default=100.0),
        "timestamp": datetime.now().isoformat(),
    }

    collaboration_mode = "_".join(collaboration_specialists)

    # P3: Create execution trace for this session
    trace_id = str(uuid.uuid4())[:12]
    complexity = state.get("complexity", 5)

    logger.info(f"Integrated Orchestrator: mode={collaboration_mode}, "
                f"specialists={collaboration_specialists}, "
                f"trace_id={trace_id}, context={shared_context}")

    return {
        "collaboration_mode": collaboration_mode,
        "collaboration_specialists": collaboration_specialists,
        "current_specialist_index": 0,
        "shared_context": shared_context,
        "original_query": query,
        "aggregation_required": True,
        "multi_agent_active": True,
        "active_specialist": collaboration_specialists[0] if collaboration_specialists else None,
        "specialist_start_time": time.time(),
        # P3: Enhanced tracking
        "trace_id": trace_id,
        "agent_timings": {"orchestrator": time.time()},
    }


# ============================================================
# COLLABORATION-AWARE SPECIALIST NODES
# ============================================================

async def collab_separation_agent_node(state: MultiAgentState, sql_agent_node) -> Command:
    """
    Separation specialist with Command-based handoff for collaboration.

    P0/P1 Changes:
    - Returns Command object for dynamic routing (instead of dict)
    - Uses SeparationResult Pydantic schema for structured output
    - Creates HandoffPayload for context passing to TEA agent

    When in collaboration mode:
    - Executes separation planning
    - Extracts structured results (sequences, solvents)
    - Returns Command(goto="collab_tea_agent") for handoff
    """
    is_collaborative = state.get("collaboration_mode") is not None
    shared_context = state.get("shared_context", {})

    # Set categories for separation
    state_copy = dict(state)
    state_copy["selected_categories"] = ["separation", "dissolution", "solvent_properties", "visualization", "safety"]

    # P2: If in collaborative mode, use task-oriented instruction
    if is_collaborative:
        polymers = shared_context.get("polymers", [])
        temperature = shared_context.get("temperature", 80.0)

        # Create task-oriented request (P2: reduces context bloat)
        sep_task = SeparationTaskRequest(
            polymers=polymers,
            temperature=temperature,
            top_k_solvents=3,
            ranking_criterion="selectivity"
        )

        # Use the schema's to_instruction() method for consistent formatting
        tool_instruction = HumanMessage(content=sep_task.to_instruction())
        messages = list(state_copy.get("messages", []))
        messages.append(tool_instruction)
        state_copy["messages"] = messages

    # Execute separation agent
    result = await sql_agent_node(state_copy)

    if is_collaborative:
        # Extract structured results from ALL messages (state + result)
        existing_messages = state.get("messages", [])
        new_messages = result.get("messages", [])

        # Combine: existing + new
        all_messages = list(existing_messages) + list(new_messages)
        all_text = ""

        # Debug: log message types
        msg_types = [type(m).__name__ for m in all_messages]
        logger.info(f"Separation: {len(all_messages)} messages, types: {set(msg_types)}")

        for msg in all_messages:
            msg_type = type(msg).__name__

            if isinstance(msg, AIMessage):
                if hasattr(msg, 'content') and msg.content:
                    all_text += str(msg.content) + "\n"
            elif msg_type == 'ToolMessage':
                # ToolMessage from LangGraph
                content = getattr(msg, 'content', '')
                if isinstance(content, str):
                    all_text += content + "\n"
                    # Log a sample of tool output for debugging
                    sample = content[:300].replace('\n', ' ')
                    logger.info(f"ToolMessage ({len(content)} chars): {sample}...")
            elif hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, str):
                    all_text += content + "\n"
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            all_text += item['text'] + "\n"
                        elif isinstance(item, str):
                            all_text += item + "\n"
                elif content:
                    all_text += str(content) + "\n"

        logger.info(f"Separation extraction from {len(all_text)} chars")
        if len(all_text) > 200:
            # Check for solvent keywords in the text
            test_solvents = ["cyclohexane", "hexane", "methanol", "dimethylformamide"]
            found_test = [s for s in test_solvents if s in all_text.lower()]
            logger.info(f"Test solvents in text: {found_test}")

        separation_results = parse_separation_results(all_text)
        separation_results["polymers"] = shared_context.get("polymers", [])
        separation_results["temperature"] = shared_context.get("temperature", 80.0)

        # Extract solvents from tool output using specific patterns
        all_text_lower = all_text.lower()

        # Known solvent names - only accept these as valid solvents
        known_solvents = {
            # Hydrocarbons
            "xylene", "toluene", "cyclohexane", "hexane", "heptane", "pentane", "octane",
            "n-heptane", "n-hexane", "n-octane", "n-decane",
            "decalin", "tetralin", "benzene", "mesitylene", "cumene",
            # Polar aprotic
            "dmf", "dmso", "nmp", "thf", "mek", "dmac", "dma", "sulfolane", "acetonitrile",
            "dimethylformamide", "dimethylsulfoxide", "n-methylpyrrolidone",
            # Alcohols
            "acetone", "ethanol", "methanol", "isopropanol", "butanol", "propanol",
            "benzyl alcohol", "phenol", "cresol", "ethylene glycol", "propylene glycol",
            "propyleneglycol", "glycol",
            # Halogenated
            "dcm", "dichloromethane", "chloroform", "trichloroethylene", "tetrachloroethylene",
            "ch2cl2", "ccl4",
            # Acids
            "formic acid", "acetic acid", "dichloroacetic acid", "trifluoroacetic acid",
            # Green solvents
            "limonene", "cymene", "pinene", "gamma-valerolactone", "gvl", "cyrene",
            "dihydrolevoglucosenone", "2-methyltetrahydrofuran", "cpme",
            # Others
            "water", "dioxane", "pyridine", "aniline"
        }

        # Pattern 1: Markdown bold pattern from tool output: **solvent_name**:
        bold_pattern_matches = re.findall(r'\*\*([a-zA-Z0-9\-]+)\*\*:', all_text_lower)
        for match in bold_pattern_matches:
            match_clean = match.strip().lower()
            if match_clean in known_solvents:
                if match_clean not in separation_results.get("solvents", []):
                    separation_results.setdefault("solvents", []).append(match_clean)

        # Pattern 2: "Solvent: {name}" from tool output
        solvent_pattern_matches = re.findall(r'solvent[:\s]+([a-zA-Z0-9\-\s]+?)(?:\s*[\|\,\n]|$)', all_text_lower)
        for match in solvent_pattern_matches:
            match_clean = match.strip().lower()
            for known in known_solvents:
                if known in match_clean:
                    if known not in separation_results.get("solvents", []):
                        separation_results.setdefault("solvents", []).append(known)

        # Pattern 3: Direct keyword matching - most reliable
        for solv in known_solvents:
            if len(solv) <= 3:
                pattern = rf'\b{re.escape(solv)}\b'
                if re.search(pattern, all_text_lower):
                    if solv not in separation_results.get("solvents", []):
                        separation_results.setdefault("solvents", []).append(solv)
            else:
                if solv in all_text_lower:
                    if solv not in separation_results.get("solvents", []):
                        separation_results.setdefault("solvents", []).append(solv)

        # Pattern 3: Extract selectivity values
        selectivity_matches = re.findall(r'selectivity[:\s]+(\d+\.?\d*)%?', all_text_lower)
        for match in selectivity_matches:
            try:
                val = float(match)
                if val not in separation_results.get("selectivities", []):
                    separation_results.setdefault("selectivities", []).append(val)
            except ValueError:
                pass

        logger.info(f"Separation Agent (collab): Found {len(separation_results.get('solvents', []))} solvents, "
                   f"{len(separation_results.get('sequences', []))} sequences")

        # P1: Create structured SeparationResult
        try:
            sep_result_obj = SeparationResult(
                sequences=separation_results.get("sequences", []),
                solvents=separation_results.get("solvents", []),
                selectivities=separation_results.get("selectivities", []),
                polymers=separation_results.get("polymers", []),
                temperature=separation_results.get("temperature", 80.0),
                best_sequence=separation_results.get("best_sequence") or (
                    separation_results.get("sequences", [[]])[0] if separation_results.get("sequences") else None
                ),
                best_solvent=separation_results.get("solvents", [None])[0] if separation_results.get("solvents") else None,
                algorithm_used=separation_results.get("algorithm_used"),
                raw_response=separation_results.get("raw_response"),
            )
            separation_results = sep_result_obj.model_dump()
        except Exception as e:
            logger.warning(f"Could not create SeparationResult schema: {e}")

        # Check if there are pending tool_calls
        result_messages = result.get("messages", [])
        has_pending_tools = False
        if result_messages:
            last_msg = result_messages[-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                has_pending_tools = True

        # Increment iteration counter and check for max iterations
        sep_iter = state.get("sep_iteration_count", 0) + 1
        max_sep_iter = 10  # Max iterations for separation agent

        if sep_iter >= max_sep_iter:
            logger.warning(f"Separation hit max iterations ({max_sep_iter}), forcing handoff")
            has_pending_tools = False  # Force final pass

        # If tools are pending, return dict (not Command) to let conditional edge route to tools
        # This prevents premature handoff before tools complete
        if has_pending_tools:
            logger.info(f"Separation: {len(last_msg.tool_calls)} tool calls pending, returning dict (iter {sep_iter})")
            return {
                **result,
                "sep_iteration_count": sep_iter,
                "separation_results": separation_results,  # Partial results
                "agent_timings": {
                    **state.get("agent_timings", {}),
                    "separation_iter": time.time(),
                },
            }

        # FINAL PASS: Tools have completed, validate and handoff
        logger.info(f"Separation: FINAL PASS (iter {sep_iter})")
        solvents_found = separation_results.get("solvents", [])

        # VALIDATION: Check solvents on FINAL pass
        if not solvents_found:
            logger.warning(f"Separation found no solvents on final pass - skipping TEA and going to aggregator")

            # Create handoff metrics for failed separation
            agent_start_time = state.get("agent_timings", {}).get("orchestrator", time.time())
            handoff_metrics_entry = create_handoff_metrics(
                from_agent="separation",
                to_agent="smart_aggregator",
                start_time=agent_start_time,
                tools_called=["plan_sequential_separation"],
                success=False,
                error_message="No solvents found in separation analysis",
                task_type="separation",
            )
            handoff_metrics_entry["query_summary"] = f"Separation of {len(shared_context.get('polymers', []))} polymers - NO SOLVENTS FOUND"

            # Skip TEA and go directly to aggregator with warning
            return Command(
                update={
                    **result,
                    "separation_results": separation_results,
                    "tea_results": {"error": "Skipped - no solvents from separation", "cost_per_kg": None},
                    "handoff_metrics": [handoff_metrics_entry],
                    "agent_timings": {
                        **state.get("agent_timings", {}),
                        "separation": time.time(),
                    },
                },
                goto="smart_aggregator"
            )

        # P2: Create task-oriented TEA request (reduces context bloat)
        tea_task = TEATaskRequest(
            solvents=separation_results.get("solvents", [])[:5],  # Top 5 solvents
            throughput_kg_hr=shared_context.get("throughput_kg_hr", 100.0),
            recovery_rate=0.95,
            include_capex=True,
            include_lca=False,
            compare_solvents=True,
            polymers=separation_results.get("polymers", []),
            temperature=separation_results.get("temperature", 80.0),
            best_sequence=separation_results.get("best_sequence"),
        )

        # P3: Create enhanced handoff metrics (consolidates P0 handoff_history)
        agent_start_time = state.get("agent_timings", {}).get("orchestrator", time.time())
        solvents_found = separation_results.get("solvents", [])

        handoff_metrics_entry = create_handoff_metrics(
            from_agent="separation",
            to_agent="tea_lca",
            start_time=agent_start_time,
            tools_called=["plan_sequential_separation"],
            success=bool(solvents_found),
            task_type="separation",
            context_size=len(str(separation_results)) if separation_results else 0,
        )
        # Add query summary to metrics for traceability
        handoff_metrics_entry["query_summary"] = f"Separation of {len(shared_context.get('polymers', []))} polymers, found {len(solvents_found)} solvents"

        # Create pending handoff with task params for TEA agent
        pending_handoff = {
            "from_agent": "separation",
            "to_agent": "tea_lca",
            "task_params": tea_task.model_dump(),
        }

        # Return Command for dynamic routing to TEA agent
        return Command(
            update={
                **result,
                "separation_results": separation_results,
                "current_specialist_index": state.get("current_specialist_index", 0) + 1,
                "pending_handoff": pending_handoff,
                "handoff_metrics": [handoff_metrics_entry],
                "agent_timings": {
                    **state.get("agent_timings", {}),
                    "separation": time.time(),
                },
            },
            goto="collab_tea_agent"
        )

    # Non-collaborative mode: return dict
    return result


async def collab_tea_agent_node(state: MultiAgentState, sql_agent_node) -> Command:
    """
    TEA specialist with Command-based handoff for collaboration.

    P0/P1 Changes:
    - Returns Command object for dynamic routing
    - Uses TEAResult Pydantic schema for structured output
    - Consumes SeparationResult from upstream agent

    When in collaboration mode:
    - Reads separation_results from state (SeparationResult schema)
    - Runs TEA for each unique solvent
    - Returns Command(goto="smart_aggregator") for final aggregation
    """
    is_collaborative = state.get("collaboration_mode") is not None
    separation_results = state.get("separation_results")
    shared_context = state.get("shared_context", {})

    # Set categories for TEA
    state_copy = dict(state)
    state_copy["selected_categories"] = ["economics", "strap", "visualization", "solvent_properties"]

    if is_collaborative and separation_results:
        # P2: Get task request from handoff payload or create from separation results
        pending_handoff = state.get("pending_handoff", {})
        task_params = pending_handoff.get("task_params", {})

        if task_params:
            # Use task-oriented request from handoff (P2: reduces context)
            tea_task = TEATaskRequest(**task_params)
            solvents = tea_task.solvents
        else:
            # Fallback: build from separation results
            solvents = separation_results.get("solvents", [])
            throughput = shared_context.get("throughput_kg_hr", 100.0)

            if not solvents:
                solvents = ["xylene", "cyclohexane", "hexane"]
                logger.warning(f"No solvents from separation, using defaults: {solvents}")

            tea_task = TEATaskRequest(
                solvents=solvents[:5],
                throughput_kg_hr=throughput,
                recovery_rate=0.95,
                include_capex=True,
                compare_solvents=True,
            )

        # P2: Use task's to_instruction() for consistent, minimal context
        context_message = HumanMessage(content=tea_task.to_instruction())
        # Append context to messages
        messages = list(state_copy.get("messages", []))
        messages.append(context_message)
        state_copy["messages"] = messages

    # Execute TEA agent
    result = await sql_agent_node(state_copy)

    if is_collaborative:
        # Check if there are pending tool_calls
        result_messages = result.get("messages", [])
        has_pending_tools = False
        if result_messages:
            last_msg = result_messages[-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                has_pending_tools = True

        # Increment iteration counter and check for max iterations
        tea_iter = state.get("tea_iteration_count", 0) + 1
        max_tea_iter = 10  # Max iterations for TEA agent

        if tea_iter >= max_tea_iter:
            logger.warning(f"TEA hit max iterations ({max_tea_iter}), forcing handoff")
            has_pending_tools = False  # Force final pass

        # If tools are pending, return dict (not Command) to let conditional edge route to tools
        if has_pending_tools:
            logger.info(f"TEA: {len(last_msg.tool_calls)} tool calls pending, returning dict (iter {tea_iter})")
            return {
                **result,
                "tea_iteration_count": tea_iter,
                "agent_timings": {
                    **state.get("agent_timings", {}),
                    "tea_iter": time.time(),
                },
            }

        # FINAL PASS: Tools have completed, extract and process results
        logger.info(f"TEA: FINAL PASS (iter {tea_iter})")

        # Extract structured TEA results from ALL messages
        messages = result.get("messages", [])
        all_text = ""
        tool_outputs = []

        for msg in messages:
            if isinstance(msg, AIMessage):
                all_text += msg.content + "\n"
            elif hasattr(msg, 'content') and isinstance(msg.content, str):
                # Tool messages - log them for debugging
                tool_outputs.append(msg.content)
                all_text += msg.content + "\n"

        # Log tool outputs for debugging cost extraction issues
        if tool_outputs:
            for i, output in enumerate(tool_outputs[:2]):  # Log first 2 tool outputs
                sample = output[:500].replace('\n', ' ')
                logger.info(f"TEA ToolMessage {i+1} ({len(output)} chars): {sample}...")

        tea_results = parse_tea_results(all_text)
        tea_results["solvents_analyzed"] = separation_results.get("solvents", []) if separation_results else []

        # Enhanced cost extraction with more patterns
        if not tea_results.get("cost_per_kg"):
            # Comprehensive patterns for cost extraction
            cost_patterns = [
                # TEA tool output format: "Cost per kg polymer: $X.XXXX/kg"
                r'cost\s+per\s+kg\s+polymer[:\s]*\$?(\d+\.?\d*)',
                # Standard patterns
                r'msp[:\s]*\$?(\d+\.?\d*)',
                r'cost[:\s]*\$?(\d+\.?\d*)\s*/?\s*kg',
                r'\$(\d+\.?\d*)\s*/\s*kg',
                # Processing cost patterns
                r'processing\s+cost[:\s]*\$?(\d+\.?\d*)',
                r'solvent\s+recovery\s+cost[:\s]*\$?(\d+\.?\d*)',
                # Total cost patterns
                r'total\s+cost[:\s]*\$?(\d+\.?\d*)',
                r'operating\s+cost[:\s]*\$?(\d+\.?\d*)\s*/?\s*kg',
                # Per kg patterns with different formats
                r'(\d+\.?\d*)\s*\$\s*/\s*kg',
                r'(\d+\.?\d*)\s*usd\s*/\s*kg',
                r'(\d+\.?\d*)\s*per\s+kg',
            ]
            for pattern in cost_patterns:
                match = re.search(pattern, all_text.lower())
                if match:
                    try:
                        cost = float(match.group(1))
                        if 0.01 < cost < 1000:  # Wider sanity range
                            tea_results["cost_per_kg"] = cost
                            logger.info(f"TEA cost extracted: ${cost}/kg (pattern: {pattern[:30]}...)")
                            break
                    except ValueError:
                        pass

        # Log if no cost found after all attempts
        if not tea_results.get("cost_per_kg"):
            logger.warning(f"TEA: No cost_per_kg extracted from {len(all_text)} chars of output")
            # Log a sample of the text for debugging
            sample = all_text[:300].replace('\n', ' ')
            logger.debug(f"TEA output sample: {sample}...")

        logger.info(f"TEA Agent (collab): cost_per_kg={tea_results.get('cost_per_kg')}, "
                   f"payback={tea_results.get('payback_years')}")

        try:
            tea_result_obj = TEAResult(
                msp_values=tea_results.get("msp_values", {}),
                best_solvent=tea_results.get("best_solvent"),
                cost_per_kg=tea_results.get("cost_per_kg"),
                total_capex=tea_results.get("total_capex"),
                total_opex=tea_results.get("total_opex"),
                payback_years=tea_results.get("payback_years"),
                cost_breakdown=tea_results.get("cost_breakdown", {}),
                solvents_analyzed=tea_results.get("solvents_analyzed", []),
                throughput_kg_hr=shared_context.get("throughput_kg_hr"),
                raw_response=tea_results.get("raw_response"),
            )
            tea_results = tea_result_obj.model_dump()
        except Exception as e:
            logger.warning(f"Could not create TEAResult schema: {e}")

        # P2: Create task-oriented aggregator request
        # Handle None values safely
        best_seq = separation_results.get("best_sequence") if separation_results else None
        sep_summary = None
        if best_seq:
            sep_summary = f"Best sequence: {' → '.join(best_seq[:5])}"

        aggregator_task = AggregatorTaskRequest(
            separation_summary=sep_summary,
            tea_summary=f"Cost: ${tea_results.get('cost_per_kg', 'N/A')}/kg" if tea_results.get('cost_per_kg') else None,
            best_solvent=tea_results.get("best_solvent"),
            best_sequence=best_seq,
            cost_per_kg=tea_results.get("cost_per_kg"),
            original_query=shared_context.get("original_query"),
        )

        # P3: Create enhanced handoff metrics (consolidates P0 handoff_history)
        sep_timing = state.get("agent_timings", {}).get("separation", time.time())
        cost_found = tea_results.get("cost_per_kg")

        handoff_metrics_entry = create_handoff_metrics(
            from_agent="tea_lca",
            to_agent="smart_aggregator",
            start_time=sep_timing,
            tools_called=["analyze_solvent_recovery_tea", "compare_solvents_tea_lca"],
            success=bool(cost_found),
            task_type="tea",
            context_size=len(str(tea_results)) if tea_results else 0,
        )
        # Add query summary to metrics for traceability
        handoff_metrics_entry["query_summary"] = f"TEA analysis complete, cost=${cost_found}/kg" if cost_found else "TEA analysis complete, no cost extracted"

        # Create pending handoff with task params for aggregator
        pending_handoff = {
            "from_agent": "tea_lca",
            "to_agent": "smart_aggregator",
            "task_params": aggregator_task.model_dump(),
        }

        # Return Command for dynamic routing to aggregator
        return Command(
            update={
                **result,
                "tea_results": tea_results,
                "current_specialist_index": state.get("current_specialist_index", 0) + 1,
                "pending_handoff": pending_handoff,
                "handoff_metrics": [handoff_metrics_entry],
                "agent_timings": {
                    **state.get("agent_timings", {}),
                    "tea": time.time(),
                },
            },
            goto="smart_aggregator"
        )

    # Non-collaborative mode: return dict
    return result


# ============================================================
# SMART AGGREGATOR FOR COLLABORATION RESULTS
# ============================================================

async def smart_aggregator_node(state: MultiAgentState) -> dict:
    """
    Combine results from multiple specialists into unified recommendation.

    P1 Changes:
    - Uses SeparationResult and TEAResult schemas for structured access
    - Includes handoff history in output for traceability

    P2 Changes:
    - Uses AggregatorTaskRequest for minimal context access
    - Extracts key metrics from task_params instead of parsing messages

    P3 Changes:
    - Finalizes execution trace with full metrics
    - Includes handoff_metrics summary in output
    - Tracks total execution time across all agents

    For Separation + TEA collaboration:
    - Merges separation sequences with cost data
    - Ranks by cost-effectiveness
    - Provides integrated recommendation
    """
    collaboration_mode = state.get("collaboration_mode")
    separation_results = state.get("separation_results")
    tea_results = state.get("tea_results")
    messages = state.get("messages", [])
    start_time = state.get("specialist_start_time")
    # P3: Get enhanced tracking data (handoff_metrics replaces handoff_history)
    trace_id = state.get("trace_id")
    handoff_metrics = state.get("handoff_metrics", [])
    agent_timings = state.get("agent_timings", {})

    # P2: Get task-oriented request from handoff
    pending_handoff = state.get("pending_handoff", {})
    task_params = pending_handoff.get("task_params", {})

    elapsed = time.time() - start_time if start_time else 0

    logger.info(f"Smart Aggregator: mode={collaboration_mode}, elapsed={elapsed:.2f}s, "
                f"handoffs={len(handoff_metrics)}, task_params={bool(task_params)}")

    # If not in collaboration mode, pass through
    if not collaboration_mode:
        return {
            "multi_agent_active": False,
            "active_specialist": None,
            "aggregation_required": False,
        }

    # Build integrated response
    output_parts = []

    if collaboration_mode == "separation_tea_lca" or collaboration_mode == "separation_tea":
        output_parts.append("# Integrated Separation + Economic Analysis\n")
        output_parts.append(f"*Analysis completed in {elapsed:.1f}s using multi-agent collaboration*\n\n")

        # P1: Use structured schema access (with dict fallback)
        # Separation Summary
        if separation_results:
            output_parts.append("## Separation Analysis Summary\n")
            polymers = separation_results.get("polymers", [])
            solvents = separation_results.get("solvents", [])
            best_sequence = separation_results.get("best_sequence", [])
            best_solvent = separation_results.get("best_solvent")
            algorithm = separation_results.get("algorithm_used", "exhaustive")

            output_parts.append(f"- **Polymers analyzed:** {', '.join(polymers) if polymers else 'N/A'}\n")
            output_parts.append(f"- **Solvents identified:** {', '.join(solvents) if solvents else 'N/A'}\n")

            sequences = separation_results.get("sequences", [])
            if sequences:
                output_parts.append(f"- **Sequences evaluated:** {len(sequences)}\n")

            if best_sequence:
                output_parts.append(f"- **Best sequence:** {' → '.join(best_sequence)}\n")

            if algorithm == "greedy":
                output_parts.append(f"- **Algorithm:** Greedy (O(n²) for {len(polymers)} polymers)\n")

            output_parts.append("\n")

        # TEA Summary (P1: Structured access)
        if tea_results:
            output_parts.append("## Economic Analysis Summary\n")
            cost_per_kg = tea_results.get("cost_per_kg")
            capex = tea_results.get("total_capex")
            opex = tea_results.get("total_opex")
            payback = tea_results.get("payback_years")
            best_tea_solvent = tea_results.get("best_solvent")
            msp_values = tea_results.get("msp_values", {})

            if best_tea_solvent:
                output_parts.append(f"- **Best solvent (cost):** {best_tea_solvent}\n")
            if cost_per_kg:
                output_parts.append(f"- **Estimated cost:** ${cost_per_kg:.2f}/kg polymer processed\n")
            if capex:
                output_parts.append(f"- **Capital investment (CAPEX):** ${capex:,.0f}\n")
            if opex:
                output_parts.append(f"- **Operating cost (OPEX):** ${opex:,.0f}/yr\n")
            if payback:
                output_parts.append(f"- **Payback period:** {payback:.1f} years\n")

            # Show MSP comparison if multiple solvents analyzed
            if len(msp_values) > 1:
                output_parts.append("\n**Cost by Solvent:**\n")
                for solvent, msp in sorted(msp_values.items(), key=lambda x: x[1]):
                    output_parts.append(f"  - {solvent}: ${msp:.2f}/kg\n")

            output_parts.append("\n")

        # Integrated Recommendation
        output_parts.append("## Integrated Recommendation\n")

        # P2: Try to get key metrics from task_params first (faster, cleaner)
        best_solvent = task_params.get("best_solvent")
        best_sequence = task_params.get("best_sequence")
        cost = task_params.get("cost_per_kg")

        # Fallback to full results if task_params incomplete
        if not best_solvent and tea_results:
            best_solvent = tea_results.get("best_solvent")
        if not best_sequence and separation_results:
            best_sequence = separation_results.get("best_sequence")
        if not cost and tea_results:
            cost = tea_results.get("cost_per_kg")
        if not best_solvent and separation_results:
            solvents = separation_results.get("solvents", [])
            best_solvent = solvents[0] if solvents else None

        if best_solvent or best_sequence:
            cost_str = f"${cost:.2f}" if cost else "N/A"
            seq_str = f" with sequence **{' → '.join(best_sequence[:5])}**" if best_sequence else ""
            output_parts.append(
                f"Based on combined separation selectivity and economic analysis, "
                f"the recommended approach uses **{best_solvent or 'N/A'}**{seq_str} "
                f"with an estimated processing cost of **{cost_str}/kg** "
                f"(see detailed breakdown above).\n"
            )
        else:
            output_parts.append("See individual specialist outputs above for details.\n")

        # P3: Include execution metrics summary (replaces P0 handoff_history trace)
        if handoff_metrics and trace_id:
            output_parts.append(f"\n<details>\n<summary>Execution Trace (trace_id: {trace_id})</summary>\n\n")
            total_handoff_time = sum(
                m.get("duration_ms", 0) or 0 for m in handoff_metrics
            )
            output_parts.append(f"- **Total handoff time:** {total_handoff_time:.0f}ms\n")
            output_parts.append(f"- **Handoffs:** {len(handoff_metrics)}\n")
            for i, metric in enumerate(handoff_metrics):
                duration = metric.get("duration_ms", 0) or 0
                success = "✓" if metric.get("success") else "✗"
                output_parts.append(
                    f"  {i+1}. {metric.get('from_agent')} → {metric.get('to_agent')} "
                    f"({duration:.0f}ms) {success}\n"
                )
            output_parts.append("\n</details>\n")

        output_parts.append("\n---\n")
        output_parts.append("*This analysis was performed by the DISSOLVE multi-agent system, "
                          "combining Separation Planning and TEA/LCA specialists.*\n")

    elif collaboration_mode == "separation_literature":
        output_parts.append("# Integrated Separation + Literature Analysis\n")
        output_parts.append(f"*Analysis completed in {elapsed:.1f}s*\n\n")
        # Similar structure for separation + literature
        output_parts.append("See specialist outputs above for detailed findings.\n")

    # P3: Log execution trace for debugging
    if trace_id:
        logger.info(f"Execution trace {trace_id}: {len(handoff_metrics)} handoffs, "
                   f"timings={agent_timings}")

    # Create aggregated message
    aggregated_content = "".join(output_parts)
    aggregated_message = AIMessage(content=aggregated_content)

    # P3: Finalize execution trace in return state
    final_trace = {
        "trace_id": trace_id,
        "total_elapsed_s": elapsed,
        "handoffs_count": len(handoff_metrics),
        "agent_timings": agent_timings,
        "completed_at": datetime.now().isoformat(),
    }

    return {
        "messages": messages + [aggregated_message],
        "multi_agent_active": False,
        "active_specialist": None,
        "aggregation_required": False,
        "collaboration_mode": None,  # Reset for next query
        # P3: Include finalized trace in state for external access
        "execution_trace": final_trace,
    }


# ============================================================
# ROUTER NODE (Updated for Integrated Path)
# ============================================================

async def multi_agent_router_node(state: dict) -> dict:
    """
    Enhanced router node with complexity scoring, specialist dispatch,
    and integrated collaboration detection.

    Performance: ~1-2ms (rule-based, no LLM call)
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "complexity": 3,
            "path": "standard",
            "specialist": None,
            "routing_reason": "No messages - defaulting to standard",
            "multi_agent_active": False
        }

    # Find the last human message
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break

    if not last_human_message:
        return {
            "complexity": 3,
            "path": "standard",
            "specialist": None,
            "routing_reason": "No human message found",
            "multi_agent_active": False
        }

    query = last_human_message.content if hasattr(last_human_message, 'content') else str(last_human_message)

    # Get routing decision
    decision = enhanced_complexity_router(query)

    logger.info(f"Router: complexity={decision.complexity}, path={decision.path}, "
                f"specialist={decision.specialist}, collaboration={decision.collaboration_specialists}, "
                f"reason={decision.reason}")

    return {
        "complexity": decision.complexity,
        "path": decision.path,
        "specialist": decision.specialist,
        "routing_reason": decision.reason,
        "selected_categories": decision.categories if decision.categories else None,
        "multi_agent_active": decision.path in ["specialist", "integrated"],
        "active_specialist": decision.specialist,
        "specialist_start_time": time.time() if decision.path in ["specialist", "integrated"] else None,
        # NEW: Collaboration fields
        "collaboration_specialists": decision.collaboration_specialists,
        "collaboration_mode": "_".join(decision.collaboration_specialists) if decision.collaboration_specialists else None,
    }


# ============================================================
# RESULTS AGGREGATOR (Legacy - for single specialists)
# ============================================================

async def results_aggregator_node(state: MultiAgentState) -> dict:
    """
    Aggregate results from specialist agents.

    For single-specialist queries, this just passes through.
    For multi-specialist queries, routes to smart_aggregator.
    """
    messages = state.get("messages", [])
    specialist = state.get("specialist")
    start_time = state.get("specialist_start_time")
    collaboration_mode = state.get("collaboration_mode")

    elapsed = time.time() - start_time if start_time else 0

    logger.info(f"Aggregator: specialist={specialist}, collaboration={collaboration_mode}, elapsed={elapsed:.2f}s")

    # If in collaboration mode, the smart aggregator handles this
    if collaboration_mode:
        return await smart_aggregator_node(state)

    # Simple pass-through for single specialists
    return {
        "multi_agent_active": False,
        "active_specialist": None
    }


# ============================================================
# BUILD MULTI-AGENT GRAPH (Updated with Integrated Path)
# ============================================================

def build_multi_agent_graph(
    sql_agent_node,  # Main agent node from agent_sql_final
    async_tool_node_class,  # AsyncToolNode class
    all_tools: list,
    tool_categories: dict,
    llm_factory  # Function to create LLM with tools
):
    """
    Build the multi-agent StateGraph with integrated collaboration support.

    Args:
        sql_agent_node: The main agent node function
        async_tool_node_class: AsyncToolNode class for tool execution
        all_tools: Complete list of all tools
        tool_categories: TOOL_CATEGORIES dict
        llm_factory: Function(tools, system_prompt) -> LLM with tools bound

    Returns:
        Compiled StateGraph
    """
    # Initialize tool subsets
    initialize_tool_subsets(tool_categories, all_tools)

    builder = StateGraph(MultiAgentState)

    # ========== NODES ==========

    # Router node (runs first)
    builder.add_node("router", multi_agent_router_node)

    # Fast path agent (uses subset of tools)
    builder.add_node("fast_agent", sql_agent_node)
    builder.add_node("fast_tools", async_tool_node_class(FAST_PATH_TOOLS or all_tools[:20]))

    # Standard path agent (uses all tools via existing router)
    builder.add_node("standard_agent", sql_agent_node)
    builder.add_node("standard_tools", async_tool_node_class(all_tools))

    # ========== SINGLE SPECIALISTS ==========

    # Separation specialist (single mode)
    async def separation_agent_node(state):
        state_copy = dict(state)
        state_copy["selected_categories"] = ["separation", "dissolution", "solvent_properties", "visualization"]
        return await sql_agent_node(state_copy)

    builder.add_node("separation_agent", separation_agent_node)
    builder.add_node("separation_tools", async_tool_node_class(SEPARATION_TOOLS or all_tools))

    # TEA/LCA specialist (single mode)
    async def tea_agent_node(state):
        state_copy = dict(state)
        state_copy["selected_categories"] = ["economics", "strap", "visualization", "solvent_properties"]
        return await sql_agent_node(state_copy)

    builder.add_node("tea_agent", tea_agent_node)
    builder.add_node("tea_tools", async_tool_node_class(TEA_LCA_TOOLS or all_tools))

    # Literature specialist (single mode)
    async def literature_agent_node(state):
        state_copy = dict(state)
        state_copy["selected_categories"] = ["literature", "rag"]
        return await sql_agent_node(state_copy)

    builder.add_node("literature_agent", literature_agent_node)
    builder.add_node("literature_tools", async_tool_node_class(LITERATURE_TOOLS or all_tools))

    # ========== INTEGRATED COLLABORATION PATH ==========

    # Integrated orchestrator (initializes collaboration)
    builder.add_node("integrated_orchestrator", integrated_orchestrator_node)

    # Collaboration-aware separation agent
    async def collab_separation_node(state):
        return await collab_separation_agent_node(state, sql_agent_node)

    builder.add_node("collab_separation_agent", collab_separation_node)
    builder.add_node("collab_separation_tools", async_tool_node_class(SEPARATION_TOOLS or all_tools))

    # Collaboration-aware TEA agent
    async def collab_tea_node(state):
        return await collab_tea_agent_node(state, sql_agent_node)

    builder.add_node("collab_tea_agent", collab_tea_node)
    builder.add_node("collab_tea_tools", async_tool_node_class(TEA_LCA_TOOLS or all_tools))

    # Smart aggregator for collaboration results
    builder.add_node("smart_aggregator", smart_aggregator_node)

    # Legacy aggregator for single specialists
    builder.add_node("aggregator", results_aggregator_node)

    # ========== ROUTING ==========

    # Specialist name to node name mapping
    SPECIALIST_TO_NODE = {
        "separation": "separation_agent",
        "tea_lca": "tea_agent",
        "literature": "literature_agent"
    }

    def route_by_path(state: MultiAgentState) -> str:
        """Route to appropriate agent based on complexity decision."""
        path = state.get("path", "standard")
        specialist = state.get("specialist")

        if path == "integrated":
            return "integrated_orchestrator"
        elif path == "fast":
            return "fast_agent"
        elif path == "specialist" and specialist:
            return SPECIALIST_TO_NODE.get(specialist, "standard_agent")
        else:
            return "standard_agent"

    # ========== EDGES ==========

    # Start -> Router
    builder.add_edge(START, "router")

    # Router -> Agent (based on path)
    builder.add_conditional_edges(
        "router",
        route_by_path,
        {
            "fast_agent": "fast_agent",
            "standard_agent": "standard_agent",
            "separation_agent": "separation_agent",
            "tea_agent": "tea_agent",
            "literature_agent": "literature_agent",
            "integrated_orchestrator": "integrated_orchestrator",
        }
    )

    # Should continue function
    def should_continue(state) -> Literal["continue", "end"]:
        """Determine if agent should continue or end."""
        messages = state.get("messages", [])
        current_iter = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 15)

        if current_iter >= max_iter:
            return "end"

        if not messages:
            return "end"

        try:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "continue"
        except (IndexError, TypeError, AttributeError):
            return "end"

        return "end"

    # Fast path loop
    builder.add_conditional_edges(
        "fast_agent",
        should_continue,
        {"continue": "fast_tools", "end": END}
    )
    builder.add_edge("fast_tools", "fast_agent")

    # Standard path loop
    builder.add_conditional_edges(
        "standard_agent",
        should_continue,
        {"continue": "standard_tools", "end": END}
    )
    builder.add_edge("standard_tools", "standard_agent")

    # Single specialist paths -> Aggregator -> END
    for specialist in ["separation", "tea", "literature"]:
        builder.add_conditional_edges(
            f"{specialist}_agent",
            should_continue,
            {"continue": f"{specialist}_tools", "end": "aggregator"}
        )
        builder.add_edge(f"{specialist}_tools", f"{specialist}_agent")

    builder.add_edge("aggregator", END)

    # ========== INTEGRATED COLLABORATION EDGES (P0: Command-based routing) ==========

    # Integrated orchestrator -> Collab Separation Agent
    builder.add_edge("integrated_orchestrator", "collab_separation_agent")

    # P0: Collab Separation Agent with Command-aware routing
    # When agent returns Command(goto="collab_tea_agent"), LangGraph routes automatically
    # When agent needs tools, we route to tools first
    def route_collab_sep(state) -> Literal["collab_separation_tools", "collab_tea_agent", "smart_aggregator", "__end__"]:
        """Route based on tool calls or Command. Command.goto takes precedence."""
        messages = state.get("messages", [])
        if not messages:
            return "collab_tea_agent"

        try:
            last_message = messages[-1]
            # If last message has tool calls, route to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "collab_separation_tools"
        except (IndexError, TypeError, AttributeError):
            pass

        # Default: hand off to TEA (Command will override this if returned)
        return "collab_tea_agent"

    builder.add_conditional_edges(
        "collab_separation_agent",
        route_collab_sep,
        {
            "collab_separation_tools": "collab_separation_tools",
            "collab_tea_agent": "collab_tea_agent",
            "smart_aggregator": "smart_aggregator",
            "__end__": END,
        }
    )
    builder.add_edge("collab_separation_tools", "collab_separation_agent")

    # P0: Collab TEA Agent with Command-aware routing
    def route_collab_tea(state) -> Literal["collab_tea_tools", "smart_aggregator", "__end__"]:
        """Route based on tool calls or Command."""
        messages = state.get("messages", [])
        if not messages:
            return "smart_aggregator"

        try:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "collab_tea_tools"
        except (IndexError, TypeError, AttributeError):
            pass

        # Default: aggregate (Command will override if returned)
        return "smart_aggregator"

    builder.add_conditional_edges(
        "collab_tea_agent",
        route_collab_tea,
        {
            "collab_tea_tools": "collab_tea_tools",
            "smart_aggregator": "smart_aggregator",
            "__end__": END,
        }
    )
    builder.add_edge("collab_tea_tools", "collab_tea_agent")

    # Smart aggregator -> END
    builder.add_edge("smart_aggregator", END)

    # ========== COMPILE ==========

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("✅ Multi-Agent Graph compiled successfully!")
    logger.info(f"  Paths: fast, standard, specialist (separation, tea, literature), INTEGRATED")
    logger.info(f"  Tool subsets: fast={len(FAST_PATH_TOOLS)}, sep={len(SEPARATION_TOOLS)}, "
                f"tea={len(TEA_LCA_TOOLS)}, lit={len(LITERATURE_TOOLS)}")
    logger.info(f"  Collaboration: separation -> tea_lca (iterative handoff)")

    return graph


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    'enhanced_complexity_router',
    'RoutingDecision',
    'MultiAgentState',
    'build_multi_agent_graph',
    'initialize_tool_subsets',
    'multi_agent_router_node',
    'integrated_orchestrator_node',
    'smart_aggregator_node',
    'collab_separation_agent_node',
    'collab_tea_agent_node',
    'SEPARATION_PLANNER_PROMPT',
    'TEA_LCA_ANALYST_PROMPT',
    'LITERATURE_RESEARCHER_PROMPT',
    'FAST_PATH_TOOLS',
    'SEPARATION_TOOLS',
    'TEA_LCA_TOOLS',
    'LITERATURE_TOOLS',
]
