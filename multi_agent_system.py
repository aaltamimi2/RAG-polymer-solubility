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
import os

# Import structured schemas for inter-agent communication
from agent_schemas import (
    SeparationResult,
    TEAResult,
    LiteratureResult,
    HandoffPayload,
    SharedContext,
    SeparationStep,
    parse_to_separation_result,
    parse_to_tea_result,
    # P2: Task-oriented handoff schemas
    TEATaskRequest,
    SeparationTaskRequest,
    LiteratureTaskRequest,
    AggregatorTaskRequest,
    # P3: Enhanced tracking schemas
    HandoffMetrics,
    ExecutionTrace,
    # P0 Enhancement: Review/Revision loop
    ReviewerFeedback,
)
import uuid

logger = logging.getLogger(__name__)


def filter_result_for_collab(result: dict) -> dict:
    """
    Filter base agent result to only include essential keys for collaboration nodes.

    This prevents concurrent update conflicts when multiple nodes try to update the same keys.
    We use an allow-list approach to be safe - only messages are passed through.
    All other state (iteration tracking, timings, etc.) is managed by the collaboration nodes themselves.
    """
    # Only pass messages - everything else is managed by collaboration nodes
    allowed_keys = {"messages"}
    return {k: v for k, v in result.items() if k in allowed_keys}


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
        "separate", "separating", "separation", "multilayer", "multi-layer", "sequence"
    ])
    has_cost_keyword = any(w in query_lower for w in [
        "cost", "economic", "tea", "capex", "opex", "payback", "msp", "cheap", "expensive"
    ])

    # Separation + Literature integration - check this BEFORE 2-way checks
    has_literature_keyword = any(w in query_lower for w in [
        "literature", "research", "papers", "studies", "publications",
        "rag", "knowledgebase", "indexed", "strap", "verify"
    ])

    # 3-WAY COLLABORATION: Separation + Literature + TEA
    # This takes priority when all three domains are needed
    if has_separation_keyword and has_literature_keyword and has_cost_keyword:
        return RoutingDecision(
            complexity=5,
            path="integrated",
            specialist=None,
            categories=["separation", "dissolution", "literature", "rag", "economics", "strap", "visualization", "solvent_properties"],
            reason="Integrated Separation + Literature + TEA analysis detected",
            collaboration_specialists=["separation", "literature", "tea_lca"]
        )

    # 2-WAY: Separation + TEA/LCA
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

    # 2-WAY: Separation + Literature
    if has_separation_keyword and has_literature_keyword:
        return RoutingDecision(
            complexity=5,
            path="integrated",
            specialist=None,
            categories=["separation", "dissolution", "literature", "rag"],
            reason="Integrated Separation + Literature research detected",
            collaboration_specialists=["separation", "literature"]
        )

    # Literature-first queries with polymer/solvent context (search first, then analyze)
    has_deinking_keyword = any(w in query_lower for w in [
        "deinking", "ink removal", "printed", "printing", "surfactant",
        "flexographic", "gravure", "coating removal"
    ])
    if has_deinking_keyword:
        return RoutingDecision(
            complexity=4,
            path="specialist",
            specialist="literature",
            categories=["literature", "rag"],
            reason="Deinking/printed plastics literature query detected"
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
        "search literature", "search the literature", "find papers", "research on",
        "publications about", "what does the literature", "scholarly articles",
        "peer-reviewed", "web of science", "google scholar", "search rag",
        "ask literature", "what papers", "recent research", "state of the art",
        "hansen solubility parameter", "in the literature", "from literature",
        "literature search", "indexed papers", "knowledgebase"
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
# P1: CHECKPOINTER CONFIGURATION
# ============================================================

class CheckpointerConfig:
    """Configuration for persistent checkpointing.

    P1 Enhancement: Supports PostgreSQL, Redis, or in-memory checkpointing.

    Environment Variables:
        CHECKPOINTER_TYPE: "memory" (default), "postgres", or "redis"
        DATABASE_URL: PostgreSQL connection string (for postgres)
        REDIS_URL: Redis connection string (for redis)
    """

    @staticmethod
    def get_checkpointer():
        """Get the appropriate checkpointer based on environment configuration."""
        checkpointer_type = os.environ.get("CHECKPOINTER_TYPE", "memory").lower()

        if checkpointer_type == "postgres":
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                database_url = os.environ.get("DATABASE_URL")
                if not database_url:
                    logger.warning("DATABASE_URL not set, falling back to MemorySaver")
                    return MemorySaver()
                logger.info("Using PostgreSQL checkpointer for persistent state")
                return PostgresSaver.from_conn_string(database_url)
            except ImportError:
                logger.warning("langgraph-checkpoint-postgres not installed, falling back to MemorySaver")
                return MemorySaver()
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL checkpointer: {e}, falling back to MemorySaver")
                return MemorySaver()

        elif checkpointer_type == "redis":
            try:
                from langgraph.checkpoint.redis import RedisSaver
                redis_url = os.environ.get("REDIS_URL")
                if not redis_url:
                    logger.warning("REDIS_URL not set, falling back to MemorySaver")
                    return MemorySaver()
                logger.info("Using Redis checkpointer for persistent state")
                return RedisSaver.from_conn_string(redis_url)
            except ImportError:
                logger.warning("langgraph-checkpoint-redis not installed, falling back to MemorySaver")
                return MemorySaver()
            except Exception as e:
                logger.warning(f"Failed to initialize Redis checkpointer: {e}, falling back to MemorySaver")
                return MemorySaver()

        else:
            # Default: in-memory
            logger.info("Using in-memory checkpointer (not persistent)")
            return MemorySaver()


# ============================================================
# P2: CROSS-SESSION STORE FOR CACHING
# ============================================================

class SessionStore:
    """
    P2 Enhancement: Cross-session store for caching common results.

    Provides:
    - Caching separation results for common polymer sets
    - Storing frequently used solvent properties
    - Sharing configurations across sessions

    Usage:
        store = SessionStore()
        store.cache_separation("PE,PP,PS", results)
        cached = store.get_cached_separation("PE,PP,PS")
    """

    _instance = None
    _cache: Dict[str, Any] = {}
    _cache_times: Dict[str, float] = {}
    _ttl_seconds: int = 3600  # 1 hour default TTL

    def __new__(cls):
        """Singleton pattern for shared store."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._cache = {}
            cls._cache_times = {}
        return cls._instance

    @classmethod
    def _normalize_polymers(cls, polymers: str) -> str:
        """Normalize polymer string (sorted, uppercase)."""
        if "," in polymers:
            parts = sorted([p.strip().upper() for p in polymers.split(",")])
            return ",".join(parts)
        return polymers.strip().upper()

    @classmethod
    def _make_key(cls, prefix: str, identifier: str) -> str:
        """Create a cache key."""
        return f"{prefix}:{identifier}"

    @classmethod
    def cache_separation(cls, polymers: str, results: Dict[str, Any], temperature: float = 80.0) -> None:
        """Cache separation results for a polymer set."""
        normalized = cls._normalize_polymers(polymers)
        key = cls._make_key("sep", f"{normalized}@{temperature}")
        cls._cache[key] = results
        cls._cache_times[key] = time.time()
        logger.debug(f"SessionStore: Cached separation for {key}")

    @classmethod
    def get_cached_separation(cls, polymers: str, temperature: float = 80.0, max_age_seconds: int = None) -> Optional[Dict[str, Any]]:
        """Get cached separation results if available and not stale."""
        normalized = cls._normalize_polymers(polymers)
        key = cls._make_key("sep", f"{normalized}@{temperature}")
        if key not in cls._cache:
            return None

        # Check staleness
        max_age = max_age_seconds or cls._ttl_seconds
        cached_time = cls._cache_times.get(key, 0)
        if time.time() - cached_time > max_age:
            logger.debug(f"SessionStore: Cache expired for {key}")
            del cls._cache[key]
            del cls._cache_times[key]
            return None

        logger.debug(f"SessionStore: Cache hit for {key}")
        return cls._cache[key]

    @classmethod
    def cache_tea(cls, solvents: str, throughput: float, results: Dict[str, Any]) -> None:
        """Cache TEA results for a solvent set."""
        normalized = cls._normalize_polymers(solvents)  # Works for solvents too
        key = cls._make_key("tea", f"{normalized}@{throughput}")
        cls._cache[key] = results
        cls._cache_times[key] = time.time()

    @classmethod
    def get_cached_tea(cls, solvents: str, throughput: float, max_age_seconds: int = None) -> Optional[Dict[str, Any]]:
        """Get cached TEA results if available."""
        normalized = cls._normalize_polymers(solvents)  # Works for solvents too
        key = cls._make_key("tea", f"{normalized}@{throughput}")
        if key not in cls._cache:
            return None

        max_age = max_age_seconds or cls._ttl_seconds
        cached_time = cls._cache_times.get(key, 0)
        if time.time() - cached_time > max_age:
            del cls._cache[key]
            del cls._cache_times[key]
            return None

        return cls._cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached data."""
        cls._cache.clear()
        cls._cache_times.clear()
        logger.info("SessionStore: Cache cleared")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(cls._cache),
            "separation_entries": sum(1 for k in cls._cache if k.startswith("sep:")),
            "tea_entries": sum(1 for k in cls._cache if k.startswith("tea:")),
            "oldest_entry_age_s": time.time() - min(cls._cache_times.values()) if cls._cache_times else 0,
        }


# ============================================================
# P2: POLYMER-SOLVENT KNOWLEDGE GRAPH
# ============================================================

class PolymerKnowledgeGraph:
    """
    P2 Enhancement: Domain-specific knowledge graph for polymer-solvent relationships.

    Provides:
    - Automatic inference of related polymers
    - Solvent compatibility checking
    - Safety constraint propagation
    - Common separation strategies

    Based on CRISPR-GPT pattern of domain knowledge integration.
    """

    # Polymer family groupings
    POLYMER_FAMILIES = {
        "polyolefins": ["PE", "LDPE", "HDPE", "PP", "LLDPE"],
        "polyesters": ["PET", "PBT", "PLA", "PBS", "PBAT"],
        "styrenics": ["PS", "ABS", "SAN", "HIPS"],
        "polyamides": ["PA6", "PA66", "Nylon6", "Nylon66"],
        "vinyl": ["PVC", "PVDC", "PVDF"],
        "engineering": ["PC", "PMMA", "POM"],
        "barrier": ["EVOH", "PVOH"],
        "biodegradable": ["PLA", "PHA", "PHB", "PBS", "PBAT"],
    }

    # Solvent-polymer compatibility matrix (simplified)
    # Higher score = better dissolution
    COMPATIBILITY = {
        "xylene": {"PE": 0.7, "LDPE": 0.8, "HDPE": 0.6, "PP": 0.7, "PS": 0.9, "ABS": 0.6},
        "toluene": {"PS": 0.95, "ABS": 0.7, "PMMA": 0.5, "PE": 0.4, "PP": 0.4},
        "cyclohexane": {"PE": 0.8, "LDPE": 0.85, "HDPE": 0.75, "PP": 0.8, "PS": 0.6},
        "thf": {"PVC": 0.9, "PMMA": 0.85, "PS": 0.7, "ABS": 0.8},
        "acetone": {"PMMA": 0.9, "ABS": 0.5, "PS": 0.3, "PVC": 0.4},
        "dmf": {"PVC": 0.7, "PMMA": 0.6, "PA6": 0.5, "PA66": 0.5},
        "nmp": {"PA6": 0.8, "PA66": 0.8, "PVC": 0.6},
        "dcm": {"PC": 0.9, "PMMA": 0.8, "PS": 0.85, "PVC": 0.7},
        "chloroform": {"PC": 0.95, "PMMA": 0.85, "PS": 0.9},
        "formic_acid": {"PA6": 0.95, "PA66": 0.95},
        "phenol": {"PET": 0.8, "PA6": 0.7},
    }

    # Solvent safety scores (GSK-based, 1-10, higher = safer)
    SOLVENT_SAFETY = {
        "water": 10, "ethanol": 9, "isopropanol": 8, "acetone": 7,
        "ethyl_acetate": 7, "methanol": 6, "toluene": 4, "xylene": 4,
        "cyclohexane": 5, "hexane": 4, "thf": 5, "dcm": 3,
        "chloroform": 2, "dmf": 4, "nmp": 5, "formic_acid": 4,
        "phenol": 3,
    }

    # Common separation strategies
    SEPARATION_STRATEGIES = {
        ("PE", "PET"): {
            "recommended_solvent": "xylene",
            "temperature": 120,
            "notes": "PE dissolves at high temp, PET remains solid",
        },
        ("PS", "PE"): {
            "recommended_solvent": "toluene",
            "temperature": 80,
            "notes": "PS dissolves readily, PE less so",
        },
        ("PVC", "PE"): {
            "recommended_solvent": "thf",
            "temperature": 60,
            "notes": "PVC dissolves in THF, PE does not",
        },
        ("PA6", "PE"): {
            "recommended_solvent": "formic_acid",
            "temperature": 25,
            "notes": "PA6 dissolves in formic acid at room temp",
        },
    }

    @classmethod
    def get_polymer_family(cls, polymer: str) -> Optional[str]:
        """Get the family a polymer belongs to."""
        polymer_upper = polymer.upper()
        for family, members in cls.POLYMER_FAMILIES.items():
            if polymer_upper in [m.upper() for m in members]:
                return family
        return None

    @classmethod
    def get_related_polymers(cls, polymer: str) -> List[str]:
        """Get polymers in the same family."""
        family = cls.get_polymer_family(polymer)
        if family:
            return cls.POLYMER_FAMILIES[family]
        return []

    @classmethod
    def get_compatible_solvents(cls, polymer: str, min_score: float = 0.5) -> List[Tuple[str, float]]:
        """Get solvents compatible with a polymer, sorted by compatibility."""
        polymer_upper = polymer.upper()
        compatible = []
        for solvent, polymers in cls.COMPATIBILITY.items():
            score = polymers.get(polymer_upper, 0)
            if score >= min_score:
                compatible.append((solvent, score))
        return sorted(compatible, key=lambda x: x[1], reverse=True)

    @classmethod
    def get_selectivity_hint(cls, target: str, others: List[str]) -> Optional[Dict[str, Any]]:
        """Get selectivity hint for separating target from others."""
        target_upper = target.upper()
        others_upper = [p.upper() for p in others]

        # Find solvents that dissolve target but not others
        hints = []
        for solvent, polymers in cls.COMPATIBILITY.items():
            target_score = polymers.get(target_upper, 0)
            other_scores = [polymers.get(p, 0) for p in others_upper]
            max_other = max(other_scores) if other_scores else 0

            selectivity = target_score - max_other
            if selectivity > 0.2:  # Significant selectivity
                hints.append({
                    "solvent": solvent,
                    "target_score": target_score,
                    "max_other_score": max_other,
                    "selectivity": selectivity,
                    "safety": cls.SOLVENT_SAFETY.get(solvent, 5),
                })

        if hints:
            # Sort by selectivity * safety
            hints.sort(key=lambda x: x["selectivity"] * x["safety"] / 10, reverse=True)
            return hints[0]
        return None

    @classmethod
    def get_separation_strategy(cls, polymer1: str, polymer2: str) -> Optional[Dict[str, Any]]:
        """Get recommended separation strategy for a polymer pair."""
        p1, p2 = sorted([polymer1.upper(), polymer2.upper()])
        key = (p1, p2)
        if key in cls.SEPARATION_STRATEGIES:
            return cls.SEPARATION_STRATEGIES[key]

        # Try reverse order
        key = (p2, p1)
        if key in cls.SEPARATION_STRATEGIES:
            return cls.SEPARATION_STRATEGIES[key]

        # Generate hint if no explicit strategy
        hint = cls.get_selectivity_hint(p1, [p2])
        if hint:
            return {
                "recommended_solvent": hint["solvent"],
                "temperature": 80,  # Default
                "notes": f"Inferred from compatibility scores (selectivity: {hint['selectivity']:.2f})",
                "inferred": True,
            }
        return None

    @classmethod
    def check_safety_constraints(cls, solvents: List[str]) -> Dict[str, Any]:
        """Check safety constraints for a list of solvents."""
        results = {
            "all_safe": True,
            "warnings": [],
            "scores": {},
        }

        for solvent in solvents:
            solvent_lower = solvent.lower().replace(" ", "_")
            score = cls.SOLVENT_SAFETY.get(solvent_lower, 5)
            results["scores"][solvent] = score

            if score <= 3:
                results["all_safe"] = False
                results["warnings"].append(f"{solvent} has low safety score ({score}/10)")
            elif score <= 5:
                results["warnings"].append(f"{solvent} has moderate safety concerns ({score}/10)")

        return results

    @classmethod
    def suggest_safer_alternatives(cls, solvent: str, target_polymer: str) -> List[Dict[str, Any]]:
        """Suggest safer alternatives for a solvent."""
        current_safety = cls.SOLVENT_SAFETY.get(solvent.lower().replace(" ", "_"), 5)
        polymer_upper = target_polymer.upper()

        alternatives = []
        for alt_solvent, polymers in cls.COMPATIBILITY.items():
            alt_safety = cls.SOLVENT_SAFETY.get(alt_solvent, 5)
            if alt_safety > current_safety:
                compat = polymers.get(polymer_upper, 0)
                if compat >= 0.3:  # At least some compatibility
                    alternatives.append({
                        "solvent": alt_solvent,
                        "safety_score": alt_safety,
                        "compatibility": compat,
                        "safety_improvement": alt_safety - current_safety,
                    })

        return sorted(alternatives, key=lambda x: x["safety_score"], reverse=True)[:3]


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

    # Separation specialist (includes advanced algorithms from tools/ module)
    SEPARATION_TOOLS = get_category_tools([
        "separation", "advanced_separation", "dissolution", "solvent_properties",
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
    # Using max reducer to handle concurrent updates (always take highest value)
    current_specialist_index: Annotated[int, lambda a, b: max(a, b)] = 0

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
    # Using Annotated with dict merge to allow multiple nodes to update
    agent_timings: Annotated[Dict[str, float], lambda a, b: {**a, **b}] = {}

    # Finalized execution trace (P3: Set by smart_aggregator)
    execution_trace: Optional[Dict[str, Any]] = None

    # =========================================================
    # P0 Enhancement: Review/Revision Loop
    # =========================================================

    # Current retry count for separation (max 2 retries)
    separation_retry_count: int = 0

    # Reviewer feedback from last review
    reviewer_feedback: Optional[Dict[str, Any]] = None

    # Modified parameters for retry (e.g., wider temperature range)
    retry_params: Dict[str, Any] = {}

    # =========================================================
    # P1 Enhancement: Parallel Execution & Supervisor
    # =========================================================

    # Supervisor decision (if using LLM-based supervisor)
    supervisor_decision: Optional[Dict[str, Any]] = None

    # Flag for parallel execution mode
    parallel_execution: bool = False

    # Results from parallel specialists (aggregated)
    parallel_results: Dict[str, Any] = {}

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
        r'at\s+(\d+)\s*degrees',  # Must have "degrees" to avoid matching "at 200 kg/hr"
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
# P1: PARALLEL ORCHESTRATOR FOR INDEPENDENT SPECIALISTS
# ============================================================

async def parallel_orchestrator_node(
    state: MultiAgentState,
    sql_agent_node,
    specialists: List[str] = None
) -> dict:
    """
    P1 Enhancement: Run independent specialists in parallel.

    For collaboration modes where specialists don't depend on each other's output
    (e.g., separation + literature), run them concurrently for faster results.

    Args:
        state: Current multi-agent state
        sql_agent_node: The SQL agent node function
        specialists: List of specialists to run in parallel

    Returns:
        Combined state with results from all parallel specialists
    """
    specialists = specialists or state.get("collaboration_specialists", [])
    shared_context = state.get("shared_context", {})

    logger.info(f"Parallel Orchestrator: Running {len(specialists)} specialists in parallel")

    # Define specialist execution functions
    async def run_separation():
        """Run separation agent."""
        state_copy = dict(state)
        state_copy["selected_categories"] = [
            "separation", "advanced_separation", "dissolution",
            "solvent_properties", "visualization", "safety"
        ]
        polymers = shared_context.get("polymers", [])
        temperature = shared_context.get("temperature", 80.0)

        sep_task = SeparationTaskRequest(
            polymers=polymers,
            temperature=temperature,
            top_k_solvents=3,
            ranking_criterion="selectivity"
        )
        tool_instruction = HumanMessage(content=sep_task.to_instruction())
        state_copy["messages"] = list(state.get("messages", [])) + [tool_instruction]

        result = await sql_agent_node(state_copy)

        # Parse separation results
        messages = result.get("messages", [])
        all_text = "\n".join(
            msg.content for msg in messages
            if hasattr(msg, 'content') and isinstance(msg.content, str)
        )
        separation_results = parse_separation_results(all_text)
        separation_results["polymers"] = polymers
        separation_results["temperature"] = temperature

        return {"separation_results": separation_results, "messages": messages}

    async def run_literature():
        """Run literature agent."""
        state_copy = dict(state)
        state_copy["selected_categories"] = ["literature", "rag"]

        query = shared_context.get("original_query", "")
        polymers = shared_context.get("polymers", [])

        from agent_schemas import LiteratureTaskRequest
        lit_task = LiteratureTaskRequest(
            search_topic=query,
            polymers=polymers,
            max_results=10,
            search_rag_first=True
        )
        tool_instruction = HumanMessage(content=lit_task.to_instruction())
        state_copy["messages"] = list(state.get("messages", [])) + [tool_instruction]

        result = await sql_agent_node(state_copy)
        messages = result.get("messages", [])

        # Extract literature results
        all_text = "\n".join(
            msg.content for msg in messages
            if hasattr(msg, 'content') and isinstance(msg.content, str)
        )
        literature_results = {
            "papers_found": all_text.count("DOI") + all_text.count("doi"),
            "key_findings": [],
            "raw_response": all_text[:2000],
        }

        return {"literature_results": literature_results, "messages": messages}

    async def run_tea():
        """Run TEA agent (requires separation results)."""
        state_copy = dict(state)
        state_copy["selected_categories"] = ["economics", "strap", "visualization", "solvent_properties"]

        # Get solvents from separation results if available
        separation_results = state.get("separation_results", {})
        solvents = separation_results.get("solvents", ["xylene", "cyclohexane"])

        tea_task = TEATaskRequest(
            solvents=solvents[:5],
            throughput_kg_hr=shared_context.get("throughput_kg_hr", 100.0),
            recovery_rate=0.95,
            include_capex=True,
            compare_solvents=True,
        )
        tool_instruction = HumanMessage(content=tea_task.to_instruction())
        state_copy["messages"] = list(state.get("messages", [])) + [tool_instruction]

        result = await sql_agent_node(state_copy)
        messages = result.get("messages", [])

        # Parse TEA results
        all_text = "\n".join(
            msg.content for msg in messages
            if hasattr(msg, 'content') and isinstance(msg.content, str)
        )
        tea_results = parse_tea_results(all_text)

        return {"tea_results": tea_results, "messages": messages}

    # Map specialists to their execution functions
    specialist_runners = {
        "separation": run_separation,
        "literature": run_literature,
        "tea_lca": run_tea,
    }

    # Determine which specialists can run in parallel
    # TEA depends on separation, so they can't be parallel
    # Separation + Literature CAN be parallel
    independent_specialists = []
    dependent_specialists = []

    for spec in specialists:
        if spec == "tea_lca":
            dependent_specialists.append(spec)  # TEA needs separation results
        else:
            independent_specialists.append(spec)

    # Run independent specialists in parallel
    parallel_results = {}
    all_messages = list(state.get("messages", []))

    if independent_specialists:
        tasks = []
        for spec in independent_specialists:
            if spec in specialist_runners:
                tasks.append(specialist_runners[spec]())

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Parallel specialist failed: {result}")
                    continue
                if isinstance(result, dict):
                    parallel_results.update(result)
                    all_messages.extend(result.get("messages", []))

    # Run dependent specialists sequentially (TEA after separation)
    for spec in dependent_specialists:
        if spec in specialist_runners:
            # Update state with parallel results before running dependent
            state_with_results = dict(state)
            state_with_results.update(parallel_results)
            state_with_results["messages"] = all_messages

            result = await specialist_runners[spec]()
            if isinstance(result, dict):
                parallel_results.update(result)
                all_messages.extend(result.get("messages", []))

    logger.info(f"Parallel Orchestrator: Completed {len(specialists)} specialists")

    return {
        **parallel_results,
        "messages": all_messages,
        "current_specialist_index": len(specialists),
        "aggregation_required": True,
    }


# ============================================================
# P1: LLM-BASED SUPERVISOR (Optional Dynamic Routing)
# ============================================================

async def supervisor_decision_node(
    state: MultiAgentState,
    llm_factory=None
) -> Command:
    """
    P1 Enhancement: LLM-based supervisor for dynamic routing decisions.

    Uses SupervisorDecision schema to determine next agent based on:
    - Current results quality
    - Remaining specialists to visit
    - Query requirements

    Only activated for complex queries (complexity >= 4) to avoid
    latency overhead for simple queries.

    Args:
        state: Current multi-agent state
        llm_factory: Function to create LLM (optional, for testing)

    Returns:
        Command with routing decision
    """
    from agent_schemas import SupervisorDecision

    complexity = state.get("complexity", 3)
    collaboration_mode = state.get("collaboration_mode")
    separation_results = state.get("separation_results", {})
    tea_results = state.get("tea_results", {})
    reviewer_feedback = state.get("reviewer_feedback", {})

    # Only use LLM supervisor for complex queries
    if complexity < 4 or not collaboration_mode:
        # Simple rule-based routing
        if separation_results and not tea_results:
            return Command(goto="collab_tea_agent")
        elif tea_results:
            return Command(goto="smart_aggregator")
        else:
            return Command(goto="collab_separation_agent")

    # Build supervisor context
    has_solvents = bool(separation_results.get("solvents", []))
    has_cost = bool(tea_results.get("cost_per_kg"))
    # Only consider quality score if reviewer feedback exists
    has_reviewer_feedback = bool(reviewer_feedback)
    quality_score = reviewer_feedback.get("quality_score", 1.0) if has_reviewer_feedback else 1.0

    # Rule-based supervisor (can be replaced with LLM call)
    if not has_solvents:
        # Need separation first
        decision = SupervisorDecision(
            next_agent="collab_separation_agent",
            reason="Separation results needed - no solvents found",
            confidence=0.9
        )
    elif has_solvents and not has_cost:
        # Have solvents, need TEA
        decision = SupervisorDecision(
            next_agent="collab_tea_agent",
            reason=f"Proceeding to TEA with {len(separation_results.get('solvents', []))} solvents",
            confidence=0.85
        )
    elif has_reviewer_feedback and quality_score < 0.5 and state.get("separation_retry_count", 0) < 2:
        # Low quality with explicit feedback, consider retry
        decision = SupervisorDecision(
            next_agent="collab_separation_agent",
            reason=f"Quality score {quality_score:.2f} below threshold, retrying",
            is_reroute=True,
            confidence=0.7
        )
    else:
        # Ready for aggregation
        decision = SupervisorDecision(
            next_agent="smart_aggregator",
            reason="All required analyses complete",
            confidence=0.95
        )

    logger.info(f"Supervisor Decision: {decision.next_agent} (confidence={decision.confidence:.2f}, reason={decision.reason})")

    return Command(
        update={"supervisor_decision": decision.model_dump()},
        goto=decision.next_agent
    )


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

    # Set categories for separation (includes advanced_separation for new algorithms)
    state_copy = dict(state)
    state_copy["selected_categories"] = ["separation", "advanced_separation", "dissolution", "solvent_properties", "visualization", "safety"]

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

        # CRITICAL FIX: When tools are pending, return dict (NOT Command) to let
        # conditional edges route to tools. Command(goto=...) bypasses conditional edges!
        if has_pending_tools:
            logger.info(f"Separation: {len(last_msg.tool_calls)} tool calls pending, returning dict (iter {sep_iter})")
            return {
                **filter_result_for_collab(result),
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
                    **filter_result_for_collab(result),
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

        # Determine next agent based on collaboration mode
        collaboration_mode = state.get("collaboration_mode", "separation_tea")
        agent_start_time = state.get("agent_timings", {}).get("orchestrator", time.time())
        solvents_found = separation_results.get("solvents", [])

        # Handle both 2-way (separation_literature) and 3-way (separation_literature_tea_lca) modes
        if collaboration_mode in ("separation_literature", "separation_literature_tea_lca"):
            # Create literature task request - include verification context for 3-way mode
            search_topic = f"Separation and solubility of {', '.join(separation_results.get('polymers', []))}"
            if collaboration_mode == "separation_literature_tea_lca":
                search_topic += " - verify dissolution selectivity for greedy sequence"

            lit_task = LiteratureTaskRequest(
                search_topic=search_topic,
                polymers=separation_results.get("polymers", []),
                solvents=solvents_found[:5] if solvents_found else [],
                max_results=10,
                search_rag_first=True,
            )

            handoff_metrics_entry = create_handoff_metrics(
                from_agent="separation",
                to_agent="literature",
                start_time=agent_start_time,
                tools_called=["plan_sequential_separation"],
                success=bool(solvents_found),
                task_type="separation",
                context_size=len(str(separation_results)) if separation_results else 0,
            )
            handoff_metrics_entry["query_summary"] = f"Separation of {len(shared_context.get('polymers', []))} polymers -> literature"

            pending_handoff = {
                "from_agent": "separation",
                "to_agent": "literature",
                "task_params": lit_task.model_dump(),
            }

            return Command(
                update={
                    **filter_result_for_collab(result),
                    "separation_results": separation_results,
                    "current_specialist_index": state.get("current_specialist_index", 0) + 1,
                    "pending_handoff": pending_handoff,
                    "handoff_metrics": [handoff_metrics_entry],
                    "agent_timings": {
                        **state.get("agent_timings", {}),
                        "separation": time.time(),
                    },
                },
                goto="collab_literature_agent"
            )

        # P3: Create enhanced handoff metrics (consolidates P0 handoff_history)
        # P0 Enhancement: Route to reviewer for quality validation before TEA
        handoff_metrics_entry = create_handoff_metrics(
            from_agent="separation",
            to_agent="separation_reviewer",  # P0: Route to reviewer first
            start_time=agent_start_time,
            tools_called=["plan_sequential_separation", "find_optimal_separation_sequence"],
            success=bool(solvents_found),
            task_type="separation",
            context_size=len(str(separation_results)) if separation_results else 0,
        )
        handoff_metrics_entry["query_summary"] = f"Separation of {len(shared_context.get('polymers', []))} polymers, found {len(solvents_found)} solvents"

        # P0 Enhancement: Route to reviewer for quality validation
        # The reviewer will decide whether to proceed to TEA or request revision
        return Command(
            update={
                **result,
                "separation_results": separation_results,
                "current_specialist_index": state.get("current_specialist_index", 0) + 1,
                "handoff_metrics": [handoff_metrics_entry],
                "agent_timings": {
                    **state.get("agent_timings", {}),
                    "separation": time.time(),
                },
            },
            goto="separation_reviewer"  # P0: Route to reviewer
        )

    # Non-collaborative mode: return dict
    return result


# ============================================================
# P0 ENHANCEMENT: SEPARATION REVIEWER NODE (Review/Revision Loop)
# ============================================================

# Quality thresholds for separation results
SEPARATION_QUALITY_THRESHOLDS = {
    "min_solvents": 2,           # Need at least 2 solvents to choose from
    "min_selectivity": 5.0,      # Minimum selectivity percentage (was 0)
    "max_retries": 2,            # Maximum retry attempts
    "temperature_expansion": 20, # Degrees to expand temperature window on retry
}


async def separation_reviewer_node(state: MultiAgentState) -> Command:
    """
    Review separation results and decide whether to proceed or revise.

    P0 Enhancement: Implements GPT-Researcher style review/revision loop.

    Quality checks:
    1. Minimum solvents found (>= 2)
    2. Selectivity thresholds met (>= 5%)
    3. Sequence completeness (all polymers covered)

    Returns:
        Command to either:
        - goto="collab_tea_agent" if acceptable
        - goto="collab_separation_agent" if revision needed (with modified params)
        - goto="smart_aggregator" if max retries exceeded
    """
    separation_results = state.get("separation_results") or {}
    shared_context = state.get("shared_context") or {}
    retry_count = min(state.get("separation_retry_count", 0), 100)  # Clamp to reasonable max
    max_retries = SEPARATION_QUALITY_THRESHOLDS["max_retries"]

    # Extract metrics
    solvents = separation_results.get("solvents", [])
    selectivities = separation_results.get("selectivities", [])
    best_sequence = separation_results.get("best_sequence", [])
    polymers = separation_results.get("polymers", []) or shared_context.get("polymers", [])
    temperature = separation_results.get("temperature", shared_context.get("temperature", 80.0))

    # Calculate quality metrics
    solvents_count = len(solvents)
    min_selectivity = min(selectivities) if selectivities else 0.0
    max_selectivity = max(selectivities) if selectivities else 0.0
    has_sequence = bool(best_sequence and len(best_sequence) > 0)

    # Build issues list
    issues = []
    suggestions = []

    # Check 1: Minimum solvents
    if solvents_count < SEPARATION_QUALITY_THRESHOLDS["min_solvents"]:
        issues.append(f"Only {solvents_count} solvents found (minimum: {SEPARATION_QUALITY_THRESHOLDS['min_solvents']})")
        suggestions.append("Try expanding temperature range to find more solvent options")

    # Check 2: Selectivity threshold
    if min_selectivity < SEPARATION_QUALITY_THRESHOLDS["min_selectivity"] and selectivities:
        issues.append(f"Minimum selectivity {min_selectivity:.1f}% is below threshold ({SEPARATION_QUALITY_THRESHOLDS['min_selectivity']}%)")
        suggestions.append("Consider different temperature or alternative solvents")

    # Check 3: Sequence completeness
    if polymers and has_sequence and len(best_sequence) < len(polymers) - 1:
        issues.append(f"Sequence covers {len(best_sequence)} steps but {len(polymers)} polymers need separation")
        suggestions.append("Ensure all polymer pairs have viable separation paths")

    # Calculate quality score (0-1)
    quality_score = 1.0
    if solvents_count < 2:
        quality_score -= 0.4
    elif solvents_count < 3:
        quality_score -= 0.2
    if min_selectivity < 5 and selectivities:
        quality_score -= 0.3
    if not has_sequence:
        quality_score -= 0.2
    quality_score = max(0.0, quality_score)

    # Decision logic
    is_acceptable = len(issues) == 0 or quality_score >= 0.6
    requires_revision = not is_acceptable and retry_count < max_retries

    # Create reviewer feedback
    feedback = ReviewerFeedback(
        is_acceptable=is_acceptable,
        quality_score=quality_score,
        issues=issues,
        suggestions=suggestions,
        requires_revision=requires_revision,
        solvents_count=solvents_count,
        min_selectivity=min_selectivity if selectivities else None,
        max_selectivity=max_selectivity if selectivities else None,
        has_sequence=has_sequence,
        retry_count=retry_count,
        max_retries=max_retries,
    )

    # Log review decision
    logger.info(f"Separation Review: quality={quality_score:.2f}, acceptable={is_acceptable}, "
                f"issues={len(issues)}, retry={retry_count}/{max_retries}")
    if issues:
        for issue in issues:
            logger.warning(f"  Issue: {issue}")

    # Create handoff metrics for review
    handoff_metrics_entry = create_handoff_metrics(
        from_agent="separation_reviewer",
        to_agent="collab_tea_agent" if is_acceptable else "collab_separation_agent",
        start_time=state.get("agent_timings", {}).get("separation", time.time()),
        tools_called=[],
        success=is_acceptable,
        error_message="; ".join(issues) if issues else None,
        task_type="review",
    )
    handoff_metrics_entry["quality_score"] = quality_score

    if requires_revision:
        # Build retry parameters (expand temperature window)
        current_temp = temperature
        temp_expansion = SEPARATION_QUALITY_THRESHOLDS["temperature_expansion"]
        new_temp_min = max(40, current_temp - temp_expansion)
        new_temp_max = min(180, current_temp + temp_expansion)

        retry_params = {
            "temperature_range": (new_temp_min, new_temp_max),
            "retry_reason": "; ".join(issues),
            "previous_solvents": solvents,
        }

        feedback.retry_params = retry_params
        feedback.revision_instructions = (
            f"Retry separation planning with expanded temperature range "
            f"({new_temp_min}°C to {new_temp_max}°C). "
            f"Previous attempt found {solvents_count} solvents with selectivity {min_selectivity:.1f}%-{max_selectivity:.1f}%."
        )

        logger.info(f"Separation Reviewer: Requesting revision (attempt {retry_count + 1}/{max_retries})")

        return Command(
            update={
                "separation_retry_count": retry_count + 1,
                "reviewer_feedback": feedback.model_dump(),
                "retry_params": retry_params,
                "handoff_metrics": [handoff_metrics_entry],
                # Clear previous results for retry
                "separation_results": None,
            },
            goto="collab_separation_agent"
        )

    elif is_acceptable:
        # Proceed to TEA agent
        logger.info(f"Separation Reviewer: Results acceptable (quality={quality_score:.2f})")

        # Create TEA task request
        tea_task = TEATaskRequest(
            solvents=solvents[:5],
            throughput_kg_hr=shared_context.get("throughput_kg_hr", 100.0),
            recovery_rate=0.95,
            include_capex=True,
            include_lca=False,
            compare_solvents=True,
            polymers=polymers,
            temperature=temperature,
            best_sequence=best_sequence,
        )

        pending_handoff = {
            "from_agent": "separation_reviewer",
            "to_agent": "tea_lca",
            "task_params": tea_task.model_dump(),
        }

        return Command(
            update={
                "reviewer_feedback": feedback.model_dump(),
                "pending_handoff": pending_handoff,
                "handoff_metrics": [handoff_metrics_entry],
                "agent_timings": {
                    **state.get("agent_timings", {}),
                    "reviewer": time.time(),
                },
            },
            goto="collab_tea_agent"
        )

    else:
        # Max retries exceeded - proceed to aggregator with warning
        logger.warning(f"Separation Reviewer: Max retries ({max_retries}) exceeded, proceeding with partial results")

        feedback.revision_instructions = f"Max retries exceeded. Issues: {'; '.join(issues)}"

        return Command(
            update={
                "reviewer_feedback": feedback.model_dump(),
                "handoff_metrics": [handoff_metrics_entry],
                "agent_timings": {
                    **state.get("agent_timings", {}),
                    "reviewer": time.time(),
                },
                # Pass through to TEA even with suboptimal results
                "pending_handoff": {
                    "from_agent": "separation_reviewer",
                    "to_agent": "tea_lca",
                    "task_params": {"solvents": solvents[:5] if solvents else ["xylene", "cyclohexane"]},
                },
            },
            goto="collab_tea_agent"
        )


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
        tea_instruction = tea_task.to_instruction()
        context_message = HumanMessage(content=tea_instruction)

        # Log what we're sending to TEA agent
        logger.info(f"TEA Agent receiving instruction for solvents: {tea_task.solvents[:5]}")
        logger.debug(f"TEA instruction: {tea_instruction[:200]}...")

        # Check if there are TEA tool results already
        current_messages = state.get("messages", [])
        from langchain_core.messages import ToolMessage

        # Find TEA-related ToolMessages by content markers
        tea_tool_messages = [
            msg for msg in current_messages
            if isinstance(msg, ToolMessage) and
               hasattr(msg, 'content') and isinstance(msg.content, str) and
               any(marker in msg.content.lower() for marker in
                   ['cost per kg polymer', 'capex', 'opex', 'payback', 'tea analysis',
                    'solvent recovery', 'annual operating'])
        ]

        has_tea_tool_results = len(tea_tool_messages) > 0
        logger.debug(f"TEA: {len(current_messages)} messages, {len(tea_tool_messages)} are TEA tool results")

        if not has_tea_tool_results:
            # First run: clean instruction only
            state_copy["messages"] = [context_message]
            logger.info("TEA: First run - using clean TEA instruction")
        else:
            # Subsequent run: Build a VALID message sequence for Gemini
            # Structure: [HumanMessage(instruction + tool results summary)]
            # This avoids the complex AIMessage/ToolMessage ordering issues
            tool_results_summary = "\n\n".join([
                f"**Tool Result:**\n{msg.content[:2000]}..."
                if len(msg.content) > 2000 else f"**Tool Result:**\n{msg.content}"
                for msg in tea_tool_messages[:5]
            ])

            combined_message = HumanMessage(content=f"""
{tea_instruction}

## Previous Tool Results (analyze these):

{tool_results_summary}

Based on the tool results above, extract and report:
1. The cost per kg polymer for each solvent analyzed
2. Which solvent is most cost-effective
3. CAPEX and OPEX figures

Do NOT call the tools again. Just analyze the results above and provide a summary.
""")
            state_copy["messages"] = [combined_message]
            logger.info(f"TEA: Subsequent run - {len(tea_tool_messages)} tool results included in instruction")

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

        # CRITICAL FIX: When tools are pending, return dict (NOT Command) to let
        # conditional edges route to TEA tools first
        if has_pending_tools:
            logger.info(f"TEA: {len(last_msg.tool_calls)} tool calls pending, returning dict (iter {tea_iter})")
            return {
                **filter_result_for_collab(result),
                "tea_iteration_count": tea_iter,
                "tea_results": {"status": "pending_tools"},
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
            logger.info(f"TEA: Found {len(tool_outputs)} tool outputs")
            for i, output in enumerate(tool_outputs[:2]):  # Log first 2 tool outputs
                sample = output[:500].replace('\n', ' ')
                logger.info(f"TEA ToolMessage {i+1} ({len(output)} chars): {sample}...")
        else:
            logger.warning(f"TEA: No tool outputs found! LLM may not have called TEA tools.")

        tea_results = parse_tea_results(all_text)
        tea_results["solvents_analyzed"] = separation_results.get("solvents", []) if separation_results else []

        # Enhanced cost extraction with more patterns
        if not tea_results.get("cost_per_kg"):
            # Comprehensive patterns for cost extraction - ordered by specificity
            cost_patterns = [
                # Most specific patterns first (from actual TEA tool output)
                r'cost\s+per\s+kg\s+polymer[:\s]*\$?(\d+\.?\d*)\s*/?\s*kg',
                r'cost\s+per\s+kg[:\s]*\$?(\d+\.?\d*)',
                # Standard patterns
                r'msp[:\s]*\$?(\d+\.?\d*)',
                r'\$(\d+\.?\d*)\s*/\s*kg',
                r'(\d+\.?\d*)\s*\$/kg',
                # Processing cost patterns
                r'processing\s+cost[:\s]*\$?(\d+\.?\d*)',
                r'solvent\s+recovery\s+cost[:\s]*\$?(\d+\.?\d*)',
                # Generic cost patterns (less specific)
                r'total\s+cost[:\s]*\$?(\d+\.?\d*)',
                r'operating\s+cost[:\s]*\$?(\d+\.?\d*)',
                # Per kg patterns with different formats
                r'(\d+\.?\d*)\s*usd\s*/\s*kg',
                r'(\d+\.?\d*)\s*per\s+kg',
            ]
            for pattern in cost_patterns:
                match = re.search(pattern, all_text.lower())
                if match:
                    try:
                        cost = float(match.group(1))
                        if 0.001 < cost < 1000:  # Wider range for small costs like $0.01/kg
                            tea_results["cost_per_kg"] = cost
                            logger.info(f"TEA cost extracted: ${cost}/kg (pattern: {pattern[:40]}...)")
                            break
                    except ValueError:
                        pass

        # Log if no cost found after all attempts
        if not tea_results.get("cost_per_kg"):
            logger.warning(f"TEA: No cost_per_kg extracted from {len(all_text)} chars of output")
            # Log more of the text for debugging
            sample = all_text[:500].replace('\n', ' ')
            logger.warning(f"TEA output sample: {sample}...")

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
                **filter_result_for_collab(result),
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
# COLLABORATIVE LITERATURE AGENT
# ============================================================

# KB keywords for auto-selection
KB_KEYWORDS = {
    "STRAP-CORE": [
        "strap", "solvent", "dissolution", "polymer recycling", "hansen",
        "selectivity", "thermodynamic", "solubility parameter", "separation",
        "multilayer", "film", "plastics recycling", "polymer-solvent"
    ],
    "printed_plastics_deinking": [
        "deinking", "ink", "printed", "printing", "pigment", "surfactant",
        "flexographic", "gravure", "coating", "adhesive", "label", "removal",
        "washing", "cleaning", "detergent", "surface", "contamination"
    ],
}


def select_knowledgebases(query: str, polymers: List[str] = None, solvents: List[str] = None) -> List[str]:
    """
    Auto-select relevant knowledgebases based on query content.

    Args:
        query: User query or search topic
        polymers: Optional list of polymers mentioned
        solvents: Optional list of solvents mentioned

    Returns:
        List of KB names to search (in priority order)
    """
    query_lower = query.lower()
    scores = {}

    for kb_name, keywords in KB_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in query_lower:
                score += 1
        scores[kb_name] = score

    # If query mentions polymers/solvents, boost STRAP-CORE
    if polymers or solvents:
        scores["STRAP-CORE"] = scores.get("STRAP-CORE", 0) + 2

    # Sort by score and filter to those with matches
    sorted_kbs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [kb for kb, score in sorted_kbs if score > 0]

    # If no matches, default to both
    if not selected:
        selected = list(KB_KEYWORDS.keys())

    logger.debug(f"KB selection for '{query[:50]}...': {selected} (scores: {scores})")
    return selected


def parse_literature_results(text: str) -> Dict[str, Any]:
    """
    Parse LLM output to extract structured literature results.

    Args:
        text: Raw LLM response text

    Returns:
        Dict with structured literature data
    """
    results = {
        "papers_found": 0,
        "key_findings": [],
        "citations": [],
        "polymers_mentioned": [],
        "solvents_mentioned": [],
        "temperatures_mentioned": [],
        "confidence_score": 0.5,
    }

    # Count papers/passages
    paper_patterns = [
        r"(\d+)\s*(?:papers?|passages?|results?|sources?)\s*found",
        r"found\s*(\d+)\s*(?:papers?|passages?|results?)",
        r"passage\s*(\d+)\s*of",
    ]
    for pattern in paper_patterns:
        match = re.search(pattern, text.lower())
        if match:
            results["papers_found"] = int(match.group(1))
            break

    # Extract key findings (bullet points)
    findings = []
    lines = text.split('\n')
    in_findings_section = False
    for line in lines:
        line = line.strip()
        if 'finding' in line.lower() or 'key' in line.lower():
            in_findings_section = True
            continue
        if in_findings_section and line.startswith(('-', '*', '•', '1', '2', '3')):
            # Clean the bullet point
            finding = re.sub(r'^[-*•\d.)\s]+', '', line).strip()
            if finding and len(finding) > 10:
                findings.append(finding[:500])  # Truncate long findings
        if in_findings_section and not line and findings:
            in_findings_section = False
    results["key_findings"] = findings[:10]  # Max 10 findings

    # Extract polymer mentions
    polymer_patterns = [
        r'\b(PE|PP|PET|PS|PVC|LDPE|HDPE|LLDPE|PA6|PA66|Nylon|EVOH|PC|ABS|PMMA|PU|PLA)\b'
    ]
    polymers = set()
    for pattern in polymer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        polymers.update([m.upper() for m in matches])
    results["polymers_mentioned"] = list(polymers)

    # Extract solvent mentions (common solvents)
    common_solvents = [
        "xylene", "toluene", "cyclohexane", "hexane", "heptane", "decalin",
        "dmf", "thf", "dmso", "acetone", "ethanol", "methanol", "chloroform",
        "dichloromethane", "nmp", "dce", "benzene", "limonene", "turpentine"
    ]
    solvents = []
    for solvent in common_solvents:
        if solvent.lower() in text.lower():
            solvents.append(solvent)
    results["solvents_mentioned"] = solvents

    # Extract temperature mentions
    temp_patterns = [
        r'(\d+)\s*°?\s*C\b',
        r'(\d+)\s*degrees?\s*(?:celsius|C)\b',
        r'at\s+(\d+)\s*degrees',
    ]
    temps = set()
    for pattern in temp_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            temp = int(m)
            if 0 < temp < 400:  # Reasonable temperature range
                temps.add(float(temp))
    results["temperatures_mentioned"] = sorted(list(temps))

    # Estimate confidence based on results quality
    confidence = 0.3
    if results["papers_found"] > 0:
        confidence += 0.2
    if len(results["key_findings"]) > 2:
        confidence += 0.2
    if results["polymers_mentioned"] or results["solvents_mentioned"]:
        confidence += 0.2
    results["confidence_score"] = min(confidence, 1.0)

    results["raw_response"] = text[:5000] if len(text) > 5000 else text

    return results


async def collab_literature_agent_node(state: MultiAgentState, sql_agent_node) -> Command:
    """
    Literature specialist with Command-based handoff for collaboration.

    Supports two output modes:
    1. Direct user: When collaboration_mode is None, returns literature directly
    2. Collaboration: Passes context to other agents (separation, TEA)

    Key features:
    - Auto-selects relevant knowledgebases based on query content
    - Searches multiple KBs when relevant
    - Extracts structured data for collaboration (polymers, solvents, temps)
    - Returns Command for dynamic routing
    """
    import rag_module as rag

    is_collaborative = state.get("collaboration_mode") is not None
    shared_context = state.get("shared_context", {})
    pending_handoff = state.get("pending_handoff", {})

    # Set categories for literature
    state_copy = dict(state)
    state_copy["selected_categories"] = ["literature", "rag"]

    # Get query from messages or handoff
    messages = state.get("messages", [])
    query = ""
    if messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content if hasattr(msg, 'content') else str(msg)
                break

    # Get task request from handoff if collaborative
    lit_task = None
    if is_collaborative and pending_handoff:
        task_params = pending_handoff.get("task_params", {})
        if task_params:
            try:
                lit_task = LiteratureTaskRequest(**task_params)
            except Exception as e:
                logger.warning(f"Could not parse LiteratureTaskRequest: {e}")

    # Auto-select KBs based on query content
    polymers = lit_task.polymers if lit_task else shared_context.get("polymers", [])
    solvents = lit_task.solvents if lit_task else []
    search_topic = lit_task.search_topic if lit_task else query

    selected_kbs = select_knowledgebases(search_topic, polymers, solvents)
    logger.info(f"Literature Agent: auto-selected KBs: {selected_kbs}")

    # Build instruction with KB context
    kb_instruction = f"Search these knowledgebases in order: {', '.join(selected_kbs)}"

    if lit_task:
        instruction = lit_task.to_instruction()
    else:
        # Build instruction from query
        instruction = f"""
**LITERATURE SEARCH TASK**

**Topic:** {search_topic}

**Search Strategy:**
1. {kb_instruction}
2. Use search_literature_rag tool for semantic search
3. Use ask_literature for specific follow-up questions

**Requirements:**
- Search for relevant papers and passages
- Extract key findings as bullet points
- Note any polymers, solvents, or temperatures mentioned
- Include source citations with page numbers

**Format Response As:**
## Key Findings
- [finding 1]
- [finding 2]
...

## Relevant Sources
- [source 1, page X]
...

## Polymers/Solvents Mentioned
[List any mentioned in the literature]
"""

    context_message = HumanMessage(content=instruction)

    # Check for existing literature tool results to avoid infinite loop
    current_messages = state.get("messages", [])
    lit_tool_messages = [
        msg for msg in current_messages
        if isinstance(msg, ToolMessage) and
           hasattr(msg, 'content') and isinstance(msg.content, str) and
           any(marker in msg.content.lower() for marker in
               ['literature search', 'passage', 'source:', 'found:', 'relevant'])
    ]

    MAX_LIT_TOOL_RESULTS = 8  # Limit tool calls to prevent infinite loops
    has_lit_tool_results = len(lit_tool_messages) > 0

    if not has_lit_tool_results:
        state_copy["messages"] = [context_message]
        logger.info("Literature: First run - using clean instruction")
    else:
        # Subsequent run: include tool results summary
        tool_results_summary = "\n\n".join([
            f"**Search Result:**\n{msg.content[:1500]}..."
            if len(msg.content) > 1500 else f"**Search Result:**\n{msg.content}"
            for msg in lit_tool_messages[:5]
        ])

        combined_message = HumanMessage(content=f"""
{instruction}

## Previous Search Results (analyze these):

{tool_results_summary}

Based on the search results above, provide a summary of:
1. Key findings about deinking conditions
2. Polymers and solvents mentioned
3. Any experimental temperatures or parameters

Do NOT search again. Summarize the results above.
""")
        state_copy["messages"] = [combined_message]
        logger.info(f"Literature: Subsequent run - {len(lit_tool_messages)} tool results included")

    # Execute literature agent
    result = await sql_agent_node(state_copy)

    # Check for pending tool calls with loop limit
    result_messages = result.get("messages", [])
    if result_messages:
        last_msg = result_messages[-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            # Check if we've reached the limit
            if len(lit_tool_messages) >= MAX_LIT_TOOL_RESULTS:
                logger.info(f"Literature: Reached max tool results ({len(lit_tool_messages)}), stopping loop")
                # Fall through to extraction
            else:
                # Return dict for tool routing
                logger.info(f"Literature: {len(last_msg.tool_calls)} tool calls pending ({len(lit_tool_messages)}/{MAX_LIT_TOOL_RESULTS})")
                return {
                    **filter_result_for_collab(result),
                    "literature_results": {"status": "pending_tools"},
                    "agent_timings": {
                        **state.get("agent_timings", {}),
                        "literature_pending": time.time(),
                    },
                }

    # Extract structured literature results
    all_text = ""
    for msg in result_messages:
        if isinstance(msg, AIMessage):
            all_text += msg.content + "\n"
        elif hasattr(msg, 'content') and isinstance(msg.content, str):
            all_text += msg.content + "\n"

    lit_results = parse_literature_results(all_text)
    lit_results["knowledgebases_searched"] = selected_kbs

    logger.info(f"Literature Agent: papers={lit_results.get('papers_found')}, "
               f"findings={len(lit_results.get('key_findings', []))}, "
               f"confidence={lit_results.get('confidence_score', 0):.2f}")

    # Create LiteratureResult schema
    try:
        lit_result_obj = LiteratureResult(
            papers_found=lit_results.get("papers_found", 0),
            key_findings=lit_results.get("key_findings", []),
            citations=lit_results.get("citations", []),
            knowledge_gaps=lit_results.get("knowledge_gaps", []),
            knowledgebases_searched=lit_results.get("knowledgebases_searched", []),
            confidence_score=lit_results.get("confidence_score", 0.5),
            polymers_mentioned=lit_results.get("polymers_mentioned", []),
            solvents_mentioned=lit_results.get("solvents_mentioned", []),
            temperatures_mentioned=lit_results.get("temperatures_mentioned", []),
            raw_response=lit_results.get("raw_response"),
        )
        lit_results = lit_result_obj.model_dump()
    except Exception as e:
        logger.warning(f"Could not create LiteratureResult schema: {e}")

    if is_collaborative:
        # Determine next agent based on collaboration mode
        collab_mode = state.get("collaboration_mode")

        if collab_mode == "separation_literature":
            # Literature → smart_aggregator (literature supports separation)
            next_agent = "smart_aggregator"
            task_summary = f"Literature search complete: {lit_results.get('papers_found', 0)} papers"
        elif collab_mode == "separation_literature_tea_lca":
            # 3-WAY: Literature → TEA (after separation, before final aggregation)
            next_agent = "collab_tea_agent"
            task_summary = f"Literature verification: {lit_results.get('papers_found', 0)} papers found"
            logger.info(f"3-way collaboration: Literature -> TEA")

            # Get solvents from separation results for TEA
            separation_results = state.get("separation_results", {})
            solvents_for_tea = separation_results.get("solvents", [])[:5]
            if not solvents_for_tea and lit_results.get("solvents_mentioned"):
                solvents_for_tea = lit_results.get("solvents_mentioned", [])[:5]

            # Create proper TEATaskRequest for the handoff
            shared_context = state.get("shared_context", {})
            tea_task = TEATaskRequest(
                solvents=solvents_for_tea if solvents_for_tea else ["cyclohexane"],  # Default fallback
                throughput_kg_hr=shared_context.get("throughput_kg_hr", 100.0),
                recovery_rate=0.95,
                include_capex=True,
            )

            # Override pending_handoff for TEA with proper format
            pending_handoff = {
                "from_agent": "literature",
                "to_agent": next_agent,
                "task_params": tea_task.model_dump(),
                "literature_context": {
                    "summary": task_summary,
                    "key_findings": lit_results.get("key_findings", [])[:5],
                    "solvents_verified": lit_results.get("solvents_mentioned", []),
                },
            }

            return Command(
                update={
                    **filter_result_for_collab(result),
                    "literature_results": lit_results,
                    "current_specialist_index": state.get("current_specialist_index", 0) + 1,
                    "pending_handoff": pending_handoff,
                    "handoff_metrics": [create_handoff_metrics(
                        from_agent="literature",
                        to_agent=next_agent,
                        start_time=state.get("agent_timings", {}).get("literature_start", time.time()),
                        tools_called=["search_literature_rag"],
                        success=lit_results.get("papers_found", 0) > 0,
                        task_type="literature",
                        context_size=len(str(lit_results)),
                    )],
                    "agent_timings": {
                        **state.get("agent_timings", {}),
                        "literature": time.time(),
                    },
                },
                goto=next_agent
            )
        elif collab_mode == "literature_separation":
            # Literature → separation (literature informs separation planning)
            next_agent = "collab_separation_agent"
            # Pass extracted context to separation
            sep_task = SeparationTaskRequest(
                polymers=lit_results.get("polymers_mentioned", []),
                temperature=lit_results.get("temperatures_mentioned", [80.0])[0] if lit_results.get("temperatures_mentioned") else 80.0,
            )
            task_summary = f"Literature found polymers: {lit_results.get('polymers_mentioned', [])}"
        else:
            # Default: return to aggregator
            next_agent = "smart_aggregator"
            task_summary = "Literature search complete"

        # Create handoff metrics
        start_time = state.get("agent_timings", {}).get("literature_start", time.time())
        handoff_metrics_entry = create_handoff_metrics(
            from_agent="literature",
            to_agent=next_agent,
            start_time=start_time,
            tools_called=["search_literature_rag", "ask_literature"],
            success=lit_results.get("papers_found", 0) > 0,
            task_type="literature",
            context_size=len(str(lit_results)),
        )
        handoff_metrics_entry["query_summary"] = task_summary

        # Create pending handoff
        pending_handoff = {
            "from_agent": "literature",
            "to_agent": next_agent,
            "task_params": {
                "literature_summary": task_summary,
                "key_findings": lit_results.get("key_findings", [])[:5],
                "polymers_mentioned": lit_results.get("polymers_mentioned", []),
                "solvents_mentioned": lit_results.get("solvents_mentioned", []),
            },
        }

        return Command(
            update={
                **filter_result_for_collab(result),
                "literature_results": lit_results,
                "current_specialist_index": state.get("current_specialist_index", 0) + 1,
                "pending_handoff": pending_handoff,
                "handoff_metrics": [handoff_metrics_entry],
                "agent_timings": {
                    **state.get("agent_timings", {}),
                    "literature": time.time(),
                },
            },
            goto=next_agent
        )

    # Non-collaborative mode: return dict
    return {
        **filter_result_for_collab(result),
        "literature_results": lit_results,
    }


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

    # P0 Enhancement: Get reviewer feedback
    reviewer_feedback = state.get("reviewer_feedback", {})
    retry_count = state.get("separation_retry_count", 0)

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

    # Handle both 2-way (separation_tea) and 3-way (separation_literature_tea_lca) modes
    if collaboration_mode in ("separation_tea_lca", "separation_tea", "separation_literature_tea_lca"):
        is_3way = collaboration_mode == "separation_literature_tea_lca"
        title = "# Integrated Separation + Literature + Economic Analysis\n" if is_3way else "# Integrated Separation + Economic Analysis\n"
        output_parts.append(title)
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

            # P0 Enhancement: Include reviewer feedback
            if reviewer_feedback:
                quality_score = reviewer_feedback.get("quality_score", 1.0)
                issues = reviewer_feedback.get("issues", [])
                output_parts.append(f"- **Quality score:** {quality_score:.0%}\n")
                if retry_count > 0:
                    output_parts.append(f"- **Retries performed:** {retry_count}\n")
                if issues:
                    output_parts.append(f"- **Review notes:** {'; '.join(issues[:2])}\n")

            output_parts.append("\n")

        # Literature Summary (3-way mode only)
        if is_3way:
            literature_results = state.get("literature_results", {})
            if literature_results:
                output_parts.append("## Literature Verification Summary\n")
                papers = literature_results.get("papers_found", 0)
                kbs = literature_results.get("knowledgebases_searched", [])
                findings = literature_results.get("key_findings", [])
                confidence = literature_results.get("confidence_score", 0)
                solvents_lit = literature_results.get("solvents_mentioned", [])

                output_parts.append(f"- **Papers found:** {papers}\n")
                output_parts.append(f"- **Knowledge bases searched:** {', '.join(kbs) if kbs else 'N/A'}\n")
                output_parts.append(f"- **Confidence score:** {confidence:.2f}\n")
                if solvents_lit:
                    output_parts.append(f"- **Solvents verified in literature:** {', '.join(solvents_lit[:5])}\n")

                if findings:
                    output_parts.append("\n**Key Findings:**\n")
                    for f in findings[:3]:
                        output_parts.append(f"- {f[:150]}...\n" if len(f) > 150 else f"- {f}\n")

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
        if is_3way:
            output_parts.append("*This analysis was performed by the DISSOLVE multi-agent system, "
                              "combining Separation Planning, Literature Verification, and TEA/LCA specialists.*\n")
        else:
            output_parts.append("*This analysis was performed by the DISSOLVE multi-agent system, "
                              "combining Separation Planning and TEA/LCA specialists.*\n")

    elif collaboration_mode == "separation_literature":
        # Get literature results from state
        literature_results = state.get("literature_results", {})

        output_parts.append("# Integrated Separation + Literature Analysis\n")
        output_parts.append(f"*Analysis completed in {elapsed:.1f}s using multi-agent collaboration*\n\n")

        # Separation Summary
        if separation_results:
            output_parts.append("## Separation Analysis\n")
            polymers = separation_results.get("polymers", [])
            solvents = separation_results.get("solvents", [])
            best_sequence = separation_results.get("best_sequence", [])

            output_parts.append(f"- **Polymers analyzed:** {', '.join(polymers) if polymers else 'N/A'}\n")
            output_parts.append(f"- **Solvents identified:** {', '.join(solvents[:5]) if solvents else 'N/A'}\n")
            if best_sequence:
                output_parts.append(f"- **Best sequence:** {' → '.join(best_sequence)}\n")
            output_parts.append("\n")

        # Literature Summary
        if literature_results:
            output_parts.append("## Literature Findings\n")
            papers = literature_results.get("papers_found", 0)
            kbs = literature_results.get("knowledgebases_searched", [])
            findings = literature_results.get("key_findings", [])
            confidence = literature_results.get("confidence_score", 0)

            output_parts.append(f"- **Papers/passages found:** {papers}\n")
            output_parts.append(f"- **Knowledgebases searched:** {', '.join(kbs) if kbs else 'N/A'}\n")
            output_parts.append(f"- **Confidence:** {confidence:.1%}\n\n")

            if findings:
                output_parts.append("**Key Findings:**\n")
                for finding in findings[:5]:
                    output_parts.append(f"- {finding}\n")
                output_parts.append("\n")

            # Show polymers/solvents mentioned in literature
            lit_polymers = literature_results.get("polymers_mentioned", [])
            lit_solvents = literature_results.get("solvents_mentioned", [])
            if lit_polymers:
                output_parts.append(f"- **Polymers in literature:** {', '.join(lit_polymers)}\n")
            if lit_solvents:
                output_parts.append(f"- **Solvents in literature:** {', '.join(lit_solvents)}\n")

        # Integrated Recommendation
        output_parts.append("\n## Integrated Recommendation\n")
        if separation_results and literature_results:
            output_parts.append(
                "The separation analysis has been enhanced with literature context. "
                "See individual sections above for detailed findings.\n"
            )
        else:
            output_parts.append("See specialist outputs above for details.\n")

        output_parts.append("\n---\n")
        output_parts.append("*This analysis was performed by the DISSOLVE multi-agent system, "
                          "combining Separation Planning and Literature Search specialists.*\n")

    elif collaboration_mode == "literature_only":
        # Literature-only collaboration (direct search)
        literature_results = state.get("literature_results", {})

        output_parts.append("# Literature Search Results\n")
        output_parts.append(f"*Search completed in {elapsed:.1f}s*\n\n")

        if literature_results:
            papers = literature_results.get("papers_found", 0)
            kbs = literature_results.get("knowledgebases_searched", [])
            findings = literature_results.get("key_findings", [])
            confidence = literature_results.get("confidence_score", 0)

            output_parts.append(f"**Papers/passages found:** {papers}\n")
            output_parts.append(f"**Knowledgebases searched:** {', '.join(kbs)}\n")
            output_parts.append(f"**Confidence:** {confidence:.1%}\n\n")

            if findings:
                output_parts.append("## Key Findings\n")
                for finding in findings:
                    output_parts.append(f"- {finding}\n")
                output_parts.append("\n")

            # Show context extracted
            lit_polymers = literature_results.get("polymers_mentioned", [])
            lit_solvents = literature_results.get("solvents_mentioned", [])
            lit_temps = literature_results.get("temperatures_mentioned", [])

            if lit_polymers or lit_solvents or lit_temps:
                output_parts.append("## Context Extracted\n")
                if lit_polymers:
                    output_parts.append(f"- **Polymers:** {', '.join(lit_polymers)}\n")
                if lit_solvents:
                    output_parts.append(f"- **Solvents:** {', '.join(lit_solvents)}\n")
                if lit_temps:
                    output_parts.append(f"- **Temperatures:** {', '.join([f'{t}°C' for t in lit_temps])}\n")

        output_parts.append("\n---\n")
        output_parts.append("*Search performed by the DISSOLVE Literature Agent.*\n")

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
        state_copy["selected_categories"] = ["separation", "advanced_separation", "dissolution", "solvent_properties", "visualization"]
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

    # P0 Enhancement: Separation reviewer for quality validation
    builder.add_node("separation_reviewer", separation_reviewer_node)

    # Collaboration-aware TEA agent
    async def collab_tea_node(state):
        return await collab_tea_agent_node(state, sql_agent_node)

    builder.add_node("collab_tea_agent", collab_tea_node)
    builder.add_node("collab_tea_tools", async_tool_node_class(TEA_LCA_TOOLS or all_tools))

    # Collaboration-aware Literature agent
    async def collab_literature_node(state):
        return await collab_literature_agent_node(state, sql_agent_node)

    builder.add_node("collab_literature_agent", collab_literature_node)
    builder.add_node("collab_literature_tools", async_tool_node_class(LITERATURE_TOOLS or all_tools))

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

    # Integrated orchestrator -> First collaboration agent (based on mode)
    def route_from_orchestrator(state) -> Literal["collab_separation_agent", "collab_literature_agent"]:
        """Route to first specialist based on collaboration mode."""
        collab_mode = state.get("collaboration_mode", "")
        specialists = state.get("collaboration_specialists", [])

        # Literature-first collaborations
        if collab_mode == "literature_separation" or (specialists and specialists[0] == "literature"):
            return "collab_literature_agent"

        # Default: separation-first
        return "collab_separation_agent"

    builder.add_conditional_edges(
        "integrated_orchestrator",
        route_from_orchestrator,
        {
            "collab_separation_agent": "collab_separation_agent",
            "collab_literature_agent": "collab_literature_agent",
        }
    )

    # P0: Collab Separation Agent with Command-aware routing
    # P0 Enhancement: Routes to separation_reviewer for quality validation before TEA
    # Also supports literature routing for 3-way collaboration
    def route_collab_sep(state) -> Literal["collab_separation_tools", "separation_reviewer", "collab_tea_agent", "collab_literature_agent", "smart_aggregator", "__end__"]:
        """Route based on tool calls or Command. Command.goto takes precedence."""
        messages = state.get("messages", [])
        collab_mode = state.get("collaboration_mode", "")

        if not messages:
            # Check collaboration mode to determine next agent
            if collab_mode in ("separation_literature", "separation_literature_tea_lca"):
                return "collab_literature_agent"
            return "separation_reviewer"  # P0: Route to reviewer

        try:
            last_message = messages[-1]
            # If last message has tool calls, route to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "collab_separation_tools"
        except (IndexError, TypeError, AttributeError):
            pass

        # Route based on collaboration mode
        if collab_mode in ("separation_literature", "separation_literature_tea_lca"):
            return "collab_literature_agent"

        # Default: hand off to reviewer (Command will override this if returned)
        return "separation_reviewer"  # P0: Route to reviewer

    builder.add_conditional_edges(
        "collab_separation_agent",
        route_collab_sep,
        {
            "collab_separation_tools": "collab_separation_tools",
            "separation_reviewer": "separation_reviewer",  # P0: Reviewer destination
            "collab_tea_agent": "collab_tea_agent",
            "collab_literature_agent": "collab_literature_agent",
            "smart_aggregator": "smart_aggregator",
            "__end__": END,
        }
    )
    builder.add_edge("collab_separation_tools", "collab_separation_agent")

    # P0 Enhancement: Separation Reviewer with Command-based routing
    # Reviewer decides: proceed to TEA, retry separation, or aggregator (max retries)
    def route_reviewer(state) -> Literal["collab_separation_agent", "collab_tea_agent", "smart_aggregator", "__end__"]:
        """Route based on reviewer decision. Command.goto takes precedence."""
        # The reviewer returns Command objects, so LangGraph will use those
        # This function is the fallback for non-Command returns
        feedback = state.get("reviewer_feedback", {})
        if feedback.get("requires_revision"):
            return "collab_separation_agent"
        elif feedback.get("is_acceptable"):
            return "collab_tea_agent"
        else:
            return "smart_aggregator"

    builder.add_conditional_edges(
        "separation_reviewer",
        route_reviewer,
        {
            "collab_separation_agent": "collab_separation_agent",
            "collab_tea_agent": "collab_tea_agent",
            "smart_aggregator": "smart_aggregator",
            "__end__": END,
        }
    )

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

    # P4: Collab Literature Agent with Command-aware routing
    def route_collab_literature(state) -> Literal["collab_literature_tools", "collab_separation_agent", "smart_aggregator", "__end__"]:
        """Route based on tool calls or Command."""
        messages = state.get("messages", [])
        if not messages:
            return "smart_aggregator"

        try:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "collab_literature_tools"
        except (IndexError, TypeError, AttributeError):
            pass

        # Check collaboration mode to determine next agent
        collab_mode = state.get("collaboration_mode")
        if collab_mode == "literature_separation":
            return "collab_separation_agent"

        # Default: aggregate (Command will override if returned)
        return "smart_aggregator"

    builder.add_conditional_edges(
        "collab_literature_agent",
        route_collab_literature,
        {
            "collab_literature_tools": "collab_literature_tools",
            "collab_separation_agent": "collab_separation_agent",
            "smart_aggregator": "smart_aggregator",
            "__end__": END,
        }
    )
    builder.add_edge("collab_literature_tools", "collab_literature_agent")

    # Smart aggregator -> END
    builder.add_edge("smart_aggregator", END)

    # ========== COMPILE ==========

    # P1: Use configurable checkpointer (supports postgres, redis, or memory)
    checkpointer = CheckpointerConfig.get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("✅ Multi-Agent Graph compiled successfully!")
    logger.info(f"  Paths: fast, standard, specialist (separation, tea, literature), INTEGRATED")
    logger.info(f"  Tool subsets: fast={len(FAST_PATH_TOOLS)}, sep={len(SEPARATION_TOOLS)}, "
                f"tea={len(TEA_LCA_TOOLS)}, lit={len(LITERATURE_TOOLS)}")
    logger.info(f"  Collaboration: separation -> reviewer -> tea_lca, separation <-> literature (multi-KB)")
    logger.info(f"  Checkpointer: {type(checkpointer).__name__}")

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
    'collab_literature_agent_node',
    'select_knowledgebases',
    'parse_literature_results',
    # P0 Enhancement: Review/Revision loop
    'separation_reviewer_node',
    'SEPARATION_QUALITY_THRESHOLDS',
    # P1 Enhancement: Parallel execution & Supervisor
    'parallel_orchestrator_node',
    'supervisor_decision_node',
    'CheckpointerConfig',
    # P2 Enhancement: Cross-session store & Knowledge graph
    'SessionStore',
    'PolymerKnowledgeGraph',
    # Prompts
    'SEPARATION_PLANNER_PROMPT',
    'TEA_LCA_ANALYST_PROMPT',
    'LITERATURE_RESEARCHER_PROMPT',
    # Tool subsets
    'FAST_PATH_TOOLS',
    'SEPARATION_TOOLS',
    'TEA_LCA_TOOLS',
    'LITERATURE_TOOLS',
    'KB_KEYWORDS',
]
