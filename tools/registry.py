"""
P3.3: Tool Registry

Central registry for all tools with input/output contracts,
discovery mechanism, and execution management.
"""

from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel
import logging
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools available in the system."""
    SEPARATION = "separation"
    TEA = "tea"
    LITERATURE = "literature"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    DATABASE = "database"
    COMPARISON = "comparison"
    OPTIMIZATION = "optimization"


class ToolCapability(str, Enum):
    """Capabilities that tools can provide."""
    QUERY_DATABASE = "query_database"
    COMPUTE_SELECTIVITY = "compute_selectivity"
    OPTIMIZE_SEQUENCE = "optimize_sequence"
    ANALYZE_COSTS = "analyze_costs"
    SEARCH_LITERATURE = "search_literature"
    GENERATE_VISUALIZATION = "generate_visualization"
    COMPARE_OPTIONS = "compare_options"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "str", "int", "float", "list", "dict", "bool"
    required: bool = True
    default: Any = None
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)  # min, max, enum, etc.


@dataclass
class ToolContract:
    """Contract defining a tool's interface."""
    name: str
    description: str
    category: ToolCategory
    capabilities: List[ToolCapability] = field(default_factory=list)

    # Input/output specs
    input_parameters: List[ToolParameter] = field(default_factory=list)
    output_schema: Optional[str] = None  # Name of Pydantic model

    # Execution metadata
    is_async: bool = False
    timeout_seconds: int = 60
    max_output_tokens: int = 15000

    # Dependencies
    requires_database: bool = False
    requires_api_key: bool = False
    depends_on_tools: List[str] = field(default_factory=list)

    # Permissions
    agents_allowed: List[str] = field(default_factory=list)  # Empty = all

    def validate_input(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate input parameters against contract."""
        errors = []

        for param in self.input_parameters:
            if param.required and param.name not in params:
                errors.append(f"Missing required parameter: {param.name}")
                continue

            if param.name in params:
                value = params[param.name]

                # Type checking
                type_map = {
                    "str": str,
                    "int": int,
                    "float": (int, float),
                    "list": list,
                    "dict": dict,
                    "bool": bool,
                }
                expected_type = type_map.get(param.type)
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"Parameter {param.name}: expected {param.type}, got {type(value).__name__}")

                # Constraints
                if "min" in param.constraints and value < param.constraints["min"]:
                    errors.append(f"Parameter {param.name}: value {value} below minimum {param.constraints['min']}")
                if "max" in param.constraints and value > param.constraints["max"]:
                    errors.append(f"Parameter {param.name}: value {value} above maximum {param.constraints['max']}")
                if "enum" in param.constraints and value not in param.constraints["enum"]:
                    errors.append(f"Parameter {param.name}: value must be one of {param.constraints['enum']}")

        return len(errors) == 0, errors


@dataclass
class RegisteredTool:
    """A registered tool with its contract and implementation."""
    contract: ToolContract
    implementation: Callable
    is_active: bool = True
    call_count: int = 0
    total_time_ms: float = 0.0
    error_count: int = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        avg_time = self.total_time_ms / max(self.call_count, 1)
        error_rate = self.error_count / max(self.call_count, 1)
        return {
            "call_count": self.call_count,
            "avg_time_ms": avg_time,
            "error_rate": error_rate,
            "is_active": self.is_active,
        }


class ToolRegistry:
    """
    Central registry for tool management.

    Provides:
    - Tool registration with contracts
    - Tool discovery by category/capability
    - Input validation
    - Execution with metrics tracking
    - Agent-based access control
    """

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._tools: Dict[str, RegisteredTool] = {}
        self._by_category: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self._by_capability: Dict[ToolCapability, List[str]] = {cap: [] for cap in ToolCapability}
        self._initialized = True

    def register(
        self,
        name: str,
        implementation: Callable,
        description: str,
        category: ToolCategory,
        capabilities: List[ToolCapability] = None,
        input_parameters: List[ToolParameter] = None,
        output_schema: Optional[str] = None,
        is_async: bool = False,
        timeout_seconds: int = 60,
        requires_database: bool = False,
        agents_allowed: List[str] = None,
    ) -> None:
        """Register a tool with its contract."""
        contract = ToolContract(
            name=name,
            description=description,
            category=category,
            capabilities=capabilities or [],
            input_parameters=input_parameters or [],
            output_schema=output_schema,
            is_async=is_async,
            timeout_seconds=timeout_seconds,
            requires_database=requires_database,
            agents_allowed=agents_allowed or [],
        )

        self._tools[name] = RegisteredTool(
            contract=contract,
            implementation=implementation,
        )

        # Index by category
        if name not in self._by_category[category]:
            self._by_category[category].append(name)

        # Index by capabilities
        for cap in (capabilities or []):
            if name not in self._by_capability[cap]:
                self._by_capability[cap].append(name)

        logger.info(f"Registered tool: {name} ({category.value})")

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name not in self._tools:
            return False

        tool = self._tools[name]

        # Remove from indices
        category = tool.contract.category
        if name in self._by_category[category]:
            self._by_category[category].remove(name)

        for cap in tool.contract.capabilities:
            if name in self._by_capability[cap]:
                self._by_capability[cap].remove(name)

        del self._tools[name]
        return True

    def get(self, name: str) -> Optional[RegisteredTool]:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def get_contract(self, name: str) -> Optional[ToolContract]:
        """Get a tool's contract by name."""
        tool = self._tools.get(name)
        return tool.contract if tool else None

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        capability: Optional[ToolCapability] = None,
        agent: Optional[str] = None,
        active_only: bool = True,
    ) -> List[str]:
        """List tools matching criteria."""
        if category:
            names = self._by_category.get(category, [])
        elif capability:
            names = self._by_capability.get(capability, [])
        else:
            names = list(self._tools.keys())

        result = []
        for name in names:
            tool = self._tools.get(name)
            if not tool:
                continue

            # Filter by active status
            if active_only and not tool.is_active:
                continue

            # Filter by agent access
            if agent and tool.contract.agents_allowed:
                if agent not in tool.contract.agents_allowed:
                    continue

            result.append(name)

        return result

    def find_by_capability(self, capability: ToolCapability) -> List[ToolContract]:
        """Find all tools with a specific capability."""
        names = self._by_capability.get(capability, [])
        return [self._tools[n].contract for n in names if n in self._tools]

    def find_for_task(self, task_description: str) -> List[ToolContract]:
        """Find tools relevant to a task description (keyword matching)."""
        keywords_to_capabilities = {
            "separation": [ToolCapability.OPTIMIZE_SEQUENCE, ToolCapability.COMPUTE_SELECTIVITY],
            "selectivity": [ToolCapability.COMPUTE_SELECTIVITY],
            "cost": [ToolCapability.ANALYZE_COSTS],
            "tea": [ToolCapability.ANALYZE_COSTS],
            "economic": [ToolCapability.ANALYZE_COSTS],
            "literature": [ToolCapability.SEARCH_LITERATURE],
            "paper": [ToolCapability.SEARCH_LITERATURE],
            "compare": [ToolCapability.COMPARE_OPTIONS],
            "plot": [ToolCapability.GENERATE_VISUALIZATION],
            "visualize": [ToolCapability.GENERATE_VISUALIZATION],
        }

        task_lower = task_description.lower()
        relevant_caps = set()

        for keyword, caps in keywords_to_capabilities.items():
            if keyword in task_lower:
                relevant_caps.update(caps)

        if not relevant_caps:
            return []

        tools = []
        seen = set()
        for cap in relevant_caps:
            for contract in self.find_by_capability(cap):
                if contract.name not in seen:
                    tools.append(contract)
                    seen.add(contract.name)

        return tools

    async def execute(
        self,
        name: str,
        params: Dict[str, Any],
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a tool with validation and metrics tracking."""
        import time

        tool = self._tools.get(name)
        if not tool:
            return {"success": False, "error": f"Tool not found: {name}"}

        if not tool.is_active:
            return {"success": False, "error": f"Tool is inactive: {name}"}

        # Check agent access
        if agent and tool.contract.agents_allowed:
            if agent not in tool.contract.agents_allowed:
                return {"success": False, "error": f"Agent {agent} not allowed to use {name}"}

        # Validate input
        is_valid, errors = tool.contract.validate_input(params)
        if not is_valid:
            return {"success": False, "error": f"Validation failed: {errors}"}

        # Execute with timing
        start = time.time()
        try:
            if tool.contract.is_async:
                result = await asyncio.wait_for(
                    tool.implementation(**params),
                    timeout=tool.contract.timeout_seconds
                )
            else:
                result = tool.implementation(**params)

            elapsed_ms = (time.time() - start) * 1000

            # Update stats
            tool.call_count += 1
            tool.total_time_ms += elapsed_ms

            return {
                "success": True,
                "result": result,
                "execution_time_ms": elapsed_ms,
                "tool_name": name,
            }

        except asyncio.TimeoutError:
            tool.call_count += 1
            tool.error_count += 1
            return {"success": False, "error": f"Tool timeout after {tool.contract.timeout_seconds}s"}

        except Exception as e:
            tool.call_count += 1
            tool.error_count += 1
            return {"success": False, "error": str(e)}

    def execute_sync(
        self,
        name: str,
        params: Dict[str, Any],
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous execution wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.execute(name, params, agent)
        )

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all tools."""
        return {name: tool.get_stats() for name, tool in self._tools.items()}

    def get_schema_summary(self) -> str:
        """Generate a summary of all registered tools for LLM context."""
        lines = ["# Available Tools\n"]

        for category in ToolCategory:
            tools = self.list_tools(category=category)
            if not tools:
                continue

            lines.append(f"\n## {category.value.title()}\n")

            for name in tools:
                contract = self.get_contract(name)
                if not contract:
                    continue

                lines.append(f"### {name}")
                lines.append(f"{contract.description}\n")

                if contract.input_parameters:
                    lines.append("**Parameters:**")
                    for param in contract.input_parameters:
                        req = "required" if param.required else "optional"
                        lines.append(f"- `{param.name}` ({param.type}, {req}): {param.description}")

                lines.append("")

        return "\n".join(lines)

    def clear(self):
        """Clear all registered tools (for testing)."""
        self._tools.clear()
        self._by_category = {cat: [] for cat in ToolCategory}
        self._by_capability = {cap: [] for cap in ToolCapability}


# Global registry instance
_registry: Optional[ToolRegistry] = None

def get_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(
    name: str,
    description: str,
    category: ToolCategory,
    capabilities: List[ToolCapability] = None,
    input_parameters: List[ToolParameter] = None,
    output_schema: Optional[str] = None,
    is_async: bool = False,
    **kwargs
):
    """Decorator to register a function as a tool."""
    def decorator(func: Callable) -> Callable:
        get_registry().register(
            name=name,
            implementation=func,
            description=description,
            category=category,
            capabilities=capabilities,
            input_parameters=input_parameters,
            output_schema=output_schema,
            is_async=is_async or asyncio.iscoroutinefunction(func),
            **kwargs
        )
        return func
    return decorator


# ============================================================
# Pre-register core tools
# ============================================================

def _register_core_tools():
    """Register core tools from the system."""
    registry = get_registry()

    # Separation tools
    registry.register(
        name="find_optimal_separation_sequence",
        implementation=lambda **kwargs: {"status": "placeholder"},  # Will be replaced
        description="Find optimal separation sequence for a set of polymers using greedy, DP, or branch-and-bound algorithms",
        category=ToolCategory.SEPARATION,
        capabilities=[ToolCapability.OPTIMIZE_SEQUENCE, ToolCapability.QUERY_DATABASE],
        input_parameters=[
            ToolParameter(name="polymers", type="str", required=True, description="Comma-separated polymer names"),
            ToolParameter(name="temperature", type="float", required=False, default=80.0, description="Temperature in °C"),
            ToolParameter(name="algorithm", type="str", required=False, default="greedy",
                         constraints={"enum": ["greedy", "dp", "branch_and_bound"]}),
        ],
        output_schema="SeparationToolOutput",
        requires_database=True,
    )

    registry.register(
        name="calculate_selectivity_detailed",
        implementation=lambda **kwargs: {"status": "placeholder"},
        description="Calculate detailed selectivity metrics for polymer pairs",
        category=ToolCategory.SEPARATION,
        capabilities=[ToolCapability.COMPUTE_SELECTIVITY, ToolCapability.QUERY_DATABASE],
        input_parameters=[
            ToolParameter(name="polymer1", type="str", required=True),
            ToolParameter(name="polymer2", type="str", required=True),
            ToolParameter(name="temperature", type="float", required=False, default=80.0),
        ],
        output_schema="SeparationToolOutput",
        requires_database=True,
    )

    # TEA tools
    registry.register(
        name="analyze_solvent_recovery_tea",
        implementation=lambda **kwargs: {"status": "placeholder"},
        description="Perform techno-economic analysis for solvent recovery process",
        category=ToolCategory.TEA,
        capabilities=[ToolCapability.ANALYZE_COSTS],
        input_parameters=[
            ToolParameter(name="solvent", type="str", required=True),
            ToolParameter(name="throughput_kg_hr", type="float", required=False, default=100.0,
                         constraints={"min": 1, "max": 10000}),
            ToolParameter(name="recovery_rate", type="float", required=False, default=0.95,
                         constraints={"min": 0.5, "max": 0.99}),
        ],
        output_schema="TEAToolOutput",
    )

    registry.register(
        name="compare_solvents_tea_lca",
        implementation=lambda **kwargs: {"status": "placeholder"},
        description="Compare multiple solvents on TEA and LCA metrics",
        category=ToolCategory.TEA,
        capabilities=[ToolCapability.ANALYZE_COSTS, ToolCapability.COMPARE_OPTIONS],
        input_parameters=[
            ToolParameter(name="solvents", type="list", required=True),
            ToolParameter(name="throughput_kg_hr", type="float", required=False, default=100.0),
        ],
        output_schema="ComparisonToolOutput",
    )

    # Literature tools
    registry.register(
        name="search_literature_rag",
        implementation=lambda **kwargs: {"status": "placeholder"},
        description="Search literature using RAG knowledgebases",
        category=ToolCategory.LITERATURE,
        capabilities=[ToolCapability.SEARCH_LITERATURE],
        input_parameters=[
            ToolParameter(name="query", type="str", required=True),
            ToolParameter(name="knowledgebase", type="str", required=False, default="auto"),
            ToolParameter(name="max_results", type="int", required=False, default=10,
                         constraints={"min": 1, "max": 50}),
        ],
        output_schema="LiteratureToolOutput",
    )

    # Comparison tools
    registry.register(
        name="rank_solvents_for_separation",
        implementation=lambda **kwargs: {"status": "placeholder"},
        description="Rank solvents based on multiple criteria",
        category=ToolCategory.COMPARISON,
        capabilities=[ToolCapability.COMPARE_OPTIONS],
        input_parameters=[
            ToolParameter(name="polymers", type="str", required=True),
            ToolParameter(name="criterion", type="str", required=False, default="selectivity",
                         constraints={"enum": ["selectivity", "cost", "safety", "combined"]}),
            ToolParameter(name="top_k", type="int", required=False, default=5),
        ],
        output_schema="ComparisonToolOutput",
        requires_database=True,
    )


# Initialize core tools on module load
_register_core_tools()


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    # Enums
    "ToolCategory",
    "ToolCapability",
    # Data classes
    "ToolParameter",
    "ToolContract",
    "RegisteredTool",
    # Registry
    "ToolRegistry",
    "get_registry",
    "register_tool",
]
