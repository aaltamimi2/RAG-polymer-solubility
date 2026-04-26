"""DISSOLVE deep agent: wires model + tools + subagents + system prompt."""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

import deepagents.graph as deepagents_graph
from dotenv import load_dotenv

load_dotenv(override=True)

# Enable LangSmith tracing if keys are present
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "strap-agent")

from deepagents.backends import FilesystemBackend  # noqa: E402
from deepagents.graph import create_deep_agent  # noqa: E402
from deepagents.middleware.subagents import SubAgent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402

# Root directory for memory / skills (where AGENTS.md lives)
_PACKAGE_DIR = Path(__file__).parent

from .guardrails import SubagentGuardMiddleware  # noqa: E402
from .direct_fast_path import DirectToolFastPathMiddleware  # noqa: E402
from .prompts import FILE_IO_DIRECTIVE, THINK_DIRECTIVE, build_system_prompt  # noqa: E402
from .result_extractor import StructuredResultExtractorMiddleware  # noqa: E402
from .routing import RoutingMiddleware, generate_routing_table  # noqa: E402
from .planning.typed_runtime_integration import TypedRuntimeMiddleware  # noqa: E402
from .session_state import (  # noqa: E402
    append_transcript_event,
    build_session_context_block,
    inject_session_context,
    load_session_context,
    save_session_context,
    session_paths,
    should_inject_session_context,
    update_session_context_from_direct_metadata,
    update_session_context_from_text,
)
from .subagent_config import load_subagent_specs  # noqa: E402
from .traced_subagent_middleware import TracedSubAgentMiddleware  # noqa: E402
from .verifier import OutputVerifierMiddleware  # noqa: E402

from .tools import get_core_tools  # noqa: E402
from .tools import (  # noqa: E402  — tool group registry for YAML loader
    get_adaptive_separation_tools,
    get_biosteam_tools,
    get_contaminant_removal_tools,
    get_database_query_tools,
    get_interpolation_tools,
    get_ml_prediction_tools,
    get_patent_tools,
    get_rag_core_tools,
    get_rag_diagnostics_tools,
    get_reflection_tools,
    get_safety_card_tools,
    get_safety_gsk_tools,
    get_safety_pubchem_tools,
    get_scholar_tools,
    get_separation_core_tools,
    get_separation_plot_tools,
    get_result_extractor_tools,
    get_sidecar_read_tools,
    get_sidecar_write_tools,
    get_solvent_lookup_tools,
    get_statistical_tools,
    get_thermal_prediction_tools,
    get_visualization_tools,
    get_waste_optimization_tools,
)

# Map YAML tool_group names → getter functions
_TOOL_GROUP_REGISTRY: dict[str, callable] = {
    "database_query": get_database_query_tools,
    "separation_core": get_separation_core_tools,
    "adaptive_separation": get_adaptive_separation_tools,
    "safety_card": get_safety_card_tools,
    "safety_gsk": get_safety_gsk_tools,
    "safety_pubchem": get_safety_pubchem_tools,
    "scholar": get_scholar_tools,
    "patent": get_patent_tools,
    "rag_core": get_rag_core_tools,
    "rag_diagnostics": get_rag_diagnostics_tools,
    "visualization": get_visualization_tools,
    "separation_plot": get_separation_plot_tools,
    "statistical": get_statistical_tools,
    "ml_prediction": get_ml_prediction_tools,
    "thermal_prediction": get_thermal_prediction_tools,
    "interpolation": get_interpolation_tools,
    "biosteam": get_biosteam_tools,
    "contaminant_removal": get_contaminant_removal_tools,
    "waste_optimization": get_waste_optimization_tools,
    "solvent_lookup": get_solvent_lookup_tools,
    "reflection": get_reflection_tools,
    "sidecar_write": get_sidecar_write_tools,
    "sidecar_read": get_sidecar_read_tools,
    "result_extractor": get_result_extractor_tools,
}


def _get_cli_version() -> str:
    """Return the installed CLI version for startup rendering."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("strap-agent")
    except Exception:
        return "0.2.0"


def _build_startup_panel(
    *,
    accent: str,
    version_text: str,
    thread_id: str,
    use_checkpointer: bool,
    model_alias: str = "",
    model_name: str = "",
) -> object:
    """Build the startup splash panel for interactive CLI sessions."""
    import getpass

    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel

    logo = Text(justify="center")
    for line in (
        "   ____  _________ __________ _    _    ______",
        "  / __ \\/  _/ ___// ___/ __ \\ |  / |  / / __/",
        " / / / // / \\__ \\/ /__/ /_/ / | /| | / / _/  ",
        "/_/ /_/___/____/\\___/\\____/|__/ |_|/_/___/  ",
    ):
        logo.append(line + "\n", style=f"bold {accent}")
    logo.append("Contaminant-aware solubility orchestration", style=f"{accent}")

    status = Table.grid(padding=(0, 1))
    status.add_row(Text(f"Welcome back {getpass.getuser()}", style="bold white"))
    status.add_row(Text(f"DISSOLVE v{version_text}", style=f"bold {accent}"))
    status.add_row(Text("Separation · Contaminants · TEA/LCA · Safety", style="white"))
    status.add_row(Text(str(Path.cwd()), style="dim"))
    if model_alias:
        model_text = f"{model_alias}"
        if model_name:
            model_text += f" ({model_name})"
        status.add_row(Text(f"Model {model_text}", style="dim"))
    if os.getenv("LANGSMITH_API_KEY"):
        status.add_row(Text("LangSmith tracing enabled", style="green"))
    else:
        status.add_row(Text("LangSmith tracing disabled", style="yellow"))
    if use_checkpointer:
        status.add_row(Text(f"Session {thread_id}", style="dim"))
    else:
        status.add_row(Text("Session persistence disabled", style="dim"))
    status.add_row(Text("Use /model to switch; quit to exit", style="dim"))

    layout = Table.grid(expand=True)
    layout.add_column(ratio=3, justify="center")
    layout.add_column(ratio=2)
    layout.add_row(logo, status)

    return Panel(
        layout,
        border_style=accent,
        title="[bold white]DISSOLVE CLI[/]",
        subtitle="[dim]interactive session[/]",
        padding=(1, 2),
    )


def _answer_prefers_plain_terminal(answer: str) -> bool:
    """Return True for pre-rendered terminal cards that Markdown would collapse."""

    text = str(answer or "").lstrip()
    if not text.startswith("╭"):
        return False
    header = text[:240]
    return "DISSOLVE SAFETY CARD" in header or "DISSOLVE SAFETY COMPARISON" in header


def _show_startup_animation(
    console,
    *,
    version_text: str,
    thread_id: str,
    use_checkpointer: bool,
    model_alias: str = "",
    model_name: str = "",
) -> None:
    """Render a short blue startup animation for interactive sessions only."""
    import sys
    import time

    from rich.live import Live

    if not (console.is_terminal and sys.stdin.isatty() and sys.stderr.isatty() and not os.getenv("CI")):
        return

    accents = ["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#3b82f6", "#2563eb"]
    with Live(console=console, transient=True, refresh_per_second=20) as live:
        for accent in accents:
            live.update(
                _build_startup_panel(
                    accent=accent,
                    version_text=version_text,
                    thread_id=thread_id,
                    use_checkpointer=use_checkpointer,
                    model_alias=model_alias,
                    model_name=model_name,
                )
            )
            time.sleep(0.07)

    console.print(
        _build_startup_panel(
            accent="#3b82f6",
            version_text=version_text,
            thread_id=thread_id,
            use_checkpointer=use_checkpointer,
            model_alias=model_alias,
            model_name=model_name,
        )
    )
    console.print()

def _create_session_scratch_dir() -> Path:
    """Create a per-session temp directory for sidecar files."""
    scratch = Path(tempfile.mkdtemp(prefix="strap_scratch_"))
    atexit.register(shutil.rmtree, scratch, True)
    logger.info("Session scratch directory: %s", scratch)
    return scratch

class SubagentOverride(TypedDict, total=False):
    """Per-subagent guardrail overrides for create_dissolve_agent().

    All fields are optional and correspond to SubagentGuardMiddleware
    constructor parameters.
    """
    max_iterations: int
    token_budget: int
    max_tool_calls: int
    synthesis_tools: set
    truncate_tool_results_after: int | None
    free_tools: set


class CliModelSpec(TypedDict, total=False):
    """CLI-facing model metadata."""

    label: str
    model: str
    provider: str
    env_var: str


_DEFAULT_CLI_MODEL_ALIAS = "gemini-flash-lite"
_CLI_MODEL_PRIMARY_ALIASES = ("gemini-flash-lite", "gemini-pro", "claude-sonnet")
_DEFAULT_CLI_INTERACTION_MODE = "review"
_CLI_INTERACTION_MODES = {
    "review": {
        "label": "Review defaults",
        "description": "Ask before applying tunable CLI defaults",
    },
    "auto": {
        "label": "Auto defaults",
        "description": "Skip CLI review and let agent/tools use defaults",
    },
}
_BIOSTEAM_CLARIFICATION_RE = re.compile(
    r"\b("
    r"biosteam|tea|lca|msp|tci|aoc|capex|opex|gwp|"
    r"techno[- ]economic|process simulation|carbon footprint|co2e?"
    r")\b",
    re.IGNORECASE,
)
_ENERGY_CASE_RE = re.compile(r"\b(?:energy\s+case\s*)?C[123]\b", re.IGNORECASE)
_ALL_ENERGY_CASES_RE = re.compile(
    r"\b(all|each|every|compare|across)\s+(?:three\s+)?energy\s+cases\b|"
    r"\bC1\s*,\s*C2\s*,?\s*(?:and\s*)?C3\b",
    re.IGNORECASE,
)
_CAPACITY_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:metric\s*)?(?:tonnes?|tons?|mt|tpy)\s*(?:/|per)?\s*(?:year|yr|annum)?\b|"
    r"\bprocessing\s+capacity\b|\bfeed\s+capacity\b|\bcapacity\b",
    re.IGNORECASE,
)
_TARGET_FRACTION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:wt\s*%|weight\s*%|%)\s*(?:[A-Za-z0-9/-]+)?\b|"
    r"\btarget_plastic_percent\b|\bfeed\s+fraction\b|\bfeed\s+composition\b",
    re.IGNORECASE,
)
_PRECIPITATION_TEMP_RE = re.compile(
    r"\bprecip(?:itation|itate|itating)?\b|\bcooling\s+temperature\b|"
    r"\bprecipitation_temp_c\b",
    re.IGNORECASE,
)
_METRIC_PATTERNS = (
    ("MSP", re.compile(r"\bmsp\b|minimum\s+selling\s+price", re.IGNORECASE)),
    ("TCI/CAPEX", re.compile(r"\btci\b|\bcapex\b|capital\s+cost", re.IGNORECASE)),
    ("AOC/OPEX", re.compile(r"\baoc\b|\bopex\b|operating\s+cost", re.IGNORECASE)),
    ("GWP", re.compile(r"\bgwp\b|carbon\s+footprint|co2e?", re.IGNORECASE)),
)
_BIOSTEAM_METRIC_PRESETS = (
    "MSP, TCI/CAPEX, AOC/OPEX, GWP",
    "MSP, GWP",
    "TCI/CAPEX, AOC/OPEX",
    "Full TEA/LCA summary",
)
_BIOSTEAM_CAPACITY_PRESETS = ("8,000 MT/yr", "20,000 MT/yr", "50,000 MT/yr", "100,000 MT/yr")
_BIOSTEAM_TARGET_FRACTION_PRESETS = (
    "10 wt% target plastic in feed",
    "25 wt% target plastic in feed",
    "50 wt% target plastic in feed",
    "60 wt% target plastic in feed",
    "75 wt% target plastic in feed",
    "90 wt% target plastic in feed",
)
_BIOSTEAM_PRECIPITATION_TEMP_PRESETS = ("5 C", "25 C", "40 C")


def _cli_model_registry() -> dict[str, CliModelSpec]:
    """Return the configured model aliases available to the CLI."""
    claude_model = os.getenv("DISSOLVE_CLAUDE_SONNET_MODEL", "anthropic:claude-sonnet-4-6")
    return {
        "gemini": {
            "label": "Gemini 3.1 Flash Lite",
            "model": "google_genai:gemini-3.1-flash-lite-preview",
            "provider": "Google",
            "env_var": "GOOGLE_API_KEY",
        },
        "gemini-flash-lite": {
            "label": "Gemini 3.1 Flash Lite",
            "model": "google_genai:gemini-3.1-flash-lite-preview",
            "provider": "Google",
            "env_var": "GOOGLE_API_KEY",
        },
        "gemini-pro": {
            "label": "Gemini 3.1 Pro",
            "model": "google_genai:gemini-3.1-pro-preview",
            "provider": "Google",
            "env_var": "GOOGLE_API_KEY",
        },
        "claude": {
            "label": "Claude Sonnet 4.6",
            "model": claude_model,
            "provider": "Anthropic",
            "env_var": "ANTHROPIC_API_KEY",
        },
        "claude-sonnet": {
            "label": "Claude Sonnet 4.6",
            "model": claude_model,
            "provider": "Anthropic",
            "env_var": "ANTHROPIC_API_KEY",
        },
    }


def _resolve_cli_model(raw_model: str | None = None) -> tuple[str, CliModelSpec]:
    """Resolve a CLI alias or provider-qualified model id."""
    registry = _cli_model_registry()
    requested = (raw_model or os.getenv("STRAP_MODEL") or _DEFAULT_CLI_MODEL_ALIAS).strip()
    if not requested:
        requested = _DEFAULT_CLI_MODEL_ALIAS

    if requested in registry:
        return requested, registry[requested]

    for alias, spec in registry.items():
        if requested == spec["model"]:
            return alias, spec

    provider = requested.split(":", 1)[0] if ":" in requested else "custom"
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "GOOGLE_API_KEY" if provider == "google_genai" else ""
    return requested, {
        "label": requested,
        "model": requested,
        "provider": provider,
        "env_var": env_var,
    }


def _model_env_status(spec: CliModelSpec) -> str:
    """Return whether the model provider appears configured."""
    env_var = spec.get("env_var", "")
    if not env_var:
        return "unknown"
    return "configured" if os.getenv(env_var) else f"missing {env_var}"


def _resolve_cli_interaction_mode(raw_mode: str | None = None) -> str:
    """Resolve CLI interaction mode aliases."""
    requested = (raw_mode or os.getenv("DISSOLVE_CLI_MODE") or _DEFAULT_CLI_INTERACTION_MODE).strip().lower()
    aliases = {
        "interactive": "review",
        "human": "review",
        "hitl": "review",
        "human-in-the-loop": "review",
        "confirm": "review",
        "review": "review",
        "auto": "auto",
        "automatic": "auto",
        "defaults": "auto",
        "fast": "auto",
    }
    return aliases.get(requested, _DEFAULT_CLI_INTERACTION_MODE)


def _biosteam_energy_case_options() -> list[dict[str, str]]:
    """Return CLI-selectable BioSTEAM energy cases."""
    try:
        from strap.vendor.biosteam_runner import _ENERGY_CASES

        cases = dict(_ENERGY_CASES)
    except Exception:
        cases = {
            "C1": "CHP (on-site boiler + turbogenerator)",
            "C2": "Grid + AMCOR (no on-site utilities)",
            "C3": "Grid + Boiler (boiler but no turbogenerator)",
        }
    return [
        {"value": code, "label": code, "description": description}
        for code, description in cases.items()
    ]


def _query_needs_energy_case_clarification(query: str) -> bool:
    """Whether a BioSTEAM-like query needs a CLI energy-case choice."""
    if not _BIOSTEAM_CLARIFICATION_RE.search(query):
        return False
    if _ALL_ENERGY_CASES_RE.search(query):
        return False
    return not bool(_ENERGY_CASE_RE.search(query))


def _biosteam_requested_metrics(query: str) -> list[str]:
    """Return explicitly requested BioSTEAM metric labels."""
    return [label for label, pattern in _METRIC_PATTERNS if pattern.search(query)]


def _cycle_choice(current: str, choices: tuple[str, ...]) -> str:
    """Return the next choice, defaulting to the first when current is unknown."""
    try:
        index = choices.index(current)
    except ValueError:
        return choices[0]
    return choices[(index + 1) % len(choices)]


def _biosteam_initial_run_settings(query: str) -> dict[str, dict[str, str]]:
    """Build editable BioSTEAM run settings from user query + defaults."""
    energy_option = _biosteam_energy_case_options()[0]
    metrics = _biosteam_requested_metrics(query)
    return {
        "energy_case": {
            "label": "Energy case",
            "value": energy_option["value"],
            "description": energy_option["description"],
            "source": "default",
        },
        "metrics": {
            "label": "Metrics",
            "value": ", ".join(metrics) if metrics else _BIOSTEAM_METRIC_PRESETS[0],
            "source": "user" if metrics else "default",
        },
        "capacity": {
            "label": "Capacity",
            "value": "provided in prompt" if _CAPACITY_RE.search(query) else "20,000 MT/yr",
            "source": "user" if _CAPACITY_RE.search(query) else "default",
        },
        "target_fraction": {
            "label": "Target fraction",
            "value": "provided in prompt" if _TARGET_FRACTION_RE.search(query) else "60 wt% target plastic in feed",
            "source": "user" if _TARGET_FRACTION_RE.search(query) else "default",
        },
        "precipitation_temp": {
            "label": "Precipitation temp",
            "value": "provided in prompt" if _PRECIPITATION_TEMP_RE.search(query) else "25 C",
            "source": "user" if _PRECIPITATION_TEMP_RE.search(query) else "default",
        },
    }


def _biosteam_settings_to_rows(settings: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Return menu rows for editable BioSTEAM run settings."""
    rows = []
    for key in ("energy_case", "metrics", "capacity", "target_fraction", "precipitation_temp"):
        setting = settings[key]
        value = setting["value"]
        if key == "energy_case":
            value = f"{setting['value']} - {setting['description']}"
        rows.append(
            {
                "key": key,
                "setting": setting["label"],
                "value": value,
                "source": setting["source"],
            }
        )
    rows.append({"key": "continue", "setting": "Continue", "value": "Run with these settings", "source": ""})
    return rows


def _biosteam_run_basis_rows(query: str) -> list[dict[str, str]]:
    """Return compact user/default basis rows for BioSTEAM clarification."""
    settings = _biosteam_initial_run_settings(query)
    return [
        {
            "parameter": "Metrics",
            "value": settings["metrics"]["value"],
            "source": settings["metrics"]["source"],
            "append": "",
        },
        {
            "parameter": "Capacity",
            "value": settings["capacity"]["value"],
            "source": settings["capacity"]["source"],
            "append": "" if settings["capacity"]["source"] == "user" else f"processing_capacity = {settings['capacity']['value']}",
        },
        {
            "parameter": "Target fraction",
            "value": settings["target_fraction"]["value"],
            "source": settings["target_fraction"]["source"],
            "append": "" if settings["target_fraction"]["source"] == "user" else f"target_plastic_percent = {settings['target_fraction']['value']}",
        },
        {
            "parameter": "Precipitation temp",
            "value": settings["precipitation_temp"]["value"],
            "source": settings["precipitation_temp"]["source"],
            "append": "" if settings["precipitation_temp"]["source"] == "user" else f"precipitation_temp_c = {settings['precipitation_temp']['value']}",
        },
    ]


def _append_biosteam_run_settings_clarification(query: str, settings: dict[str, dict[str, str]]) -> str:
    """Append selected BioSTEAM run settings as explicit agent context."""
    energy = settings["energy_case"]
    clarification = (
        f"{query.rstrip()}\n\n"
        f"CLI clarification: use BioSTEAM energy case {energy['value']} ({energy['description']})."
    )
    basis_lines = []
    for key in ("metrics", "capacity", "target_fraction", "precipitation_temp"):
        setting = settings[key]
        if setting["source"] in {"default", "override"}:
            basis_lines.append(f"{setting['label']} = {setting['value']} ({setting['source']})")
    if basis_lines:
        clarification += (
            "\nCLI run basis: "
            + "; ".join(basis_lines)
            + ". State this run basis in the final answer."
        )
    return clarification


def _append_energy_case_clarification(
    query: str,
    option: dict[str, str],
    basis_rows: list[dict[str, str]] | None = None,
) -> str:
    """Append the selected energy case and default basis as explicit context."""
    value = option["value"]
    description = option["description"]
    clarification = (
        f"{query.rstrip()}\n\n"
        f"CLI clarification: use BioSTEAM energy case {value} ({description})."
    )
    default_assumptions = [
        row["append"]
        for row in (basis_rows or _biosteam_run_basis_rows(query))
        if row.get("source") == "default" and row.get("append")
    ]
    if default_assumptions:
        clarification += (
            "\nCLI assumptions/defaults because these were not specified: "
            + "; ".join(default_assumptions)
            + ". State these defaults in the final answer."
        )
    return clarification


def _resolve_tools(group_names: list[str], registry: dict | None = None) -> list:
    """Resolve YAML tool_group names to actual tool function lists."""
    r = registry if registry is not None else _TOOL_GROUP_REGISTRY
    tools = []
    for name in group_names:
        getter = r.get(name)
        if getter:
            tools.extend(getter())
    return tools


# Tools that should never count against the subagent tool-call budget.
# Keep file handoff helpers free, but bill exploratory filesystem tools.
_ALWAYS_FREE_TOOLS = {
    "write_file", "read_file",  # inter-agent communication
    "write_todos",  # planning/meta
}


def _resolve_guardrails(cfg: dict | None, *, agent_name: str | None = None) -> list:
    """Build middleware list from YAML guardrails config."""
    if cfg is None:
        return [SubagentGuardMiddleware(
            free_tools=_ALWAYS_FREE_TOOLS.copy(),
            agent_name=agent_name,
        )]
    free = set(cfg["free_tools"]) if cfg.get("free_tools") else set()
    free |= _ALWAYS_FREE_TOOLS
    return [SubagentGuardMiddleware(
        max_iterations=cfg.get("max_iterations", 25),
        token_budget=cfg.get("token_budget", 200_000),
        max_tool_calls=cfg.get("max_tool_calls", 10),
        synthesis_tools=set(cfg["synthesis_tools"]) if cfg.get("synthesis_tools") else set(),
        truncate_tool_results_after=cfg.get("truncate_tool_results_after"),
        free_tools=free,
        agent_name=agent_name,
    )]


def _build_subagents(
    yaml_path: str | Path | None = None,
    overrides: dict[str, SubagentOverride] | None = None,
) -> list[SubAgent]:
    """Load subagent definitions from YAML config.

    Falls back to the shared subagent config manifest next to this module.
    The ``THINK_DIRECTIVE`` is appended to every system prompt automatically.

    Args:
        yaml_path: Optional manifest, directory, or legacy YAML path.
        overrides: Optional dict mapping subagent name -> guardrail overrides.
            Override keys mirror ``SubagentGuardMiddleware`` parameters:
            ``max_tool_calls``, ``max_iterations``, ``token_budget``,
            ``synthesis_tools``, ``truncate_tool_results_after``, ``free_tools``.
    """
    specs = load_subagent_specs(yaml_path)

    _overrides = overrides or {}

    # Warn about override keys that don't match any known subagent name
    known_names = {spec["name"] for spec in specs}
    for name in _overrides:
        if name not in known_names:
            warnings.warn(
                f"subagent_overrides contains unknown agent '{name}'. "
                f"Known agents: {sorted(known_names)}",
                stacklevel=3,
            )

    subagents: list[SubAgent] = []
    for spec in specs:
        missing_field = False
        for required_field in ("name", "system_prompt", "description"):
            if required_field not in spec:
                logger.warning(
                    "subagents.yaml entry missing required field '%s': %s",
                    required_field,
                    spec.get("name", "<unnamed>"),
                )
                missing_field = True
        if missing_field:
            continue
        agent_name = spec["name"]
        guardrail_cfg = dict(spec.get("guardrails") or {})
        if agent_name in _overrides:
            guardrail_cfg.update(_overrides[agent_name])

        prompt = spec["system_prompt"].rstrip() + FILE_IO_DIRECTIVE + THINK_DIRECTIVE
        sa = SubAgent(
            name=agent_name,
            description=spec["description"].strip(),
            system_prompt=prompt,
            tools=_resolve_tools(spec.get("tool_groups", [])),
            middleware=_resolve_guardrails(guardrail_cfg, agent_name=agent_name),
        )
        subagents.append(sa)

    return subagents


def _get_or_create_thread_id(session_arg: str | None = None) -> str:
    """Return *session_arg* unchanged, or generate a fresh 12-hex-char thread ID."""
    if session_arg:
        return session_arg
    return uuid.uuid4().hex[:12]


def create_dissolve_agent(
    model_name: str = os.getenv("STRAP_MODEL", "google_genai:gemini-3.1-flash-lite-preview"),
    subagent_overrides: dict[str, SubagentOverride] | None = None,
    checkpointer=None,
    enable_persistence: bool = False,
):
    """Create and return a compiled DISSOLVE deep agent with subagents.

    Uses progressive loading:
    - ``memory`` (AGENTS.md) is always injected into the system prompt.
    - ``skills`` (skills/\\*) are loaded on demand by SkillsMiddleware.
    - ``system_prompt`` carries only the dynamic routing table.

    Args:
        model_name: LangChain model identifier (provider:model-name format).
        subagent_overrides: Optional per-subagent guardrail overrides.
            Dict mapping subagent name -> override fields.
            See :class:`SubagentOverride` for valid keys.
        checkpointer: Optional LangGraph checkpointer instance.  When provided,
            the agent graph is compiled with cross-turn memory.  Pass any
            ``BaseCheckpointSaver`` (e.g. ``MemorySaver``, ``SqliteSaver``).
        enable_persistence: When ``True`` and *checkpointer* is ``None``,
            automatically create an in-process ``MemorySaver`` checkpointer so
            that conversation state survives across ``invoke()`` calls within
            the same Python process.  For durable disk persistence, pass a
            ``SqliteSaver`` via *checkpointer* instead.

            Example::

                create_dissolve_agent(subagent_overrides={
                    "separation-engineer": {"max_tool_calls": 30},
                })
    """
    if enable_persistence and checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    model = init_chat_model(model_name)

    # Lightweight Gemini Flash model shared by both the routing classifier
    # and the output verifier — single instance, no extra cost.
    flash_model = init_chat_model("google_genai:gemini-3-flash-preview")

    # Semantic routing: LLM-based classifier with keyword fallback
    routing = RoutingMiddleware(classifier_model=flash_model)

    # Deterministic direct-tool fast path: simple structured lookups can
    # render core tool output without spending a model call on synthesis.
    direct_fast_path = DirectToolFastPathMiddleware()

    # Opt-in typed runtime: selected complex workflows compile to a typed plan
    # and execute through production wrappers before legacy advisory routing.
    typed_runtime = TypedRuntimeMiddleware()

    # Output verifier: single reflection pass on the orchestrator's
    # final synthesis to catch unsupported claims / missing caveats.
    output_verifier = OutputVerifierMiddleware(verifier_model=flash_model)

    # Orchestrator-level guardrails: cap total token usage across the run.
    # task/read_file/write_file/write_todos are free so delegation chains
    # don't eat the budget — only analysis tools count.
    orchestrator_guard = SubagentGuardMiddleware(
        max_iterations=50,
        token_budget=500_000,
        max_tool_calls=12,
        truncate_tool_results_after=3000,
        free_tools={"think", "task", "read_file", "write_file", "write_todos",
                    "get_subagent_result", "get_subagent_results",
                    "get_all_subagent_results", "list_handoffs",
                    "get_handoff_details", "build_handoff"},
    )

    # Set up per-agent scratch root for scoped sidecar artifacts
    scratch_dir = _create_session_scratch_dir()

    # Structured result extractor: intercepts task() ToolMessages and
    # extracts <STRUCTURED_RESULT> JSON blocks into a per-invocation registry.
    result_extractor = StructuredResultExtractorMiddleware(artifact_root=scratch_dir)

    # Middleware order (innermost → outermost):
    #   direct_fast_path → typed_runtime → routing → output_verifier → result_extractor → orchestrator_guard
    original_subagent_middleware = deepagents_graph.SubAgentMiddleware
    deepagents_graph.SubAgentMiddleware = TracedSubAgentMiddleware
    try:
        agent = create_deep_agent(
            model=model,
            tools=get_core_tools() + get_result_extractor_tools(),
            subagents=_build_subagents(overrides=subagent_overrides),
            system_prompt=build_system_prompt(generate_routing_table()),
            memory=["./AGENTS.md"],
            skills=["./skills/"],
            backend=FilesystemBackend(root_dir=str(_PACKAGE_DIR)),
            middleware=[direct_fast_path, typed_runtime, routing, output_verifier, result_extractor, orchestrator_guard],
            name="dissolve-agent",
            checkpointer=checkpointer,
        )
    finally:
        deepagents_graph.SubAgentMiddleware = original_subagent_middleware
    return agent


# Keep backward-compatible alias
create_strap_agent = create_dissolve_agent


def _extract_text(content) -> str:
    """Extract plain text from an AI message content field.

    Handles both plain strings and list-of-dicts (Gemini format).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def main():
    """Interactive CLI — clean output inspired by Claude Code / Codex."""
    import argparse
    import logging
    import readline  # noqa: F401 — enables arrow-key editing in input()
    import sys
    import time

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.spinner import Spinner as RichSpinner
    from rich.live import Live
    from rich.text import Text

    # ── Parse CLI arguments ──
    parser = argparse.ArgumentParser(
        prog="dissolve",
        description="DISSOLVE — Data Integrated Solubility Solver via LLM Evaluation",
        add_help=True,
    )
    parser.add_argument(
        "--session",
        metavar="SESSION_ID",
        default=None,
        help="Resume a previous session by its ID (printed at startup). "
             "Omit to start a new session.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        default=False,
        help="Disable in-process MemorySaver persistence (use raw history list instead).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help="Initial model alias or provider-qualified model id. Use /model in the CLI to switch.",
    )
    parser.add_argument(
        "--mode",
        metavar="MODE",
        default=None,
        choices=("review", "auto"),
        help="CLI interaction mode: review asks before tunable defaults; auto skips clarification.",
    )
    args = parser.parse_args()

    use_checkpointer = not args.no_persist
    thread_id = _get_or_create_thread_id(args.session)
    durable_session_paths = session_paths(thread_id)
    durable_session_paths["dir"].mkdir(parents=True, exist_ok=True)
    session_context = load_session_context(thread_id)
    version_text = _get_cli_version()
    current_model_alias, current_model_spec = _resolve_cli_model(args.model)
    current_model_name = current_model_spec["model"]
    current_interaction_mode = _resolve_cli_interaction_mode(args.mode)

    console = Console(stderr=True)

    # ── Suppress all library logging for clean CLI output ──
    logging.disable(logging.CRITICAL)

    # ── Banner / startup splash ──
    _show_startup_animation(
        console,
        version_text=version_text,
        thread_id=thread_id,
        use_checkpointer=use_checkpointer,
        model_alias=current_model_alias,
        model_name=current_model_name,
    )
    if not (console.is_terminal and sys.stdin.isatty() and sys.stderr.isatty() and not os.getenv("CI")):
        console.print()
        console.print(
            Text.assemble(
                ("DISSOLVE", "bold cyan"),
                (f" v{version_text}", "dim"),
            )
        )
        console.print("[dim]Data Integrated Solubility Solver via LLM Evaluation[/]")
        if os.getenv("LANGSMITH_API_KEY"):
            console.print("[dim]LangSmith tracing:[/] [green]enabled[/]")
        if use_checkpointer:
            console.print(
                f"[dim]Session ID:[/] [bold]{thread_id}[/bold]  "
                "[dim](pass --session {id} to resume)[/]"
            )
            console.print(f"[dim]Session files:[/] {durable_session_paths['dir']}")
        console.print(
            f"[dim]Model:[/] [bold]{current_model_alias}[/bold] "
            f"[dim]({current_model_name})[/]"
        )
        mode_spec = _CLI_INTERACTION_MODES[current_interaction_mode]
        console.print(
            f"[dim]Mode:[/] [bold]{current_interaction_mode}[/bold] "
            f"[dim]({mode_spec['label']})[/]"
        )
        console.print("[dim]Type [bold]quit[/bold] to exit. Use [bold]/model[/bold] or [bold]/mode[/bold] to configure.[/]\n")

    # ── Load agent with spinner ──
    checkpointer = None
    if use_checkpointer:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            sqlite_saver = SqliteSaver.from_conn_string(str(durable_session_paths["checkpoint"]))
            if hasattr(sqlite_saver, "__enter__"):
                checkpointer = sqlite_saver.__enter__()
                atexit.register(sqlite_saver.__exit__, None, None, None)
            else:
                checkpointer = sqlite_saver
        except Exception:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()

    with Live(
        RichSpinner("dots", text=Text(f"Loading {current_model_alias}...", style="dim")),
        console=console,
        transient=True,
    ):
        agent = create_dissolve_agent(
            model_name=current_model_name,
            checkpointer=checkpointer,
        )

    out = Console()  # stdout console for answers
    history: list = []  # used only when checkpointer is disabled
    prompt_session = None

    if out.is_terminal and sys.stdin.isatty() and sys.stdout.isatty() and not os.getenv("CI"):
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.key_binding import KeyBindings

            prompt_keys = KeyBindings()

            @prompt_keys.add("enter")
            def _(event) -> None:
                event.current_buffer.validate_and_handle()

            prompt_session = PromptSession(
                multiline=True,
                key_bindings=prompt_keys,
                prompt_continuation="  ",
            )
        except Exception:
            prompt_session = None

    # Build LangGraph config for checkpointer-aware invocations
    invoke_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 150}
    # When no checkpointer, keep recursion_limit but skip thread_id
    plain_config = {"recursion_limit": 150}

    def _model_rows() -> list[dict[str, object]]:
        registry = _cli_model_registry()
        rows: list[dict[str, object]] = []
        seen_models: set[str] = set()
        for alias in _CLI_MODEL_PRIMARY_ALIASES:
            spec = registry[alias]
            model_name = spec["model"]
            if model_name in seen_models:
                continue
            seen_models.add(model_name)
            alias_group = [
                name
                for name, candidate in registry.items()
                if candidate["model"] == model_name
            ]
            rows.append(
                {
                    "alias": alias,
                    "selector_alias": f"*{alias}" if current_model_name == model_name else alias,
                    "aliases": alias_group,
                    "spec": spec,
                    "model": model_name,
                    "provider": spec["provider"],
                    "status": _model_env_status(spec),
                    "active": current_model_name == model_name,
                }
            )

        if current_model_name not in seen_models:
            rows.append(
                {
                    "alias": current_model_alias,
                    "selector_alias": f"*{current_model_alias}",
                    "aliases": [current_model_alias],
                    "spec": current_model_spec,
                    "model": current_model_name,
                    "provider": current_model_spec.get("provider", "custom"),
                    "status": _model_env_status(current_model_spec),
                    "active": True,
                }
            )
        return rows

    def _mode_rows() -> list[dict[str, object]]:
        return [
            {
                "mode": mode,
                "label": spec["label"],
                "description": spec["description"],
                "status": "active" if mode == current_interaction_mode else "",
                "active": mode == current_interaction_mode,
            }
            for mode, spec in _CLI_INTERACTION_MODES.items()
        ]

    def _print_model_table() -> None:
        from rich.table import Table

        table = Table(title="DISSOLVE Models", show_lines=False)
        table.add_column("Alias", style="bold cyan")
        table.add_column("Model")
        table.add_column("Provider")
        table.add_column("Status")
        for row in _model_rows():
            aliases_text = ", ".join(row["aliases"])
            marker = "*" if row["active"] else ""
            table.add_row(
                f"{marker}{aliases_text}",
                row["model"],
                row["provider"],
                row["status"],
            )
        out.print(table)
        out.print("[dim]Use /model to open the selector, /model <alias> to switch directly, or /model current to show the active model.[/]")

    def _print_mode_table() -> None:
        from rich.table import Table

        table = Table(title="DISSOLVE Modes", show_lines=False)
        table.add_column("Mode", style="bold cyan")
        table.add_column("Label")
        table.add_column("Description")
        table.add_column("Status")
        for row in _mode_rows():
            marker = "*" if row["active"] else ""
            table.add_row(
                f"{marker}{row['mode']}",
                row["label"],
                row["description"],
                row["status"],
            )
        out.print(table)
        out.print("[dim]Use /mode to open the selector, /mode review for human-in-the-loop defaults, or /mode auto to skip CLI review.[/]")

    def _switch_model(target: str) -> bool:
        nonlocal agent, current_model_alias, current_model_spec, current_model_name

        next_alias, next_spec = _resolve_cli_model(target)
        next_model_name = next_spec["model"]
        env_status = _model_env_status(next_spec)
        if env_status.startswith("missing "):
            out.print(
                f"[red]Cannot switch to {next_alias}:[/] {env_status}. "
                "Set the provider API key in your environment or .env file."
            )
            return False
        if next_model_name == current_model_name:
            out.print(f"[dim]Already using {current_model_alias} ({current_model_name}).[/]")
            return True

        with Live(
            RichSpinner("dots", text=Text(f"Switching to {next_alias}...", style="dim")),
            console=console,
            transient=True,
        ):
            try:
                next_agent = create_dissolve_agent(
                    model_name=next_model_name,
                    checkpointer=checkpointer,
                )
            except Exception as e:
                out.print(f"[red]Model switch failed:[/] {e}")
                return False

        agent = next_agent
        current_model_alias = next_alias
        current_model_spec = next_spec
        current_model_name = next_model_name
        out.print(f"[green]Switched model:[/] {current_model_alias} [dim]({current_model_name})[/]")
        return True

    def _switch_mode(target: str) -> bool:
        nonlocal current_interaction_mode

        next_mode = _resolve_cli_interaction_mode(target)
        if next_mode == current_interaction_mode:
            spec = _CLI_INTERACTION_MODES[current_interaction_mode]
            out.print(f"[dim]Already using {current_interaction_mode} mode ({spec['label']}).[/]")
            return True
        current_interaction_mode = next_mode
        spec = _CLI_INTERACTION_MODES[current_interaction_mode]
        out.print(f"[green]Switched mode:[/] {current_interaction_mode} [dim]({spec['label']})[/]")
        return True

    def _render_choice_selector(
        *,
        title: str,
        rows: list[dict[str, object]],
        columns: list[tuple[str, str, int]],
        selected: int,
        top: int,
        visible_count: int,
        footer: str,
        note=None,
    ):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        visible_rows = rows[top:top + visible_count]
        table = Table.grid(expand=True)
        table.add_column(width=2)
        for _, _, ratio in columns:
            table.add_column(ratio=ratio)
        table.add_row("", *[f"[bold]{header}[/]" for _, header, _ in columns])
        for offset, row in enumerate(visible_rows):
            index = top + offset
            is_selected = index == selected
            is_active = bool(row.get("active"))
            cursor = ">" if is_selected else " "
            style = "reverse bold cyan" if is_selected else "bold cyan" if is_active else ""
            table.add_row(
                cursor,
                *[str(row.get(field, "")) for field, _, _ in columns],
                style=style,
            )

        if len(rows) > visible_count:
            footer += f"  Showing {top + 1}-{top + len(visible_rows)} of {len(rows)}"
        body = Group(table, Text(), note) if note is not None else table
        return Panel(
            body,
            title=f"[bold]{title}[/]",
            subtitle=f"[dim]{footer}[/]",
            border_style="cyan",
            padding=(1, 2),
        )

    def _read_selector_key(stdin_fd: int) -> str:
        import os
        import select
        import time

        first = os.read(stdin_fd, 1)
        if first in (b"\r", b"\n"):
            return "enter"
        if first == b"\x1b":
            sequence = first
            deadline = time.monotonic() + 0.5
            while len(sequence) < 8:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not select.select([stdin_fd], [], [], min(0.05, remaining))[0]:
                    continue
                sequence += os.read(stdin_fd, 1)
                if sequence in {b"\x1b[A", b"\x1bOA", b"\x1b[B", b"\x1bOB"}:
                    break
            if sequence in {b"\x1b[A", b"\x1bOA"}:
                return "up"
            if sequence in {b"\x1b[B", b"\x1bOB"}:
                return "down"
            return "escape"
        return first.decode(errors="ignore")

    def _select_cli_choice(
        *,
        title: str,
        rows: list[dict[str, object]],
        columns: list[tuple[str, str, int]],
        footer: str,
        selected: int = 0,
        visible_limit: int = 8,
        note=None,
    ) -> dict[str, object] | None:
        import sys
        import termios
        import tty

        if not rows:
            return None

        selected = max(0, min(selected, len(rows) - 1))
        selected = next((i for i, row in enumerate(rows) if row.get("active")), selected)
        visible_count = min(visible_limit, len(rows))
        top = max(0, min(selected, len(rows) - visible_count))
        stdin_fd = sys.stdin.fileno()
        original_settings = termios.tcgetattr(stdin_fd)
        try:
            tty.setcbreak(stdin_fd)
            with Live(
                _render_choice_selector(
                    title=title,
                    rows=rows,
                    columns=columns,
                    selected=selected,
                    top=top,
                    visible_count=visible_count,
                    footer=footer,
                    note=note,
                ),
                console=out,
                refresh_per_second=20,
                transient=True,
            ) as live:
                while True:
                    key = _read_selector_key(stdin_fd)
                    if key == "escape":
                        return None
                    if key == "up":
                        selected = (selected - 1) % len(rows)
                    elif key == "down":
                        selected = (selected + 1) % len(rows)
                    elif key == "enter":
                        live.stop()
                        return rows[selected]
                    else:
                        continue

                    if selected < top:
                        top = selected
                    elif selected >= top + visible_count:
                        top = selected - visible_count + 1
                    live.update(
                        _render_choice_selector(
                            title=title,
                            rows=rows,
                            columns=columns,
                            selected=selected,
                            top=top,
                            visible_count=visible_count,
                            footer=footer,
                            note=note,
                        )
                    )
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, original_settings)

    def _open_model_selector() -> None:
        import sys

        if not (console.is_terminal and out.is_terminal and sys.stdin.isatty()):
            _print_model_table()
            return

        rows = _model_rows()
        if not rows:
            out.print("[yellow]No models are configured.[/]")
            return

        selected = next((i for i, row in enumerate(rows) if row["active"]), 0)
        selected_row = _select_cli_choice(
            title="DISSOLVE Models",
            rows=rows,
            columns=[
                ("selector_alias", "Alias", 2),
                ("model", "Model", 4),
                ("provider", "Provider", 1),
                ("status", "Status", 2),
            ],
            footer="Up/Down select  Enter switch  Esc back",
            selected=selected,
        )
        if selected_row is None:
            return
        _switch_model(str(selected_row["alias"]))

    def _open_mode_selector() -> None:
        import sys

        if not (console.is_terminal and out.is_terminal and sys.stdin.isatty()):
            _print_mode_table()
            return

        rows = _mode_rows()
        selected = next((i for i, row in enumerate(rows) if row["active"]), 0)
        selected_row = _select_cli_choice(
            title="DISSOLVE Modes",
            rows=rows,
            columns=[
                ("mode", "Mode", 1),
                ("label", "Label", 2),
                ("description", "Description", 4),
                ("status", "Status", 1),
            ],
            footer="Up/Down select  Enter switch  Esc back",
            selected=selected,
        )
        if selected_row is None:
            return
        _switch_mode(str(selected_row["mode"]))

    def _advance_biosteam_setting(settings: dict[str, dict[str, str]], key: str) -> None:
        if key == "energy_case":
            options = _biosteam_energy_case_options()
            values = tuple(option["value"] for option in options)
            next_value = _cycle_choice(settings[key]["value"], values)
            next_option = next(option for option in options if option["value"] == next_value)
            settings[key]["value"] = next_option["value"]
            settings[key]["description"] = next_option["description"]
            settings[key]["source"] = "default" if next_option["value"] == "C1" else "override"
            return

        if key == "metrics":
            next_value = _cycle_choice(settings[key]["value"], _BIOSTEAM_METRIC_PRESETS)
            settings[key]["value"] = next_value
            settings[key]["source"] = "override"
            return

        presets_by_key = {
            "capacity": _BIOSTEAM_CAPACITY_PRESETS,
            "target_fraction": _BIOSTEAM_TARGET_FRACTION_PRESETS,
            "precipitation_temp": _BIOSTEAM_PRECIPITATION_TEMP_PRESETS,
        }
        defaults_by_key = {
            "capacity": "20,000 MT/yr",
            "target_fraction": "60 wt% target plastic in feed",
            "precipitation_temp": "25 C",
        }
        presets = presets_by_key[key]
        next_value = _cycle_choice(settings[key]["value"], presets)
        settings[key]["value"] = next_value
        settings[key]["source"] = "default" if next_value == defaults_by_key[key] else "override"

    def _edit_biosteam_run_settings(user_input: str) -> dict[str, dict[str, str]] | None:
        settings = _biosteam_initial_run_settings(user_input)
        selected = 0
        while True:
            rows = []
            for index, row in enumerate(_biosteam_settings_to_rows(settings)):
                row = dict(row)
                row["index"] = index
                rows.append(row)

            selected_row = _select_cli_choice(
                title="Configure BioSTEAM Run",
                rows=rows,
                columns=[
                    ("setting", "Setting", 2),
                    ("value", "Value", 5),
                    ("source", "Source", 1),
                ],
                footer="Up/Down move  Enter change/continue  Esc cancel",
                selected=selected,
                visible_limit=8,
            )
            if selected_row is None:
                return None
            selected = int(selected_row["index"])
            key = str(selected_row["key"])
            if key == "continue":
                return settings
            _advance_biosteam_setting(settings, key)

    def _handle_model_command(command: str) -> None:
        parts = command.split(maxsplit=1)
        target = parts[1].strip() if len(parts) > 1 else ""
        if not target:
            _open_model_selector()
            return
        if target in {"list", "ls"}:
            _print_model_table()
            return
        if target in {"current", "show"}:
            out.print(f"[bold]Current model:[/] {current_model_alias} [dim]({current_model_name})[/]")
            return
        if target in {"help", "-h", "--help"}:
            out.print("[bold]/model[/] opens the selector; [bold]/model list[/] lists models; [bold]/model current[/] shows the active model; [bold]/model <alias>[/] switches.")
            return

        _switch_model(target)

    def _handle_mode_command(command: str) -> None:
        parts = command.split(maxsplit=1)
        target = parts[1].strip() if len(parts) > 1 else ""
        if not target:
            _open_mode_selector()
            return
        if target in {"list", "ls"}:
            _print_mode_table()
            return
        if target in {"current", "show"}:
            spec = _CLI_INTERACTION_MODES[current_interaction_mode]
            out.print(f"[bold]Current mode:[/] {current_interaction_mode} [dim]({spec['label']})[/]")
            return
        if target in {"help", "-h", "--help"}:
            out.print("[bold]/mode[/] opens the selector; [bold]/mode review[/] asks before tunable defaults; [bold]/mode auto[/] skips CLI review.")
            return

        _switch_mode(target)

    def _handle_context_command(command: str) -> None:
        nonlocal session_context

        parts = command.split(maxsplit=1)
        target = parts[1].strip().lower() if len(parts) > 1 else "show"
        if target in {"show", "current", ""}:
            block = build_session_context_block(session_context)
            if block:
                out.print(block)
            else:
                out.print("[dim]No structured session context yet.[/]")
            return
        if target == "path":
            out.print(f"[bold]Session directory:[/] {durable_session_paths['dir']}")
            out.print(f"[dim]Context:[/] {durable_session_paths['context']}")
            out.print(f"[dim]Transcript:[/] {durable_session_paths['transcript']}")
            return
        if target == "clear":
            session_context = load_session_context(thread_id)
            session_context["feedstock"] = {}
            session_context["process"] = {}
            session_context["analysis"] = {}
            session_context["last_user_query"] = ""
            save_session_context(thread_id, session_context)
            out.print("[green]Cleared structured session context.[/]")
            return
        if target in {"help", "-h", "--help"}:
            out.print("[bold]/context[/] shows compact session context; [bold]/context clear[/] resets it; [bold]/context path[/] shows durable files.")
            return
        out.print("[yellow]Unknown /context command. Use /context help.[/]")

    def _clarify_user_input(user_input: str) -> str | None:
        if current_interaction_mode == "auto":
            return user_input
        if not _query_needs_energy_case_clarification(user_input):
            return user_input
        if not (console.is_terminal and out.is_terminal and sys.stdin.isatty()):
            return user_input

        out.print("[dim]BioSTEAM run settings need confirmation.[/]")
        settings = _edit_biosteam_run_settings(user_input)
        if settings is None:
            out.print("[dim]BioSTEAM run settings cancelled.[/]")
            return None
        return _append_biosteam_run_settings_clarification(user_input, settings)

    def _read_user_input() -> str:
        if prompt_session is not None:
            return prompt_session.prompt("> ")
        return out.input("[bold]> [/]")

    # ── REPL ──
    while True:
        try:
            user_input = _read_user_input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        if user_input.startswith("/model"):
            _handle_model_command(user_input)
            continue
        if user_input.startswith("/mode"):
            _handle_mode_command(user_input)
            continue
        if user_input.startswith("/context"):
            _handle_context_command(user_input)
            continue

        clarified_input = _clarify_user_input(user_input)
        if clarified_input is None:
            continue
        user_input = clarified_input
        inject_context = should_inject_session_context(user_input, session_context)
        agent_user_input = inject_session_context(user_input, session_context) if inject_context else user_input
        session_context = update_session_context_from_text(session_context, user_input, role="user")
        save_session_context(thread_id, session_context)
        append_transcript_event(
            thread_id,
            "user",
            user_input,
            injected_context=inject_context,
            mode=current_interaction_mode,
            model=current_model_name,
        )

        t0 = time.time()
        with Live(
            RichSpinner("dots", text=Text("Thinking...", style="dim")),
            console=console,
            transient=True,
        ):
            try:
                if use_checkpointer:
                    # Checkpointer maintains full history — only send the new message
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": agent_user_input}]},
                        config=invoke_config,
                    )
                else:
                    # No checkpointer — manually accumulate history each turn
                    history.append({"role": "user", "content": agent_user_input})
                    result = agent.invoke(
                        {"messages": list(history)},
                        config=plain_config,
                    )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]\n")
                if not use_checkpointer:
                    history.pop()
                continue
            except Exception as e:
                console.print(f"\n[red]Error:[/] {e}\n")
                if not use_checkpointer:
                    history.pop()
                continue

        elapsed = time.time() - t0

        # Extract last AI message text
        answer = None
        answer_metadata = {}
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                answer = _extract_text(msg.content)
                metadata = getattr(msg, "additional_kwargs", None)
                answer_metadata = metadata if isinstance(metadata, dict) else {}
                break

        if answer:
            if not use_checkpointer:
                history.append({"role": "assistant", "content": answer})
            session_context = update_session_context_from_text(
                session_context,
                answer,
                role="assistant",
            )
            session_context = update_session_context_from_direct_metadata(
                session_context,
                answer_metadata,
            )
            save_session_context(thread_id, session_context)
            append_transcript_event(
                thread_id,
                "assistant",
                answer,
                elapsed_time=elapsed,
                model=current_model_name,
                route_decision=answer_metadata.get("strap_route_decision"),
                artifacts=answer_metadata.get("strap_artifacts"),
            )
            out.print()
            if _answer_prefers_plain_terminal(answer):
                out.print(answer)
            else:
                out.print(Markdown(answer))
            console.print(f"\n[dim]({elapsed:.1f}s)[/]\n")
        else:
            console.print("\n[dim]No response.[/]\n")


if __name__ == "__main__":
    main()
