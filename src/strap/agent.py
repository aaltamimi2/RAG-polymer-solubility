"""DISSOLVE deep agent: wires model + tools + subagents + system prompt."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

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
from .result_extractor import StructuredResultExtractorMiddleware  # noqa: E402
from .routing import RoutingMiddleware, generate_routing_table  # noqa: E402
from .verifier import OutputVerifierMiddleware  # noqa: E402
import yaml  # noqa: E402

from .tools import get_core_tools  # noqa: E402
from .tools import (  # noqa: E402  — tool group registry for YAML loader
    get_adaptive_separation_tools,
    get_biosteam_tools,
    get_database_query_tools,
    get_interpolation_tools,
    get_ml_prediction_tools,
    get_patent_tools,
    get_rag_core_tools,
    get_rag_diagnostics_tools,
    get_reflection_tools,
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
)

# Map YAML tool_group names → getter functions
_TOOL_GROUP_REGISTRY: dict[str, callable] = {
    "database_query": get_database_query_tools,
    "separation_core": get_separation_core_tools,
    "adaptive_separation": get_adaptive_separation_tools,
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
    "solvent_lookup": get_solvent_lookup_tools,
    "reflection": get_reflection_tools,
    "sidecar_write": get_sidecar_write_tools,
    "sidecar_read": get_sidecar_read_tools,
    "result_extractor": get_result_extractor_tools,
}

def _create_session_scratch_dir() -> Path:
    """Create a per-session temp directory for sidecar files."""
    scratch = Path(tempfile.mkdtemp(prefix="strap_scratch_"))
    atexit.register(shutil.rmtree, scratch, True)
    logger.info("Session scratch directory: %s", scratch)
    return scratch


SYSTEM_PROMPT = """\
{routing_table}

## Delegation policy
- You can answer simple queries directly using your tools, or delegate complex analysis
  to specialists via task(). After specialists return, synthesize their results.
- Your direct tools handle: listing polymers/solvents, solvent properties, solubility
  predictions, and separation selectivity rankings. Delegate to subagents for
  complex planning (multi-scheme design, sequential separation, TEA/LCA, safety, etc.).
- For "what solvents separate X from Y" queries, rank_solvents_selectivity returns
  selectivity, solubilities, boiling points, and atmospheric feasibility in one table.
- When separating a binary pair A/B, ALWAYS check BOTH directions:
  rank_solvents_selectivity(target=A, other=B) AND rank_solvents_selectivity(target=B, other=A).
  One direction may have excellent solvents while the other has none.
- When the user asks for "multiple schemes" or "compare strategies", delegate ONCE to
  separation-engineer — it has a multi-scheme tool that generates multiple options in
  a single invocation.
- Prefer delegating to subagents one at a time. You may launch three task() calls in
  parallel when the tasks are independent (e.g. separation + safety).
  Never launch more than three task() calls at once.
- When the user asks for a diagram, plot, or visualization of a separation sequence,
  delegate to visualization-specialist with a short instruction like
  "Create a separation tree plot for LDPE,HDPE,PS,PVC at 120C".
  The specialist has matplotlib tools (create_separation_tree_plot,
  create_process_flow_diagram) that produce publication-quality PNG plots.
  NEVER ask it to generate Mermaid or text-based diagrams.

## Multi-polymer pipeline protocol
When asked to propose separation sequences AND test them in BioSTEAM:
1. Delegate to separation-engineer with: "Plan sequential separation for <polymers> at <temp>C"
2. From the result, extract the `top_k_sequences` — each has `sequence` (polymer order) and `solvent_mapping` (polymer→solvent dict).
3. For EACH sequence, build a polymers_json array:
   [{{"polymer":"<P1>","solvent":"<S1>"}},{{"polymer":"<P2>","solvent":"<S2>"}},...]
4. Delegate to biosteam-analyst with ALL sequences to test, e.g.:
   "Run multi-polymer BioSTEAM for these alternative sequences:
    Seq 1: [{{"polymer":"LDPE","solvent":"Xylene"}},{{"polymer":"PET","solvent":"Toluene"}},{{"polymer":"EVOH","solvent":"Ethylene Glycol"}}]
    Seq 2: [{{"polymer":"PET","solvent":"Toluene"}},{{"polymer":"LDPE","solvent":"Xylene"}},{{"polymer":"EVOH","solvent":"Ethylene Glycol"}}]
    ..."
5. Synthesize: rank sequences by blended MSP and GWP.

## Structured results from subagents
Each subagent appends a <STRUCTURED_RESULT> JSON block at the end of its response.
Use it as the authoritative source for numeric data when delegating downstream.

Sequential handoff — extract from <STRUCTURED_RESULT> and include in next task():
- separation-engineer → biosteam-analyst: extract top_k_sequences[*].solvent_mapping
- separation-engineer → visualization-specialist: extract best_sequence and solvent_mapping
- statistics-ml → visualization-specialist: extract summary stats or prediction results
- safety-analyst results: read g_score and hazard_codes per solvent

Parallel synthesis — when combining results from parallel agents:
- Prefer numeric fields from <STRUCTURED_RESULT> over parsing prose for comparisons.

You can also call get_subagent_result(agent_name) to retrieve the structured
JSON from a completed subagent, or get_all_subagent_results() to see all results.
These are faster than re-parsing prose when you need specific numeric fields.

Fallback: if no <STRUCTURED_RESULT> block is present, extract data from the prose as before.
""".format(routing_table=generate_routing_table())


_THINK_DIRECTIVE = (
    "\n\n## REFLECTION\n"
    "After each tool call, use think() to assess findings and decide whether to "
    "continue or synthesize. Use domain tools first, never rely on general knowledge alone."
)

_FILE_IO_DIRECTIVE = (
    "\n\n## FILE I/O\n"
    "In multi-agent chains: read_file referenced paths FIRST, write_file your findings LAST."
)


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


def _resolve_tools(group_names: list[str], registry: dict | None = None) -> list:
    """Resolve YAML tool_group names to actual tool function lists."""
    r = registry if registry is not None else _TOOL_GROUP_REGISTRY
    tools = []
    for name in group_names:
        getter = r.get(name)
        if getter:
            tools.extend(getter())
    return tools


# Tools that should never count against the subagent tool-call budget
_ALWAYS_FREE_TOOLS = {
    "write_file", "read_file",  # inter-agent communication
    "ls", "glob", "edit_file", "grep", "execute", "write_todos",  # filesystem/meta
}


def _resolve_guardrails(cfg: dict | None) -> list:
    """Build middleware list from YAML guardrails config."""
    if cfg is None:
        return [SubagentGuardMiddleware(free_tools=_ALWAYS_FREE_TOOLS.copy())]
    free = set(cfg["free_tools"]) if cfg.get("free_tools") else set()
    free |= _ALWAYS_FREE_TOOLS
    return [SubagentGuardMiddleware(
        max_iterations=cfg.get("max_iterations", 25),
        token_budget=cfg.get("token_budget", 200_000),
        max_tool_calls=cfg.get("max_tool_calls", 10),
        synthesis_tools=set(cfg["synthesis_tools"]) if cfg.get("synthesis_tools") else set(),
        truncate_tool_results_after=cfg.get("truncate_tool_results_after"),
        free_tools=free,
    )]


def _build_subagents(
    yaml_path: str | Path | None = None,
    overrides: dict[str, SubagentOverride] | None = None,
) -> list[SubAgent]:
    """Load subagent definitions from YAML config.

    Falls back to ``subagents.yaml`` next to this module.
    The ``_THINK_DIRECTIVE`` is appended to every system prompt automatically.

    Args:
        yaml_path: Path to YAML config file. Defaults to subagents.yaml.
        overrides: Optional dict mapping subagent name -> guardrail overrides.
            Override keys mirror ``SubagentGuardMiddleware`` parameters:
            ``max_tool_calls``, ``max_iterations``, ``token_budget``,
            ``synthesis_tools``, ``truncate_tool_results_after``, ``free_tools``.
    """
    if yaml_path is None:
        yaml_path = _PACKAGE_DIR / "subagents.yaml"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Support both the old flat-list format and the new mapping format
    # (with top-level "subagents" and "execution_pairs" keys).
    if isinstance(data, dict):
        specs = data["subagents"]
    else:
        specs = data

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

        prompt = spec["system_prompt"].rstrip() + _FILE_IO_DIRECTIVE + _THINK_DIRECTIVE
        sa = SubAgent(
            name=agent_name,
            description=spec["description"].strip(),
            system_prompt=prompt,
            tools=_resolve_tools(spec.get("tool_groups", [])),
            middleware=_resolve_guardrails(guardrail_cfg),
        )
        subagents.append(sa)

    return subagents


def _get_or_create_thread_id(session_arg: str | None = None) -> str:
    """Return *session_arg* unchanged, or generate a fresh 12-hex-char thread ID."""
    if session_arg:
        return session_arg
    return uuid.uuid4().hex[:12]


def create_dissolve_agent(
    model_name: str = os.getenv("STRAP_MODEL", "google_genai:gemini-2.5-pro"),
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
                    "get_subagent_result", "get_all_subagent_results"},
    )

    # Structured result extractor: intercepts task() ToolMessages and
    # extracts <STRUCTURED_RESULT> JSON blocks into a per-invocation registry.
    result_extractor = StructuredResultExtractorMiddleware()

    # Set up per-session scratch directory for sidecar tools
    scratch_dir = _create_session_scratch_dir()
    from .tools.sidecar import set_scratch_dir
    set_scratch_dir(scratch_dir)

    # Middleware order (innermost → outermost):
    #   routing → output_verifier → result_extractor → orchestrator_guard
    agent = create_deep_agent(
        model=model,
        tools=get_core_tools() + get_result_extractor_tools(),
        subagents=_build_subagents(overrides=subagent_overrides),
        system_prompt=SYSTEM_PROMPT,
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        backend=FilesystemBackend(root_dir=str(_PACKAGE_DIR)),
        middleware=[routing, output_verifier, result_extractor, orchestrator_guard],
        name="dissolve-agent",
        checkpointer=checkpointer,
    )
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
    from rich.panel import Panel
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
    args = parser.parse_args()

    use_checkpointer = not args.no_persist
    thread_id = _get_or_create_thread_id(args.session)

    console = Console(stderr=True)

    # ── Suppress all library logging for clean CLI output ──
    logging.disable(logging.CRITICAL)

    # ── Banner ──
    console.print()
    console.print(
        Text.assemble(
            ("DISSOLVE", "bold cyan"),
            (" v0.2.0", "dim"),
        )
    )
    console.print("[dim]Data Integrated Solubility Solver via LLM Evaluation[/]")
    if os.getenv("LANGSMITH_API_KEY"):
        console.print("[dim]LangSmith tracing:[/] [green]enabled[/]")
    if use_checkpointer:
        console.print(f"[dim]Session ID:[/] [bold]{thread_id}[/bold]  "
                      "[dim](pass --session {id} to resume)[/]")
    console.print("[dim]Type [bold]quit[/bold] to exit.[/]\n")

    # ── Load agent with spinner ──
    with Live(
        RichSpinner("dots", text=Text("Loading agent...", style="dim")),
        console=console,
        transient=True,
    ):
        agent = create_dissolve_agent(enable_persistence=use_checkpointer)

    out = Console()  # stdout console for answers
    history: list = []  # used only when checkpointer is disabled

    # Build LangGraph config for checkpointer-aware invocations
    invoke_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 150}
    # When no checkpointer, keep recursion_limit but skip thread_id
    plain_config = {"recursion_limit": 150}

    # ── REPL ──
    while True:
        try:
            user_input = out.input("[bold]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

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
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=invoke_config,
                    )
                else:
                    # No checkpointer — manually accumulate history each turn
                    history.append({"role": "user", "content": user_input})
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
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                answer = _extract_text(msg.content)
                break

        if answer:
            if not use_checkpointer:
                history.append({"role": "assistant", "content": answer})
            out.print()
            out.print(Markdown(answer))
            console.print(f"\n[dim]({elapsed:.1f}s)[/]\n")
        else:
            console.print("\n[dim]No response.[/]\n")


if __name__ == "__main__":
    main()
