"""DISSOLVE deep agent: wires model + tools + subagents + system prompt."""

from __future__ import annotations

import os
from pathlib import Path

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
from .routing import RoutingMiddleware, generate_routing_table  # noqa: E402
from .verifier import OutputVerifierMiddleware  # noqa: E402
import yaml  # noqa: E402

from .tools import get_core_tools  # noqa: E402
from .tools import (  # noqa: E402  — tool group registry for YAML loader
    get_adaptive_separation_tools,
    get_biosteam_tools,
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
    get_solvent_lookup_tools,
    get_statistical_tools,
    get_thermal_prediction_tools,
    get_visualization_tools,
)

# Map YAML tool_group names → getter functions
_TOOL_GROUP_REGISTRY: dict[str, callable] = {
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
}

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
- Prefer delegating to subagents one at a time. You may launch two task() calls in
  parallel when the tasks are independent (e.g. separation + safety).
  Never launch more than two task() calls at once.
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


def _resolve_tools(group_names: list[str]) -> list:
    """Resolve YAML tool_group names to actual tool function lists."""
    tools = []
    for name in group_names:
        getter = _TOOL_GROUP_REGISTRY.get(name)
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
) -> list[SubAgent]:
    """Load subagent definitions from YAML config.

    Falls back to ``subagents.yaml`` next to this module.
    The ``_THINK_DIRECTIVE`` is appended to every system prompt automatically.
    """
    if yaml_path is None:
        yaml_path = _PACKAGE_DIR / "subagents.yaml"

    with open(yaml_path) as f:
        specs = yaml.safe_load(f)

    subagents: list[SubAgent] = []
    for spec in specs:
        prompt = spec["system_prompt"].rstrip() + _FILE_IO_DIRECTIVE + _THINK_DIRECTIVE
        sa = SubAgent(
            name=spec["name"],
            description=spec["description"].strip(),
            system_prompt=prompt,
            tools=_resolve_tools(spec.get("tool_groups", [])),
            middleware=_resolve_guardrails(spec.get("guardrails")),
        )
        subagents.append(sa)

    return subagents


def create_dissolve_agent(model_name: str = os.getenv("STRAP_MODEL", "google_genai:gemini-2.5-pro")):
    """Create and return a compiled DISSOLVE deep agent with subagents.

    Uses progressive loading:
    - ``memory`` (AGENTS.md) is always injected into the system prompt.
    - ``skills`` (skills/\*) are loaded on demand by SkillsMiddleware.
    - ``system_prompt`` carries only the dynamic routing table.
    """
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
        free_tools={"think", "task", "read_file", "write_file", "write_todos"},
    )

    # Middleware order (innermost → outermost):
    #   routing → output_verifier → orchestrator_guard
    agent = create_deep_agent(
        model=model,
        tools=get_core_tools(),
        subagents=_build_subagents(),
        system_prompt=SYSTEM_PROMPT,
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        backend=FilesystemBackend(root_dir=str(_PACKAGE_DIR)),
        middleware=[routing, output_verifier, orchestrator_guard],
        name="dissolve-agent",
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
    console.print("[dim]Type [bold]quit[/bold] to exit.[/]\n")

    # ── Load agent with spinner ──
    with Live(
        RichSpinner("dots", text=Text("Loading agent...", style="dim")),
        console=console,
        transient=True,
    ):
        agent = create_dissolve_agent()

    out = Console()  # stdout console for answers
    history: list = []  # accumulated conversation history across turns

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

        history.append({"role": "user", "content": user_input})

        t0 = time.time()
        with Live(
            RichSpinner("dots", text=Text("Thinking...", style="dim")),
            console=console,
            transient=True,
        ):
            try:
                result = agent.invoke({"messages": list(history)})
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]\n")
                # Remove the unanswered user message
                history.pop()
                continue
            except Exception as e:
                console.print(f"\n[red]Error:[/] {e}\n")
                history.pop()
                continue

        elapsed = time.time() - t0

        # Extract last AI message text and append to history
        answer = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                answer = _extract_text(msg.content)
                break

        if answer:
            history.append({"role": "assistant", "content": answer})
            out.print()
            out.print(Markdown(answer))
            console.print(f"\n[dim]({elapsed:.1f}s)[/]\n")
        else:
            console.print("\n[dim]No response.[/]\n")


if __name__ == "__main__":
    main()
