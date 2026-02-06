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
from .routing import generate_routing_table, routing_middleware  # noqa: E402
import yaml  # noqa: E402

from .tools import get_core_tools  # noqa: E402
from .tools import (  # noqa: E402  — tool group registry for YAML loader
    get_adaptive_separation_tools,
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
    get_statistical_tools,
    get_strap_process_tools,
    get_tea_lca_tools,
    get_visualization_tools,
)

# Map YAML tool_group names → getter functions
_TOOL_GROUP_REGISTRY: dict[str, callable] = {
    "separation_core": get_separation_core_tools,
    "adaptive_separation": get_adaptive_separation_tools,
    "safety_gsk": get_safety_gsk_tools,
    "safety_pubchem": get_safety_pubchem_tools,
    "tea_lca": get_tea_lca_tools,
    "strap_process": get_strap_process_tools,
    "scholar": get_scholar_tools,
    "patent": get_patent_tools,
    "rag_core": get_rag_core_tools,
    "rag_diagnostics": get_rag_diagnostics_tools,
    "visualization": get_visualization_tools,
    "separation_plot": get_separation_plot_tools,
    "statistical": get_statistical_tools,
    "ml_prediction": get_ml_prediction_tools,
    "interpolation": get_interpolation_tools,
    "reflection": get_reflection_tools,
}

SYSTEM_PROMPT = """\
{routing_table}

## Delegation policy
- When a [ROUTING: ...] hint appears, your VERY NEXT action must be the task() call
  it specifies. Do NOT call query_database, list_tables, or any other tool before
  delegating — the subagent already has everything it needs.
- After a subagent returns results, synthesize them into a final answer for the user.
  Do NOT run additional database queries to validate or expand subagent results.
- Your role is to route, synthesize subagent results, and answer simple data lookups.
  Use your direct tools only for quick lookups (e.g. "list polymers", "what is the
  boiling point of toluene") that don't need a specialist.
- Delegate to subagents one at a time UNLESS the [ROUTING] hint explicitly instructs
  you to launch two task() calls in parallel. In that case, include both task() calls
  in a single response. Never launch more than two task() calls at once.

## Inter-agent file state (sequential chains only)
- When routing hints instruct subagents to write findings to /chain_state/ files,
  include that instruction in the task description you pass to each subagent.
- Between sequential steps, use read_file to read the prior subagent's output file
  before delegating to the next subagent. Include a brief summary (2-3 sentences)
  AND the file path in the next task description, so the subagent can read_file
  for full details.
- After the final subagent completes, read_file all /chain_state/ files and
  synthesize a comprehensive answer.
- Do NOT paste entire file contents into the task description — pass the file path
  and a summary instead.
""".format(routing_table=generate_routing_table())


_THINK_DIRECTIVE = (
    "\n\n## REFLECTION PROTOCOL\n"
    "You have a `think` tool. Use it AFTER every domain tool call to assess:\n"
    "- What concrete data did I just obtain? (cite specific numbers)\n"
    "- Is my finding grounded in tool output, or am I relying on general knowledge?\n"
    "- What is still missing? Should I call another tool or synthesize now?\n"
    "NEVER answer a question using only your general knowledge — always use your "
    "domain tools first, then reflect with `think`, then synthesize."
)

_FILE_IO_DIRECTIVE = (
    "\n\n## FILE I/O IN CHAINS\n"
    "When you are part of a sequential multi-agent chain:\n"
    "- **READ FIRST**: If the task description references a file path from a prior "
    "step (e.g. /chain_state/step_1_*.md), your VERY FIRST action must be "
    "read_file on that path. Base your analysis on the full file contents, not "
    "just the brief summary in the task description.\n"
    "- **WRITE LAST**: If the task description instructs you to write your findings "
    "to a specific file path, use write_file to save your complete findings "
    "(data, analysis, recommendations) to that path as your FINAL action before "
    "returning. Format the file as structured markdown with clear section headings."
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
_ALWAYS_FREE_TOOLS = {"write_file", "read_file"}


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


def create_dissolve_agent(model_name: str = "google_genai:gemini-3-flash-preview"):
    """Create and return a compiled DISSOLVE deep agent with subagents.

    Uses progressive loading:
    - ``memory`` (AGENTS.md) is always injected into the system prompt.
    - ``skills`` (skills/\*) are loaded on demand by SkillsMiddleware.
    - ``system_prompt`` carries only the dynamic routing table.
    """
    model = init_chat_model(model_name)
    agent = create_deep_agent(
        model=model,
        tools=get_core_tools(),
        subagents=_build_subagents(),
        system_prompt=SYSTEM_PROMPT,
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        backend=FilesystemBackend(root_dir=str(_PACKAGE_DIR)),
        middleware=[routing_middleware],
        name="dissolve-agent",
    )
    return agent


# Keep backward-compatible alias
create_strap_agent = create_dissolve_agent


def main():
    """Interactive CLI loop."""
    agent = create_dissolve_agent()
    print("DISSOLVE Agent ready. Type your question (or 'quit' to exit).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )

        # Extract the last AI message
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                print(f"\nDISSOLVE: {msg.content}\n")
                break


if __name__ == "__main__":
    main()
