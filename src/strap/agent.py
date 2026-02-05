"""DISSOLVE deep agent: wires model + tools + subagents + system prompt."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Enable LangSmith tracing if keys are present
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "strap-agent")

from deepagents.graph import create_deep_agent  # noqa: E402
from deepagents.middleware.subagents import SubAgent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402

from .guardrails import SubagentGuardMiddleware  # noqa: E402
from .routing import generate_routing_table, routing_middleware  # noqa: E402
from .tools import (  # noqa: E402
    get_adaptive_separation_tools,
    get_core_tools,
    get_ml_prediction_tools,
    get_patent_tools,
    get_rag_core_tools,
    get_rag_diagnostics_tools,
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

SYSTEM_PROMPT = """\
You are DISSOLVE — Data Integrated Solubility Solver via LLM Evaluation.

You help researchers and engineers design solvent-based separation processes
for mixed polymer waste streams (e.g. multilayer packaging, automotive shredder
residue, e-waste plastics).

## Available data
The database contains polymer–solvent dissolution data (solubility vs temperature)
for a range of commodity and engineering polymers and common organic solvents.
Use the tools to discover which polymers and solvents are available.

## Your direct tools (always loaded)
- **Database query tools** — list tables, describe schemas, run SQL, validate data
- **Listing tools** — discover available polymers and solvents
- **Solvent property tools** — look up boiling point, LogP, Cp, Energy, rank by property

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

## Guidelines
- Selectivity >= 5 is the minimum viability threshold.
- Always state the temperature used.
- **NEVER recommend a solvent at a temperature above its boiling point.** All
  separations operate at atmospheric pressure — no pressurized vessels. If the
  user requests a temperature, exclude any solvent whose boiling point is at or
  below that temperature.
- When uncertain, run a broad ranking first, then zoom in with selectivity.
- Suggest multi-step separation cascades for challenging mixtures.
- Mention safety and environmental concerns only if the user asks.
""".format(routing_table=generate_routing_table())


def _build_subagents() -> list[SubAgent]:
    """Build the specialist subagent definitions."""
    return [
        SubAgent(
            name="separation-engineer",
            description=(
                "Advanced separation specialist. Use for: optimal separation sequences "
                "(greedy/DP/branch-and-bound), sequential separation planning, "
                "differential precipitation, antisolvent methods, "
                "atmospheric feasibility checks, polymer dissolution analysis, "
                "selective solubility analysis, finding optimal separation conditions."
            ),
            system_prompt=(
                "You are a separation engineering specialist.\n\n"
                "## WORKFLOW (follow this sequence exactly)\n"
                "1. PLAN: Call plan_sequential_separation or "
                "find_optimal_separation_sequence with the full polymer list "
                "and target conditions.\n"
                "2. VERIFY (optional, max 2 calls): If needed, call "
                "calculate_selectivity_detailed for ONE or TWO critical pairs "
                "to confirm the plan. Do NOT verify every pair.\n"
                "3. SYNTHESIZE: Write your final answer summarizing the "
                "separation plan, key selectivities, recommended temperatures, "
                "and any caveats.\n\n"
                "## HARD RULES\n"
                "- Maximum 8 tool calls total. After that you MUST synthesize.\n"
                "- After plan_sequential_separation or "
                "find_optimal_separation_sequence returns, go DIRECTLY to "
                "step 3 (synthesize). At most 2 verification calls.\n"
                "- Do NOT call calculate_selectivity_detailed, "
                "analyze_selective_solubility_enhanced, or "
                "analyze_precipitation_temperature more than 2 times each.\n"
                "- Do NOT call build_compatibility_matrix unless specifically "
                "asked for a compatibility overview.\n\n"
                "## KEY TOOLS\n"
                "- find_optimal_separation_sequence: optimal separation order "
                "(greedy/DP/branch-and-bound). "
                "algorithm='greedy'/'dp'/'branch_and_bound'/'auto'/'compare'.\n"
                "- plan_sequential_separation: detailed multi-step separation "
                "planning\n"
                "- calculate_selectivity_detailed: single-pair selectivity\n"
                "- rank_solvents_for_separation: multi-criteria solvent ranking\n"
                "- find_optimal_separation_conditions: adaptive condition finder\n"
                "- Precipitation/antisolvent tools for differential dissolution\n\n"
                "## CONSTRAINT\n"
                "Never recommend a solvent at a temperature above its boiling "
                "point. All operations are at atmospheric pressure."
            ),
            tools=(
                get_separation_core_tools()
                + get_adaptive_separation_tools()
            ),
            middleware=[SubagentGuardMiddleware(
                max_tool_calls=8,
                synthesis_tools={
                    "plan_sequential_separation",
                    "find_optimal_separation_sequence",
                },
                truncate_tool_results_after=2000,
            )],
        ),
        SubAgent(
            name="safety-analyst",
            description=(
                "Safety and environmental specialist. Use for: GSK solvent G-scores, "
                "PubChem GHS hazard data, toxicity data (LD50, LC50, biodegradation), "
                "safety comparisons between solvents, safety visualizations."
            ),
            system_prompt=(
                "You are a chemical safety analyst. You have tools for GSK G-score "
                "lookups, PubChem safety/toxicity data retrieval, and safety "
                "visualizations. Provide thorough safety assessments."
            ),
            tools=(
                get_safety_gsk_tools()
                + get_safety_pubchem_tools()
            ),
            middleware=[SubagentGuardMiddleware()],
        ),
        SubAgent(
            name="tea-lca-analyst",
            description=(
                "Techno-economic and environmental specialist. Use for: solvent recovery "
                "TEA (capital costs, operating costs, payback), LCA (GHG emissions, energy), "
                "STRAP process analysis, minimum selling price, scale economics, "
                "scenario comparisons, TEA/LCA visualizations."
            ),
            system_prompt=(
                "You are a techno-economic and life cycle analysis specialist. You have "
                "tools for TEA, LCA, and STRAP process economics. Provide detailed "
                "economic and environmental assessments."
            ),
            tools=(
                get_tea_lca_tools()
                + get_strap_process_tools()
            ),
            middleware=[SubagentGuardMiddleware()],
        ),
        SubAgent(
            name="scholar-researcher",
            description=(
                "Scientific literature specialist. Use for: searching Google Scholar, "
                "Web of Science, downloading open-access papers, saving articles to RAG."
            ),
            system_prompt=(
                "You are a scientific literature specialist. You have tools for "
                "searching Google Scholar and Web of Science. You can also save "
                "open-access papers directly to RAG for later analysis. Find and "
                "summarize relevant research."
            ),
            tools=get_scholar_tools(),
            middleware=[SubagentGuardMiddleware()],
        ),
        SubAgent(
            name="patent-researcher",
            description=(
                "Patent search specialist. Use for: searching Google Patents, looking up "
                "specific patents by number, downloading patent PDFs, saving patents to RAG."
            ),
            system_prompt=(
                "You are a patent research specialist. You have tools for searching "
                "Google Patents and looking up specific patents by number. You can "
                "also save patent PDFs to RAG for later analysis."
            ),
            tools=get_patent_tools(),
            middleware=[SubagentGuardMiddleware()],
        ),
        SubAgent(
            name="rag-analyst",
            description=(
                "RAG (Retrieval-Augmented Generation) specialist. Use for: ingesting PDFs "
                "into RAG, asking questions over indexed literature, searching indexed "
                "documents, chunk quality checks, retrieval diagnostics, embedding analysis."
            ),
            system_prompt=(
                "You are a RAG analysis specialist. You have tools for ingesting "
                "documents, searching indexed literature, answering questions from "
                "literature, and running diagnostics on retrieval quality. Help users "
                "build and query their literature knowledge base."
            ),
            tools=(
                get_rag_core_tools()
                + get_rag_diagnostics_tools()
            ),
            middleware=[SubagentGuardMiddleware()],
        ),
        SubAgent(
            name="visualization-specialist",
            description=(
                "Data visualization specialist. Use for: solubility vs temperature plots, "
                "interactive Plotly charts, selectivity heatmaps, multi-panel analysis, "
                "comparison dashboards, precipitation curves, process flow diagrams, "
                "separation tree plots, atmospheric feasibility plots."
            ),
            system_prompt=(
                "You are a data visualization specialist. You have tools for creating "
                "publication-quality plots of solubility data, heatmaps, dashboards, "
                "process diagrams, and separation-specific visualizations (separation "
                "trees, selectivity heatmaps, precipitation curves, atmospheric "
                "feasibility). Create clear, informative visualizations."
            ),
            tools=(
                get_visualization_tools()
                + get_separation_plot_tools()
            ),
            middleware=[SubagentGuardMiddleware()],
        ),
        SubAgent(
            name="statistics-ml",
            description=(
                "Statistics and ML specialist. Use for: statistical summaries with "
                "confidence intervals, correlation analysis, regression, group comparisons "
                "(hypothesis testing), ML-based solubility prediction using Hansen "
                "Solubility Parameters."
            ),
            system_prompt=(
                "You are a statistics and machine learning specialist. You have tools "
                "for statistical analysis and ML-based solubility prediction. Provide "
                "rigorous statistical assessments."
            ),
            tools=(
                get_statistical_tools()
                + get_ml_prediction_tools()
            ),
            middleware=[SubagentGuardMiddleware()],
        ),
    ]


def create_dissolve_agent(model_name: str = "google_genai:gemini-3-flash-preview"):
    """Create and return a compiled DISSOLVE deep agent with subagents."""
    model = init_chat_model(model_name)
    agent = create_deep_agent(
        model=model,
        tools=get_core_tools(),
        subagents=_build_subagents(),
        system_prompt=SYSTEM_PROMPT,
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
