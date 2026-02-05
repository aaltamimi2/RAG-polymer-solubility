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
- Always delegate to subagents ONE AT A TIME. Never launch multiple task() calls in a
  single response. Wait for each subagent to finish before launching the next one.

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
                "You are a separation engineering specialist. Your key tools:\n"
                "- find_optimal_separation_sequence: finds optimal separation order "
                "using greedy, DP, or branch-and-bound algorithms. Supports "
                "algorithm='greedy'/'dp'/'branch_and_bound'/'auto'/'compare'.\n"
                "- plan_sequential_separation: detailed multi-step separation planning\n"
                "- calculate_selectivity_detailed: single-pair selectivity check\n"
                "- rank_solvents_for_separation: rank solvents by multi-criteria scoring\n"
                "- find_optimal_separation_conditions: adaptive separation condition finder\n"
                "- analyze_selective_solubility_enhanced: enhanced selective solubility analysis\n"
                "- Precipitation/antisolvent tools for differential dissolution\n"
                "Always use find_optimal_separation_sequence when asked for a "
                "separation sequence or to optimize separation order.\n\n"
                "IMPORTANT: After calling find_optimal_separation_sequence or "
                "plan_sequential_separation and receiving results, synthesize your "
                "findings into a clear final answer immediately. Do NOT exhaustively "
                "validate results with other tools. Limit yourself to 5-8 tool calls "
                "total. Use build_compatibility_matrix only when specifically asked "
                "for a compatibility overview.\n\n"
                "CONSTRAINT: Never recommend a solvent at a temperature above its "
                "boiling point. All operations are at atmospheric pressure — no "
                "pressurized vessels. Exclude any solvent whose boiling point is at "
                "or below the requested temperature."
            ),
            tools=(
                get_separation_core_tools()
                + get_adaptive_separation_tools()
            ),
            middleware=[SubagentGuardMiddleware()],
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
