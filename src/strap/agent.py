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

from .tools import (  # noqa: E402
    get_core_tools,
    get_advanced_separation_tools,
    get_statistical_tools,
    get_safety_gsk_tools,
    get_safety_pubchem_tools,
    get_tea_lca_tools,
    get_strap_process_tools,
    get_scholar_tools,
    get_patent_tools,
    get_rag_core_tools,
    get_rag_diagnostics_tools,
    get_visualization_tools,
    get_ml_prediction_tools,
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
- **Adaptive separation tools** — find optimal conditions, selective solubility analysis

## Specialist subagents (spawned on demand)
You can delegate to these specialists when needed:

1. **separation-engineer** — advanced separation algorithms (greedy/DP/branch-and-bound),
   sequence planning, integrated analysis, precipitation & antisolvent methods
2. **safety-analyst** — GSK G-scores, PubChem GHS hazard data, toxicity (LD50/LC50),
   safety comparisons and visualizations
3. **tea-lca-analyst** — techno-economic analysis, life cycle assessment, STRAP process
   economics, MSP calculation, scale economics, scenario comparison
4. **scholar-researcher** — Google Scholar, Web of Science search, save papers to RAG
5. **patent-researcher** — Google Patents search, patent lookup, save patents to RAG
6. **rag-analyst** — RAG ingestion, literature Q&A, chunk quality, diagnostics
7. **visualization-specialist** — solubility curves, selectivity heatmaps, multi-panel
   analysis, comparison dashboards, precipitation curves
8. **statistics-ml** — statistical summaries, correlation, regression, group comparisons,
   ML solubility prediction via Hansen parameters

## Delegation rules
- **Always delegate** to the separation-engineer for: separation sequences, multi-polymer
  separation planning, greedy/DP/branch-and-bound algorithms, precipitation, antisolvents.
  The separation-engineer has `find_optimal_separation_sequence` which handles sequence
  optimization — do NOT attempt to replicate this with your core tools.
- **Always delegate** to the safety-analyst, tea-lca-analyst, visualization-specialist,
  statistics-ml, scholar-researcher, patent-researcher, or rag-analyst when the query
  falls within their domain.
- Your core tools are for quick lookups and simple selectivity analysis only.

## Guidelines
- Selectivity >= 5 is the minimum viability threshold.
- Always state the temperature used.
- When uncertain, run a broad ranking first, then zoom in with selectivity.
- Suggest multi-step separation cascades for challenging mixtures.
- Flag safety and environmental concerns for each recommended solvent.
"""


def _build_subagents() -> list[SubAgent]:
    """Build the specialist subagent definitions."""
    return [
        SubAgent(
            name="separation-engineer",
            description=(
                "Advanced separation specialist. Use for: optimal separation sequences "
                "(greedy/DP/branch-and-bound), sequential separation planning, integrated "
                "separation analysis, differential precipitation, antisolvent methods, "
                "atmospheric feasibility checks, polymer dissolution analysis."
            ),
            system_prompt=(
                "You are a separation engineering specialist. Your key tools:\n"
                "- find_optimal_separation_sequence: finds optimal separation order "
                "using greedy, DP, or branch-and-bound algorithms. Supports "
                "algorithm='greedy'/'dp'/'branch_and_bound'/'auto'/'compare'.\n"
                "- plan_sequential_separation: detailed multi-step separation planning\n"
                "- calculate_selectivity_detailed: single-pair selectivity check\n"
                "- rank_solvents_for_separation: rank solvents by multi-criteria scoring\n"
                "- Precipitation/antisolvent tools for differential dissolution\n"
                "Always use find_optimal_separation_sequence when asked for a "
                "separation sequence or to optimize separation order."
            ),
            tools=get_advanced_separation_tools(),
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
        ),
        SubAgent(
            name="visualization-specialist",
            description=(
                "Data visualization specialist. Use for: solubility vs temperature plots, "
                "interactive Plotly charts, selectivity heatmaps, multi-panel analysis, "
                "comparison dashboards, precipitation curves, process flow diagrams."
            ),
            system_prompt=(
                "You are a data visualization specialist. You have tools for creating "
                "publication-quality plots of solubility data, heatmaps, dashboards, "
                "and process diagrams. Create clear, informative visualizations."
            ),
            tools=get_visualization_tools(),
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
        ),
    ]


def create_dissolve_agent(model_name: str = "anthropic:claude-sonnet-4-5-20250929"):
    """Create and return a compiled DISSOLVE deep agent with subagents."""
    model = init_chat_model(model_name)
    agent = create_deep_agent(
        model=model,
        tools=get_core_tools(),
        subagents=_build_subagents(),
        system_prompt=SYSTEM_PROMPT,
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
