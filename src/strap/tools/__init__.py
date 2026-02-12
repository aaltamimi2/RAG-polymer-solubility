"""STRAP tool categories for deep agent and subagent wiring.

Each submodule contains plain functions (no @tool decorator needed for deepagents).
Import tool lists by category for selective loading into subagents.
"""

from __future__ import annotations


def _safe_import(module_path: str, names: list[str]) -> list:
    """Import tool functions, returning empty list if module unavailable."""
    try:
        mod = __import__(module_path, fromlist=names)
        return [getattr(mod, n) for n in names if hasattr(mod, n)]
    except Exception:
        return []


# ------------------------------------------------------------------
# Tool lists by category (lazy-imported)
# ------------------------------------------------------------------

def get_database_query_tools() -> list:
    return _safe_import("strap.tools.database_query", [
        "list_tables", "describe_table", "check_column_values",
        "query_database", "validate_and_query",
    ])


def get_adaptive_separation_tools() -> list:
    return _safe_import("strap.tools.adaptive_separation", [
        "find_optimal_separation_conditions",
        "analyze_selective_solubility_enhanced",
    ])


def get_statistical_tools() -> list:
    return _safe_import("strap.tools.statistical", [
        "statistical_summary", "correlation_analysis",
        "compare_groups_statistically", "regression_analysis",
    ])


def get_visualization_tools() -> list:
    return _safe_import("strap.tools.visualization", [
        "plot_solubility_vs_temperature",
        "plot_solubility_vs_temperature_interactive",
        "plot_selectivity_heatmap",
        "plot_multi_panel_analysis",
        "plot_comparison_dashboard",
        "plot_solvent_properties",
        "plot_interpolation_vs_sql",
    ])


def get_solvent_property_tools() -> list:
    return _safe_import("strap.tools.solvent_properties", [
        "get_solvent_properties",
        "rank_solvents_by_property",
    ])


def get_safety_gsk_tools() -> list:
    return _safe_import("strap.tools.safety_gsk", [
        "get_solvent_gscore", "get_family_alternatives",
        "visualize_gscores",
    ])


def get_safety_pubchem_tools() -> list:
    return _safe_import("strap.tools.safety_pubchem", [
        "get_pubchem_safety_info", "compare_pubchem_safety",
        "visualize_pubchem_safety", "get_pubchem_toxicity",
    ])


def get_listing_tools() -> list:
    return _safe_import("strap.tools.listing", [
        "list_available_solvents", "list_available_polymers",
    ])


def get_interpolation_tools() -> list:
    return _safe_import("strap.tools.interpolation", [
        "predict_solubility",
        "predict_solubility_range",
        "list_interpolation_coverage",
        "rank_solvents_selectivity",
    ])


def get_ml_prediction_tools() -> list:
    return _safe_import("strap.tools.ml_prediction", [
        "predict_solubility_ml",
    ])


def get_thermal_prediction_tools() -> list:
    return _safe_import("strap.tools.thermal_prediction", [
        "predict_thermal_properties",
        "generate_solubility_for_new_polymer",
        "list_generated_polymers",
    ])


def get_tea_lca_tools() -> list:
    return _safe_import("strap.tools.tea_lca", [
        "analyze_solvent_recovery_tea", "analyze_solvent_recovery_lca",
        "compare_solvents_tea_lca",
        "generate_tea_lca_visualizations",
    ])


def get_strap_process_tools() -> list:
    return _safe_import("strap.tools.strap_process", [
        "analyze_strap_process", "calculate_strap_msp",
        "compare_strap_scenarios",
        "generate_strap_visualizations",
    ])


def get_literature_tools() -> list:
    return _safe_import("strap.tools.literature", [
        "search_google_scholar", "search_google_patents",
        "lookup_patent", "search_web_of_science",
        "search_arxiv", "search_patentsview",
    ])


def get_scholar_tools() -> list:
    return _safe_import("strap.tools.literature", [
        "search_google_scholar", "search_web_of_science", "search_arxiv",
    ])


def get_patent_tools() -> list:
    return _safe_import("strap.tools.literature", [
        "search_google_patents", "lookup_patent", "search_patentsview",
    ])


def get_rag_core_tools() -> list:
    return _safe_import("strap.tools.rag_core", [
        "search_literature_rag", "ingest_pdf_to_rag",
        "get_rag_status", "ask_literature", "clear_rag_index",
        "visualize_rag_chunks", "check_rag_chunk_quality",
        "get_rag_chunk_report",
        "download_pdf_to_rag",
    ])


def get_rag_diagnostics_tools() -> list:
    return _safe_import("strap.tools.rag_diagnostics", [
        "analyze_search_diagnostics", "visualize_retrieval_patterns",
        "visualize_embedding_space", "analyze_document_similarity",
        "analyze_dense_vs_sparse", "analyze_reranking_impact",
        "analyze_section_boost", "analyze_query_expansion",
        "run_full_rag_diagnostics",
    ])


def get_advanced_separation_tools() -> list:
    return _safe_import("strap.tools.advanced_separation", [
        "find_optimal_separation_sequence",
        "optimize_separation_temperature",
        "calculate_selectivity_detailed",
        "rank_solvents_for_separation",
        "build_compatibility_matrix",
        "find_challenging_polymer_pairs",
        "create_separation_tree_plot",
        "create_selectivity_heatmap",
        "create_process_flow_diagram",
        "find_differential_precipitation_solvents",
        "analyze_multi_polymer_precipitation",
        "analyze_precipitation_temperature",
        "plot_precipitation_curves",
        "plot_atmospheric_feasibility",
        "compare_polymer_pairs_precipitation",
        "check_atmospheric_feasibility",
        "check_multi_polymer_atmospheric_feasibility",
        "find_antisolvents",
        "find_antisolvent_pairs",
        "analyze_selective_antisolvent_precipitation",
        "plan_sequential_separation",
        "plan_multiple_separation_schemes",
        "analyze_integrated_separation",
        "view_alternative_separation_sequence",
    ])


def get_separation_core_tools() -> list:
    """Advanced separation tools excluding visualization (for separation-engineer).

    17 tools: advanced_separation minus 5 viz tools minus analyze_integrated_separation.
    """
    return _safe_import("strap.tools.advanced_separation", [
        "find_optimal_separation_sequence",
        "optimize_separation_temperature",
        "calculate_selectivity_detailed",
        "rank_solvents_for_separation",
        "build_compatibility_matrix",
        "find_challenging_polymer_pairs",
        "plan_sequential_separation",
        "plan_multiple_separation_schemes",
        "view_alternative_separation_sequence",
        "find_differential_precipitation_solvents",
        "analyze_multi_polymer_precipitation",
        "analyze_precipitation_temperature",
        "compare_polymer_pairs_precipitation",
        "check_atmospheric_feasibility",
        "check_multi_polymer_atmospheric_feasibility",
        "find_antisolvents",
        "find_antisolvent_pairs",
        "analyze_selective_antisolvent_precipitation",
    ])


def get_thermal_prediction_tools() -> list:
    return _safe_import("strap.tools.thermal_prediction", [
        "lookup_tg",
    ])


def get_reflection_tools() -> list:
    return _safe_import("strap.tools.reflection", ["think"])


def get_separation_plot_tools() -> list:
    """Separation visualization tools (for visualization-specialist)."""
    return _safe_import("strap.tools.advanced_separation", [
        "create_separation_tree_plot",
        "create_selectivity_heatmap",
        "create_process_flow_diagram",
        "plot_precipitation_curves",
        "plot_atmospheric_feasibility",
    ])


# ------------------------------------------------------------------
# Grouped getters for subagent wiring
# ------------------------------------------------------------------

def get_core_tools() -> list:
    """Tools always available to the main agent.

    Core = database_query (5) + listing (2) + solvent_property (2)
         + interpolation (4) = 13 tools.
    Adaptive separation tools are now exclusive to separation-engineer to
    avoid tool overlap that causes the orchestrator to handle specialist
    queries directly.
    """
    return (
        get_database_query_tools()
        + get_listing_tools()
        + get_solvent_property_tools()
        + get_interpolation_tools()
    )


def get_all_tools() -> list:
    """Every tool across all categories."""
    getters = [
        get_database_query_tools,
        get_adaptive_separation_tools,
        get_statistical_tools,
        get_visualization_tools,
        get_solvent_property_tools,
        get_safety_gsk_tools,
        get_safety_pubchem_tools,
        get_listing_tools,
        get_interpolation_tools,
        get_ml_prediction_tools,
        get_tea_lca_tools,
        get_strap_process_tools,
        get_literature_tools,
        get_rag_core_tools,
        get_rag_diagnostics_tools,
        get_advanced_separation_tools,
    ]
    all_tools = []
    for getter in getters:
        all_tools.extend(getter())
    return all_tools
