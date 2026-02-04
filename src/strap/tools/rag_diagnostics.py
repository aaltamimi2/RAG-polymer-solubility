"""RAG diagnostics tools."""

from __future__ import annotations

import json
import logging
import asyncio
import os
from typing import Optional

from strap.tools._helpers import safe_tool_wrapper, truncate_output

try:
    from strap.vendor import rag
except ImportError:
    rag = None

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Diagnostics tools
# ------------------------------------------------------------------

@safe_tool_wrapper
async def analyze_search_diagnostics(
    query: str,
    top_k: int = 10
) -> str:
    """Analyze search score breakdown showing dense, sparse, boost, and reranking contributions.

    Args:
        query: Search query to analyze
        top_k: Number of results to analyze (default: 10)

    WHEN TO USE:
    - "Analyze search scores for 'polymer dissolution temperature'"
    - "Why are these results ranking this way for 'Hansen parameters'?"
    - "Debug search scores for 'PET recycling'"
    """
    try:
        analysis = rag.analyze_search_scores(query=query, top_k=top_k)

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = [f"# 🔍 Search Score Analysis\n"]
        output.append(f"**Query:** {query}")
        output.append(f"**Results Analyzed:** {analysis['num_results']}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Score statistics
        stats = analysis.get("score_stats", {})
        output.append("## Score Statistics\n")
        output.append(f"- **Avg Dense Score:** {stats.get('dense_mean', 0):.3f}")
        output.append(f"- **Avg Sparse Score:** {stats.get('sparse_mean', 0):.3f}")
        output.append(f"- **Reranking Improved:** {stats.get('rerank_improved', 0)} results\n")

        # Top results breakdown
        output.append("## Top Results Breakdown\n")
        output.append("| Rank | Source | Section | Dense | Sparse | Boost | Final |")
        output.append("|------|--------|---------|-------|--------|-------|-------|")

        for i, r in enumerate(analysis.get("results", [])[:5], 1):
            output.append(f"| {i} | {r['source'][:15]}... | {r['section'][:10]} | "
                         f"{r['dense_score']:.3f} | {r['sparse_score']:.3f} | "
                         f"{r['section_boost']:.3f} | {r['final_score']:.3f} |")

        output.append("\n**Interpretation:**")
        if stats.get('dense_mean', 0) > stats.get('sparse_mean', 0):
            output.append("- Dense (semantic) search is contributing more than sparse (keyword)")
        else:
            output.append("- Sparse (keyword) search is contributing more than dense (semantic)")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Search diagnostics error: {e}")
        return f"❌ Failed to analyze search: {str(e)}"


@safe_tool_wrapper
async def visualize_retrieval_patterns() -> str:
    """Analyze retrieval patterns to identify document and section frequency biases.

    WHEN TO USE:
    - "Show me retrieval patterns across my documents"
    - "Which documents are retrieved most often?"
    - "Analyze section distribution in search results"
    """
    try:
        analysis = rag.analyze_retrieval_patterns()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 📊 Retrieval Pattern Analysis\n"]
        output.append(f"**Queries Tested:** {analysis['num_queries']}")
        output.append(f"**Test Queries:** {', '.join(analysis.get('queries_tested', [])[:3])}...\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Most retrieved documents
        output.append("## Most Retrieved Documents\n")
        for doc, count in analysis.get("most_retrieved_docs", []):
            output.append(f"- **{doc[:40]}...**: {count} times")

        # Section distribution
        output.append("\n## Section Distribution\n")
        section_dist = analysis.get("section_distribution", {})
        total = sum(section_dist.values())
        for section, count in sorted(section_dist.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total if total > 0 else 0
            output.append(f"- **{section.replace('_', ' ').title()}**: {count} ({pct:.1f}%)")

        # Top score average
        output.append(f"\n**Avg Top-1 Score:** {analysis.get('avg_top_score', 0):.3f}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Retrieval patterns error: {e}")
        return f"❌ Failed to analyze retrieval patterns: {str(e)}"


@safe_tool_wrapper
async def visualize_embedding_space(
    sample_size: int = 500,
    method: str = "tsne"
) -> str:
    """Visualize document embeddings in 2D space via dimensionality reduction.

    Args:
        sample_size: Number of chunks to sample (default: 500)
        method: Reduction method, "tsne" or "umap" (default: tsne)

    WHEN TO USE:
    - "Visualize my document embeddings"
    - "Show embedding space clustering"
    - "Create t-SNE plot of my RAG documents"
    """
    try:
        analysis = rag.visualize_embedding_space(sample_size=sample_size, method=method)

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = [f"# 🗺️ Embedding Space Visualization\n"]
        output.append(f"**Method:** {analysis['method'].upper()}")
        output.append(f"**Embeddings Visualized:** {analysis['num_embeddings']}")
        output.append(f"**Embedding Dimension:** {analysis['embedding_dim']}")
        output.append(f"**Unique Documents:** {analysis['unique_sources']}")
        output.append(f"**Unique Sections:** {analysis['unique_sections']}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        output.append("**Interpretation:**")
        output.append("- Clusters indicate semantically similar content")
        output.append("- Documents should cluster if they cover similar topics")
        output.append("- Scattered points may indicate diverse content")
        output.append("- Outliers may be unique or poorly extracted content")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Embedding visualization error: {e}")
        return f"❌ Failed to visualize embeddings: {str(e)}"


@safe_tool_wrapper
async def analyze_document_similarity() -> str:
    """Compute and visualize a document-level similarity matrix.

    WHEN TO USE:
    - "Which documents in my RAG are most similar?"
    - "Show document similarity matrix"
    - "Find related papers in my collection"
    """
    try:
        analysis = rag.compute_document_similarity_matrix()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 📐 Document Similarity Analysis\n"]
        output.append(f"**Documents Analyzed:** {analysis['num_documents']}")
        output.append(f"**Average Similarity:** {analysis['avg_similarity']:.3f}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Most similar pairs
        output.append("## Most Similar Document Pairs\n")
        for pair in analysis.get("most_similar_pairs", [])[:5]:
            output.append(f"- **{pair['doc1'][:25]}...** ↔ **{pair['doc2'][:25]}...**: {pair['similarity']:.3f}")

        # Least similar pairs
        output.append("\n## Least Similar Document Pairs\n")
        for pair in analysis.get("least_similar_pairs", [])[:3]:
            output.append(f"- **{pair['doc1'][:25]}...** ↔ **{pair['doc2'][:25]}...**: {pair['similarity']:.3f}")

        output.append("\n**Interpretation:**")
        output.append("- Similarity > 0.8: Very related content, possible overlap")
        output.append("- Similarity 0.5-0.8: Related topics")
        output.append("- Similarity < 0.5: Different topics (good diversity)")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Document similarity error: {e}")
        return f"❌ Failed to compute similarity: {str(e)}"


@safe_tool_wrapper
async def analyze_dense_vs_sparse() -> str:
    """Compare dense (semantic) vs sparse (keyword) retrieval performance and correlation.

    WHEN TO USE:
    - "Compare dense and sparse retrieval"
    - "Is semantic search or keyword search working better?"
    - "Analyze hybrid search components"
    """
    try:
        analysis = rag.analyze_dense_vs_sparse()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# ⚖️ Dense vs Sparse Retrieval Analysis\n"]
        output.append(f"**Queries Tested:** {analysis['num_queries']}")
        output.append(f"**Results Analyzed:** {analysis['num_results']}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Key statistics
        output.append("## Key Statistics\n")
        output.append(f"- **Correlation:** {analysis['correlation']:.3f}")
        output.append(f"- **Avg Dense Score:** {analysis['avg_dense_score']:.3f}")
        output.append(f"- **Avg Sparse Score:** {analysis['avg_sparse_score']:.3f}")
        output.append(f"- **Dense Wins:** {analysis['dense_wins']} ({analysis['dense_win_rate']*100:.1f}%)")
        output.append(f"- **Sparse Wins:** {analysis['sparse_wins']} ({(1-analysis['dense_win_rate'])*100:.1f}%)")

        # Interpretation
        output.append("\n**Interpretation:**")
        if analysis['correlation'] > 0.7:
            output.append("- High correlation: Both methods agree on relevance")
        elif analysis['correlation'] > 0.4:
            output.append("- Moderate correlation: Methods complement each other")
        else:
            output.append("- Low correlation: Methods capture different signals (good for hybrid)")

        if analysis['dense_win_rate'] > 0.7:
            output.append("- Semantic search is dominant - consider adjusting sparse weight")
        elif analysis['dense_win_rate'] < 0.3:
            output.append("- Keyword search is dominant - documents may have strong keywords")
        else:
            output.append("- Balanced contribution - hybrid search working well")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Dense vs sparse error: {e}")
        return f"❌ Failed to analyze: {str(e)}"


@safe_tool_wrapper
async def analyze_reranking_impact() -> str:
    """Analyze how cross-encoder reranking changes result ordering and scores.

    WHEN TO USE:
    - "How much does reranking help my search results?"
    - "Analyze reranking impact"
    - "Is the cross-encoder improving ranking?"
    """
    try:
        analysis = rag.analyze_reranking_impact()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 🔄 Reranking Impact Analysis\n"]
        output.append(f"**Queries Tested:** {analysis['num_queries']}")
        output.append(f"**Results Analyzed:** {analysis['total_results']}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Position changes
        output.append("## Position Changes\n")
        output.append(f"- **Results with position change:** {analysis['results_with_position_change']}")
        output.append(f"- **Moved up:** {analysis['moved_up']}")
        output.append(f"- **Moved down:** {analysis['moved_down']}")
        output.append(f"- **Unchanged:** {analysis['unchanged']}")
        output.append(f"- **Avg position change:** {analysis['avg_position_change']:.2f}")

        # Interpretation
        output.append("\n**Interpretation:**")
        if analysis['moved_up'] > analysis['moved_down']:
            output.append("- Reranking is promoting relevant results ✅")
        elif analysis['moved_up'] < analysis['moved_down']:
            output.append("- Reranking may be demoting good results ⚠️")
        else:
            output.append("- Reranking has balanced effect")

        pct_changed = analysis['results_with_position_change'] / analysis['total_results'] * 100 if analysis['total_results'] > 0 else 0
        if pct_changed > 50:
            output.append(f"- High impact: {pct_changed:.0f}% of results changed position")
        else:
            output.append(f"- Moderate impact: {pct_changed:.0f}% of results changed position")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Reranking analysis error: {e}")
        return f"❌ Failed to analyze reranking: {str(e)}"


@safe_tool_wrapper
async def analyze_section_boost() -> str:
    """Analyze section-based score boosting impact on final ranking.

    WHEN TO USE:
    - "How does section boosting affect my results?"
    - "Analyze section boost impact"
    - "Are abstracts being prioritized correctly?"
    """
    try:
        analysis = rag.analyze_section_boost_impact()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 📑 Section Boost Analysis\n"]
        output.append(f"**Results Analyzed:** {analysis['total_results']}")
        output.append(f"**Avg Boost Contribution:** {analysis['avg_boost_contribution']*100:.1f}%\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Configured boosts
        output.append("## Configured Section Boosts\n")
        for section, boost in analysis.get("configured_boosts", {}).items():
            output.append(f"- **{section.title()}:** +{boost:.2f}")

        # Section performance
        output.append("\n## Section Performance\n")
        output.append("| Section | Count | Avg Boost | Avg Rank |")
        output.append("|---------|-------|-----------|----------|")
        for section, stats in analysis.get("section_stats", {}).items():
            output.append(f"| {section[:12]} | {stats['count']} | {stats['avg_boost']:.3f} | {stats['avg_rank']:.1f} |")

        output.append("\n**Interpretation:**")
        output.append("- Lower avg rank = appearing higher in results")
        output.append("- Higher boost = more priority given to section type")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Section boost error: {e}")
        return f"❌ Failed to analyze section boost: {str(e)}"


@safe_tool_wrapper
async def analyze_query_expansion() -> str:
    """Analyze query expansion effectiveness on retrieval results.

    WHEN TO USE:
    - "Is query expansion helping my searches?"
    - "Analyze query expansion effectiveness"
    - "How do expanded terms affect results?"
    """
    try:
        analysis = rag.analyze_query_expansion()

        if analysis.get("error"):
            return f"❌ {analysis['error']}"

        output = ["# 🔀 Query Expansion Analysis\n"]
        output.append(f"**Queries Tested:** {analysis['num_queries']}")
        output.append(f"**Net Result Change:** {analysis['net_change']:+d}\n")

        if analysis.get("plot_path"):
            output.append(f"**Visualization:** `{analysis['plot_path']}`\n")

        # Summary
        output.append("## Summary\n")
        output.append(f"- **Total New Results:** {analysis['total_new_results']}")
        output.append(f"- **Total Lost Results:** {analysis['total_lost_results']}")
        output.append(f"- **Avg Score Change:** {analysis['avg_score_improvement']:+.3f}")

        # Per-query details
        output.append("\n## Query Details\n")
        for detail in analysis.get("expansion_details", [])[:5]:
            output.append(f"\n**Query:** {detail['original_query']}")
            output.append(f"- Expansions: {detail['num_expansions']}")
            output.append(f"- New results: +{detail['new_results']}, Lost: -{detail['lost_results']}")
            if detail.get('expanded_queries'):
                output.append(f"- Expanded to: {', '.join(detail['expanded_queries'][:3])}...")

        # Interpretation
        output.append("\n**Interpretation:**")
        if analysis['net_change'] > 0:
            output.append("- Expansion is adding relevant results ✅")
        elif analysis['net_change'] < 0:
            output.append("- Expansion may be diluting results ⚠️")
        else:
            output.append("- Expansion has neutral effect")

        if analysis['avg_score_improvement'] > 0:
            output.append("- Expanded queries have higher avg scores ✅")
        else:
            output.append("- Original queries may be sufficient")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Query expansion error: {e}")
        return f"❌ Failed to analyze query expansion: {str(e)}"


@safe_tool_wrapper
async def run_full_rag_diagnostics() -> str:
    """Run all RAG diagnostics and generate a comprehensive report with visualizations.

    WHEN TO USE:
    - "Run full RAG diagnostics"
    - "Generate complete RAG system report"
    - "Health check my RAG system"
    """
    try:
        results = rag.generate_full_rag_diagnostics()

        output = ["# 🔬 Full RAG System Diagnostics\n"]
        output.append(f"**Generated:** {results['timestamp']}")
        output.append(f"**Plots Created:** {len(results.get('all_plots', []))}\n")

        # List all diagnostics
        output.append("## Diagnostics Run\n")

        diagnostics = results.get("diagnostics", {})

        # Chunk distribution
        if "chunk_distribution" in diagnostics:
            cd = diagnostics["chunk_distribution"]
            summary = cd.get("summary", {})
            output.append(f"### 1. Chunk Distribution")
            output.append(f"- Total chunks: {summary.get('total_chunks', 0)}")
            output.append(f"- Documents: {summary.get('total_documents', 0)}")
            output.append(f"- Plot: `{cd.get('plot_path', 'N/A')}`\n")

        # Chunk quality
        if "chunk_quality" in diagnostics:
            cq = diagnostics["chunk_quality"]
            output.append(f"### 2. Chunk Quality")
            output.append(f"- Status: {'✅' if cq.get('status') == 'healthy' else '⚠️'} {cq.get('status', 'unknown')}")
            output.append(f"- Issues: {len(cq.get('issues', []))}")
            output.append(f"- Warnings: {len(cq.get('warnings', []))}\n")

        # Retrieval patterns
        if "retrieval_patterns" in diagnostics:
            rp = diagnostics["retrieval_patterns"]
            output.append(f"### 3. Retrieval Patterns")
            output.append(f"- Plot: `{rp.get('plot_path', 'N/A')}`\n")

        # Embedding space
        if "embedding_space" in diagnostics:
            es = diagnostics["embedding_space"]
            output.append(f"### 4. Embedding Space")
            output.append(f"- Embeddings: {es.get('num_embeddings', 0)}")
            output.append(f"- Plot: `{es.get('plot_path', 'N/A')}`\n")

        # Document similarity
        if "document_similarity" in diagnostics:
            ds = diagnostics["document_similarity"]
            output.append(f"### 5. Document Similarity")
            output.append(f"- Avg similarity: {ds.get('avg_similarity', 0):.3f}")
            output.append(f"- Plot: `{ds.get('plot_path', 'N/A')}`\n")

        # Dense vs sparse
        if "dense_vs_sparse" in diagnostics:
            dvs = diagnostics["dense_vs_sparse"]
            output.append(f"### 6. Dense vs Sparse")
            output.append(f"- Correlation: {dvs.get('correlation', 0):.3f}")
            output.append(f"- Dense win rate: {dvs.get('dense_win_rate', 0)*100:.0f}%")
            output.append(f"- Plot: `{dvs.get('plot_path', 'N/A')}`\n")

        # Reranking
        if "reranking_impact" in diagnostics:
            ri = diagnostics["reranking_impact"]
            output.append(f"### 7. Reranking Impact")
            output.append(f"- Position changes: {ri.get('results_with_position_change', 0)}")
            output.append(f"- Plot: `{ri.get('plot_path', 'N/A')}`\n")

        # Section boost
        if "section_boost" in diagnostics:
            sb = diagnostics["section_boost"]
            output.append(f"### 8. Section Boost")
            output.append(f"- Plot: `{sb.get('plot_path', 'N/A')}`\n")

        # Query expansion
        if "query_expansion" in diagnostics:
            qe = diagnostics["query_expansion"]
            output.append(f"### 9. Query Expansion")
            output.append(f"- Net result change: {qe.get('net_change', 0):+d}")
            output.append(f"- Plot: `{qe.get('plot_path', 'N/A')}`\n")

        # All plots
        output.append("## Generated Plots\n")
        for plot_path in results.get("all_plots", []):
            if plot_path:
                output.append(f"- `{plot_path}`")

        output.append("\n✅ Full diagnostics complete! Review plots for detailed analysis.")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Full diagnostics error: {e}")
        return f"❌ Failed to run diagnostics: {str(e)}"
