"""Thin adapter layer around the vendored RAG module."""

from __future__ import annotations

from typing import Any


def _get_vendor_rag() -> Any:
    try:
        from strap.vendor import rag as rag_module
    except ImportError as exc:  # pragma: no cover - exercised via callers
        raise RuntimeError("Vendored RAG module is unavailable") from exc
    return rag_module


def get_rag_system() -> Any:
    return _get_vendor_rag().get_rag_system()


def get_rag_status() -> dict[str, Any]:
    return _get_vendor_rag().get_rag_status()


def format_rag_context_cross_kb(*args, **kwargs):
    return _get_vendor_rag().format_rag_context_cross_kb(*args, **kwargs)


def analyze_search_scores(*args, **kwargs):
    return _get_vendor_rag().analyze_search_scores(*args, **kwargs)


def analyze_retrieval_patterns(*args, **kwargs):
    return _get_vendor_rag().analyze_retrieval_patterns(*args, **kwargs)


def visualize_embedding_space(*args, **kwargs):
    return _get_vendor_rag().visualize_embedding_space(*args, **kwargs)


def compute_document_similarity_matrix(*args, **kwargs):
    return _get_vendor_rag().compute_document_similarity_matrix(*args, **kwargs)


def analyze_dense_vs_sparse(*args, **kwargs):
    return _get_vendor_rag().analyze_dense_vs_sparse(*args, **kwargs)


def analyze_reranking_impact(*args, **kwargs):
    return _get_vendor_rag().analyze_reranking_impact(*args, **kwargs)


def analyze_section_boost_impact(*args, **kwargs):
    return _get_vendor_rag().analyze_section_boost_impact(*args, **kwargs)


def analyze_query_expansion(*args, **kwargs):
    return _get_vendor_rag().analyze_query_expansion(*args, **kwargs)
