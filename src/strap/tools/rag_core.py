"""RAG (Retrieval-Augmented Generation) core tools for literature search."""
from __future__ import annotations
import json
import logging
import asyncio
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from strap.services.rag_service import (
    format_rag_context_cross_kb,
    get_rag_status,
    get_rag_system,
)
from strap.tools._helpers import safe_tool_wrapper, truncate_output
logger = logging.getLogger(__name__)
@safe_tool_wrapper(structured_output=True)
async def search_literature_rag(
    query: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
    knowledgebase: Optional[str] = None,
    include_context: bool = True
) -> str:
    """Search indexed scientific literature using hybrid semantic and keyword RAG search.
    Searches across ALL knowledgebases by default (STRAP-CORE, user-library, etc.)
    to find the most relevant passages regardless of which KB they are in.
    Args:
        query: Natural language question or search query
        top_k: Number of relevant passages to return (default: 5)
        source_filter: Comma-separated document names to search within
        knowledgebase: Search only this KB (default: None = search all KBs)
        include_context: Whether to include expanded parent context (default: True)
    WHEN TO USE:
    - "Search literature for Hansen solubility parameters of polyethylene"
    - "What does the literature say about toluene toxicity?"
    - "Find passages about multilayer film separation"
    """
    try:
        # Get RAG system
        rag_system = get_rag_system()
        # Check if any KB has content
        kbs_with_content = [
            kb.name for kb in rag_system.kb_manager.list_kbs()
            if kb.chunk_count > 0
        ]
        if not kbs_with_content:
            return ("⚠️ **RAG System Not Ready**\n\n"
                    "No documents have been indexed yet.\n\n"
                    "**To index documents:**\n"
                    "1. Add PDF files to the `./rag_pdfs/` directory\n"
                    "2. Use the `ingest_pdf_to_rag` tool to index them\n\n"
                    "Alternatively, use `search_google_scholar` or `search_web_of_science` "
                    "for online literature search.")
        # Parse source filter
        sources = None
        if source_filter:
            sources = [s.strip() for s in source_filter.split(",")]
        # Search: single KB or cross-KB
        kb_filter = [knowledgebase] if knowledgebase else None
        results = rag_system.search_across_kbs(
            query=query,
            top_k=top_k,
            kb_names=kb_filter,
            source_filter=sources,
            return_parent_context=include_context,
        )
        if not results:
            return (f"No relevant passages found for: '{query}'\n\n"
                    "**Suggestions:**\n"
                    "- Try different search terms\n"
                    "- Use more specific or broader query\n"
                    "- Check if relevant documents are indexed with `get_rag_status`")
        # Format output
        searched_kbs = kb_filter or kbs_with_content
        output = [f"# Literature Search Results\n"]
        output.append(f"**Query:** {query}")
        output.append(f"**Found:** {len(results)} relevant passages")
        output.append(f"**KBs searched:** {', '.join(searched_kbs)}\n")
        for i, result in enumerate(results, 1):
            output.append(f"\n---\n")
            output.append(f"### Passage {i}")
            # Source info with KB tag
            kb_tag = ""
            if result.metadata and result.metadata.get("kb_name"):
                kb_tag = f" [{result.metadata['kb_name']}]"
            source_info = f"**Source:** {result.source}{kb_tag}"
            if result.page_number:
                source_info += f" (Page {result.page_number})"
            output.append(source_info)
            # Relevance score
            score = result.rerank_score if result.rerank_score is not None else result.score
            output.append(f"**Relevance:** {score:.3f}")
            # Text content
            text = result.parent_text if include_context and result.parent_text else result.text
            # Truncate very long passages
            if len(text) > 1500:
                text = text[:1500] + "..."
            output.append(f"\n{text}\n")
        # Add summary
        sources_used = list(set(r.source for r in results))
        output.append(f"\n---\n**Sources searched:** {', '.join(sources_used)}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"Literature search failed: {str(e)}\n\nPlease try again with a simpler query."
@safe_tool_wrapper(structured_output=True)
async def ingest_pdf_to_rag(
    pdf_paths: Optional[str] = None,
    use_ocr: bool = False,
    recreate_index: bool = False
) -> str:
    """Ingest PDF documents into the RAG system for semantic literature search.
    Args:
        pdf_paths: Comma-separated paths to PDF files, or None to scan ./rag_pdfs/
        use_ocr: Enable OCR for scanned documents (default: False)
        recreate_index: Delete existing index and start fresh (default: False)
    WHEN TO USE:
    - "Ingest all PDFs in the rag_pdfs folder"
    - "Add paper.pdf to the RAG index"
    - "Re-index all documents with OCR enabled"
    """
    try:
        import glob as glob_module
        # Determine PDF paths
        if pdf_paths:
            paths = [p.strip() for p in pdf_paths.split(",")]
        else:
            # Scan default directory
            paths = glob_module.glob(os.path.join("./rag_pdfs", "*.pdf"))
        if not paths:
            return (f"❌ **No PDFs Found**\n\n"
                    f"No PDF files found in `./rag_pdfs/`\n\n"
                    "**To add documents:**\n"
                    "1. Place PDF files in the `./rag_pdfs/` directory\n"
                    "2. Or specify paths: `ingest_pdf_to_rag(pdf_paths='path/to/file.pdf')`")
        # Perform ingestion
        rag_system = get_rag_system()
        result = rag_system.ingest_pdfs(
            pdf_paths=paths,
            use_ocr=use_ocr,
            recreate_collection=recreate_index
        )
        if not result.get("success"):
            return f"❌ **Ingestion Failed**\n\n{result.get('error', 'Unknown error')}"
        # Format output
        output = ["# 📥 PDF Ingestion Complete\n"]
        output.append(f"**Documents Processed:** {len(result.get('processed_files', []))}")
        output.append(f"**Documents Failed:** {len(result.get('failed_files', []))}")
        output.append(f"\n**Indexing Summary:**")
        output.append(f"- Base chunks: {result.get('base_chunks', 0)}")
        output.append(f"- Parent chunks: {result.get('parent_chunks', 0)}")
        output.append(f"- Child chunks (indexed): {result.get('child_chunks', 0)}")
        # Filter stats
        filter_stats = result.get('filter_stats', {})
        if filter_stats:
            output.append(f"\n**Filtering:**")
            output.append(f"- Processed: {filter_stats.get('total_processed', 0)}")
            output.append(f"- Retained: {filter_stats.get('retained', 0)}")
            output.append(f"- Filtered: headers={filter_stats.get('header_footer', 0)}, "
                         f"citations={filter_stats.get('citation_heavy', 0)}, "
                         f"duplicates={filter_stats.get('duplicate', 0)}")
        # List processed files
        processed = result.get('processed_files', [])
        if processed:
            output.append(f"\n**Processed Files:**")
            for p in processed[:10]:  # Limit to 10
                output.append(f"- {Path(p).name}")
            if len(processed) > 10:
                output.append(f"- ... and {len(processed) - 10} more")
        # Failed files
        failed = result.get('failed_files', [])
        if failed:
            output.append(f"\n**Failed Files:**")
            for f in failed[:5]:
                output.append(f"- {Path(f).name}")
        output.append(f"\n✅ RAG system ready for search. Use `search_literature_rag` to query.")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"RAG ingestion error: {e}")
        return f"❌ Ingestion failed: {str(e)}"
@safe_tool_wrapper(structured_output=True)
async def get_rag_status() -> str:
    """Retrieve current RAG system status, indexed documents, and configuration.
    WHEN TO USE:
    - "Check if documents are indexed before searching"
    - "What documents are available in my RAG index?"
    - "Is the RAG system healthy?"
    """
    try:
        status = get_rag_status()
        output = ["# 📊 RAG System Status\n"]
        # Overall status
        if status.get('ready'):
            output.append("**Status:** ✅ Ready for search")
        elif status.get('initialized'):
            output.append("**Status:** ⚠️ Initialized but no documents indexed")
        else:
            output.append("**Status:** ❌ Not initialized")
        # Dependencies
        output.append(f"\n**Dependencies:**")
        output.append(f"- Embeddings: {'✅' if status.get('embeddings_available') else '❌'}")
        output.append(f"- Vector DB (Qdrant): {'✅' if status.get('qdrant_available') else '❌'}")
        output.append(f"- PDF Processing: {'✅' if status.get('pdf_processing_available') else '❌'}")
        output.append(f"- Reranking: {'✅' if status.get('reranking_enabled') else '❌'}")
        # Collection info
        collection = status.get('collection', {})
        output.append(f"\n**Vector Database:**")
        output.append(f"- Collection: {collection.get('collection_name', 'N/A')}")
        output.append(f"- Indexed chunks: {collection.get('points_count', 0)}")
        output.append(f"- Status: {collection.get('status', 'unknown')}")
        # Chunk store
        chunk_store = status.get('chunk_store', {})
        output.append(f"\n**Document Statistics:**")
        output.append(f"- Total chunks: {chunk_store.get('total_chunks', 0)}")
        output.append(f"- Total sources: {chunk_store.get('total_sources', 0)}")
        output.append(f"- Parent chunks: {chunk_store.get('parent_chunks', 0)}")
        output.append(f"- Child chunks: {chunk_store.get('child_chunks', 0)}")
        output.append(f"- Parent-child mode: {'Yes' if chunk_store.get('parent_child_enabled') else 'No'}")
        # Sources
        sources = chunk_store.get('sources', [])
        if sources:
            output.append(f"\n**Indexed Documents ({len(sources)}):**")
            chunks_per_source = chunk_store.get('chunks_per_source', {})
            for source in sources[:15]:  # Limit to 15
                count = chunks_per_source.get(source, 'N/A')
                output.append(f"- {source}: {count} chunks")
            if len(sources) > 15:
                output.append(f"- ... and {len(sources) - 15} more")
        else:
            output.append(f"\n**No documents indexed yet.**")
            output.append(f"\nTo add documents:")
            output.append(f"1. Place PDF files in `./rag_pdfs/`")
            output.append(f"2. Run `ingest_pdf_to_rag` tool")
        # Configuration
        config = status.get('config', {})
        output.append(f"\n**Configuration:**")
        output.append(f"- Embedding model: {config.get('dense_model', 'N/A')}")
        output.append(f"- Chunk strategy: {config.get('chunk_strategy', 'N/A')}")
        output.append(f"- Chunk size: {config.get('chunk_size', 'N/A')} tokens")
        output.append(f"- Parent-child: {'Yes' if config.get('use_parent_child') else 'No'}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"RAG status error: {e}")
        return f"❌ Failed to get RAG status: {str(e)}"
@safe_tool_wrapper(structured_output=True)
async def ask_literature(
    question: str,
    top_k: int = 5,
    max_context_tokens: int = 3000,
    knowledgebase: Optional[str] = None,
) -> str:
    """Answer a question using synthesized context from indexed scientific literature.
    Searches across ALL knowledgebases by default to find the best context.
    Args:
        question: Natural language question to answer
        top_k: Number of passages to consider (default: 5)
        max_context_tokens: Maximum tokens for context (default: 3000)
        knowledgebase: Search only this KB (default: None = search all KBs)
    WHEN TO USE:
    - "What are the environmental impacts of toluene?"
    - "How does temperature affect polymer solubility?"
    - "What solvents are recommended for PET recycling?"
    """
    try:
        rag_system = get_rag_system()
        kbs_with_content = [
            kb.name for kb in rag_system.kb_manager.list_kbs()
            if kb.chunk_count > 0
        ]
        if not kbs_with_content:
            return ("**Literature Database Not Ready**\n\n"
                    "No documents have been indexed.\n\n"
                    "Use `ingest_pdf_to_rag` to add scientific papers first.")
        # Get context and sources (cross-KB by default)
        kb_filter = [knowledgebase] if knowledgebase else None
        context, sources = format_rag_context_cross_kb(
            query=question,
            top_k=top_k,
            max_tokens=max_context_tokens,
            kb_names=kb_filter,
        )
        if not context:
            return (f"No relevant information found for: '{question}'\n\n"
                    "Try rephrasing your question or check indexed documents with `get_rag_status`.")
        # Format response
        output = [f"# Literature Answer\n"]
        output.append(f"**Question:** {question}\n")
        output.append(f"---\n")
        output.append(f"## Relevant Information from Literature\n")
        output.append(context)
        output.append(f"\n---\n")
        output.append(f"## Sources ({len(sources)} passages)")
        for i, src in enumerate(sources, 1):
            source_name = src.get('source', 'Unknown')
            page = src.get('page_number')
            score = src.get('score', 0)
            kb_tag = ""
            meta = src.get('metadata', {})
            if meta and meta.get('kb_name'):
                kb_tag = f" [{meta['kb_name']}]"
            source_str = f"{i}. **{source_name}**{kb_tag}"
            if page:
                source_str += f" (Page {page})"
            source_str += f" - Relevance: {score:.2f}"
            output.append(source_str)
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Ask literature error: {e}")
        return f"Failed to search literature: {str(e)}"
@safe_tool_wrapper(structured_output=True)
async def clear_rag_index() -> str:
    """Clear the RAG index and remove all indexed documents from the vector database.
    WHEN TO USE:
    - "Reset the RAG index to remove all documents"
    - "Fix a corrupted RAG index"
    - "Start over with a new document set"
    """
    try:
        rag_system = get_rag_system()
        # Get current status
        status = rag_system.get_status()
        collection = status.get('collection', {})
        points_before = collection.get('points_count', 0)
        # Clear the index
        if rag_system.vector_db:
            success = rag_system.vector_db.delete_collection()
            if not success:
                return "❌ Failed to delete collection"
        # Clear chunk store
        rag_system.chunk_store.clear()
        output = ["# 🗑️ RAG Index Cleared\n"]
        output.append(f"**Deleted:** {points_before} indexed chunks")
        output.append(f"\nThe RAG system is now empty.")
        output.append(f"\nTo re-index documents:")
        output.append(f"1. Ensure PDFs are in `./rag_pdfs/`")
        output.append(f"2. Run `ingest_pdf_to_rag`")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Clear RAG index error: {e}")
        return f"❌ Failed to clear index: {str(e)}"
@safe_tool_wrapper(structured_output=True)
async def visualize_rag_chunks() -> str:
    """Generate a 6-panel visualization of indexed chunk distributions and statistics.
    WHEN TO USE:
    - "Show me visualization of the indexed chunks"
    - "Generate chunk distribution plots"
    - "Visualize the RAG document breakdown"
    """
    try:
        # Generate plot
        plot_path = rag.plot_chunk_distributions()
        if plot_path is None:
            return ("⚠️ **No chunks to visualize**\n\n"
                    "The RAG index is empty. Use `ingest_pdf_to_rag` to add documents first.")
        # Get summary stats
        summary = rag.get_chunk_summary()
        output = ["# 📊 RAG Chunk Visualization\n"]
        output.append(f"**Plot saved to:** `{plot_path}`\n")
        # Quick stats
        output.append("## Summary Statistics\n")
        output.append(f"- **Total Chunks:** {summary['total_chunks']:,}")
        output.append(f"- **Documents:** {summary['total_documents']}")
        output.append(f"- **Total Tokens:** {summary['token_stats']['total']:,}")
        ts = summary['token_stats']
        output.append(f"\n**Token Distribution:**")
        output.append(f"- Mean: {ts['mean']:.1f} | Median: {ts['median']:.1f}")
        output.append(f"- Range: {ts['min']} - {ts['max']}")
        output.append(f"- Std Dev: {ts['std']:.1f}")
        # Section breakdown
        output.append(f"\n**Section Types:**")
        for section, count in sorted(summary['section_distribution'].items(),
                                      key=lambda x: x[1], reverse=True)[:5]:
            pct = 100 * count / summary['total_chunks']
            output.append(f"- {section.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        output.append(f"\n**View the full visualization at:** `{plot_path}`")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Chunk visualization error: {e}")
        return f"❌ Failed to generate visualization: {str(e)}"
@safe_tool_wrapper(structured_output=True)
async def check_rag_chunk_quality() -> str:
    """Run quality checks on ingested RAG chunks and identify potential issues.
    WHEN TO USE:
    - "Check the quality of my RAG chunks"
    - "Are my indexed documents properly chunked?"
    - "Run quality diagnostics on the RAG index"
    """
    try:
        quality = rag.check_chunk_quality()
        if quality['status'] == 'error':
            return (f"⚠️ **{quality['message']}**\n\n"
                    f"**Recommendations:**\n" +
                    "\n".join(f"- {r}" for r in quality['recommendations']))
        output = ["# 🔍 RAG Chunk Quality Report\n"]
        # Status indicator
        if quality['status'] == 'healthy':
            output.append("**Status:** ✅ All checks passed\n")
        elif quality['status'] == 'warnings':
            output.append("**Status:** ⚠️ Warnings detected\n")
        else:
            output.append("**Status:** ❌ Issues found\n")
        output.append(f"**Total Chunks:** {quality['total_chunks']:,}\n")
        # Issues
        if quality['issues']:
            output.append("## ❌ Issues\n")
            for issue in quality['issues']:
                output.append(f"- {issue}")
            output.append("")
        # Warnings
        if quality['warnings']:
            output.append("## ⚠️ Warnings\n")
            for warning in quality['warnings']:
                output.append(f"- {warning}")
            output.append("")
        # Recommendations
        if quality['recommendations']:
            output.append("## 💡 Recommendations\n")
            for rec in quality['recommendations']:
                output.append(f"- {rec}")
            output.append("")
        # Quality metrics
        summary = quality.get('summary', {})
        if summary:
            qm = summary.get('quality_metrics', {})
            output.append("## 📊 Quality Metrics\n")
            output.append(f"- **Tiny chunks (<20 tokens):** {qm.get('tiny_chunks', 0)} ({qm.get('tiny_percentage', 0):.1f}%)")
            output.append(f"- **Large chunks (>1000 tokens):** {qm.get('large_chunks', 0)} ({qm.get('large_percentage', 0):.1f}%)")
            output.append(f"- **Empty chunks:** {qm.get('empty_chunks', 0)}")
            output.append(f"- **Coefficient of Variation:** {qm.get('cv', 0):.2f}")
            if qm.get('cv', 0) < 0.5:
                output.append("  (Good - consistent chunk sizes)")
            elif qm.get('cv', 0) < 1.0:
                output.append("  (Moderate - some variation)")
            else:
                output.append("  (High - inconsistent chunking)")
        if quality['status'] == 'healthy':
            output.append("\n✅ Your RAG index is healthy and ready for searching!")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Chunk quality check error: {e}")
        return f"❌ Failed to check quality: {str(e)}"
@safe_tool_wrapper(structured_output=True)
async def get_rag_chunk_report() -> str:
    """Generate a comprehensive report of all RAG chunk statistics.
    WHEN TO USE:
    - "Give me a full report on my RAG chunks"
    - "Generate detailed RAG index statistics"
    - "What's the complete breakdown of my indexed literature?"
    """
    try:
        report = rag.generate_chunk_report()
        return report
    except Exception as e:
        logger.error(f"Chunk report error: {e}")
        return f"❌ Failed to generate report: {str(e)}"
# ------------------------------------------------------------------
# Download / ingestion tools (moved from rag_diagnostics)
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
async def download_pdf_to_rag(
    url: str,
    filename: Optional[str] = None,
    auto_ingest: bool = True
) -> str:
    """Download a PDF from a URL and save it to the RAG system.
    Args:
        url: Direct URL to the PDF file
        filename: Custom filename without .pdf extension (auto-generated if omitted)
        auto_ingest: Automatically ingest into RAG after download (default: True)
    WHEN TO USE:
    - "Download this PDF to RAG: https://arxiv.org/pdf/2301.00001.pdf"
    - "Save the PDF from this link to my literature collection"
    """
    import requests
    import re
    from pathlib import Path
    from urllib.parse import urlparse, unquote
    try:
        # Validate URL
        if not url or not url.startswith(('http://', 'https://')):
            return "❌ Invalid URL. Please provide a valid HTTP/HTTPS URL."
        # Create RAG pdfs directory if it doesn't exist
        pdf_dir = Path(rag.RAG_PDF_DIR)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        # Generate filename if not provided
        if not filename:
            # Try to extract from URL
            parsed_url = urlparse(url)
            url_filename = unquote(Path(parsed_url.path).name)
            if url_filename and url_filename.endswith('.pdf'):
                filename = url_filename[:-4]  # Remove .pdf
            else:
                # Generate unique name
                import hashlib
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"downloaded_{url_hash}"
        # Clean filename
        filename = re.sub(r'[^\w\-_]', '_', filename)
        filepath = pdf_dir / f"{filename}.pdf"
        # Check if file already exists
        if filepath.exists():
            return (f"⚠️ File `{filename}.pdf` already exists in RAG directory.\n\n"
                    f"Use a different filename or delete the existing file first.")
        # Download the PDF
        logger.info(f"Downloading PDF from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()
        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
            return (f"⚠️ URL does not appear to be a PDF.\n"
                    f"Content-Type: {content_type}\n\n"
                    "Please provide a direct link to a PDF file.")
        # Save the file
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        output = [f"# ✅ PDF Downloaded Successfully\n"]
        output.append(f"**Filename:** {filename}.pdf")
        output.append(f"**Size:** {file_size_mb:.2f} MB")
        output.append(f"**Location:** {filepath}")
        # Auto-ingest if requested
        if auto_ingest:
            output.append(f"\n**Ingesting into RAG...**")
            rag_system = get_rag_system()
            result = rag_system.ingest_pdfs(
                pdf_paths=[str(filepath)],
                use_ocr=False,
                recreate_collection=False
            )
            if result.get("success"):
                output.append(f"✅ Successfully indexed!")
                output.append(f"- Chunks created: {result.get('total_chunks', 0)}")
                output.append(f"\nYou can now search this document with `search_literature_rag`")
            else:
                output.append(f"⚠️ Ingestion had issues: {result.get('error', 'Unknown')}")
                output.append(f"The PDF is saved. Try `ingest_pdf_to_rag` manually.")
        else:
            output.append(f"\n📁 PDF saved. To index it, run `ingest_pdf_to_rag`")
        return "\n".join(output)
    except requests.exceptions.RequestException as e:
        return f"❌ Download failed: {str(e)}\n\nCheck if the URL is accessible and points to a PDF."
    except Exception as e:
        logger.error(f"PDF download error: {e}")
        return f"❌ Error downloading PDF: {str(e)}"
