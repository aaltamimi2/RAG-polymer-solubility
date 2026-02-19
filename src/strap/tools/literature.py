"""Literature search tools: Google Scholar, Google Patents, PatentsView,
Web of Science, arXiv — with relevancy filtering and selective RAG ingestion."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from strap.tools._helpers import safe_tool_wrapper, truncate_output

logger = logging.getLogger(__name__)

# Default max papers to auto-save to RAG
DEFAULT_MAX_SAVE = 2

# Maximum PDF download size to prevent disk exhaustion
_MAX_PDF_BYTES = 100 * 1024 * 1024  # 100 MB

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

_DL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def _apply_relevancy_filter(
    results: list[dict],
    filter_relevant: bool,
    text_field: str = "snippet",
    title_field: str = "title",
    keywords_field: Optional[str] = None,
) -> tuple[list[dict], list, str]:
    """Optionally filter results by domain relevancy.

    Returns (results_to_show, all_scores, summary_text).
    When filter_relevant is False, returns originals unchanged.
    """
    if not filter_relevant or not results:
        return results, [], ""

    from strap.tools.relevancy import filter_results, format_relevancy_summary

    kept, all_scores = filter_results(
        results,
        text_field=text_field,
        title_field=title_field,
        keywords_field=keywords_field,
        min_category="MEDIUM",
    )
    summary = format_relevancy_summary(all_scores, len(kept), len(results))
    return kept, all_scores, summary


def _get_relevancy_badge(item: dict) -> str:
    """Return a short relevancy badge string if scoring info is attached."""
    rel = item.get("_relevancy")
    if not rel:
        return ""
    terms = ", ".join(rel.matched_terms[:5])
    return f"**Relevancy:** {rel.category} ({rel.score:.2f}){' -- ' + terms if terms else ''}"


def _download_and_ingest(
    download_items: list[dict],
    max_save: int,
    knowledgebase: Optional[str],
    source_prefix: str,
) -> list[str]:
    """Download PDFs and ingest into RAG. Returns output lines.

    Each item in *download_items* must have keys:
      - 'url': PDF download URL
      - 'filename': safe filename stem (no .pdf)
      - 'title': display title
    """
    import requests as _requests

    pdf_dir = Path("rag_pdfs")
    pdf_dir.mkdir(exist_ok=True)

    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    output: list[str] = []

    for item in download_items:
        if len(downloaded) >= max_save:
            break

        title = item["title"]
        pdf_url = item.get("url")
        if not pdf_url:
            skipped.append(f"{title[:50]}... (no PDF)")
            continue

        fname = re.sub(r"[^\w\-]", "_", item["filename"][:60])
        fpath = pdf_dir / f"{source_prefix}_{fname}.pdf"

        if fpath.exists():
            skipped.append(f"{title[:50]}... (exists)")
            continue

        try:
            resp = _requests.get(
                pdf_url, headers=_DL_HEADERS, timeout=120,
                stream=True, allow_redirects=True,
            )
            # For Google Patents: fall back to storage URL if we get HTML
            if (resp.status_code != 200
                    or "html" in resp.headers.get("Content-Type", "").lower()):
                alt = item.get("alt_url")
                if alt:
                    resp = _requests.get(
                        alt, headers=_DL_HEADERS, timeout=120, stream=True
                    )
            resp.raise_for_status()

            total_bytes = 0
            with open(fpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > _MAX_PDF_BYTES:
                        logger.warning(
                            "PDF exceeds %d byte limit, skipping: %s",
                            _MAX_PDF_BYTES, pdf_url,
                        )
                        break

            if total_bytes > _MAX_PDF_BYTES:
                fpath.unlink(missing_ok=True)
                continue  # skip adding to downloaded list
            downloaded.append(str(fpath))
        except Exception as e:
            failed.append(f"{title[:40]}... ({str(e)[:40]})")

    # Ingest into RAG
    if downloaded:
        try:
            from strap.vendor import rag as rag_mod

            rag_sys = rag_mod.get_rag_system()

            # Default to user-library when no KB specified
            if not knowledgebase:
                knowledgebase = "user-library"

            # Check readonly — redirect to user-library unless explicitly forced
            kb_info = rag_sys.kb_manager.get_kb(knowledgebase)
            if kb_info and getattr(kb_info, "readonly", False):
                logger.info(
                    f"KB '{knowledgebase}' is readonly, redirecting to user-library"
                )
                knowledgebase = "user-library"
                kb_info = rag_sys.kb_manager.get_kb(knowledgebase)

            # Switch or create knowledgebase
            prev_kb = None
            try:
                rag_sys.switch_kb(knowledgebase)
            except ValueError:
                rag_sys.create_kb(
                    knowledgebase,
                    description="User-ingested literature",
                    switch_to=True,
                )
            prev_kb = knowledgebase

            # Auto-detect image-based PDFs (e.g. USPTO patents) and enable OCR
            needs_ocr = False
            try:
                import PyPDF2

                for p in downloaded:
                    reader = PyPDF2.PdfReader(p)
                    sample_text = "".join(
                        reader.pages[i].extract_text() or ""
                        for i in range(min(3, len(reader.pages)))
                    )
                    if len(sample_text.strip()) < 50:
                        needs_ocr = True
                        break
            except Exception:
                pass

            result = rag_sys.ingest_pdfs(
                downloaded, incremental=True, use_ocr=needs_ocr,
                interpret_figures=False,          # Fast: skip Gemini figure interpretation
                use_contextual_enrichment=False,  # Fast: skip LLM context enrichment
            )
            n_chunks = result.get("total_chunks", "?")
            kb_used = result.get("kb_name", knowledgebase or "default")
            output.append(
                f"\n**RAG:** Saved {len(downloaded)} PDFs -> "
                f"knowledgebase **{kb_used}** ({n_chunks} chunks)"
            )
        except Exception as e:
            logger.error(f"RAG ingest failed: {e}")
            output.append(f"\n**RAG ingest failed:** {e}")
    else:
        output.append("\n**RAG:** No PDFs could be downloaded.")

    if skipped:
        output.append(f"**Skipped:** {len(skipped)} papers")
    if failed:
        output.append(f"**Failed:** {'; '.join(failed)}")

    return output


# ============================================================
# Google Scholar Search (SerpAPI)
# ============================================================

@safe_tool_wrapper
def search_google_scholar(
    query: str,
    max_results: int = 10,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None,
    filter_relevant: bool = True,
    save_to_rag: bool = False,
    max_save: int = DEFAULT_MAX_SAVE,
    knowledgebase: Optional[str] = None,
) -> str:
    """Search Google Scholar for academic research articles via SerpAPI.

    Args:
        query: Search query (e.g., "polymer dissolution", "Hansen solubility parameters")
        max_results: Maximum results to return (default: 10, max: 20)
        year_low: Minimum publication year (optional)
        year_high: Maximum publication year (optional)
        filter_relevant: Filter results by domain relevancy (default: True)
        save_to_rag: Download open-access PDFs and ingest into RAG (default: False)
        max_save: Max papers to save when save_to_rag is True (default: 2)
        knowledgebase: RAG knowledgebase name (default: active KB; creates new if needed)

    WHEN TO USE:
    - "Search Google Scholar for polymer dissolution articles"
    - "Find papers on Hansen solubility parameters"
    - "Search and save the 3 best papers on PET recycling to RAG"
    """
    if not query or not query.strip():
        return "Error: search query cannot be empty."
    try:
        from strap.vendor.serpapi_scholar import GoogleScholarClient

        client = GoogleScholarClient()

        results = client.search(
            query=query,
            num_results=min(max_results, 20),
            year_low=year_low,
            year_high=year_high,
            sort_by="date",
        )

        organic_results = results.get("organic_results", [])
        if not organic_results:
            return (
                f"No results found for query: '{query}'\n\n"
                "Try:\n- Using simpler search terms\n- Removing year filters\n- Checking spelling"
            )

        # Parse into dicts
        parsed = [client._parse_article(r) for r in organic_results]

        # Relevancy filter
        parsed, all_scores, rel_summary = _apply_relevancy_filter(
            parsed, filter_relevant, text_field="snippet", title_field="title"
        )

        if not parsed:
            return (
                f"No relevant results for: '{query}'\n\n"
                "All results were filtered as low-relevancy. "
                "Try broader terms or set filter_relevant=False."
            )

        # Format output
        output = [f"# Google Scholar Results: {query}\n"]
        output.append(f"**Found:** {len(parsed)} articles\n")

        if year_low or year_high:
            output.append(f"**Year Range:** {year_low or '...'}-{year_high or '...'}\n")

        if rel_summary:
            output.append(rel_summary + "\n")

        output.append("\n## Articles\n")

        for i, article in enumerate(parsed, 1):
            title = article.get("title", "N/A")
            link = article.get("link", "#")
            output.append(f"\n### {i}. [{title}]({link})")

            badge = _get_relevancy_badge(article)
            if badge:
                output.append(badge)

            authors = article.get("authors", [])
            author_str = ", ".join(authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} total)"
            output.append(f"**Authors:** {author_str or 'Not available'}")

            pub = article.get("publication_info", "")
            if pub:
                output.append(f"**Publication:** {pub}")

            output.append(f"**Year:** {article.get('year', 'N/A')}")
            output.append(f"**Citations:** {article.get('cited_by_count', 0)}")

            pdf_link = article.get("pdf_link")
            if pdf_link:
                output.append(f"**[PDF Available]({pdf_link})**")

            snippet = article.get("snippet", "")
            if snippet:
                output.append(f"*{snippet[:300]}{'...' if len(snippet) > 300 else ''}*")

        output.append("\n\n---")
        output.append(f"**Search Query:** `{query}`")
        output.append(f"**Results Shown:** {len(parsed)}")

        # --- save_to_rag ---
        if save_to_rag:
            # Sort by relevancy score desc, take top max_save with PDFs
            candidates = sorted(
                parsed,
                key=lambda a: getattr(a.get("_relevancy"), "score", 0),
                reverse=True,
            )

            dl_items = []
            for a in candidates:
                pdf_url = a.get("pdf_link")
                if not pdf_url:
                    continue
                year_val = a.get("year", "unknown")
                dl_items.append({
                    "url": pdf_url,
                    "filename": f"{year_val}_{a.get('title', 'paper')[:50]}",
                    "title": a.get("title", "Unknown"),
                })

            output.extend(
                _download_and_ingest(dl_items, max_save, knowledgebase, "scholar")
            )

        return "\n".join(output)

    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return (
                "Google Scholar search requires a SerpAPI key.\n\n"
                "**Setup:**\n1. Get key from: https://serpapi.com/\n"
                "2. Set `SERPAPI_KEY=your-key`\n3. Restart"
            )
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Google Scholar search error: {e}")
        return f"Search failed: {e}\n\nPlease try again or simplify your query."


# ============================================================
# Google Patents Search (SerpAPI)
# ============================================================

@safe_tool_wrapper
def search_google_patents(
    query: str,
    max_results: int = 10,
    after: Optional[str] = None,
    before: Optional[str] = None,
    assignee: Optional[str] = None,
    inventor: Optional[str] = None,
    country: Optional[str] = None,
    filter_relevant: bool = True,
    save_to_rag: bool = False,
    max_save: int = DEFAULT_MAX_SAVE,
    knowledgebase: Optional[str] = None,
) -> str:
    """Search Google Patents for patent documents via SerpAPI.

    Args:
        query: Search query (e.g., "polymer dissolution solvent recovery")
        max_results: Maximum results to return (default: 10, max: 20)
        after: Filter patents filed after date (YYYYMMDD)
        before: Filter patents filed before date (YYYYMMDD)
        assignee: Filter by company/assignee (e.g., "Dow", "BASF")
        inventor: Filter by inventor name
        country: Filter by country code (US, EP, WO, CN, JP, etc.)
        filter_relevant: Filter by domain relevancy (default: True)
        save_to_rag: Download patent PDFs and ingest into RAG (default: False)
        max_save: Max patents to save when save_to_rag is True (default: 2)
        knowledgebase: RAG knowledgebase name (default: active KB; creates new if needed)

    WHEN TO USE:
    - "Search patents for polymer dissolution processes"
    - "Find and save the best 3 patents on solvent-based PET recycling"
    - "What patents does Eastman have on polymer recycling?"
    """
    if not query or not query.strip():
        return "Error: search query cannot be empty."
    try:
        from strap.vendor.serpapi_patents import GooglePatentsClient

        client = GooglePatentsClient()

        results = client.search(
            query=query,
            num_results=min(max_results, 20),
            after=after,
            before=before,
            assignee=assignee,
            inventor=inventor,
            country=country,
        )

        organic_results = results.get("organic_results", [])
        if not organic_results:
            return (
                f"No patents found for query: '{query}'\n\nTry:\n"
                "- Using different search terms\n"
                "- Removing filters (date, assignee, country)"
            )

        parsed = [client._parse_patent(r) for r in organic_results]

        # Relevancy filter
        parsed, all_scores, rel_summary = _apply_relevancy_filter(
            parsed, filter_relevant, text_field="snippet", title_field="title"
        )

        if not parsed:
            return (
                f"No relevant patents for: '{query}'\n\n"
                "All results filtered as low-relevancy. "
                "Try broader terms or set filter_relevant=False."
            )

        # Format output
        output = [f"# Patent Search Results: {query}\n"]
        output.append(f"**Found:** {len(parsed)} patents\n")

        filters = []
        if after:
            filters.append(f"After: {after}")
        if before:
            filters.append(f"Before: {before}")
        if assignee:
            filters.append(f"Assignee: {assignee}")
        if inventor:
            filters.append(f"Inventor: {inventor}")
        if country:
            filters.append(f"Country: {country}")
        if filters:
            output.append(f"**Filters:** {', '.join(filters)}\n")

        if rel_summary:
            output.append(rel_summary + "\n")

        output.append("\n## Patents\n")

        for i, patent in enumerate(parsed, 1):
            patent_id = patent.get("patent_id", "N/A")
            title = patent.get("title", "N/A")
            link = patent.get("link", f"https://patents.google.com/patent/{patent_id}")
            output.append(f"\n### {i}. [{patent_id}: {title}]({link})")

            badge = _get_relevancy_badge(patent)
            if badge:
                output.append(badge)

            output.append(f"**Assignee:** {patent.get('assignee', 'N/A')}")

            inventors_list = patent.get("inventors", [])
            inv_str = ", ".join(inventors_list[:3])
            if len(inventors_list) > 3:
                inv_str += f" et al. ({len(inventors_list)} total)"
            output.append(f"**Inventors:** {inv_str or 'Not available'}")

            filing_date = patent.get("filing_date", "N/A")
            grant_date = patent.get("grant_date", "N/A")
            if filing_date != "N/A":
                output.append(f"**Filed:** {filing_date}")
            if grant_date != "N/A":
                output.append(f"**Granted:** {grant_date}")

            output.append(f"**Country:** {patent.get('country', 'Unknown')}")

            pdf_link = patent.get("pdf_link")
            if pdf_link:
                output.append(f"[PDF Available]({pdf_link})")

            snippet = patent.get("snippet", "")
            if snippet:
                output.append(f"*{snippet[:400]}{'...' if len(snippet) > 400 else ''}*")

        output.append("\n\n---")
        output.append(f"**Search Query:** `{query}`")
        output.append(f"**Results Shown:** {len(parsed)}")

        # --- save_to_rag ---
        if save_to_rag:
            candidates = sorted(
                parsed,
                key=lambda p: getattr(p.get("_relevancy"), "score", 0),
                reverse=True,
            )

            dl_items = []
            for p in candidates:
                pid = p.get("patent_id", "")
                # Prefer USPTO direct PDF; fall back to Google Patents
                num = pid.lstrip("US") if pid.startswith("US") else pid
                pdf_url = (
                    p.get("pdf_link")
                    or f"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{num}"
                )
                dl_items.append({
                    "url": pdf_url,
                    "alt_url": f"https://patents.google.com/patent/{pid}/download",
                    "filename": f"{pid}_{p.get('title', 'patent')[:40]}",
                    "title": p.get("title", pid),
                })

            output.extend(
                _download_and_ingest(dl_items, max_save, knowledgebase, "patent")
            )

        return "\n".join(output)

    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return (
                "Google Patents search requires a SerpAPI key.\n\n"
                "**Setup:**\n1. Get key from: https://serpapi.com/\n"
                "2. Set `SERPAPI_KEY=your-key`\n3. Restart"
            )
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Google Patents search error: {e}")
        return f"Search failed: {e}\n\nPlease try again or simplify your query."


# ============================================================
# Structured Polymer Patent Search (Google Patents / SerpAPI)
# ============================================================

@safe_tool_wrapper
def search_polymer_patents(
    polymer: str,
    solvent: Optional[str] = None,
    process_type: Optional[str] = None,
    max_results: int = 10,
    save_to_rag: bool = False,
    max_save: int = DEFAULT_MAX_SAVE,
    knowledgebase: Optional[str] = None,
) -> str:
    """Search Google Patents for polymer-related patents using structured queries.

    Builds a smarter combined query from polymer name, optional solvent, and
    optional process type — more targeted than a free-text search.

    Args:
        polymer: Polymer name (e.g., "PET", "polystyrene", "HDPE")
        solvent: Solvent name (e.g., "toluene", "DMF") — optional
        process_type: Process keyword (e.g., "dissolution", "recycling", "recovery") — optional
        max_results: Maximum results to return (default: 10, max: 20)
        save_to_rag: Download patent PDFs and ingest into RAG (default: False)
        max_save: Max patents to save when save_to_rag is True (default: 2)
        knowledgebase: RAG knowledgebase name (default: active KB; creates new if needed)

    WHEN TO USE:
    - "Find patents on PET dissolution in DMF"
    - "Search patents for HDPE recycling"
    - "What patents exist for polystyrene solvent recovery?"
    - "Find polymer dissolution patents and save the best 3 to RAG"
    """
    try:
        from strap.vendor.serpapi_patents import GooglePatentsClient

        client = GooglePatentsClient()

        patents = client.search_polymer_patents(
            polymer_name=polymer,
            solvent_name=solvent,
            process_type=process_type,
            max_results=min(max_results, 20),
        )

        if not patents:
            parts = [polymer]
            if solvent:
                parts.append(solvent)
            if process_type:
                parts.append(process_type)
            query_desc = " + ".join(parts)
            return (
                f"No patents found for: {query_desc}\n\n"
                "Try:\n- Using a common name (e.g. 'polystyrene' instead of 'PS')\n"
                "- Removing the solvent or process_type filter\n"
                "- Using search_google_patents for a free-text search"
            )

        # Format output
        parts = [polymer]
        if solvent:
            parts.append(solvent)
        if process_type:
            parts.append(process_type)
        query_desc = " + ".join(parts)

        output = [f"# Polymer Patent Search: {query_desc}\n"]
        output.append(f"**Found:** {len(patents)} patents\n")

        output.append("\n## Patents\n")

        for i, patent in enumerate(patents, 1):
            patent_id = patent.get("patent_id", "N/A")
            title = patent.get("title", "N/A")
            link = patent.get("link", f"https://patents.google.com/patent/{patent_id}")
            output.append(f"\n### {i}. [{patent_id}: {title}]({link})")

            output.append(f"**Assignee:** {patent.get('assignee', 'N/A')}")

            inventors_list = patent.get("inventors", [])
            inv_str = ", ".join(inventors_list[:3])
            if len(inventors_list) > 3:
                inv_str += f" et al. ({len(inventors_list)} total)"
            output.append(f"**Inventors:** {inv_str or 'Not available'}")

            filing_date = patent.get("filing_date", "N/A")
            grant_date = patent.get("grant_date", "N/A")
            if filing_date != "N/A":
                output.append(f"**Filed:** {filing_date}")
            if grant_date != "N/A":
                output.append(f"**Granted:** {grant_date}")

            output.append(f"**Country:** {patent.get('country', 'Unknown')}")

            pdf_link = patent.get("pdf_link")
            if pdf_link:
                output.append(f"[PDF Available]({pdf_link})")

            snippet = patent.get("snippet", "")
            if snippet:
                output.append(f"*{snippet[:400]}{'...' if len(snippet) > 400 else ''}*")

        output.append("\n\n---")
        output.append(f"**Polymer:** {polymer}")
        if solvent:
            output.append(f"**Solvent:** {solvent}")
        if process_type:
            output.append(f"**Process:** {process_type}")
        output.append(f"**Results Shown:** {len(patents)}")

        # --- save_to_rag ---
        if save_to_rag:
            dl_items = []
            for p in patents:
                pid = p.get("patent_id", "")
                num = pid.lstrip("US") if pid.startswith("US") else pid
                pdf_url = (
                    p.get("pdf_link")
                    or f"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{num}"
                )
                dl_items.append({
                    "url": pdf_url,
                    "alt_url": f"https://patents.google.com/patent/{pid}/download",
                    "filename": f"{pid}_{p.get('title', 'patent')[:40]}",
                    "title": p.get("title", pid),
                })

            output.extend(
                _download_and_ingest(dl_items, max_save, knowledgebase, "patent")
            )

        return "\n".join(output)

    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return (
                "Polymer patent search requires a SerpAPI key.\n\n"
                "**Setup:**\n1. Get key from: https://serpapi.com/\n"
                "2. Set `SERPAPI_KEY=your-key`\n3. Restart"
            )
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Polymer patent search error: {e}")
        return f"Search failed: {e}\n\nPlease try again or simplify your query."


# ============================================================
# PatentsView Search (US USPTO granted patents)
# ============================================================

@safe_tool_wrapper
def search_patentsview(
    query: str,
    max_results: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    assignee: Optional[str] = None,
    filter_relevant: bool = True,
    save_to_rag: bool = False,
    max_save: int = DEFAULT_MAX_SAVE,
    knowledgebase: Optional[str] = None,
) -> str:
    """Search PatentsView for US granted patents (USPTO) with full abstracts.

    This searches only US granted patents from the USPTO database. For
    international or application-stage patents, use search_google_patents instead.

    Args:
        query: Search terms (e.g., "polymer dissolution solvent recovery")
        max_results: Maximum results to return (default: 10)
        date_from: Start date filter (YYYY-MM-DD, e.g. "2020-01-01")
        date_to: End date filter (YYYY-MM-DD, e.g. "2024-12-31")
        assignee: Filter by assignee/company name (substring match)
        filter_relevant: Filter by domain relevancy (default: True)
        save_to_rag: Download patent PDFs and ingest into RAG (default: False)
        max_save: Max patents to save when save_to_rag is True (default: 2)
        knowledgebase: RAG knowledgebase name (default: active KB; creates new if needed)

    WHEN TO USE:
    - "Search US patents for polymer dissolution"
    - "Find and save the top 3 USPTO patents on solvent-based recycling"
    - "What US patents does Eastman have on polymer recycling?"
    """
    if not query or not query.strip():
        return "Error: search query cannot be empty."
    try:
        from strap.vendor.patentsview_client import PatentsViewClient

        client = PatentsViewClient()

        patents = client.search(
            query_text=query,
            max_results=max_results,
            date_from=date_from,
            date_to=date_to,
            assignee=assignee,
        )

        if not patents:
            return (
                f"No US patents found for query: '{query}'\n\n"
                "Try:\n- Using different search terms\n"
                "- Removing filters\n"
                "- Note: PatentsView covers only US granted patents."
            )

        # Relevancy filter
        patents, all_scores, rel_summary = _apply_relevancy_filter(
            patents, filter_relevant, text_field="abstract", title_field="title"
        )

        if not patents:
            return (
                f"No relevant US patents for: '{query}'\n\n"
                "All results filtered as low-relevancy. "
                "Try broader terms or set filter_relevant=False."
            )

        # Format output
        output = [f"# PatentsView Results (US Granted Patents): {query}\n"]
        output.append(f"**Found:** {len(patents)} US patents\n")

        filters = []
        if date_from:
            filters.append(f"From: {date_from}")
        if date_to:
            filters.append(f"To: {date_to}")
        if assignee:
            filters.append(f"Assignee: {assignee}")
        if filters:
            output.append(f"**Filters:** {', '.join(filters)}\n")

        if rel_summary:
            output.append(rel_summary + "\n")

        output.append("\n## Patents\n")

        for i, patent in enumerate(patents, 1):
            patent_id = patent.get("patent_id", "N/A")
            title = patent.get("title", "N/A")
            output.append(f"\n### {i}. {patent_id}: {title}")

            badge = _get_relevancy_badge(patent)
            if badge:
                output.append(badge)

            assignee_val = patent.get("assignee", "N/A")
            ac = patent.get("assignee_country", "")
            if ac and ac != "N/A":
                output.append(f"**Assignee:** {assignee_val} ({ac})")
            else:
                output.append(f"**Assignee:** {assignee_val}")

            inventors = patent.get("inventors", [])
            inv_str = ", ".join(inventors[:5])
            if len(inventors) > 5:
                inv_str += f" et al. ({len(inventors)} total)"
            output.append(f"**Inventors:** {inv_str or 'Not available'}")

            output.append(f"**Grant Date:** {patent.get('date', 'N/A')}")
            ptype = patent.get("type", "N/A")
            if ptype != "N/A":
                output.append(f"**Type:** {ptype}")

            abstract = patent.get("abstract", "")
            if abstract:
                output.append(f"\n*{abstract[:500]}{'...' if len(abstract) > 500 else ''}*")

        output.append("\n\n---")
        output.append(f"**Search Query:** `{query}`")
        output.append(f"**Results Shown:** {len(patents)}")
        output.append("**Source:** PatentsView (USPTO) — US granted patents only")

        # --- save_to_rag (construct USPTO PDF URLs from patent IDs) ---
        if save_to_rag:
            candidates = sorted(
                patents,
                key=lambda p: getattr(p.get("_relevancy"), "score", 0),
                reverse=True,
            )

            dl_items = []
            for p in candidates:
                pid = p.get("patent_id", "")
                # Strip "US" prefix for USPTO URL
                num = pid.lstrip("US") if pid.startswith("US") else pid
                dl_items.append({
                    "url": f"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{num}",
                    "alt_url": f"https://patents.google.com/patent/{pid}/download",
                    "filename": f"{pid}_{p.get('title', 'patent')[:40]}",
                    "title": p.get("title", pid),
                })

            output.extend(
                _download_and_ingest(dl_items, max_save, knowledgebase, "patent")
            )

        return "\n".join(output)

    except ValueError as e:
        if "PATENTSVIEW_API_KEY" in str(e):
            return (
                "PatentsView search requires an API key.\n\n"
                "**Setup:**\n1. Get key from: https://patentsview.org/apis\n"
                "2. Set `PATENTSVIEW_API_KEY=your-key`\n3. Restart"
            )
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"PatentsView search error: {e}")
        return f"Search failed: {e}\n\nPlease try again or simplify your query."


# ============================================================
# Patent Lookup (single patent by number)
# ============================================================

@safe_tool_wrapper
def lookup_patent(
    patent_number: str,
    save_to_rag: bool = False,
    knowledgebase: Optional[str] = None,
) -> str:
    """Look up a specific patent by its number and retrieve full details.

    Args:
        patent_number: Patent number (e.g., US10123456, EP1234567, WO2020123456)
        save_to_rag: Download patent PDF and ingest into RAG (default: False)
        knowledgebase: RAG knowledgebase name (default: active KB)

    WHEN TO USE:
    - "Look up patent US10123456"
    - "Get details for EP1234567"
    - "Save patent US10457803 to RAG"
    """
    try:
        from strap.vendor.serpapi_patents import GooglePatentsClient

        client = GooglePatentsClient()
        patent = client.get_patent(patent_number)

        if patent.get("error"):
            return f"{patent['error']}\n\nTry checking the patent number format or searching by keywords instead."

        # Format output
        output = [f"# Patent Details: {patent.get('patent_id', patent_number)}\n"]

        title = patent.get("title", "N/A")
        output.append(f"## {title}\n")

        link = patent.get("link", "")
        if link:
            output.append(f"**Full Patent:** [{link}]({link})\n")

        output.append("### Key Information\n")
        output.append(f"**Assignee/Owner:** {patent.get('assignee', 'N/A')}")

        inventors = patent.get("inventors", [])
        output.append(f"**Inventors:** {', '.join(inventors) if inventors else 'Not available'}")

        output.append("\n### Dates\n")
        for label, key in [("Filing Date", "filing_date"), ("Grant Date", "grant_date"),
                           ("Publication Date", "publication_date"), ("Priority Date", "priority_date")]:
            val = patent.get(key, "N/A")
            if val != "N/A":
                output.append(f"**{label}:** {val}")

        output.append(f"**Country/Office:** {patent.get('country', 'Unknown')}")

        claims = patent.get("claims_count")
        if claims:
            output.append(f"**Number of Claims:** {claims}")

        cited_by = patent.get("cited_by_count", 0)
        if cited_by:
            output.append(f"**Cited By:** {cited_by} patents")

        snippet = patent.get("snippet", "")
        if snippet:
            output.append("\n### Abstract\n")
            output.append(f"*{snippet}*")

        pdf_link = patent.get("pdf_link")
        if pdf_link:
            output.append(f"\n**[Download PDF]({pdf_link})**")

        # --- save_to_rag ---
        if save_to_rag:
            pid = patent.get("patent_id", patent_number)
            dl_items = [{
                "url": pdf_link or f"https://patents.google.com/patent/{pid}/download",
                "alt_url": f"https://patentimages.storage.googleapis.com/pdfs/{pid}.pdf",
                "filename": f"{pid}",
                "title": title,
            }]
            output.extend(
                _download_and_ingest(dl_items, 1, knowledgebase, "patent")
            )

        return "\n".join(output)

    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return (
                "Patent lookup requires a SerpAPI key.\n\n"
                "**Setup:**\n1. Get key from: https://serpapi.com/\n"
                "2. Set `SERPAPI_KEY=your-key`\n3. Restart"
            )
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Patent lookup error: {e}")
        return f"Lookup failed: {e}\n\nPlease check the patent number format."


# ============================================================
# Web of Science Literature Search (Starter API)
# ============================================================

@safe_tool_wrapper
def search_web_of_science(
    query: str,
    polymer_name: Optional[str] = None,
    solvent_name: Optional[str] = None,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None,
    max_results: int = 10,
    filter_relevant: bool = True,
) -> str:
    """Search Web of Science for peer-reviewed research articles via Clarivate API.

    Args:
        query: Search query, natural language or WoS syntax (e.g., TS=, TI=, AU=)
        polymer_name: Specific polymer name (e.g., "polyethylene", "PET")
        solvent_name: Specific solvent name (e.g., "toluene", "NMP")
        year_low: Minimum publication year (e.g., 2020)
        year_high: Maximum publication year (e.g., 2026)
        max_results: Number of results (default: 10, max: 50)
        filter_relevant: Filter by domain relevancy (default: True)

    WHEN TO USE:
    - "Search Web of Science for PET dissolution articles"
    - "Find peer-reviewed papers on polymer recycling"
    - "What journal articles exist on Hansen solubility parameters?"
    """
    if not query or not query.strip():
        return "Error: search query cannot be empty."
    try:
        from strap.vendor.wos_client import WebOfScienceClient

        client = WebOfScienceClient()

        if polymer_name or solvent_name:
            year_range = None
            if year_low or year_high:
                year_range = f"{year_low or 1900}-{year_high or 2030}"
            articles = client.search_polymer_solubility_articles(
                polymer_name=polymer_name,
                solvent_name=solvent_name,
                year_range=year_range,
                max_results=max_results,
            )
        else:
            wos_query = query.strip()
            has_field_tag = any(tag in wos_query.upper() for tag in ["TS=", "TI=", "AU=", "SO=", "PY="])

            if not has_field_tag:
                has_boolean = any(op in wos_query.upper() for op in [" AND ", " OR ", " NOT "])
                if has_boolean:
                    wos_query = f"TS=({wos_query})"
                else:
                    words = wos_query.split()
                    if len(words) > 1 and '"' not in wos_query:
                        wos_query = f'TS=({" AND ".join(words)})'
                    else:
                        wos_query = f"TS=({wos_query})"

            if (year_low or year_high) and "PY=" not in wos_query.upper():
                wos_query += f" AND PY=({year_low or 1900}-{year_high or 2030})"

            logger.info(f"WoS query: {wos_query}")

            results = client.search_articles(query=wos_query, count=max_results, sort_field="PY+D")
            articles = [client._parse_hit(h) for h in results.get("hits", [])]

        if not articles:
            return (
                f"No Web of Science articles found for: '{query}'\n\n"
                "**Suggestions:**\n- Try broader search terms\n"
                "- Remove year restrictions\n- Check spelling"
            )

        # Relevancy filter (title + keywords only — WoS Starter has no abstracts)
        articles, all_scores, rel_summary = _apply_relevancy_filter(
            articles, filter_relevant, text_field="title", title_field="title",
            keywords_field="keywords",
        )

        if not articles:
            return (
                f"No relevant WoS articles for: '{query}'\n\n"
                "All results filtered as low-relevancy. "
                "Try broader terms or set filter_relevant=False."
            )

        # Format output
        output = [f"# Web of Science Results: {query}\n"]
        output.append(f"**Found:** {len(articles)} peer-reviewed articles\n")

        if year_low or year_high:
            output.append(f"**Year Range:** {year_low or '...'}-{year_high or '...'}\n")

        if rel_summary:
            output.append(rel_summary + "\n")

        output.append("\n## Articles\n")

        for i, article in enumerate(articles, 1):
            title = article.get("title", "N/A")
            uid = article.get("uid", "")
            link = f"https://www.webofscience.com/wos/woscc/full-record/{uid}" if uid else "#"
            output.append(f"\n### {i}. [{title}]({link})")

            badge = _get_relevancy_badge(article)
            if badge:
                output.append(badge)

            authors = article.get("authors", [])
            author_str = ", ".join(authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} total)"
            output.append(f"**Authors:** {author_str or 'Not available'}")

            output.append(f"**Journal:** {article.get('source', 'N/A')}")
            output.append(f"**Year:** {article.get('year', 'N/A')}")

            doi = article.get("doi", "N/A")
            if doi and doi != "N/A":
                output.append(f"**DOI:** [{doi}](https://doi.org/{doi})")
            else:
                output.append("**DOI:** Not available")

            output.append(f"**Times Cited:** {article.get('citations', 0)}")

            keywords = article.get("keywords", [])
            if keywords:
                output.append(f"**Keywords:** {', '.join(keywords[:8])}")

        output.append("\n\n---")
        output.append(f"**Search Query:** `{query}`")
        output.append(f"**Results:** {len(articles)} peer-reviewed articles from Web of Science")

        return "\n".join(output)

    except ValueError as e:
        if "WOS_STARTER_API_KEY" in str(e):
            return (
                "Web of Science search requires an API key.\n\n"
                "**Setup:**\n1. Get key from: https://developer.clarivate.com/\n"
                "2. Set `WOS_STARTER_API_KEY=your-key`\n3. Restart"
            )
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Web of Science search error: {e}")
        return f"Search failed: {e}\n\nPlease try again or simplify your query."


# ============================================================
# arXiv Search (open-access papers, free PDFs)
# ============================================================

@safe_tool_wrapper
def search_arxiv(
    query: str,
    max_results: int = 10,
    categories: Optional[str] = None,
    sort_by: str = "submitted",
    filter_relevant: bool = True,
    save_to_rag: bool = False,
    max_save: int = DEFAULT_MAX_SAVE,
    knowledgebase: Optional[str] = None,
) -> str:
    """Search arXiv for open-access preprints and papers with free PDF downloads.

    All arXiv papers are open-access — PDFs can always be downloaded. Use this
    when you need freely downloadable papers for RAG ingestion.

    Args:
        query: Search query (e.g., "polymer dissolution solvent")
        max_results: Maximum results to return (default: 10)
        categories: Comma-separated arXiv categories (e.g., "cond-mat,physics.chem-ph")
        sort_by: Sort order: "submitted", "updated", or "relevance" (default: "submitted")
        filter_relevant: Filter by domain relevancy (default: True)
        save_to_rag: Download PDFs and ingest into RAG (default: False)
        max_save: Max papers to save when save_to_rag is True (default: 2)
        knowledgebase: RAG knowledgebase name (default: active KB; creates new if needed)

    WHEN TO USE:
    - "Search arXiv for polymer dissolution papers"
    - "Find and save open-access papers on Hansen solubility parameters"
    - "Search arXiv and ingest the best 3 papers on PET recycling to RAG"
    """
    if not query or not query.strip():
        return "Error: search query cannot be empty."
    try:
        from strap.vendor.arxiv_client import ArxivClient

        client = ArxivClient()

        cat_list = None
        if categories:
            cat_list = [c.strip() for c in categories.split(",") if c.strip()]

        papers = client.search(
            query=query,
            categories=cat_list,
            max_results=max_results,
            sort_by=sort_by,
        )

        if not papers:
            return (
                f"No arXiv results for query: '{query}'\n\n"
                "Try:\n- Using different search terms\n"
                "- Removing category filters\n"
                "- Using sort_by='relevance'"
            )

        # Relevancy filter (arXiv has full abstracts — excellent for scoring)
        papers, all_scores, rel_summary = _apply_relevancy_filter(
            papers, filter_relevant, text_field="abstract", title_field="title"
        )

        if not papers:
            return (
                f"No relevant arXiv results for: '{query}'\n\n"
                "All results filtered as low-relevancy. "
                "Try broader terms or set filter_relevant=False."
            )

        # Format output
        output = [f"# arXiv Results: {query}\n"]
        output.append(f"**Found:** {len(papers)} papers (open-access)\n")

        if categories:
            output.append(f"**Categories:** {categories}\n")

        if rel_summary:
            output.append(rel_summary + "\n")

        output.append("\n## Papers\n")

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "N/A")
            url = paper.get("url", "#")
            output.append(f"\n### {i}. [{title}]({url})")

            badge = _get_relevancy_badge(paper)
            if badge:
                output.append(badge)

            authors = paper.get("authors", [])
            author_str = ", ".join(authors[:5])
            if len(authors) > 5:
                author_str += f" et al. ({len(authors)} total)"
            output.append(f"**Authors:** {author_str or 'Not available'}")

            output.append(f"**arXiv ID:** {paper.get('arxiv_id', 'N/A')}")
            output.append(f"**Date:** {paper.get('date', 'N/A')}")

            cats = paper.get("categories", [])
            if cats:
                output.append(f"**Categories:** {', '.join(cats[:5])}")

            doi = paper.get("doi")
            if doi:
                output.append(f"**DOI:** [{doi}](https://doi.org/{doi})")

            journal = paper.get("journal_ref")
            if journal:
                output.append(f"**Journal:** {journal}")

            pdf_url = paper.get("pdf_url", "")
            if pdf_url:
                output.append(f"**[PDF (open-access)]({pdf_url})**")

            abstract = paper.get("abstract", "")
            if abstract:
                output.append(f"\n*{abstract[:400]}{'...' if len(abstract) > 400 else ''}*")

        output.append("\n\n---")
        output.append(f"**Search Query:** `{query}`")
        output.append(f"**Results Shown:** {len(papers)}")
        output.append("**Source:** arXiv (open-access, free PDF downloads)")

        # --- save_to_rag (arXiv PDFs are always available) ---
        if save_to_rag:
            candidates = sorted(
                papers,
                key=lambda p: getattr(p.get("_relevancy"), "score", 0),
                reverse=True,
            )

            dl_items = []
            for p in candidates:
                arxiv_id = p.get("arxiv_id", "")
                dl_items.append({
                    "url": p.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}.pdf"),
                    "filename": f"{arxiv_id}_{p.get('title', 'paper')[:40]}",
                    "title": p.get("title", arxiv_id),
                })

            output.extend(
                _download_and_ingest(dl_items, max_save, knowledgebase, "arxiv")
            )

        return "\n".join(output)

    except ImportError:
        return (
            "arXiv search requires the `arxiv` package.\n\n"
            "Install with: `pip install arxiv`"
        )
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return f"Search failed: {e}\n\nPlease try again or simplify your query."
