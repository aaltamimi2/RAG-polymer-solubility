"""Literature search tools: Google Scholar, Google Patents, Web of Science."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Any

from strap.tools._helpers import safe_tool_wrapper, truncate_output

try:
    from strap.vendor.serpapi_scholar import SerpAPIScholarClient
except Exception:
    SerpAPIScholarClient = None

try:
    from strap.vendor.serpapi_patents import SerpAPIPatentsClient
except Exception:
    SerpAPIPatentsClient = None

try:
    from strap.vendor.wos_client import WoSAPIClient
except Exception:
    WoSAPIClient = None

logger = logging.getLogger(__name__)


# ============================================================
# Google Scholar Search (SerpAPI)
# ============================================================

@safe_tool_wrapper
def search_google_scholar(
    query: str,
    max_results: int = 10,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None,
    save_to_rag: bool = False,
    max_downloads: int = 5
) -> str:
    """Search Google Scholar for academic research articles via SerpAPI.

    Args:
        query: Search query (e.g., "polymer dissolution", "Hansen solubility parameters")
        max_results: Maximum results to return (default: 10, max: 20)
        year_low: Minimum publication year (optional)
        year_high: Maximum publication year (optional)
        save_to_rag: Download open-access PDFs and ingest into RAG (default: False)
        max_downloads: Max PDFs to download when save_to_rag is True (default: 5)

    WHEN TO USE:
    - "Search Google Scholar for polymer dissolution articles"
    - "Find papers on Hansen solubility parameters"
    - "Download open-access papers on PET recycling to RAG"
    """
    try:
        from serpapi_scholar_client import GoogleScholarClient

        # Initialize client (uses SERPAPI_KEY from environment)
        client = GoogleScholarClient()

        # Perform search
        results = client.search(
            query=query,
            num_results=min(max_results, 20),  # Cap at 20
            year_low=year_low,
            year_high=year_high,
            sort_by='date'  # Get most recent articles
        )

        # Parse results
        organic_results = results.get('organic_results', [])

        if not organic_results:
            return f"No results found for query: '{query}'\n\nTry:\n- Using simpler search terms\n- Removing year filters\n- Checking spelling"

        # Format output
        output = [f"# Google Scholar Results: {query}\n"]
        output.append(f"**Found:** {len(organic_results)} articles\n")

        if year_low or year_high:
            year_range = f"{year_low or '...'}-{year_high or '...'}"
            output.append(f"**Year Range:** {year_range}\n")

        output.append("\n## Articles\n")

        for i, result in enumerate(organic_results, 1):
            article = client._parse_article(result)

            # Title with link
            title = article.get('title', 'N/A')
            link = article.get('link', '#')
            output.append(f"\n### {i}. [{title}]({link})")

            # Authors - always show
            authors = article.get('authors', [])
            if authors:
                author_str = ', '.join(authors[:5])
                if len(authors) > 5:
                    author_str += f" et al. ({len(authors)} total)"
            else:
                author_str = "Not available"
            output.append(f"**Authors:** {author_str}")

            # Publication info
            publication = article.get('publication_info', '')
            if publication:
                output.append(f"**Publication:** {publication}")

            # Year - always show
            year = article.get('year', 'N/A')
            output.append(f"**Year:** {year}")

            # Citations - always show
            citations = article.get('cited_by_count', 0)
            output.append(f"**Citations:** {citations}")

            # PDF link if available
            pdf_link = article.get('pdf_link')
            if pdf_link:
                output.append(f"**[PDF Available]({pdf_link})**")

            # Snippet
            snippet = article.get('snippet', '')
            if snippet:
                # Truncate long snippets
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                output.append(f"*{snippet}*")

        # Footer
        output.append(f"\n\n---")
        output.append(f"**Search Query:** `{query}`")
        output.append(f"**Results Shown:** {len(organic_results)} of {results.get('search_metadata', {}).get('total_results', 'many')}")
        output.append(f"\n**Beta Feature:** This uses SerpAPI with limited monthly searches. Use wisely!")

        # --- save_to_rag: download open-access PDFs and ingest ---
        if save_to_rag:
            import re
            import requests as _requests
            from pathlib import Path

            pdf_dir = Path("rag_pdfs")
            pdf_dir.mkdir(exist_ok=True)

            dl_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            downloaded = []
            skipped = []
            failed = []

            for result in organic_results:
                if len(downloaded) >= max_downloads:
                    break

                article = client._parse_article(result)
                art_title = article.get('title', 'Unknown')
                pdf_url = article.get('pdf_link')

                if not pdf_url:
                    skipped.append(f"{art_title[:50]}... (no PDF)")
                    continue

                try:
                    safe_title = re.sub(r'[^\w\-]', '_', art_title[:50])
                    year_val = article.get('year', 'unknown')
                    fname = f"scholar_{year_val}_{safe_title}.pdf"
                    fpath = pdf_dir / fname

                    if fpath.exists():
                        skipped.append(f"{art_title[:50]}... (exists)")
                        continue

                    resp = _requests.get(pdf_url, headers=dl_headers, timeout=60, stream=True)
                    resp.raise_for_status()

                    ctype = resp.headers.get('Content-Type', '')
                    if 'pdf' not in ctype.lower() and not pdf_url.endswith('.pdf'):
                        skipped.append(f"{art_title[:50]}... (not PDF)")
                        continue

                    with open(fpath, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    downloaded.append(str(fpath))
                except Exception as dl_e:
                    failed.append(f"{art_title[:40]}... ({str(dl_e)[:30]})")

            # Ingest downloaded PDFs
            if downloaded:
                from strap.tools.rag_core import ingest_pdf_to_rag
                ingest_result = ingest_pdf_to_rag(pdf_paths=",".join(downloaded))
                output.append(f"\n**RAG:** Downloaded {len(downloaded)} PDFs.")
                output.append(f"**Ingest:** {ingest_result[:200]}")
            else:
                output.append(f"\n**RAG:** No open-access PDFs could be downloaded.")

            if skipped:
                output.append(f"**Skipped:** {len(skipped)} papers")
            if failed:
                output.append(f"**Failed:** {len(failed)} downloads")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("Google Scholar search is not available. The `serpapi_scholar_client` module is not installed.\n\n"
                "This is a BETA feature that requires SerpAPI integration.")
    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return ("Google Scholar search requires a SerpAPI key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://serpapi.com/\n"
                    "2. Set environment variable: `SERPAPI_KEY=your-key`\n"
                    "3. Restart the application")
        else:
            return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Google Scholar search error: {e}")
        return f"Search failed: {str(e)}\n\nPlease try again or simplify your query."


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
    country: Optional[str] = None
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

    WHEN TO USE:
    - "Search patents for polymer dissolution processes"
    - "Find patents on solvent-based PET recycling"
    - "What patents does Eastman have on polymer recycling?"
    """
    try:
        from serpapi_patents_client import GooglePatentsClient

        # Initialize client (uses SERPAPI_KEY from environment)
        client = GooglePatentsClient()

        # Perform search
        results = client.search(
            query=query,
            num_results=min(max_results, 20),
            after=after,
            before=before,
            assignee=assignee,
            inventor=inventor,
            country=country
        )

        # Parse results
        organic_results = results.get('organic_results', [])

        if not organic_results:
            return f"No patents found for query: '{query}'\n\nTry:\n- Using different search terms\n- Removing filters (date, assignee, country)\n- Using more general terminology"

        # Format output
        output = [f"# Patent Search Results: {query}\n"]
        output.append(f"**Found:** {len(organic_results)} patents\n")

        # Show active filters
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

        output.append("\n## Patents\n")

        for i, result in enumerate(organic_results, 1):
            patent = client._parse_patent(result)

            # Patent ID with link
            patent_id = patent.get('patent_id', 'N/A')
            title = patent.get('title', 'N/A')
            link = patent.get('link', f'https://patents.google.com/patent/{patent_id}')
            output.append(f"\n### {i}. [{patent_id}: {title}]({link})")

            # Assignee - always show
            assignee_val = patent.get('assignee', 'N/A')
            output.append(f"**Assignee:** {assignee_val}")

            # Inventors - always show
            inventors = patent.get('inventors', [])
            if inventors:
                inventor_str = ', '.join(inventors[:3])
                if len(inventors) > 3:
                    inventor_str += f" et al. ({len(inventors)} total)"
            else:
                inventor_str = "Not available"
            output.append(f"**Inventors:** {inventor_str}")

            # Dates
            filing_date = patent.get('filing_date', 'N/A')
            grant_date = patent.get('grant_date', 'N/A')
            if filing_date != 'N/A':
                output.append(f"**Filed:** {filing_date}")
            if grant_date != 'N/A':
                output.append(f"**Granted:** {grant_date}")

            # Country
            country_code = patent.get('country', 'Unknown')
            output.append(f"**Country:** {country_code}")

            # PDF link if available
            pdf_link = patent.get('pdf_link')
            if pdf_link:
                output.append(f"[PDF Available]({pdf_link})")

            # Snippet/Abstract
            snippet = patent.get('snippet', '')
            if snippet:
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                output.append(f"*{snippet}*")

        # Footer
        output.append(f"\n\n---")
        output.append(f"** Search Query:** `{query}`")
        output.append(f"** Results Shown:** {len(organic_results)}")
        output.append(f"\n **Beta Feature:** This uses SerpAPI with limited monthly searches. Use wisely!")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("Google Patents search is not available. The `serpapi_patents_client` module is not installed.\n\n"
                "This is a BETA feature that requires SerpAPI integration.")
    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return ("Google Patents search requires a SerpAPI key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://serpapi.com/\n"
                    "2. Set environment variable: `SERPAPI_KEY=your-key`\n"
                    "3. Restart the application")
        else:
            return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Google Patents search error: {e}")
        return f"Search failed: {str(e)}\n\nPlease try again or simplify your query."


@safe_tool_wrapper
def lookup_patent(
    patent_number: str,
    save_to_rag: bool = False
) -> str:
    """Look up a specific patent by its number and retrieve full details.

    Args:
        patent_number: Patent number (e.g., US10123456, EP1234567, WO2020123456)
        save_to_rag: Download patent PDF and ingest into RAG (default: False)

    WHEN TO USE:
    - "Look up patent US10123456"
    - "Get details for EP1234567"
    - "Save patent US10457803 to RAG"
    """
    try:
        from serpapi_patents_client import GooglePatentsClient

        # Initialize client
        client = GooglePatentsClient()

        # Look up specific patent
        patent = client.get_patent(patent_number)

        if patent.get('error'):
            return f"{patent['error']}\n\nTry checking the patent number format or searching by keywords instead."

        # Format output
        output = [f"# Patent Details: {patent.get('patent_id', patent_number)}\n"]

        # Title
        title = patent.get('title', 'N/A')
        output.append(f"## {title}\n")

        # Link
        link = patent.get('link', '')
        if link:
            output.append(f"**Full Patent:** [{link}]({link})\n")

        # Key Information
        output.append("### Key Information\n")

        # Assignee
        assignee = patent.get('assignee', 'N/A')
        output.append(f"**Assignee/Owner:** {assignee}")

        # Inventors
        inventors = patent.get('inventors', [])
        if inventors:
            output.append(f"**Inventors:** {', '.join(inventors)}")
        else:
            output.append("**Inventors:** Not available")

        # Dates
        output.append("\n### Dates\n")
        filing_date = patent.get('filing_date', 'N/A')
        grant_date = patent.get('grant_date', 'N/A')
        publication_date = patent.get('publication_date', 'N/A')
        priority_date = patent.get('priority_date', 'N/A')

        output.append(f"**Filing Date:** {filing_date}")
        if grant_date != 'N/A':
            output.append(f"**Grant Date:** {grant_date}")
        if publication_date != 'N/A':
            output.append(f"**Publication Date:** {publication_date}")
        if priority_date != 'N/A' and priority_date != filing_date:
            output.append(f"**Priority Date:** {priority_date}")

        # Country
        country = patent.get('country', 'Unknown')
        output.append(f"**Country/Office:** {country}")

        # Claims count if available
        claims_count = patent.get('claims_count')
        if claims_count:
            output.append(f"**Number of Claims:** {claims_count}")

        # Citations
        cited_by = patent.get('cited_by_count', 0)
        if cited_by:
            output.append(f"**Cited By:** {cited_by} patents")

        # Abstract/Snippet
        snippet = patent.get('snippet', '')
        if snippet:
            output.append("\n### Abstract\n")
            output.append(f"*{snippet}*")

        # PDF link
        pdf_link = patent.get('pdf_link')
        if pdf_link:
            output.append(f"\n**[Download PDF]({pdf_link})**")

        # --- save_to_rag: download patent PDF and ingest into RAG ---
        if save_to_rag:
            import re
            import requests
            from pathlib import Path

            patent_id = patent.get('patent_id', patent_number)

            pdf_dir = Path("rag_pdfs")
            pdf_dir.mkdir(exist_ok=True)

            safe_patent_id = re.sub(r'[^\w\-]', '_', patent_id)
            filepath = pdf_dir / f"patent_{safe_patent_id}.pdf"

            if filepath.exists():
                output.append(f"\n**RAG:** Patent PDF already exists at `{filepath}`.")
            else:
                # Determine PDF URL
                dl_url = pdf_link or f"https://patents.google.com/patent/{patent_id}/download"

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                try:
                    resp = requests.get(dl_url, headers=headers, timeout=120,
                                        stream=True, allow_redirects=True)
                    if resp.status_code != 200 or 'html' in resp.headers.get('Content-Type', '').lower():
                        alt_url = f"https://patentimages.storage.googleapis.com/pdfs/{patent_id}.pdf"
                        resp = requests.get(alt_url, headers=headers, timeout=120, stream=True)
                    resp.raise_for_status()

                    with open(filepath, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    output.append(f"\n**RAG:** Downloaded patent PDF ({size_mb:.2f} MB)")

                    # Ingest into RAG (local import to avoid circular dependency)
                    from strap.tools.rag_core import ingest_pdf_to_rag
                    ingest_result = ingest_pdf_to_rag(pdf_paths=str(filepath))
                    output.append(f"**RAG Ingest:** {ingest_result[:200]}")

                except requests.exceptions.RequestException as dl_err:
                    output.append(f"\n**RAG:** PDF download failed: {dl_err}")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("Patent lookup is not available. The `serpapi_patents_client` module is not installed.\n\n"
                "This is a BETA feature that requires SerpAPI integration.")
    except ValueError as e:
        if "SERPAPI_KEY" in str(e):
            return ("Patent lookup requires a SerpAPI key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://serpapi.com/\n"
                    "2. Set environment variable: `SERPAPI_KEY=your-key`\n"
                    "3. Restart the application")
        else:
            return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Patent lookup error: {e}")
        return f"Lookup failed: {str(e)}\n\nPlease check the patent number format."


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
    max_results: int = 10
) -> str:
    """Search Web of Science for peer-reviewed research articles via Clarivate API.

    Args:
        query: Search query, natural language or WoS syntax (e.g., TS=, TI=, AU=)
        polymer_name: Specific polymer name (e.g., "polyethylene", "PET")
        solvent_name: Specific solvent name (e.g., "toluene", "NMP")
        year_low: Minimum publication year (e.g., 2020)
        year_high: Maximum publication year (e.g., 2026)
        max_results: Number of results (default: 10, max: 50)

    WHEN TO USE:
    - "Search Web of Science for PET dissolution articles"
    - "Find peer-reviewed papers on polymer recycling"
    - "What journal articles exist on Hansen solubility parameters?"
    """
    try:
        from wos_starter_client import WebOfScienceStarterClient

        # Initialize client (uses WOS_STARTER_API_KEY from environment)
        client = WebOfScienceStarterClient()

        # Determine search strategy
        if polymer_name or solvent_name:
            # Use specialized polymer search
            articles = client.search_polymer_articles(
                polymer_name=polymer_name,
                solvent_name=solvent_name,
                year_low=year_low,
                year_high=year_high,
                max_results=max_results
            )
        elif "hansen" in query.lower() and "TS=" not in query.upper():
            # Hansen-specific search (only if not already using WoS syntax)
            articles = client.search_hansen_parameters(
                year_low=year_low,
                year_high=year_high,
                max_results=max_results
            )
        else:
            # Build WoS query - check if user provided WoS syntax
            wos_query = query.strip()

            # Check if query already has WoS field tags
            has_field_tag = any(tag in wos_query.upper() for tag in ['TS=', 'TI=', 'AU=', 'SO=', 'PY='])

            if not has_field_tag:
                # Convert natural language to WoS syntax
                # Check for boolean operators already in query
                has_boolean = any(op in wos_query.upper() for op in [' AND ', ' OR ', ' NOT '])

                if has_boolean:
                    # Query has boolean operators - wrap in TS=()
                    # Handle quoted phrases properly
                    wos_query = f'TS=({wos_query})'
                else:
                    # Simple query - wrap as topic search
                    # If multiple words without quotes, treat as AND search
                    words = wos_query.split()
                    if len(words) > 1 and '"' not in wos_query:
                        # Multi-word query without quotes - join with AND
                        wos_query = f'TS=({" AND ".join(words)})'
                    else:
                        wos_query = f'TS=({wos_query})'

            # Add year range if specified and not already in query
            if (year_low or year_high) and 'PY=' not in wos_query.upper():
                year_start = year_low or 1900
                year_end = year_high or 2030
                wos_query += f' AND PY=({year_start}-{year_end})'

            logger.info(f"WoS query: {wos_query}")

            results = client.search_documents(
                query=wos_query,
                limit=max_results,
                sort_field='PY+D'
            )

            # Parse results
            articles = []
            for record in results.get('hits', []):
                articles.append(client._parse_article(record))

        if not articles:
            return f"No Web of Science articles found for: '{query}'\n\n**Suggestions:**\n- Try broader search terms\n- Remove year restrictions\n- Check spelling of polymer/solvent names"

        # Format output
        output = [f"# Web of Science Results: {query}\n"]
        output.append(f"**Found:** {len(articles)} peer-reviewed articles\n")

        if year_low or year_high:
            year_range = f"{year_low or '...'}-{year_high or '...'}"
            output.append(f"**Year Range:** {year_range}\n")

        output.append("\n## Articles\n")

        for i, article in enumerate(articles, 1):
            # Title with WoS link
            title = article.get('title', 'N/A')
            link = article.get('link', '#')
            output.append(f"\n### {i}. [{title}]({link})")

            # Authors - always show, even if empty
            authors = article.get('authors', [])
            if authors:
                author_str = ', '.join(authors[:5])
                if len(authors) > 5:
                    author_str += f" et al. ({len(authors)} total)"
            else:
                author_str = "Not available"
            output.append(f"**Authors:** {author_str}")

            # Journal and year - always show
            journal = article.get('journal', 'N/A')
            year = article.get('year', 'N/A')
            output.append(f"**Journal:** {journal}")
            output.append(f"**Year:** {year}")

            # Volume and pages if available
            volume = article.get('volume', '')
            pages = article.get('pages', '')
            if volume or pages:
                vol_page = []
                if volume:
                    vol_page.append(f"Vol. {volume}")
                if pages:
                    vol_page.append(f"pp. {pages}")
                output.append(f"**Volume/Pages:** {', '.join(vol_page)}")

            # DOI - always show
            doi = article.get('doi', 'N/A')
            if doi and doi != 'N/A':
                output.append(f"**DOI:** [{doi}](https://doi.org/{doi})")
            else:
                output.append(f"**DOI:** Not available")

            # Citations - always show
            times_cited = article.get('times_cited', 0)
            output.append(f"**Times Cited:** {times_cited}")

            # Abstract snippet
            abstract = article.get('abstract', '')
            if abstract and abstract != 'N/A':
                # Truncate long abstracts
                if len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                output.append(f"*{abstract}*")

        # Footer
        output.append(f"\n\n---")
        output.append(f"**Search Query:** `{query}`")
        output.append(f"**Results:** {len(articles)} peer-reviewed articles from Web of Science")
        output.append(f"**Source:** Clarivate Web of Science Starter API")

        return "\n".join(output)

    except ModuleNotFoundError:
        return ("Web of Science search is not available. The `wos_starter_client` module is not installed.\n\n"
                "Please ensure the WoS client is properly configured.")
    except ValueError as e:
        if "WOS_STARTER_API_KEY" in str(e):
            return ("Web of Science search requires an API key.\n\n"
                    "**Setup:**\n"
                    "1. Get API key from: https://developer.clarivate.com/\n"
                    "2. Set environment variable: `WOS_STARTER_API_KEY=your-key`\n"
                    "3. Restart the application")
        else:
            return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Web of Science search error: {e}")
        return f"Search failed: {str(e)}\n\nPlease try again or simplify your query."
