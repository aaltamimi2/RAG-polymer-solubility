"""
arXiv API Client for open-access paper search and PDF download.

Uses the ``arxiv`` Python package (https://pypi.org/project/arxiv/).
All arXiv papers are open-access — PDFs can always be downloaded.
"""

import logging
import re
from typing import Dict, List, Optional

import arxiv

logger = logging.getLogger(__name__)


class ArxivClient:
    """Lightweight client for arXiv search + PDF URL construction."""

    def __init__(self):
        self._client = arxiv.Client(
            page_size=20,
            delay_seconds=1.0,  # arXiv rate limit: 1 req/sec
            num_retries=3,
        )
        logger.info("ArxivClient initialized")

    def search(
        self,
        query: str,
        categories: Optional[list[str]] = None,
        max_results: int = 10,
        sort_by: str = "submitted",
    ) -> List[Dict]:
        """Search arXiv for papers.

        Args:
            query: Search query (boolean operators supported).
            categories: arXiv categories to filter (e.g. ["cond-mat", "physics.chem-ph"]).
            max_results: Max papers to return.
            sort_by: "submitted" (default), "updated", or "relevance".

        Returns:
            List of parsed paper dicts.
        """
        logger.info(f"Searching arXiv for: {query}")

        search_query = query
        if categories:
            cat_query = " OR ".join(f"cat:{c}" for c in categories)
            search_query = f"({search_query}) AND ({cat_query})"

        sort_map = {
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "updated": arxiv.SortCriterion.LastUpdatedDate,
            "relevance": arxiv.SortCriterion.Relevance,
        }
        criterion = sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate)

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in self._client.results(search):
            papers.append(self._parse_result(result))

        logger.info(f"Found {len(papers)} arXiv papers")
        return papers

    @staticmethod
    def _extract_arxiv_id(entry_id: str) -> str:
        """Extract arXiv ID from entry URL (e.g. 'http://arxiv.org/abs/2301.00001v1')."""
        match = re.search(r"arxiv\.org/abs/([^v]+)", entry_id)
        if match:
            return match.group(1)
        return entry_id.split("/")[-1].split("v")[0]

    def _parse_result(self, result: arxiv.Result) -> Dict:
        arxiv_id = self._extract_arxiv_id(result.entry_id)
        pub_date = result.published
        return {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors],
            "date": pub_date.strftime("%Y-%m-%d") if pub_date else "N/A",
            "year": pub_date.year if pub_date else "N/A",
            "doi": result.doi,
            "journal_ref": result.journal_ref,
            "categories": result.categories,
            "url": result.entry_id,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        }

    def test_connection(self) -> bool:
        try:
            logger.info("Testing arXiv API connection...")
            results = self.search(query="polymer", max_results=1)
            if results:
                logger.info("arXiv API connection successful!")
                return True
            return False
        except Exception as e:
            logger.error(f"arXiv API connection failed: {e}")
            return False
