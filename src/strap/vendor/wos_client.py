"""
Web of Science Starter API Client
API key authentication for article search via the WoS Starter endpoint.

Docs: https://developer.clarivate.com/apis/wos-starter
Endpoint: https://api.clarivate.com/apis/wos-starter/v1/
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class WebOfScienceClient:
    """
    Client for searching Web of Science using the Starter API.

    Authentication: X-ApiKey header
    Base URL: https://api.clarivate.com/apis/wos-starter/v1
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Web of Science Starter API client.

        Args:
            api_key: WoS Starter API key (defaults to WOS_STARTER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("WOS_STARTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "WoS Starter API key not provided. Set WOS_STARTER_API_KEY "
                "environment variable or pass it to the constructor."
            )

        self.base_url = "https://api.clarivate.com/apis/wos-starter/v1"
        logger.info("WebOfScienceClient initialized (Starter API)")

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make authenticated GET request to the WoS Starter API.

        Args:
            endpoint: API path (e.g., '/documents')
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        headers = {
            "X-ApiKey": self.api_key,
            "Accept": "application/json",
        }

        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"WoS API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def search_articles(
        self,
        query: str,
        database: str = "WOS",
        count: int = 10,
        first_record: int = 1,
        sort_field: Optional[str] = None,
    ) -> Dict:
        """
        Search for articles in Web of Science.

        Args:
            query: WoS search query (e.g., 'TS=(polymer solubility)')
            database: Database to search (default: 'WOS')
            count: Number of results (max 50 per page)
            first_record: Starting record number (1-based)
            sort_field: Sort field (e.g., 'PY+D' for year descending)

        Returns:
            Raw API response dict with 'metadata' and 'hits' keys
        """
        logger.info(f"Searching WoS for: {query}")

        params = {
            "db": database,
            "q": query,
            "limit": min(count, 50),
            "page": ((first_record - 1) // count) + 1,
        }

        if sort_field:
            params["sortField"] = sort_field

        return self._make_request("/documents", params)

    def get_article_by_uid(self, uid: str) -> Dict:
        """
        Retrieve article metadata by unique identifier.

        Args:
            uid: Web of Science unique identifier (e.g., 'WOS:000123456789')

        Returns:
            Article metadata dictionary
        """
        logger.info(f"Fetching article: {uid}")
        return self._make_request(f"/documents/{uid}")

    def search_polymer_solubility_articles(
        self,
        polymer_name: Optional[str] = None,
        solvent_name: Optional[str] = None,
        year_range: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict]:
        """
        Search for polymer solubility related articles.

        Args:
            polymer_name: Specific polymer name
            solvent_name: Specific solvent name
            year_range: Publication year range (e.g., '2020-2024')
            max_results: Maximum number of results

        Returns:
            List of parsed article dictionaries
        """
        query_parts = []

        if polymer_name and solvent_name:
            query_parts.append(f'TS=("{polymer_name}" AND "{solvent_name}")')
        elif polymer_name:
            query_parts.append(
                f'TS=("{polymer_name}" AND (solubility OR dissolution))'
            )
        elif solvent_name:
            query_parts.append(
                f'TS=("{solvent_name}" AND polymer AND solubility)'
            )
        else:
            query_parts.append(
                "TS=(polymer solubility OR Hansen solubility parameters)"
            )

        if year_range:
            query_parts.append(f"PY=({year_range})")

        query = " AND ".join(query_parts)
        logger.info(f"Polymer solubility article search: {query}")

        try:
            results = self.search_articles(
                query=query, count=max_results, sort_field="PY+D"
            )

            articles = []
            for hit in results.get("hits", []):
                articles.append(self._parse_hit(hit))

            logger.info(f"Found {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Article search failed: {e}")
            return []

    def _parse_hit(self, hit: Dict) -> Dict:
        """
        Parse a WoS Starter API hit into simplified format.

        Args:
            hit: Raw hit from the 'hits' array

        Returns:
            Simplified article metadata
        """
        try:
            uid = hit.get("uid", "N/A")
            title = hit.get("title", "N/A")

            # Authors
            names = hit.get("names", {})
            authors_data = names.get("authors", [])
            authors = [a.get("displayName", "") for a in authors_data]

            # Source / year
            source_info = hit.get("source", {})
            pub_year = source_info.get("publishYear", "N/A")
            source = source_info.get("sourceTitle", "N/A")

            # Identifiers
            identifiers = hit.get("identifiers", {})
            doi = identifiers.get("doi", "N/A")

            # Citations
            citations = hit.get("citations", [])
            cite_count = citations[0].get("count", 0) if citations else 0

            # Keywords
            kw_data = hit.get("keywords", {})
            keywords = kw_data.get("authorKeywords", [])

            return {
                "uid": uid,
                "title": title,
                "authors": authors,
                "year": pub_year,
                "source": source,
                "doi": doi,
                "citations": cite_count,
                "keywords": keywords,
            }

        except Exception as e:
            logger.warning(f"Failed to parse hit: {e}")
            return {
                "uid": hit.get("uid", "N/A"),
                "title": "Parse Error",
                "authors": [],
                "year": "N/A",
                "source": "N/A",
                "doi": "N/A",
                "citations": 0,
                "keywords": [],
            }

    def test_connection(self) -> bool:
        """
        Test Web of Science API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Web of Science Starter API connection...")
            results = self.search_articles(query="TS=(polymer)", count=1)
            total = results.get("metadata", {}).get("total", 0)
            logger.info(
                f"Web of Science API connection successful! "
                f"({total} total results for test query)"
            )
            return True
        except Exception as e:
            logger.error(f"Web of Science API connection failed: {e}")
            return False


def test_wos_connection(api_key: str = None) -> bool:
    """Quick test of Web of Science API connection."""
    try:
        client = WebOfScienceClient(api_key=api_key)
        return client.test_connection()
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("Web of Science Starter API Client - Connection Test")
    print("=" * 70)

    if not os.getenv("WOS_STARTER_API_KEY"):
        print("\nCredentials not found!")
        print("Set: export WOS_STARTER_API_KEY='your_key'")
    else:
        success = test_wos_connection()
        if success:
            print("\nReady to use Web of Science API!")
        else:
            print("\nConnection failed. Check credentials and try again.")
