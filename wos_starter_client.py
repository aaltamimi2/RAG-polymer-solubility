"""
Web of Science Starter API Client
Simple API key authentication for article searching
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class WebOfScienceStarterClient:
    """
    Client for Web of Science Starter API

    Authentication: API Key (X-ApiKey header)
    Endpoint: https://api.clarivate.com/apis/wos-starter/v1/
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize WoS Starter API client

        Args:
            api_key: WoS Starter API key (defaults to WOS_STARTER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("WOS_STARTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "WoS Starter API key not provided. Set WOS_STARTER_API_KEY environment variable "
                "or pass it to the constructor."
            )

        self.base_url = "https://api.clarivate.com/apis/wos-starter/v1"
        logger.info("WebOfScienceStarterClient initialized")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict:
        """
        Make authenticated request to WoS Starter API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON body data

        Returns:
            Response JSON

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        headers = {
            'X-ApiKey': self.api_key,
            'Accept': 'application/json'
        }

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ WoS Starter API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise

    def search_documents(
        self,
        query: str,
        limit: int = 10,
        page: int = 1,
        db: str = "WOS",
        sort_field: Optional[str] = None
    ) -> Dict:
        """
        Search for documents in Web of Science

        Args:
            query: Search query (uses WoS query syntax)
            limit: Number of results per page (default: 10, max: 50)
            page: Page number (default: 1)
            db: Database to search (default: 'WOS')
            sort_field: Sort field (e.g., 'PY+D' for publication year descending)

        Returns:
            Search results dictionary with hits, metadata, and records

        Example query formats:
            - Field: 'TS=(Hansen solubility parameters)'
            - Complex: 'TS=(polymer) AND PY=(2020-2024)'
        """
        logger.info(f"Searching WoS Starter API for: {query}")

        params = {
            'q': query,
            'limit': min(limit, 50),
            'page': page,
            'db': db
        }

        if sort_field:
            params['sortField'] = sort_field

        return self._make_request('GET', '/documents', params=params)

    def get_document_by_id(self, doc_id: str) -> Dict:
        """
        Retrieve document metadata by ID

        Args:
            doc_id: Document identifier (e.g., 'WOS:000123456789')

        Returns:
            Document metadata dictionary
        """
        logger.info(f"Fetching document: {doc_id}")
        return self._make_request('GET', f'/documents/{doc_id}')

    def search_polymer_articles(
        self,
        polymer_name: Optional[str] = None,
        solvent_name: Optional[str] = None,
        year_low: Optional[int] = None,
        year_high: Optional[int] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for polymer solubility related articles

        Args:
            polymer_name: Specific polymer name
            solvent_name: Specific solvent name
            year_low: Start year for results
            year_high: End year for results
            max_results: Maximum number of results

        Returns:
            List of parsed article dictionaries
        """
        # Build query with WoS field tags
        query_parts = []

        if polymer_name and solvent_name:
            query_parts.append(f'TS=("{polymer_name}" AND "{solvent_name}")')
        elif polymer_name:
            query_parts.append(f'TS=("{polymer_name}" AND (solubility OR dissolution))')
        elif solvent_name:
            query_parts.append(f'TS=("{solvent_name}" AND polymer AND solubility)')
        else:
            query_parts.append('TS=(polymer solubility)')

        # Add year range if specified
        if year_low or year_high:
            year_start = year_low or 1900
            year_end = year_high or 2030
            query_parts.append(f'PY=({year_start}-{year_end})')

        query = ' AND '.join(query_parts)

        logger.info(f"Polymer article search: {query}")

        try:
            results = self.search_documents(
                query=query,
                limit=max_results,
                sort_field='PY+D'  # Publication year descending
            )

            # Parse results
            articles = []
            for record in results.get('hits', []):
                articles.append(self._parse_article(record))

            logger.info(f"Found {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Article search failed: {e}")
            return []

    def search_hansen_parameters(
        self,
        polymer_name: Optional[str] = None,
        year_low: Optional[int] = None,
        year_high: Optional[int] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for Hansen solubility parameters research

        Args:
            polymer_name: Optional specific polymer
            year_low: Start year
            year_high: End year
            max_results: Max results to return

        Returns:
            List of parsed article dictionaries
        """
        # Build query with WoS field tags
        if polymer_name:
            query = f'TS=("Hansen solubility parameters" AND "{polymer_name}")'
        else:
            query = 'TS=("Hansen solubility parameters")'

        # Add year range
        if year_low or year_high:
            year_start = year_low or 1900
            year_end = year_high or 2030
            query += f' AND PY=({year_start}-{year_end})'

        logger.info(f"Hansen parameters search: {query}")

        try:
            results = self.search_documents(query=query, limit=max_results)

            articles = []
            for record in results.get('hits', []):
                articles.append(self._parse_article(record))

            logger.info(f"Found {len(articles)} Hansen parameter articles")
            return articles

        except Exception as e:
            logger.error(f"Hansen search failed: {e}")
            return []

    def _parse_article(self, record: Dict) -> Dict:
        """
        Parse WoS Starter API record into simplified format

        Args:
            record: Raw API record

        Returns:
            Simplified article dictionary
        """
        try:
            # Extract basic info
            uid = record.get('uid', 'N/A')
            title = record.get('title', 'N/A')

            # Authors
            authors_data = record.get('authors', [])
            if isinstance(authors_data, list):
                authors = [a.get('displayName', '') for a in authors_data if isinstance(a, dict)]
            else:
                authors = []

            # Publication info
            source = record.get('source', {})
            if isinstance(source, dict):
                journal = source.get('sourceTitle', 'N/A')
                pub_year = source.get('publishYear', 'N/A')
            else:
                journal = 'N/A'
                pub_year = 'N/A'

            # DOI
            doi = record.get('doi', 'N/A')

            # Abstract
            abstract = record.get('abstract', 'N/A')

            # Citations
            times_cited = record.get('timesCited', 0)

            return {
                'uid': uid,
                'title': title,
                'authors': authors,
                'year': pub_year,
                'journal': journal,
                'doi': doi,
                'abstract': abstract,
                'times_cited': times_cited,
                'link': f"https://www.webofscience.com/wos/woscc/full-record/{uid}" if uid != 'N/A' else 'N/A'
            }

        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return {
                'uid': record.get('uid', 'N/A'),
                'title': record.get('title', 'Parse Error'),
                'authors': [],
                'year': 'N/A',
                'journal': 'N/A',
                'doi': 'N/A',
                'abstract': 'N/A',
                'times_cited': 0,
                'link': 'N/A'
            }

    def test_connection(self) -> bool:
        """
        Test WoS Starter API connection and API key validity

        Returns:
            True if connection successful
        """
        try:
            logger.info("Testing WoS Starter API connection...")

            # Try a simple search with proper WoS query syntax
            results = self.search_documents(query='TS=(polymer)', limit=1)

            if 'hits' in results or 'data' in results:
                logger.info("✅ WoS Starter API connection successful!")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response format: {results.keys()}")
                return False

        except Exception as e:
            logger.error(f"❌ WoS Starter API connection failed: {e}")
            return False


# Convenience function for quick testing
def test_wos_starter(api_key: str = None) -> bool:
    """
    Quick test of WoS Starter API connection

    Args:
        api_key: API key (optional, uses env var if not provided)

    Returns:
        True if connection successful
    """
    try:
        client = WebOfScienceStarterClient(api_key=api_key)
        return client.test_connection()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("Web of Science Starter API Client - Connection Test")
    print("=" * 70)

    # Check for API key
    if not os.getenv("WOS_STARTER_API_KEY"):
        print("\n⚠️  API key not found!")
        print("\nTo test, set environment variable:")
        print("  export WOS_STARTER_API_KEY='your-api-key'")
        print("\nOr add to .env file:")
        print("  WOS_STARTER_API_KEY=your-api-key")
    else:
        success = test_wos_starter()
        if success:
            print("\n✅ Ready to use WoS Starter API!")

            # Try a sample search
            client = WebOfScienceStarterClient()
            print("\n📚 Testing sample search: 'TS=(polymer solubility)'")
            try:
                results = client.search_documents('TS=(polymer solubility)', limit=3)
                print(f"✅ Found results: {results.get('metadata', {}).get('total', 0)} total")
            except Exception as e:
                print(f"❌ Sample search failed: {e}")
        else:
            print("\n❌ Connection failed. Check API key and try again.")
