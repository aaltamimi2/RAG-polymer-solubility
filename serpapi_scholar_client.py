"""
Google Scholar API Client using SerpAPI
Simple API key authentication for academic article searching
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class GoogleScholarClient:
    """
    Client for searching Google Scholar using SerpAPI

    Authentication: API Key (query parameter)
    Endpoint: https://serpapi.com/search
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google Scholar API client

        Args:
            api_key: SerpAPI key (defaults to SERPAPI_KEY env var)
        """
        self.api_key = api_key or os.getenv("SERPAPI_KEY")

        if not self.api_key:
            raise ValueError(
                "SerpAPI key not provided. Set SERPAPI_KEY environment variable "
                "or pass it to the constructor."
            )

        self.base_url = "https://serpapi.com/search"
        logger.info("GoogleScholarClient initialized")

    def _make_request(self, params: Dict) -> Dict:
        """
        Make request to SerpAPI

        Args:
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            requests.HTTPError: If request fails
        """
        # Add API key and engine
        params['api_key'] = self.api_key
        params['engine'] = 'google_scholar'

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ SerpAPI request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def search(
        self,
        query: str,
        num_results: int = 10,
        year_low: Optional[int] = None,
        year_high: Optional[int] = None,
        sort_by: Optional[str] = None,
        include_patents: bool = False,
        include_citations: bool = False
    ) -> Dict:
        """
        Search Google Scholar for articles

        Args:
            query: Search query
            num_results: Number of results to return (max 20 per page)
            year_low: Start year for date range filter
            year_high: End year for date range filter
            sort_by: Sort order ('date' for recent first, None for relevance)
            include_patents: Include patents in results
            include_citations: Include citations in results

        Returns:
            Search results dictionary with organic_results, pagination, etc.

        Example queries:
            - Simple: "polymer solubility"
            - Author: "author:hansen solubility parameters"
            - Title: "intitle:hansen parameters"
            - Exact phrase: '"Hansen solubility parameters"'
        """
        logger.info(f"Searching Google Scholar for: {query}")

        params = {
            'q': query,
            'num': min(num_results, 20)  # Max 20 per page
        }

        # Add optional filters
        if year_low is not None or year_high is not None:
            as_ylo = year_low if year_low else ""
            as_yhi = year_high if year_high else ""
            params['as_ylo'] = as_ylo
            params['as_yhi'] = as_yhi

        if sort_by == 'date':
            params['scisbd'] = 1  # Sort by date

        if include_patents:
            params['as_sdt'] = '0,5'  # Include patents

        if include_citations:
            params['as_vis'] = 1  # Include citations

        return self._make_request(params)

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
        # Build query
        query_parts = []

        if polymer_name and solvent_name:
            query_parts.append(f'"{polymer_name}" "{solvent_name}" solubility')
        elif polymer_name:
            query_parts.append(f'"{polymer_name}" solubility')
        elif solvent_name:
            query_parts.append(f'"{solvent_name}" polymer solubility')
        else:
            query_parts.append('polymer solubility')

        query = ' '.join(query_parts)

        logger.info(f"Polymer article search: {query}")

        try:
            results = self.search(
                query=query,
                num_results=max_results,
                year_low=year_low,
                year_high=year_high,
                sort_by='date'  # Get recent articles
            )

            # Parse organic results
            articles = []
            for result in results.get('organic_results', []):
                articles.append(self._parse_article(result))

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
        if polymer_name:
            query = f'"Hansen solubility parameters" "{polymer_name}"'
        else:
            query = '"Hansen solubility parameters"'

        logger.info(f"Hansen parameters search: {query}")

        try:
            results = self.search(
                query=query,
                num_results=max_results,
                year_low=year_low,
                year_high=year_high
            )

            articles = []
            for result in results.get('organic_results', []):
                articles.append(self._parse_article(result))

            logger.info(f"Found {len(articles)} Hansen parameter articles")
            return articles

        except Exception as e:
            logger.error(f"Hansen search failed: {e}")
            return []

    def get_author_articles(
        self,
        author_name: str,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for articles by specific author

        Args:
            author_name: Author name (e.g., "Hansen CM")
            max_results: Max results to return

        Returns:
            List of parsed article dictionaries
        """
        query = f'author:"{author_name}"'
        logger.info(f"Author search: {query}")

        try:
            results = self.search(query=query, num_results=max_results)

            articles = []
            for result in results.get('organic_results', []):
                articles.append(self._parse_article(result))

            logger.info(f"Found {len(articles)} articles by {author_name}")
            return articles

        except Exception as e:
            logger.error(f"Author search failed: {e}")
            return []

    def _parse_article(self, result: Dict) -> Dict:
        """
        Parse Google Scholar result into simplified format

        Args:
            result: Raw SerpAPI result

        Returns:
            Simplified article dictionary
        """
        try:
            # Extract basic info
            title = result.get('title', 'N/A')
            link = result.get('link', 'N/A')
            snippet = result.get('snippet', 'N/A')

            # Publication info
            pub_info = result.get('publication_info', {})
            authors_raw = pub_info.get('authors', [])
            authors = [a.get('name', '') for a in authors_raw] if isinstance(authors_raw, list) else []
            summary = pub_info.get('summary', '')

            # Extract year from summary (e.g., "J Smith - Journal, 2024")
            year = 'N/A'
            if summary:
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', summary)
                if year_match:
                    year = year_match.group(0)

            # Citation info
            inline_links = result.get('inline_links', {})
            cited_by_count = inline_links.get('cited_by', {}).get('total', 0)
            cited_by_link = inline_links.get('cited_by', {}).get('link', '')

            # Related versions
            versions_count = inline_links.get('versions', {}).get('total', 0)

            # Resources (PDF links, etc.)
            resources = result.get('resources', [])
            pdf_link = None
            for resource in resources:
                if resource.get('file_format', '').upper() == 'PDF':
                    pdf_link = resource.get('link')
                    break

            return {
                'title': title,
                'authors': authors,
                'year': year,
                'snippet': snippet,
                'link': link,
                'pdf_link': pdf_link,
                'cited_by_count': cited_by_count,
                'cited_by_link': cited_by_link,
                'versions_count': versions_count,
                'publication_info': summary
            }

        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return {
                'title': result.get('title', 'Parse Error'),
                'authors': [],
                'year': 'N/A',
                'snippet': result.get('snippet', ''),
                'link': result.get('link', 'N/A'),
                'pdf_link': None,
                'cited_by_count': 0,
                'cited_by_link': '',
                'versions_count': 0,
                'publication_info': ''
            }

    def test_connection(self) -> bool:
        """
        Test SerpAPI connection and API key validity

        Returns:
            True if connection successful
        """
        try:
            logger.info("Testing SerpAPI connection...")

            results = self.search(query='polymer', num_results=1)

            if 'organic_results' in results:
                logger.info("✅ SerpAPI connection successful!")
                return True
            else:
                logger.error("❌ Unexpected response format")
                return False

        except Exception as e:
            logger.error(f"❌ SerpAPI connection failed: {e}")
            return False

    def get_account_info(self) -> Dict:
        """
        Get SerpAPI account information and usage stats

        Returns:
            Account info dictionary
        """
        try:
            url = "https://serpapi.com/account"
            params = {'api_key': self.api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            account_data = response.json()
            logger.info(f"Account info retrieved: {account_data.get('plan', 'N/A')} plan")
            return account_data

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}


# Convenience function for quick testing
def test_scholar_search(api_key: str = None) -> bool:
    """
    Quick test of Google Scholar search

    Args:
        api_key: SerpAPI key (optional, uses env var if not provided)

    Returns:
        True if test successful
    """
    try:
        client = GoogleScholarClient(api_key=api_key)
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
    print("Google Scholar API Client (SerpAPI) - Connection Test")
    print("=" * 70)

    # Check for API key
    if not os.getenv("SERPAPI_KEY"):
        print("\n⚠️  SerpAPI key not found!")
        print("\nTo test, set environment variable:")
        print("  export SERPAPI_KEY='your-serpapi-key'")
        print("\nOr add to .env file:")
        print("  SERPAPI_KEY=your-serpapi-key")
        print("\nGet your API key from: https://serpapi.com/")
    else:
        success = test_scholar_search()
        if success:
            print("\n✅ Ready to search Google Scholar!")

            # Show account info
            client = GoogleScholarClient()
            account = client.get_account_info()
            if account:
                print(f"\n📊 Account: {account.get('account_email', 'N/A')}")
                print(f"   Plan: {account.get('plan', 'N/A')}")
                print(f"   Searches this month: {account.get('this_month_usage', 0)}")
                print(f"   Total searches: {account.get('total_searches_left', 'N/A')}")
        else:
            print("\n❌ Connection failed. Check API key and try again.")
