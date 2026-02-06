"""
Google Patents API Client using SerpAPI
Enables searching Google Patents for patent documents and looking up specific patent numbers

SerpAPI Google Patents API Documentation: https://serpapi.com/google-patents-api
"""

import os
import logging
import re
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class GooglePatentsClient:
    """
    Client for searching Google Patents using SerpAPI

    Authentication: API Key (query parameter)
    Endpoint: https://serpapi.com/search?engine=google_patents

    Features:
    - Search patents by keywords
    - Look up specific patent numbers (US, EP, WO, etc.)
    - Filter by date, inventor, assignee, country
    - Get patent details including claims, abstract, citations
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google Patents API client

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
        logger.info("GooglePatentsClient initialized")

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
        params['engine'] = 'google_patents'

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"SerpAPI request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def search(
        self,
        query: str,
        num_results: int = 10,
        before: Optional[str] = None,
        after: Optional[str] = None,
        inventor: Optional[str] = None,
        assignee: Optional[str] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        type_filter: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict:
        """
        Search Google Patents for patent documents

        Args:
            query: Search query (keywords, concepts, or patent number)
            num_results: Number of results to return (max 100 per page)
            before: Filter patents filed before date (format: YYYYMMDD)
            after: Filter patents filed after date (format: YYYYMMDD)
            inventor: Filter by inventor name
            assignee: Filter by assignee/company name
            country: Filter by country code (US, EP, WO, CN, JP, etc.)
            language: Filter by language (en, de, fr, ja, etc.)
            type_filter: Filter by type (patent, application, design)
            status: Filter by status (grant, application)

        Returns:
            Search results dictionary with organic_results, pagination, etc.

        Example queries:
            - Keywords: "polymer dissolution solvent"
            - Patent number: "US10123456"
            - Company: assignee:dow polymer
            - Inventor: inventor:smith polymer
        """
        logger.info(f"Searching Google Patents for: {query}")

        params = {
            'q': query,
            'num': max(10, min(num_results, 100))
        }

        # Add optional filters
        if before:
            params['before'] = before
        if after:
            params['after'] = after
        if inventor:
            params['inventor'] = inventor
        if assignee:
            params['assignee'] = assignee
        if country:
            params['country'] = country
        if language:
            params['language'] = language
        if type_filter:
            params['type'] = type_filter
        if status:
            params['status'] = status

        return self._make_request(params)

    def get_patent(self, patent_number: str) -> Dict:
        """
        Look up a specific patent by its number

        Args:
            patent_number: Patent number (e.g., "US10123456", "EP1234567", "WO2020123456")
                          Accepts various formats:
                          - US patents: US10123456, US 10,123,456, US2020/0123456
                          - European: EP1234567, EP 1234567 A1
                          - PCT: WO2020123456, WO 2020/123456
                          - Others: CN, JP, KR, AU, etc.

        Returns:
            Patent details including title, abstract, claims, citations
        """
        # Normalize patent number (remove spaces, commas, slashes)
        normalized = self._normalize_patent_number(patent_number)
        logger.info(f"Looking up patent: {normalized}")

        # Search for specific patent
        results = self.search(query=normalized, num_results=5)

        # Try to find exact match
        organic_results = results.get('organic_results', [])

        for result in organic_results:
            result_patent_id = result.get('patent_id', '')
            result_normalized = self._normalize_patent_number(result_patent_id)
            if normalized.upper() in result_normalized.upper():
                return self._parse_patent(result)

        # If no exact match, return first result if any
        if organic_results:
            return self._parse_patent(organic_results[0])

        return {'error': f'Patent {patent_number} not found'}

    def _normalize_patent_number(self, patent_number: str) -> str:
        """
        Normalize patent number by removing formatting characters

        Args:
            patent_number: Raw patent number

        Returns:
            Normalized patent number (uppercase, no spaces/commas/slashes)
        """
        # Remove spaces, commas, slashes, dashes
        normalized = re.sub(r'[\s,/\-]', '', patent_number)
        # Remove common suffixes like A1, B2, etc. for comparison
        # but keep them in the actual search
        return normalized.upper()

    def search_polymer_patents(
        self,
        polymer_name: Optional[str] = None,
        solvent_name: Optional[str] = None,
        process_type: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        assignee: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for polymer-related patents

        Args:
            polymer_name: Specific polymer name (e.g., "polystyrene", "PET")
            solvent_name: Specific solvent name (e.g., "toluene", "DMF")
            process_type: Type of process (e.g., "dissolution", "recycling", "recovery")
            after: Patents filed after date (YYYYMMDD)
            before: Patents filed before date (YYYYMMDD)
            assignee: Company/assignee filter
            max_results: Maximum number of results

        Returns:
            List of parsed patent dictionaries
        """
        # Build query
        query_parts = []

        if polymer_name and solvent_name:
            query_parts.append(f'"{polymer_name}" "{solvent_name}"')
        elif polymer_name:
            query_parts.append(f'"{polymer_name}"')
        elif solvent_name:
            query_parts.append(f'"{solvent_name}" polymer')

        if process_type:
            query_parts.append(process_type)
        else:
            # Default to polymer-solvent related terms
            query_parts.append('(dissolution OR solubility OR recycling OR recovery)')

        query = ' '.join(query_parts)
        logger.info(f"Polymer patent search: {query}")

        try:
            results = self.search(
                query=query,
                num_results=max_results,
                after=after,
                before=before,
                assignee=assignee
            )

            patents = []
            for result in results.get('organic_results', []):
                patents.append(self._parse_patent(result))

            logger.info(f"Found {len(patents)} patents")
            return patents

        except Exception as e:
            logger.error(f"Patent search failed: {e}")
            return []

    def _parse_patent(self, result: Dict) -> Dict:
        """
        Parse Google Patents result into simplified format

        Args:
            result: Raw SerpAPI result

        Returns:
            Simplified patent dictionary
        """
        try:
            # Basic info
            patent_id = result.get('patent_id', 'N/A')
            title = result.get('title', 'N/A')
            snippet = result.get('snippet', 'N/A')
            link = result.get('link', f'https://patents.google.com/patent/{patent_id}')

            # PDF link
            pdf_link = result.get('pdf', None)

            # Filing and grant dates
            filing_date = result.get('filing_date', 'N/A')
            grant_date = result.get('grant_date', 'N/A')
            publication_date = result.get('publication_date', 'N/A')

            # Parties
            inventor_raw = result.get('inventor', '')
            inventors = [inventor_raw] if isinstance(inventor_raw, str) else inventor_raw
            if isinstance(inventors, str):
                inventors = [inv.strip() for inv in inventors.split(',')]

            assignee = result.get('assignee', 'N/A')

            # Priority date for determining actual filing priority
            priority_date = result.get('priority_date', filing_date)

            # Extract year from filing date
            year = 'N/A'
            for date_field in [filing_date, publication_date, grant_date]:
                if date_field and date_field != 'N/A':
                    year_match = re.search(r'(19|20)\d{2}', str(date_field))
                    if year_match:
                        year = year_match.group(0)
                        break

            # Claims count if available
            claims_count = result.get('claims_count', None)

            # Citations
            cited_by_count = result.get('cited_by', {}).get('count', 0) if isinstance(result.get('cited_by'), dict) else 0

            # Thumbnail
            thumbnail = result.get('thumbnail', None)

            # Country from patent ID
            country = 'Unknown'
            country_match = re.match(r'^([A-Z]{2})', patent_id)
            if country_match:
                country = country_match.group(1)

            return {
                'patent_id': patent_id,
                'title': title,
                'snippet': snippet,
                'link': link,
                'pdf_link': pdf_link,
                'filing_date': filing_date,
                'grant_date': grant_date,
                'publication_date': publication_date,
                'priority_date': priority_date,
                'year': year,
                'inventors': inventors if isinstance(inventors, list) else [inventors],
                'assignee': assignee,
                'country': country,
                'claims_count': claims_count,
                'cited_by_count': cited_by_count,
                'thumbnail': thumbnail
            }

        except Exception as e:
            logger.warning(f"Failed to parse patent: {e}")
            return {
                'patent_id': result.get('patent_id', 'Parse Error'),
                'title': result.get('title', 'N/A'),
                'snippet': result.get('snippet', ''),
                'link': result.get('link', 'N/A'),
                'pdf_link': None,
                'filing_date': 'N/A',
                'grant_date': 'N/A',
                'publication_date': 'N/A',
                'priority_date': 'N/A',
                'year': 'N/A',
                'inventors': [],
                'assignee': 'N/A',
                'country': 'Unknown',
                'claims_count': None,
                'cited_by_count': 0,
                'thumbnail': None
            }

    def test_connection(self) -> bool:
        """
        Test SerpAPI connection and API key validity for Google Patents

        Returns:
            True if connection successful
        """
        try:
            logger.info("Testing SerpAPI Google Patents connection...")

            results = self.search(query='polymer', num_results=1)

            if 'organic_results' in results or 'search_metadata' in results:
                logger.info("SerpAPI Google Patents connection successful!")
                return True
            else:
                logger.error("Unexpected response format")
                return False

        except Exception as e:
            logger.error(f"SerpAPI connection failed: {e}")
            return False


# Convenience function for quick testing
def test_patents_search(api_key: str = None) -> bool:
    """
    Quick test of Google Patents search

    Args:
        api_key: SerpAPI key (optional, uses env var if not provided)

    Returns:
        True if test successful
    """
    try:
        client = GooglePatentsClient(api_key=api_key)
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
    print("Google Patents API Client (SerpAPI) - Connection Test")
    print("=" * 70)

    # Check for API key
    if not os.getenv("SERPAPI_KEY"):
        print("\n  SerpAPI key not found!")
        print("\nTo test, set environment variable:")
        print("  export SERPAPI_KEY='your-serpapi-key'")
        print("\nOr add to .env file:")
        print("  SERPAPI_KEY=your-serpapi-key")
        print("\nGet your API key from: https://serpapi.com/")
    else:
        success = test_patents_search()
        if success:
            print("\n Ready to search Google Patents!")

            # Example search
            print("\n Example Search: 'polymer dissolution'")
            client = GooglePatentsClient()
            results = client.search("polymer dissolution", num_results=3)

            for i, result in enumerate(results.get('organic_results', [])[:3], 1):
                patent = client._parse_patent(result)
                print(f"\n{i}. {patent['patent_id']}: {patent['title'][:60]}...")
                print(f"   Assignee: {patent['assignee']}")
                print(f"   Filed: {patent['filing_date']}")
        else:
            print("\n Connection failed. Check API key and try again.")
