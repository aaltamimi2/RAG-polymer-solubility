"""
Web of Science API Client
OAuth 2.0 Client Credentials Flow implementation for article extraction
"""

import os
import time
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class WoSToken:
    """Web of Science access token with expiry tracking"""
    access_token: str
    token_type: str
    expires_at: datetime

    def is_expired(self) -> bool:
        """Check if token is expired (with 60 second buffer)"""
        return datetime.now() >= (self.expires_at - timedelta(seconds=60))


class WebOfScienceClient:
    """
    Client for interacting with Web of Science API using OAuth 2.0

    Authentication: OAuth 2.0 Client Credentials Flow
    Base URL: https://api.clarivate.com
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        auth_method: str = "realms/api",
        api_name: str = "wos"
    ):
        """
        Initialize Web of Science API client

        Args:
            client_id: OAuth 2.0 client ID (defaults to WOS_CLIENT_ID env var)
            client_secret: OAuth 2.0 client secret (defaults to WOS_CLIENT_SECRET env var)
            auth_method: Authentication realm (default: 'realms/api')
            api_name: API name (default: 'wos')
        """
        self.client_id = client_id or os.getenv("WOS_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("WOS_CLIENT_SECRET")
        self.auth_method = auth_method
        self.api_name = api_name

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "WoS credentials not provided. Set WOS_CLIENT_ID and WOS_CLIENT_SECRET "
                "environment variables or pass them to the constructor."
            )

        self.base_url = "https://api.clarivate.com"
        self.token: Optional[WoSToken] = None

        logger.info("WebOfScienceClient initialized")

    def _get_token_url(self) -> str:
        """Construct OAuth token endpoint URL"""
        return f"{self.base_url}/auth/{self.auth_method}/api/{self.api_name}/token"

    def authenticate(self) -> WoSToken:
        """
        Obtain OAuth 2.0 access token using client credentials flow

        Returns:
            WoSToken with access token and expiry information

        Raises:
            requests.HTTPError: If authentication fails
        """
        logger.info("Requesting new access token from Web of Science API")

        token_url = self._get_token_url()

        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            response = requests.post(token_url, data=data, headers=headers, timeout=30)
            response.raise_for_status()

            token_data = response.json()

            # Calculate token expiry time
            expires_in = token_data.get('expires_in', 3600)
            expires_at = datetime.now() + timedelta(seconds=expires_in)

            self.token = WoSToken(
                access_token=token_data['access_token'],
                token_type=token_data.get('token_type', 'bearer'),
                expires_at=expires_at
            )

            logger.info(f"✅ Access token obtained, expires at {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
            return self.token

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Authentication failed: {e}")
            raise

    def _ensure_valid_token(self):
        """Ensure we have a valid access token, refresh if needed"""
        if self.token is None or self.token.is_expired():
            logger.info("Token expired or missing, obtaining new token")
            self.authenticate()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict:
        """
        Make authenticated request to Web of Science API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., '/api/wos/search')
            params: Query parameters
            data: Form data
            json_data: JSON body data

        Returns:
            Response JSON

        Raises:
            requests.HTTPError: If request fails
        """
        self._ensure_valid_token()

        url = f"{self.base_url}{endpoint}"

        headers = {
            'Authorization': f'{self.token.token_type.capitalize()} {self.token.access_token}',
            'Accept': 'application/json'
        }

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def search_articles(
        self,
        query: str,
        database: str = "WOS",
        count: int = 10,
        first_record: int = 1,
        sort_field: Optional[str] = None
    ) -> Dict:
        """
        Search for articles in Web of Science

        Args:
            query: Search query (e.g., 'TS=(polymer solubility)')
            database: Database to search (default: 'WOS')
            count: Number of results to return (default: 10)
            first_record: Starting record number (default: 1)
            sort_field: Sort field (e.g., 'PY+D' for publication year descending)

        Returns:
            Search results dictionary

        Example query formats:
            - Topic search: 'TS=(polymer solubility)'
            - Author search: 'AU=(Smith J)'
            - Title search: 'TI=(Hansen solubility parameters)'
            - Combined: 'TS=(polymer) AND PY=(2020-2024)'
        """
        logger.info(f"Searching WoS for: {query}")

        params = {
            'databaseId': database,
            'usrQuery': query,
            'count': count,
            'firstRecord': first_record
        }

        if sort_field:
            params['sortField'] = sort_field

        return self._make_request('GET', '/api/wos', params=params)

    def get_article_by_uid(self, uid: str) -> Dict:
        """
        Retrieve article metadata by unique identifier

        Args:
            uid: Web of Science unique identifier (e.g., 'WOS:000123456789')

        Returns:
            Article metadata dictionary
        """
        logger.info(f"Fetching article: {uid}")
        return self._make_request('GET', f'/api/wos/{uid}')

    def search_polymer_solubility_articles(
        self,
        polymer_name: Optional[str] = None,
        solvent_name: Optional[str] = None,
        year_range: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for polymer solubility related articles

        Args:
            polymer_name: Specific polymer name to search for
            solvent_name: Specific solvent name to search for
            year_range: Publication year range (e.g., '2020-2024')
            max_results: Maximum number of results to return

        Returns:
            List of article metadata dictionaries
        """
        # Build query
        query_parts = []

        if polymer_name and solvent_name:
            query_parts.append(f'TS=("{polymer_name}" AND "{solvent_name}")')
        elif polymer_name:
            query_parts.append(f'TS=("{polymer_name}" AND (solubility OR dissolution))')
        elif solvent_name:
            query_parts.append(f'TS=("{solvent_name}" AND polymer AND solubility)')
        else:
            query_parts.append('TS=(polymer solubility OR Hansen solubility parameters)')

        if year_range:
            query_parts.append(f'PY=({year_range})')

        query = ' AND '.join(query_parts)

        logger.info(f"Polymer solubility article search: {query}")

        try:
            results = self.search_articles(
                query=query,
                count=max_results,
                sort_field='PY+D'  # Sort by publication year descending
            )

            # Extract article records
            articles = []
            if 'Data' in results and 'Records' in results['Data']:
                for record in results['Data']['Records'].get('records', {}).get('REC', []):
                    articles.append(self._parse_article_record(record))

            logger.info(f"Found {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Article search failed: {e}")
            return []

    def _parse_article_record(self, record: Dict) -> Dict:
        """
        Parse Web of Science article record into simplified format

        Args:
            record: Raw WoS record

        Returns:
            Simplified article metadata
        """
        try:
            uid = record.get('UID', 'N/A')

            # Extract static data
            static = record.get('static_data', {})
            summary = static.get('summary', {})
            fullrecord_metadata = static.get('fullrecord_metadata', {})

            # Extract title
            titles = summary.get('titles', {}).get('title', [])
            title = titles[0].get('content', 'N/A') if titles else 'N/A'

            # Extract authors
            authors_data = summary.get('names', {}).get('name', [])
            authors = [
                f"{author.get('first_name', '')} {author.get('last_name', '')}".strip()
                for author in authors_data
            ] if authors_data else []

            # Extract publication info
            pub_info = summary.get('pub_info', {})
            pub_year = pub_info.get('pubyear', 'N/A')

            # Extract source
            source_title = summary.get('titles', {}).get('title', [])
            source = source_title[1].get('content', 'N/A') if len(source_title) > 1 else 'N/A'

            # Extract DOI
            identifiers = fullrecord_metadata.get('identifiers', {}).get('identifier', [])
            doi = 'N/A'
            for identifier in identifiers:
                if identifier.get('type') == 'doi':
                    doi = identifier.get('value', 'N/A')
                    break

            # Extract abstract
            abstract_data = fullrecord_metadata.get('abstracts', {}).get('abstract', [])
            abstract = abstract_data[0].get('abstract_text', {}).get('p', 'N/A') if abstract_data else 'N/A'

            return {
                'uid': uid,
                'title': title,
                'authors': authors,
                'year': pub_year,
                'source': source,
                'doi': doi,
                'abstract': abstract if isinstance(abstract, str) else str(abstract)
            }

        except Exception as e:
            logger.warning(f"Failed to parse article record: {e}")
            return {
                'uid': record.get('UID', 'N/A'),
                'title': 'Parse Error',
                'authors': [],
                'year': 'N/A',
                'source': 'N/A',
                'doi': 'N/A',
                'abstract': 'N/A'
            }

    def test_connection(self) -> bool:
        """
        Test Web of Science API connection and authentication

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Web of Science API connection...")
            self.authenticate()

            # Try a simple search
            results = self.search_articles(
                query='TS=(polymer)',
                count=1
            )

            logger.info("✅ Web of Science API connection successful!")
            return True

        except Exception as e:
            logger.error(f"❌ Web of Science API connection failed: {e}")
            return False


# Convenience function for quick testing
def test_wos_connection(client_id: str = None, client_secret: str = None) -> bool:
    """
    Quick test of Web of Science API connection

    Args:
        client_id: OAuth client ID (optional, uses env var if not provided)
        client_secret: OAuth client secret (optional, uses env var if not provided)

    Returns:
        True if connection successful
    """
    try:
        client = WebOfScienceClient(client_id=client_id, client_secret=client_secret)
        return client.test_connection()
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test connection
    print("=" * 70)
    print("Web of Science API Client - Connection Test")
    print("=" * 70)

    # Check for credentials
    if not os.getenv("WOS_CLIENT_ID") or not os.getenv("WOS_CLIENT_SECRET"):
        print("\n⚠️  Credentials not found!")
        print("\nTo test, set environment variables:")
        print("  export WOS_CLIENT_ID='your_client_id'")
        print("  export WOS_CLIENT_SECRET='your_client_secret'")
        print("\nThen run: python wos_api_client.py")
    else:
        success = test_wos_connection()
        if success:
            print("\n✅ Ready to use Web of Science API!")
        else:
            print("\n❌ Connection failed. Check credentials and try again.")
