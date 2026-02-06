"""
PatentsView Search API Client
Searches US granted patents (USPTO data) with full abstracts.

Docs: https://search.patentsview.org/docs
Endpoint: https://search.patentsview.org/api/v1
"""

import os
import json
import logging
import time
import requests
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Fields to request (patent_num_claims causes 400 errors — omit)
_PATENT_FIELDS = [
    "patent_id",
    "patent_title",
    "patent_date",
    "patent_abstract",
    "patent_type",
    "assignees.assignee_organization",
    "assignees.assignee_country",
    "inventors.inventor_name_first",
    "inventors.inventor_name_last",
]


class PatentsViewClient:
    """
    Client for the PatentsView Search API (US USPTO granted patents).

    Authentication: X-Api-Key header
    Rate limit: 45 req/min
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PATENTSVIEW_API_KEY")

        if not self.api_key:
            raise ValueError(
                "PatentsView API key not provided. Set PATENTSVIEW_API_KEY "
                "environment variable or pass it to the constructor."
            )

        self.base_url = "https://search.patentsview.org/api/v1"
        logger.info("PatentsViewClient initialized")

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        retries: int = 2,
    ) -> Dict:
        """Make authenticated GET request with basic retry on 429."""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "X-Api-Key": self.api_key,
            "Accept": "application/json",
        }

        for attempt in range(retries + 1):
            try:
                response = requests.get(
                    url, headers=headers, params=params, timeout=30
                )

                if response.status_code == 429 and attempt < retries:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < retries and getattr(e, "response", None) is not None:
                    if e.response.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                logger.error(f"PatentsView API request failed: {e}")
                if hasattr(e, "response") and e.response is not None:
                    logger.error(f"Response: {e.response.text[:500]}")
                raise

        return {}

    def search(
        self,
        query_text: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        assignee: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict]:
        """
        Full-text search of US granted patents.

        Args:
            query_text: Search terms (matched against patent_title)
            date_from: Start date filter (YYYY-MM-DD)
            date_to: End date filter (YYYY-MM-DD)
            assignee: Filter by assignee organization (substring match)
            max_results: Number of results to return (max 1000)

        Returns:
            List of parsed patent dicts
        """
        logger.info(f"Searching PatentsView for: {query_text}")

        # Build query
        conditions = [{"_text_any": {"patent_title": query_text}}]

        if date_from:
            conditions.append({"_gte": {"patent_date": date_from}})
        if date_to:
            conditions.append({"_lte": {"patent_date": date_to}})
        if assignee:
            conditions.append(
                {"_contains": {"assignees.assignee_organization": assignee}}
            )

        if len(conditions) == 1:
            q = conditions[0]
        else:
            q = {"_and": conditions}

        params = {
            "q": json.dumps(q),
            "f": json.dumps(_PATENT_FIELDS),
            "s": json.dumps([{"patent_date": "desc"}]),
            "o": json.dumps({"size": min(max_results, 1000)}),
        }

        data = self._make_request("/patent/", params=params)

        patents = []
        for hit in data.get("patents", []):
            patents.append(self._parse_patent(hit))

        logger.info(f"Found {len(patents)} US patents")
        return patents

    def get_patent(self, patent_id: str) -> Dict:
        """
        Look up a specific patent by its patent number.

        Args:
            patent_id: USPTO patent number (e.g., "12345678" or "US12345678")

        Returns:
            Parsed patent dict, or dict with 'error' key if not found
        """
        # Strip leading "US" prefix if present
        clean_id = patent_id.strip().upper()
        if clean_id.startswith("US"):
            clean_id = clean_id[2:]

        logger.info(f"Looking up PatentsView patent: {clean_id}")

        try:
            data = self._make_request(f"/patent/{clean_id}/")
            return self._parse_patent(data)
        except requests.exceptions.HTTPError as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 404:
                    return {"error": f"Patent {patent_id} not found in USPTO database"}
            return {"error": f"Patent lookup failed: {e}"}
        except Exception as e:
            return {"error": f"Patent lookup failed: {e}"}

    def _parse_patent(self, hit: Dict) -> Dict:
        """Extract simplified dict from API hit."""
        try:
            patent_id = hit.get("patent_id", "N/A")
            title = hit.get("patent_title", "N/A")
            date = hit.get("patent_date", "N/A")
            patent_type = hit.get("patent_type", "N/A")
            abstract = hit.get("patent_abstract", "")

            # Assignees
            assignees_raw = hit.get("assignees", []) or []
            assignee_org = "N/A"
            assignee_country = "N/A"
            if assignees_raw:
                first = assignees_raw[0]
                assignee_org = first.get("assignee_organization", "N/A") or "N/A"
                assignee_country = first.get("assignee_country", "N/A") or "N/A"

            # Inventors
            inventors_raw = hit.get("inventors", []) or []
            inventors = []
            for inv in inventors_raw:
                first_name = inv.get("inventor_name_first", "")
                last_name = inv.get("inventor_name_last", "")
                full = f"{first_name} {last_name}".strip()
                if full:
                    inventors.append(full)

            return {
                "patent_id": f"US{patent_id}" if patent_id != "N/A" else "N/A",
                "title": title,
                "date": date,
                "type": patent_type,
                "abstract": abstract,
                "assignee": assignee_org,
                "assignee_country": assignee_country,
                "inventors": inventors,
            }

        except Exception as e:
            logger.warning(f"Failed to parse patent: {e}")
            return {
                "patent_id": hit.get("patent_id", "Parse Error"),
                "title": "N/A",
                "date": "N/A",
                "type": "N/A",
                "abstract": "",
                "assignee": "N/A",
                "assignee_country": "N/A",
                "inventors": [],
            }

    def test_connection(self) -> bool:
        """Test PatentsView API connection."""
        try:
            logger.info("Testing PatentsView API connection...")
            results = self.search(query_text="polymer", max_results=1)
            if results:
                logger.info("PatentsView API connection successful!")
                return True
            else:
                logger.error("Unexpected empty response")
                return False
        except Exception as e:
            logger.error(f"PatentsView API connection failed: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("PatentsView Search API Client - Connection Test")
    print("=" * 70)

    if not os.getenv("PATENTSVIEW_API_KEY"):
        print("\nPatentsView API key not found!")
        print("Set: export PATENTSVIEW_API_KEY='your_key'")
    else:
        try:
            client = PatentsViewClient()
            success = client.test_connection()
            if success:
                print("\nReady to search PatentsView!")
                results = client.search("polymer dissolution", max_results=3)
                for i, p in enumerate(results, 1):
                    print(f"\n{i}. {p['patent_id']}: {p['title'][:60]}...")
                    print(f"   Assignee: {p['assignee']}")
                    print(f"   Date: {p['date']}")
            else:
                print("\nConnection failed.")
        except Exception as e:
            print(f"\nError: {e}")
