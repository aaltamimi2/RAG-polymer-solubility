"""
Test script for Google Scholar API integration using SerpAPI
Demonstrates article search and extraction functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv
from serpapi_scholar_client import GoogleScholarClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_basic_connection():
    """Test basic connection and API key validity"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Connection and API Key Validation")
    print("=" * 70)

    try:
        client = GoogleScholarClient()
        success = client.test_connection()

        if success:
            print("✅ Connection successful!")

            # Get account info
            account = client.get_account_info()
            if account:
                print(f"\n📊 Account Information:")
                print(f"   Email: {account.get('account_email', 'N/A')}")
                print(f"   Plan: {account.get('plan', 'N/A')}")
                print(f"   Searches this month: {account.get('this_month_usage', 0)}")
                print(f"   Total searches left: {account.get('total_searches_left', 'Unlimited')}")

            return client
        else:
            print("❌ Connection failed!")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_simple_search(client: GoogleScholarClient):
    """Test simple search query"""
    print("\n" + "=" * 70)
    print("TEST 2: Simple Search - 'Polymer Solubility'")
    print("=" * 70)

    try:
        results = client.search(query="polymer solubility", num_results=5)

        # Display search metadata
        search_meta = results.get('search_metadata', {})
        print(f"\nSearch ID: {search_meta.get('id', 'N/A')}")
        print(f"Status: {search_meta.get('status', 'N/A')}")
        print(f"Processing time: {search_meta.get('total_time_taken', 'N/A')}s")

        # Display results
        organic_results = results.get('organic_results', [])
        print(f"\n📊 Found {len(organic_results)} results\n")

        for i, result in enumerate(organic_results[:3], 1):
            article = client._parse_article(result)
            print(f"{i}. {article['title']}")
            print(f"   Authors: {', '.join(article['authors'][:3])}")
            print(f"   Year: {article['year']} | Cited by: {article['cited_by_count']}")
            print(f"   Link: {article['link'][:60]}...")
            if article['pdf_link']:
                print(f"   📄 PDF Available")
            print()

        return True

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False


def test_polymer_search(client: GoogleScholarClient):
    """Test polymer-specific searches"""
    print("\n" + "=" * 70)
    print("TEST 3: Polymer Solubility Article Search")
    print("=" * 70)

    try:
        # Test 1: General polymer solubility (recent)
        print("\n🔍 Search: Recent polymer solubility articles (2020-2024)")
        articles = client.search_polymer_articles(
            year_low=2020,
            year_high=2024,
            max_results=3
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Authors: {', '.join(article['authors'][:2])}")
            print(f"   Year: {article['year']} | Citations: {article['cited_by_count']}")
            print(f"   Snippet: {article['snippet'][:100]}...")
            print()

        # Test 2: Specific polymer-solvent pair
        print("\n🔍 Search: Cellulose acetate + Acetone")
        articles = client.search_polymer_articles(
            polymer_name="cellulose acetate",
            solvent_name="acetone",
            max_results=3
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Year: {article['year']}")
            print()

        return True

    except Exception as e:
        print(f"❌ Polymer search failed: {e}")
        return False


def test_hansen_parameters(client: GoogleScholarClient):
    """Test Hansen solubility parameters search"""
    print("\n" + "=" * 70)
    print("TEST 4: Hansen Solubility Parameters Research")
    print("=" * 70)

    try:
        # General Hansen search
        print("\n🔍 Search: Hansen solubility parameters (2020-2024)")
        articles = client.search_hansen_parameters(
            year_low=2020,
            year_high=2024,
            max_results=5
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Authors: {', '.join(article['authors'][:2])}")
            print(f"   Year: {article['year']} | Citations: {article['cited_by_count']}")
            if article['pdf_link']:
                print(f"   📄 PDF: {article['pdf_link'][:60]}...")
            print()

        # Specific polymer Hansen search
        print("\n🔍 Search: Hansen parameters for Polyethylene")
        articles = client.search_hansen_parameters(
            polymer_name="polyethylene",
            year_low=2015,
            max_results=3
        )

        print(f"📊 Found {len(articles)} polyethylene-specific articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']} ({article['year']})")

        return True

    except Exception as e:
        print(f"❌ Hansen search failed: {e}")
        return False


def test_author_search(client: GoogleScholarClient):
    """Test author-specific search"""
    print("\n" + "=" * 70)
    print("TEST 5: Author Search")
    print("=" * 70)

    try:
        # Search for Charles M. Hansen (pioneer of Hansen parameters)
        print("\n🔍 Search: Articles by Charles Hansen")
        articles = client.get_author_articles(
            author_name="Charles Hansen",
            max_results=5
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles[:3], 1):
            print(f"{i}. {article['title']}")
            print(f"   Year: {article['year']} | Citations: {article['cited_by_count']}")
            print()

        return True

    except Exception as e:
        print(f"❌ Author search failed: {e}")
        return False


def test_advanced_search(client: GoogleScholarClient):
    """Test advanced search features"""
    print("\n" + "=" * 70)
    print("TEST 6: Advanced Search Features")
    print("=" * 70)

    try:
        # Exact phrase search
        print("\n🔍 Advanced: Exact phrase + year filter")
        results = client.search(
            query='"Hansen solubility parameters" polymer',
            num_results=5,
            year_low=2020,
            sort_by='date'  # Sort by most recent
        )

        organic = results.get('organic_results', [])
        print(f"📊 Found {len(organic)} recent articles with exact phrase\n")

        for i, result in enumerate(organic[:3], 1):
            article = client._parse_article(result)
            print(f"{i}. {article['title']}")
            print(f"   Year: {article['year']}")
            print(f"   Publication: {article['publication_info'][:70]}...")
            print()

        return True

    except Exception as e:
        print(f"❌ Advanced search failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Google Scholar API Integration - Test Suite (SerpAPI)")
    print("=" * 70)

    # Check API key
    if not os.getenv("SERPAPI_KEY"):
        print("\n❌ Error: SerpAPI key not found!")
        print("\n📋 To run this test:")
        print("1. Get your API key from: https://serpapi.com/")
        print("   - Free tier: 100 searches/month")
        print("   - Sign up with: aaltamimi2@wisc.edu")
        print("\n2. Add to .env file:")
        print("   SERPAPI_KEY=your-api-key")
        print("\n3. Run: python test_scholar_integration.py")
        sys.exit(1)

    # Run tests
    client = test_basic_connection()

    if not client:
        print("\n❌ Cannot proceed without valid connection")
        sys.exit(1)

    # Run remaining tests
    test_simple_search(client)
    test_polymer_search(client)
    test_hansen_parameters(client)
    test_author_search(client)
    test_advanced_search(client)

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("- Integrate with agent_sql_final_1212_patched.py")
    print("- Add Google Scholar search tools to the agent")
    print("- Create UI components for article browsing")
    print("- Combine with Web of Science API when available")

    # Show remaining searches
    account = client.get_account_info()
    if account:
        searches_left = account.get('total_searches_left', 'Unknown')
        print(f"\n📊 Searches remaining this month: {searches_left}")


if __name__ == "__main__":
    main()
