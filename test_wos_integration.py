"""
Test script for Web of Science API integration
Demonstrates article search and extraction functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv
from wos_api_client import WebOfScienceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_basic_connection():
    """Test basic authentication and connection"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Connection and Authentication")
    print("=" * 70)

    try:
        client = WebOfScienceClient()
        success = client.test_connection()

        if success:
            print("✅ Connection successful!")
            return client
        else:
            print("❌ Connection failed!")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_simple_search(client: WebOfScienceClient):
    """Test simple topic search"""
    print("\n" + "=" * 70)
    print("TEST 2: Simple Topic Search")
    print("=" * 70)

    try:
        query = "TS=(polymer solubility)"
        print(f"\nQuery: {query}")

        results = client.search_articles(query=query, count=5)

        # Display basic stats
        if 'Data' in results:
            total = results['Data'].get('total', 0)
            print(f"\n📊 Total results: {total}")

            records = results['Data'].get('Records', {}).get('records', {}).get('REC', [])
            print(f"📄 Retrieved: {len(records)} records\n")

            # Display first result
            if records:
                first = client._parse_article_record(records[0])
                print("First result:")
                print(f"  Title: {first['title']}")
                print(f"  Authors: {', '.join(first['authors'][:3])}...")
                print(f"  Year: {first['year']}")
                print(f"  Source: {first['source']}")
                print(f"  DOI: {first['doi']}")

        return True

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False


def test_polymer_search(client: WebOfScienceClient):
    """Test polymer-specific search"""
    print("\n" + "=" * 70)
    print("TEST 3: Polymer Solubility Article Search")
    print("=" * 70)

    try:
        # Test 1: General polymer solubility
        print("\n🔍 Search: General polymer solubility (2020-2024)")
        articles = client.search_polymer_solubility_articles(
            year_range="2020-2024",
            max_results=3
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Authors: {', '.join(article['authors'][:2])}...")
            print(f"   Year: {article['year']} | DOI: {article['doi']}\n")

        # Test 2: Specific polymer
        print("\n🔍 Search: Polyethylene solubility")
        articles = client.search_polymer_solubility_articles(
            polymer_name="polyethylene",
            max_results=3
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Year: {article['year']}\n")

        return True

    except Exception as e:
        print(f"❌ Polymer search failed: {e}")
        return False


def test_hansen_parameters_search(client: WebOfScienceClient):
    """Test Hansen solubility parameters search"""
    print("\n" + "=" * 70)
    print("TEST 4: Hansen Solubility Parameters Search")
    print("=" * 70)

    try:
        query = 'TS=("Hansen solubility parameters") AND PY=(2020-2024)'
        print(f"\nQuery: {query}")

        results = client.search_articles(query=query, count=5, sort_field='TC+D')

        if 'Data' in results:
            records = results['Data'].get('Records', {}).get('records', {}).get('REC', [])
            print(f"\n📊 Found {len(records)} articles on Hansen parameters\n")

            for i, record in enumerate(records[:3], 1):
                article = client._parse_article_record(record)
                print(f"{i}. {article['title']}")
                print(f"   Authors: {', '.join(article['authors'][:3])}")
                print(f"   Year: {article['year']} | Source: {article['source']}")
                if article['abstract'] != 'N/A':
                    abstract_preview = article['abstract'][:150] + "..."
                    print(f"   Abstract: {abstract_preview}")
                print()

        return True

    except Exception as e:
        print(f"❌ Hansen search failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Web of Science API Integration - Test Suite")
    print("=" * 70)

    # Check credentials
    if not os.getenv("WOS_CLIENT_ID") or not os.getenv("WOS_CLIENT_SECRET"):
        print("\n❌ Error: WoS credentials not found!")
        print("\n📋 To run this test:")
        print("1. Get your credentials from: https://developer.clarivate.com/")
        print("2. Add to .env file:")
        print("   WOS_CLIENT_ID=your-client-id")
        print("   WOS_CLIENT_SECRET=your-client-secret")
        print("3. Run: python test_wos_integration.py")
        sys.exit(1)

    # Run tests
    client = test_basic_connection()

    if not client:
        print("\n❌ Cannot proceed without valid connection")
        sys.exit(1)

    # Run remaining tests
    test_simple_search(client)
    test_polymer_search(client)
    test_hansen_parameters_search(client)

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("- Integrate with agent_sql_final_1212_patched.py")
    print("- Add article search tools to the agent")
    print("- Create UI components for article browsing")


if __name__ == "__main__":
    main()
