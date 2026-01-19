"""
Test script for Web of Science Starter API
Demonstrates connection, search, and article extraction
"""

import os
import sys
import logging
from dotenv import load_dotenv
from wos_starter_client import WebOfScienceStarterClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_basic_connection():
    """Test basic API connection"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Connection and Authentication")
    print("=" * 70)

    try:
        client = WebOfScienceStarterClient()
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


def test_simple_search(client: WebOfScienceStarterClient):
    """Test simple search"""
    print("\n" + "=" * 70)
    print("TEST 2: Simple Search - Polymer Solubility")
    print("=" * 70)

    try:
        query = "TS=(polymer solubility)"
        print(f"\nQuery: {query}")

        results = client.search_documents(query=query, limit=5)

        # Display stats
        metadata = results.get('metadata', {})
        hits = results.get('hits', [])

        print(f"\n📊 Total results: {metadata.get('total', 0)}")
        print(f"📄 Retrieved: {len(hits)} records\n")

        # Display first 3 results
        for i, hit in enumerate(hits[:3], 1):
            article = client._parse_article(hit)
            print(f"{i}. {article['title']}")
            print(f"   Authors: {', '.join(article['authors'][:3])}...")
            print(f"   Year: {article['year']} | Journal: {article['journal']}")
            print(f"   Times Cited: {article['times_cited']}")
            print(f"   DOI: {article['doi']}\n")

        return True

    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_polymer_specific_search(client: WebOfScienceStarterClient):
    """Test polymer-specific searches"""
    print("\n" + "=" * 70)
    print("TEST 3: Polymer-Specific Article Search")
    print("=" * 70)

    try:
        # Test 1: Polyethylene
        print("\n🔍 Search: Polyethylene solubility")
        articles = client.search_polymer_articles(
            polymer_name="polyethylene",
            max_results=3
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Authors: {', '.join(article['authors'][:2])}...")
            print(f"   Year: {article['year']}\n")

        # Test 2: PET with year range
        print("\n🔍 Search: PET solubility (2020-2024)")
        articles = client.search_polymer_articles(
            polymer_name="PET",
            year_low=2020,
            year_high=2024,
            max_results=3
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Year: {article['year']}\n")

        return True

    except Exception as e:
        print(f"❌ Polymer search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hansen_parameters(client: WebOfScienceStarterClient):
    """Test Hansen solubility parameters search"""
    print("\n" + "=" * 70)
    print("TEST 4: Hansen Solubility Parameters Search")
    print("=" * 70)

    try:
        print("\n🔍 Search: Hansen solubility parameters (2020-2024)")
        articles = client.search_hansen_parameters(
            year_low=2020,
            year_high=2024,
            max_results=5
        )

        print(f"\n📊 Found {len(articles)} articles on Hansen parameters\n")

        for i, article in enumerate(articles[:3], 1):
            print(f"{i}. {article['title']}")
            print(f"   Authors: {', '.join(article['authors'][:3])}")
            print(f"   Year: {article['year']} | Journal: {article['journal']}")
            print(f"   Times Cited: {article['times_cited']}")

            if article['abstract'] != 'N/A':
                abstract_preview = str(article['abstract'])[:200] + "..."
                print(f"   Abstract: {abstract_preview}")
            print()

        return True

    except Exception as e:
        print(f"❌ Hansen search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_polymer_solvent_combination(client: WebOfScienceStarterClient):
    """Test searching for specific polymer-solvent combinations"""
    print("\n" + "=" * 70)
    print("TEST 5: Polymer-Solvent Combination Search")
    print("=" * 70)

    try:
        print("\n🔍 Search: PET + NMP")
        articles = client.search_polymer_articles(
            polymer_name="PET",
            solvent_name="NMP",
            max_results=3
        )

        print(f"📊 Found {len(articles)} articles\n")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Year: {article['year']} | Journal: {article['journal']}")
            print()

        return True

    except Exception as e:
        print(f"❌ Combination search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_api_info(client: WebOfScienceStarterClient):
    """Display API usage information"""
    print("\n" + "=" * 70)
    print("API Information")
    print("=" * 70)

    print("\n📊 WoS Starter API Details:")
    print(f"   Endpoint: {client.base_url}")
    print(f"   API Key: {client.api_key[:20]}..." if len(client.api_key) > 20 else f"   API Key: {client.api_key}")
    print("\n⚠️  Note: API key should become active within 6 hours of provisioning")
    print("   If searches fail, the key may still be activating.")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Web of Science Starter API - Test Suite")
    print("=" * 70)

    # Check for API key
    if not os.getenv("WOS_STARTER_API_KEY"):
        print("\n❌ Error: WoS Starter API key not found!")
        print("\n📋 To run this test:")
        print("1. Add your API key to .env file:")
        print("   WOS_STARTER_API_KEY=your-api-key-here")
        print("2. Run: python test_wos_starter.py")
        sys.exit(1)

    # Run tests
    client = test_basic_connection()

    if not client:
        print("\n⚠️  Connection test failed.")
        print("\n💡 Possible reasons:")
        print("   1. API key is still being activated (allow 6 hours)")
        print("   2. API endpoint or authentication method may be different")
        print("   3. Network/firewall issues")
        print("\n📚 Check documentation at: https://developer.clarivate.com/")
        sys.exit(1)

    # Display API info
    display_api_info(client)

    # Run remaining tests
    print("\n" + "=" * 70)
    print("Running Search Tests...")
    print("=" * 70)

    test_simple_search(client)
    test_polymer_specific_search(client)
    test_hansen_parameters(client)
    test_polymer_solvent_combination(client)

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("- Integrate with agent_sql_final_1212_patched.py")
    print("- Add WoS search tool alongside Google Scholar")
    print("- Update frontend to offer both literature search options")


if __name__ == "__main__":
    main()
