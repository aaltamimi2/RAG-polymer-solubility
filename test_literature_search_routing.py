"""
Test script to verify Web of Science and Google Scholar routing in the agent
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agent components
from agent_sql_final_1212_patched import agent_graph, HumanMessage

async def test_wos_search():
    """Test Web of Science search"""
    print("\n" + "=" * 70)
    print("TEST 1: Web of Science Search (should route to search_web_of_science)")
    print("=" * 70)

    query = "Search Web of Science for peer-reviewed articles on PET dissolution"
    print(f"\nQuery: {query}\n")

    state = {
        "messages": [HumanMessage(content=query)]
    }

    config = {"configurable": {"thread_id": "test_wos"}}

    try:
        result = await agent_graph.ainvoke(state, config)

        final_message = result["messages"][-1].content
        print("Response:")
        print(final_message[:500] + "..." if len(final_message) > 500 else final_message)

        # Check if WoS was used
        if "Web of Science" in final_message or "WoS" in final_message:
            print("\n✅ SUCCESS: Correctly routed to Web of Science")
            return True
        else:
            print("\n❌ FAILED: Did not use Web of Science")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


async def test_google_scholar_search():
    """Test Google Scholar search"""
    print("\n" + "=" * 70)
    print("TEST 2: Google Scholar Search (should route to search_google_scholar)")
    print("=" * 70)

    query = "Find Google Scholar papers on Hansen solubility parameters"
    print(f"\nQuery: {query}\n")

    state = {
        "messages": [HumanMessage(content=query)]
    }

    config = {"configurable": {"thread_id": "test_scholar"}}

    try:
        result = await agent_graph.ainvoke(state, config)

        final_message = result["messages"][-1].content
        print("Response:")
        print(final_message[:500] + "..." if len(final_message) > 500 else final_message)

        # Check if Google Scholar was used
        if "Google Scholar" in final_message:
            print("\n✅ SUCCESS: Correctly routed to Google Scholar")
            return True
        else:
            print("\n❌ FAILED: Did not use Google Scholar")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


async def test_generic_search():
    """Test generic search (agent should choose appropriate tool)"""
    print("\n" + "=" * 70)
    print("TEST 3: Generic Search (agent chooses best tool)")
    print("=" * 70)

    query = "Find recent research on polymer dissolution mechanisms"
    print(f"\nQuery: {query}\n")

    state = {
        "messages": [HumanMessage(content=query)]
    }

    config = {"configurable": {"thread_id": "test_generic"}}

    try:
        result = await agent_graph.ainvoke(state, config)

        final_message = result["messages"][-1].content
        print("Response:")
        print(final_message[:500] + "..." if len(final_message) > 500 else final_message)

        # Check which tool was used
        if "Web of Science" in final_message or "WoS" in final_message:
            print("\n✅ Agent chose Web of Science")
            return True
        elif "Google Scholar" in final_message:
            print("\n✅ Agent chose Google Scholar")
            return True
        else:
            print("\n⚠️  Agent may not have used literature search")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("LITERATURE SEARCH ROUTING TEST SUITE")
    print("=" * 70)
    print("\nTesting agent's ability to route between WoS and Google Scholar")

    # Check API keys
    wos_key = os.getenv("WOS_STARTER_API_KEY")
    serp_key = os.getenv("SERPAPI_KEY")

    print(f"\n🔑 WOS_STARTER_API_KEY: {'✅ Set' if wos_key else '❌ Missing'}")
    print(f"🔑 SERPAPI_KEY: {'✅ Set' if serp_key else '❌ Missing'}")

    if not wos_key or not serp_key:
        print("\n⚠️  WARNING: Some API keys are missing. Tests may fail.")

    # Run tests
    results = []

    results.append(await test_wos_search())
    results.append(await test_google_scholar_search())
    results.append(await test_generic_search())

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

    if all(results):
        print("\n✅ ALL TESTS PASSED - Literature search routing is working correctly!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Check routing logic in agent")


if __name__ == "__main__":
    asyncio.run(main())
