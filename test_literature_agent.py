"""
Test Literature Agent Implementation

Tests:
1. KB auto-selection based on query content
2. Literature search via collaborative path
3. Multi-KB search support
4. Collaboration with separation agent
"""
import asyncio
import time
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)

# Test queries
TEST_QUERIES = [
    {
        "name": "Deinking literature search",
        "query": "What does the literature say about surfactants for deinking printed PE films?",
        "expected_path": "specialist",
        "expected_specialist": "literature",
        "expected_kbs": ["printed_plastics_deinking"],
        "category": "literature",
    },
    {
        "name": "STRAP solubility literature",
        "query": "Search the literature for Hansen solubility parameters of polyethylene",
        "expected_path": "specialist",
        "expected_specialist": "literature",
        "expected_kbs": ["STRAP-CORE"],
        "category": "literature",
    },
    {
        "name": "Separation + Literature integration",
        "query": "Find research papers about separating LDPE and PET from multilayer films",
        "expected_path": "integrated",
        "expected_specialist": None,
        "expected_kbs": ["STRAP-CORE"],
        "category": "integrated",
    },
]


def test_kb_selection():
    """Test the KB auto-selection logic."""
    from multi_agent_system import select_knowledgebases

    print("\n" + "="*60)
    print("TEST: KB Auto-Selection Logic")
    print("="*60)

    test_cases = [
        ("surfactants for deinking printed PE", ["printed_plastics_deinking"]),
        ("Hansen parameters for polymer dissolution", ["STRAP-CORE"]),
        ("flexographic ink removal from LDPE", ["printed_plastics_deinking", "STRAP-CORE"]),
        ("random unrelated query", ["STRAP-CORE", "printed_plastics_deinking"]),  # Both as default
    ]

    passed = 0
    for query, expected in test_cases:
        result = select_knowledgebases(query)
        # Check if expected KBs are at the top of the result
        match = all(kb in result[:len(expected)] for kb in expected)
        status = "✓" if match else "✗"
        print(f"  [{status}] '{query[:50]}...'")
        print(f"      Expected: {expected}")
        print(f"      Got: {result}")
        if match:
            passed += 1

    print(f"\n  KB Selection: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_routing():
    """Test that queries route correctly."""
    from multi_agent_system import enhanced_complexity_router

    print("\n" + "="*60)
    print("TEST: Routing Logic")
    print("="*60)

    passed = 0
    for test in TEST_QUERIES:
        decision = enhanced_complexity_router(test["query"])
        path_ok = decision.path == test["expected_path"]
        specialist_ok = decision.specialist == test.get("expected_specialist")

        status = "✓" if path_ok and specialist_ok else "✗"
        print(f"  [{status}] {test['name']}")
        print(f"      Path: {decision.path} (expected {test['expected_path']})")
        print(f"      Specialist: {decision.specialist} (expected {test.get('expected_specialist')})")
        print(f"      Collaboration: {decision.collaboration_specialists}")

        if path_ok and specialist_ok:
            passed += 1

    print(f"\n  Routing: {passed}/{len(TEST_QUERIES)} passed")
    return passed == len(TEST_QUERIES)


async def test_literature_search():
    """Test actual literature search through the multi-agent graph."""
    print("\n" + "="*60)
    print("TEST: Literature Agent E2E (single query)")
    print("="*60)

    from agent_sql_final_1212_patched import multi_agent_graph
    from langchain_core.messages import HumanMessage

    query = "What does the literature say about surfactants for deinking printed plastics?"
    print(f"  Query: {query}")

    state = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": f"lit_test_{int(time.time())}"}, "recursion_limit": 25}

    start = time.time()
    try:
        result = await asyncio.wait_for(
            multi_agent_graph.ainvoke(state, config),
            timeout=90.0
        )
        elapsed = time.time() - start

        path = result.get("path", "unknown")
        literature_results = result.get("literature_results", {})
        messages = result.get("messages", [])

        print(f"\n  Results ({elapsed:.2f}s):")
        print(f"    Path: {path}")

        if literature_results:
            papers = literature_results.get("papers_found", 0)
            kbs = literature_results.get("knowledgebases_searched", [])
            findings = literature_results.get("key_findings", [])
            confidence = literature_results.get("confidence_score", 0)

            print(f"    Papers found: {papers}")
            print(f"    KBs searched: {kbs}")
            print(f"    Key findings: {len(findings)}")
            print(f"    Confidence: {confidence:.2f}")

            if findings:
                print(f"\n    Sample findings:")
                for f in findings[:3]:
                    print(f"      - {f[:100]}...")
        else:
            print("    [!] No literature_results in state")

        # Get final response
        if messages:
            final = messages[-1]
            content = final.content if hasattr(final, 'content') else str(final)
            print(f"\n    Response preview ({len(content)} chars):")
            print(f"    {content[:500]}...")

        return True

    except asyncio.TimeoutError:
        print(f"  TIMEOUT after {time.time() - start:.2f}s")
        return False
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("="*60)
    print("LITERATURE AGENT TEST SUITE")
    print("="*60)

    results = []

    # Unit tests
    results.append(("KB Selection", test_kb_selection()))
    results.append(("Routing", test_routing()))

    # Integration test
    results.append(("Literature E2E", await test_literature_search()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {total_passed}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
