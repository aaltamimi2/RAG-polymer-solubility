#!/usr/bin/env python3
"""
Test full ingestion pipeline with STRAP paper.

Verifies:
1. Hierarchical enrichment (LLM for parents only)
2. Knowledgebase management
3. Paper tracking
4. LLM call logging
"""

import os
import sys
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

print(f"GOOGLE_API_KEY set: {'Yes' if os.environ.get('GOOGLE_API_KEY') else 'No'}")

from rag_module import RAGSystem

def main():
    print("="*70)
    print("FULL INGESTION TEST - STRAP Paper")
    print("="*70)

    # Test PDF path
    pdf_path = "/home/aaltamimi2/LLM-test/papers/STRAP/Recycling of multilayer plastic packaging materials bysolvent-targeted recovery and precipitation.pdf"

    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found: {pdf_path}")
        return

    print(f"\nTest PDF: {os.path.basename(pdf_path)}")
    print(f"Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")

    # Initialize RAG system
    print("\n1. Initializing RAG system...")
    rag = RAGSystem(auto_init=False)

    # Create STRAP-CORE knowledgebase
    print("\n2. Creating STRAP-CORE knowledgebase...")
    try:
        kb = rag.create_kb("STRAP-CORE", "Core STRAP recycling papers", switch_to=True)
        print(f"   Created KB: {kb.name} (collection: {kb.collection_name})")
    except ValueError as e:
        print(f"   KB already exists, switching to it...")
        rag.switch_kb("STRAP-CORE")

    print(f"   Active KB: {rag.get_active_kb()}")

    # Check if paper is already ingested
    print("\n3. Checking paper status...")
    if rag.is_paper_ingested(pdf_path):
        print("   Paper already ingested! Clearing for fresh test...")
        # For testing, we'll recreate the collection
        recreate = True
    else:
        print("   Paper not yet ingested")
        recreate = False

    # Run ingestion
    print("\n4. Running ingestion pipeline...")
    print("   - Chunking strategy: recursive")
    print("   - Contextual enrichment: enabled (hierarchical)")
    print("   - Figure interpretation: enabled")
    print()

    result = rag.ingest_pdfs(
        pdf_paths=[pdf_path],
        use_ocr=False,
        recreate_collection=True,  # Fresh test
        interpret_figures=True,
        chunking_strategy="recursive",
        use_contextual_enrichment=True,
        incremental=False  # Force re-ingestion for testing
    )

    # Print results
    print("\n" + "="*70)
    print("INGESTION RESULTS")
    print("="*70)

    if result.get('success'):
        print(f"\n   Status: SUCCESS")
        print(f"   KB: {result.get('kb_name')}")
        print(f"\n   Files:")
        print(f"     - Processed: {len(result.get('processed_files', []))}")
        print(f"     - Skipped: {len(result.get('skipped_files', []))}")
        print(f"     - Failed: {len(result.get('failed_files', []))}")

        print(f"\n   Chunks:")
        print(f"     - Total: {result.get('total_chunks')}")
        print(f"     - Indexed: {result.get('indexed_chunks')}")
        print(f"     - By level: {result.get('chunks_by_level', {})}")

        llm_stats = result.get('llm_stats', {})
        print(f"\n   LLM API Calls:")
        print(f"     - Total: {llm_stats.get('total_llm_calls', 0)}")
        print(f"     - Figure interpretation: {llm_stats.get('figure_interpretation_calls', 0)}")
        print(f"     - Parent context: {llm_stats.get('parent_context_calls', 0)}")

        # Per-paper breakdown
        per_paper = llm_stats.get('per_paper', [])
        if per_paper:
            print(f"\n   Per-paper breakdown:")
            for p in per_paper:
                print(f"     - {p['source_name']}:")
                print(f"         Sections: {p.get('section_chunks', 0)}")
                print(f"         Paragraphs: {p.get('paragraph_chunks', 0)}")
                print(f"         Figure LLM calls: {p.get('figure_llm_calls', 0)}")
                print(f"         Parent LLM calls: {p.get('parent_context_llm_calls', 0)}")
                print(f"         Total LLM calls: {p.get('total_llm_calls', 0)}")

        print(f"\n   Collection info:")
        coll_info = result.get('collection_info', {})
        print(f"     - Status: {coll_info.get('status')}")
        print(f"     - Points: {coll_info.get('points_count')}")

        print(f"\n   Ingestion log: {result.get('ingestion_log')}")

        # Verify hierarchical enrichment worked
        print("\n" + "="*70)
        print("HIERARCHICAL ENRICHMENT VERIFICATION")
        print("="*70)

        parent_calls = llm_stats.get('parent_context_calls', 0)
        total_chunks = result.get('total_chunks', 0)
        indexed_chunks = result.get('indexed_chunks', 0)

        if parent_calls > 0 and parent_calls < indexed_chunks:
            savings = (1 - parent_calls / indexed_chunks) * 100
            print(f"\n   Hierarchical enrichment WORKING!")
            print(f"   - Parent contexts generated: {parent_calls}")
            print(f"   - Child chunks (inherit): {indexed_chunks - parent_calls}")
            print(f"   - LLM call savings: {savings:.1f}%")
        else:
            print(f"\n   WARNING: Check hierarchical enrichment")
            print(f"   - Parent calls: {parent_calls}")
            print(f"   - Indexed chunks: {indexed_chunks}")

    else:
        print(f"\n   Status: FAILED")
        print(f"   Error: {result.get('error')}")

    # Test search
    print("\n" + "="*70)
    print("SEARCH TEST")
    print("="*70)

    if rag.is_ready():
        queries = [
            "polymer dissolution solvent",
            "STRAP recycling process",
            "polystyrene recovery"
        ]

        for query in queries:
            print(f"\n   Query: '{query}'")
            results = rag.search(query, top_k=2)
            for i, r in enumerate(results):
                print(f"     {i+1}. [{r.section_type}] Score: {r.score:.3f}")
                print(f"        {r.text[:100]}...")
    else:
        print("   RAG system not ready for search")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    # Show ingestion log location
    print(f"\nCheck ingestion log at: ./rag_data/ingestion_log_STRAP-CORE.json")


if __name__ == "__main__":
    main()
