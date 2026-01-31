#!/usr/bin/env python3
"""
Ingest deinking/printed plastics papers into RAG system.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rag_module import RAGSystem

def main():
    print("="*70)
    print("DEINKING PAPERS RAG INGESTION")
    print("="*70)

    # Find all PDFs in the deinking directory
    pdf_dir = "/home/aaltamimi2/polymer-solubility-app/data/printed_plastics_deinking_pdfs"
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    if not pdf_paths:
        print(f"ERROR: No PDFs found in {pdf_dir}")
        return

    print(f"\nFound {len(pdf_paths)} PDFs to ingest:")
    for p in pdf_paths:
        size_kb = os.path.getsize(p) / 1024
        print(f"  - {os.path.basename(p)} ({size_kb:.1f} KB)")

    # Initialize RAG system
    print("\n1. Initializing RAG system...")
    rag = RAGSystem(auto_init=False)

    # Use existing printed_plastics_deinking KB (already created, just needs papers)
    kb_name = "printed_plastics_deinking"
    print(f"\n2. Switching to existing {kb_name} knowledgebase...")
    try:
        rag.switch_kb(kb_name)
        print(f"   Switched to KB: {kb_name}")
    except Exception as e:
        print(f"   KB doesn't exist, creating it...")
        kb = rag.create_kb(kb_name, "Literature on deinking, ink removal, and printed plastics recycling", switch_to=True)
        print(f"   Created KB: {kb.name} (collection: {kb.collection_name})")

    print(f"   Active KB: {rag.get_active_kb()}")

    # Verify STRAP-CORE is safe
    print("\n   Verifying STRAP-CORE integrity...")
    strap_info = rag.kb_manager.get_kb("STRAP-CORE")
    if strap_info:
        print(f"   STRAP-CORE: {strap_info.paper_count} papers, {strap_info.chunk_count} chunks (PROTECTED)")

    # Check current deinking KB status
    print(f"\n3. Checking current {kb_name} status...")
    deinking_info = rag.kb_manager.get_kb(kb_name)
    if deinking_info:
        print(f"   Current state: {deinking_info.paper_count} papers, {deinking_info.chunk_count} chunks")

    # Force re-ingestion to clear out invalid figure chunks
    needs_ingestion = pdf_paths
    print(f"\n   Re-ingesting all {len(needs_ingestion)} papers (clearing invalid figure chunks):")
    for p in needs_ingestion:
        print(f"     - {os.path.basename(p)}")

    # Clear old figure interpretations so they get regenerated with new filtering
    import shutil
    rag_figures_dir = "./rag_figures"
    if os.path.exists(rag_figures_dir):
        for paper_path in needs_ingestion:
            paper_name = os.path.splitext(os.path.basename(paper_path))[0][:45]
            figure_dir = os.path.join(rag_figures_dir, paper_name)
            if os.path.exists(figure_dir):
                shutil.rmtree(figure_dir)
                print(f"     Cleared old figures: {paper_name}")

    # Run ingestion
    print("\n4. Running ingestion pipeline (recreate=True to clear invalid chunks)...")
    print("   - Chunking strategy: recursive")
    print("   - Contextual enrichment: enabled (hierarchical)")
    print("   - Figure interpretation: enabled")
    print()

    result = rag.ingest_pdfs(
        pdf_paths=needs_ingestion,
        use_ocr=False,
        recreate_collection=True,  # Recreate to clear invalid figure chunks
        interpret_figures=True,
        chunking_strategy="recursive",
        use_contextual_enrichment=True,
        incremental=False  # Force re-ingestion
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

        print(f"\n   Collection info:")
        coll_info = result.get('collection_info', {})
        print(f"     - Status: {coll_info.get('status')}")
        print(f"     - Points: {coll_info.get('points_count')}")

    else:
        print(f"\n   Status: FAILED")
        print(f"   Error: {result.get('error')}")

    # Test search
    print("\n" + "="*70)
    print("SEARCH TEST")
    print("="*70)

    if rag.is_ready():
        queries = [
            "deinking printed plastics",
            "surfactant ink removal",
            "multilayer packaging recycling"
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
    print("INGESTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
