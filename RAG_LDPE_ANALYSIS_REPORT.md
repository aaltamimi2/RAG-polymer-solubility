# RAG Engine Analysis Report: LDPE Dissolution

## Executive Summary

This report evaluates the RAG (Retrieval-Augmented Generation) engine's ability to answer technical questions about LDPE dissolution from the STRAP-CORE knowledge base containing 2 indexed papers and 304 chunks.

---

## Test Queries and Results

### Query 1: General LDPE Dissolution Search
**Query:** "Search the indexed literature for LDPE dissolution"

**RAG Response Quality:** ✅ Good
- Found relevant passages from the review paper
- Mentioned specific solvents: ethyl acetate, toluene, gamma-valerolactone, CreaSolv
- Referenced temperature ranges (room temperature to 200°C)
- Mentioned xylene at 85°C for PE recovery from carton packaging

### Query 2: Specific Solvents and Temperatures
**Query:** "What specific solvents dissolve LDPE and at what temperatures?"

**RAG Response Quality:** ✅ Good
- **Xylene:** 85°C for PE dissolution
- **CreaSolv formulations:** Various temperatures
- Temperature range mentioned: 85°C to 220°C depending on solvent
- Mentioned supercritical butane and propane (without specific temps)

### Query 3: Recovery Efficiency
**Query:** "What is the recovery efficiency for PE dissolution?"

**RAG Response Quality:** ✅ Excellent
- Found specific quantitative data: **98.5% recovery efficiency**
- Process details: STRAP process with toluene at 110°C
- This demonstrates the RAG can extract precise numerical data

### Query 4: Hansen Solubility Parameters
**Query:** "What are Hansen solubility parameters for LDPE?"

**RAG Response Quality:** ⚠️ Partial
- Correctly identified HSP relevance to solubility prediction
- Did NOT find specific numerical HSP values for LDPE
- This indicates a gap in the indexed literature or extraction quality

---

## Dense vs Sparse Vector Search: How It Works

### Automatic Hybrid Search Architecture

The RAG system uses a **hybrid search approach** that combines both methods automatically:

```
User Query → Query Expansion → [Dense Search] + [Sparse Search] → Fusion → Reranking → Results
```

### 1. Dense Vector Search (Semantic)
- **Model:** BAAI/bge-base-en-v1.5 (768-dimensional embeddings)
- **How it works:**
  - Converts query and documents into dense numerical vectors
  - Finds documents with similar meaning even if words differ
  - Example: "LDPE dissolution" matches "polyethylene solubility"
- **Strength:** Understands synonyms, paraphrases, and conceptual similarity
- **Storage:** Qdrant vector database

### 2. Sparse Vector Search (Lexical)
- **Model:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **How it works:**
  - Matches exact keywords and their statistical importance
  - Weights rare technical terms higher than common words
  - Example: "LDPE" as exact match, "xylene" as specific solvent
- **Strength:** Precise matching for technical terms, chemical names, acronyms

### 3. Automatic Fusion
The system automatically:
1. **Expands queries** with scientific synonyms (e.g., "LDPE" → "low-density polyethylene")
2. **Runs both searches** in parallel
3. **Combines results** using Reciprocal Rank Fusion (RRF)
4. **Reranks** with BAAI/bge-reranker-base for final relevance scoring

### Why Hybrid Works Better

| Query Type | Dense Only | Sparse Only | Hybrid |
|------------|------------|-------------|--------|
| "polymer dissolution" | ✅ Good | ⚠️ Misses synonyms | ✅ Best |
| "LDPE at 85°C" | ⚠️ Fuzzy match | ✅ Exact match | ✅ Best |
| "Hansen parameters for PE" | ✅ Conceptual | ✅ Keyword | ✅ Best |

---

## RAG System Configuration

| Component | Value |
|-----------|-------|
| Knowledge Base | STRAP-CORE |
| Papers Indexed | 2 |
| Total Chunks | 304 |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| Query Expansion | Enabled |
| Contextual Enrichment | Hierarchical (parent-child) |

---

## Observations and Recommendations

### Strengths
1. ✅ Quantitative data extraction works well (98.5% recovery efficiency)
2. ✅ Solvent-temperature relationships correctly retrieved
3. ✅ Contextual enrichment provides section-level context
4. ✅ Hybrid search handles both technical terms and concepts

### Areas for Improvement
1. ⚠️ **Query expansion latency:** ~7 seconds per embedding batch
2. ⚠️ **Chunk store sync:** Only contains 1 of 2 papers (document similarity limited)
3. ⚠️ **Specific HSP values:** Not found in indexed content (may need additional papers)

### Recommended Next Steps
1. Index more papers with specific HSP data for polymers
2. Consider caching query embeddings for frequently-used terms
3. Re-ingest papers to sync chunk store with vector DB

---

## Conclusion

The RAG engine demonstrates **good retrieval quality** for technical polymer dissolution queries. The hybrid dense+sparse search is handled **automatically** with no user configuration required. The system successfully retrieves:
- Specific solvents and temperatures
- Quantitative recovery efficiencies
- Process descriptions and methodologies

The main limitation is the current corpus size (2 papers) - expanding the knowledge base will improve coverage of specific technical parameters like HSP values.

---

*Report generated: 2026-01-23*
*Knowledge Base: STRAP-CORE (2 papers, 304 chunks)*
