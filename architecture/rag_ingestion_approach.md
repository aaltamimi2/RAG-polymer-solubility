# RAG Ingestion Approach

## Pipeline Overview

```
PDF  -->  Extract (text + figures)  -->  Figure LLM Interpretation
     -->  Chunking (recursive, hierarchical)
     -->  Parent-only LLM Enrichment (children inherit)
     -->  Filtering  -->  Embed (dense + sparse)  -->  Qdrant
```

## 1. PDF Extraction

- **Primary:** Unstructured.io (fallback: PyPDF2)
- Extracts text by page, tables, and figures with captions
- Figures saved to `./rag_figures/{paper_name}/` as PNGs
- OCR fallback via Tesseract/PaddleOCR for scanned pages

## 2. Figure Interpretation (LLM Vision Call)

Papers containing figures trigger a **Gemini 2.5 Flash** vision call per figure:

- Each figure image is sent alongside its caption and surrounding paper text
- The LLM returns: figure type, key visual elements, trends, scientific interpretation, significance
- Interpretations saved to `{paper}/interpretations.json` and appended to the document text before chunking
- Tracked as `figure_llm_calls` in the ingestion log

## 3. Chunking

Two strategies available; **Recursive Context-Aware** is the default:

| Parameter | Value |
|-----------|-------|
| Target size | 600 tokens |
| Max size | 1,000 tokens |
| Min size | 100 tokens |
| Overlap | 150 tokens |

- **Section detection** classifies chunks into Abstract, Methods, Results, Discussion, etc.
- **Parent chunks** = section-level (~2,000 tokens max)
- **Child chunks** = paragraph-level (600 token target, 150 overlap)
- Each child stores a `parent_id`; each parent stores `child_ids`
- Context headers (paper title, section path) prepended to each chunk
- Figure/table references injected into chunks that cite them

## 4. LLM Enrichment of Parent Chunks

The key cost-saving design: **LLM is called only on parent (section) chunks**; children inherit.

- **Model:** Gemini 2.0 Flash
- **Per parent chunk**, the LLM generates a 4-6 sentence (~150 word) context summary covering:
  - Main topic, key entities (polymers, solvents, chemicals)
  - Quantitative data (temperatures, concentrations)
  - Methods/techniques (FTIR, DSC, etc.)
  - Key findings and paper context
- **Child chunks** receive the parent's context prepended to their text before embedding -- no additional LLM call
- **Impact:** ~80-90% reduction in LLM costs; 35% fewer retrieval failures (67% with reranking)

**Example cost:** A paper with 4 figures and 17 sections = **21 total LLM calls**, while its 112 child chunks get context for free.

## 5. Filtering

`ScientificChunkFilter` removes low-quality chunks:

- Headers, footers, boilerplate
- Citation-heavy fragments
- Content quality checks
- Near-duplicate detection

## 6. Embedding

| Type | Model | Dimensions |
|------|-------|-----------|
| Dense | `BAAI/bge-base-en-v1.5` | 768 |
| Sparse | TF-IDF (scikit-learn) | 10,000 features, trigrams |
| Reranker | `BAAI/bge-reranker-base` | CrossEncoder (fetch 15 -> rerank to 5) |

- Query prefix: `"Represent this sentence for searching relevant passages: "`
- TF-IDF model persisted to `./rag_embeddings/tfidf_model_v2.pkl`

## 7. Vector Storage (Qdrant)

- **Location:** `./rag_qdrant_db/` (local, file-backed)
- **Hybrid search:** dense (weight 0.7) + sparse (weight 0.3)
- **Section boosting:** Abstract/Results prioritised over References
- Payloads carry full metadata: source, section type, year, journal, DOI, quality score, parent/child IDs

## 8. Metadata per Chunk

```
source, page_number, section_type, section_title,
level (document | section | paragraph),
parent_id, child_ids,
year, journal, doi, quality_score
```

## Storage Layout

```
./rag_pdfs/              # input PDFs
./rag_figures/           # extracted figure images + interpretations
./rag_data/
  chunk_store_v2.pkl     # all chunk objects
  paper_tracker.json     # per-KB paper tracking
  ingestion_log_{KB}.json
  knowledgebases.json    # KB metadata
./rag_embeddings/        # TF-IDF model
./rag_qdrant_db/         # vector DB collections
```
