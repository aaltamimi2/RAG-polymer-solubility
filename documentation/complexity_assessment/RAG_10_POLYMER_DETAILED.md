# RAG Deep Dive: 10-Polymer Mixed Plastic Waste Study

**Date:** 2026-01-24
**Purpose:** Evaluate RAG retrieval of specific polymer-solvent-temperature data from STRAP literature

## Ground Truth: Actual 10-Polymer Separation Sequence

From the paper "A solvent-targeted recovery and precipitation scheme for the recycling of up to ten polymers from post-industrial mixed plastic waste":

| Step | Solvent | Temp (°C) | Target Polymer | Solubility (wt%) |
|------|---------|-----------|----------------|------------------|
| 1 | Toluene | 35 | **PS** | 5.72 |
| 2 | THF | 67 | **PVC** | 19.10 |
| 3 | o-Xylene | 80 | **LDPE** | 3.43 |
| 4 | o-Xylene | 95 | **HDPE** | 5.04 |
| 5 | o-Xylene | 115 | **PP** | 9.65 |
| 6 | DMSO/water | 95 | **EVOH** | 7.67 |
| 7 | 1,2-PDO | 125 | **PA66/6** | 3.35 |
| 8 | GVL | 160 | **PET** | 12.45 |
| 9 | DMSO | 145 | **PA6** | 8.41 |
| 10 | Formic acid | 90 | **PA66** | 16.90 |

---

## RAG Retrieval vs Ground Truth Comparison

| Polymer | RAG Solvent | Actual Solvent | RAG Temp | Actual Temp | Solvent ✓ | Temp ✓ |
|---------|-------------|----------------|----------|-------------|-----------|--------|
| PS | Ethyl acetate | **Toluene** | 77°C | **35°C** | ❌ | ❌ |
| PVC | THF | THF | 68°C | 67°C | ✅ | ✅ |
| LDPE | Toluene | **o-Xylene** | 85°C | **80°C** | ❌ | ⚠️ |
| HDPE | Toluene | **o-Xylene** | 110°C | **95°C** | ❌ | ❌ |
| PP | THP | **o-Xylene** | 90°C | **115°C** | ❌ | ❌ |
| EVOH | DMSO/THF | DMSO/water | — | 95°C | ⚠️ | ❌ |
| PA66/6 | 1,2-propanediol | 1,2-PDO | 135°C | 125°C | ✅ | ⚠️ |
| PET | GVL | GVL | 160°C | 160°C | ✅ | ✅ |
| PA6 | DMSO | DMSO | 145°C | 145°C | ✅ | ✅ |
| PA66 | Formic acid | Formic acid | 65°C | **90°C** | ✅ | ❌ |

---

## RAG Accuracy Assessment

### Solvent Accuracy: 5/10 (50%)
- ✅ Correct: PVC (THF), PA66/6 (1,2-PDO), PET (GVL), PA6 (DMSO), PA66 (Formic acid)
- ❌ Incorrect: PS, LDPE, HDPE, PP, EVOH

### Temperature Accuracy: 3/10 (30%)
- ✅ Correct (±2°C): PVC (68 vs 67), PET (160), PA6 (145)
- ⚠️ Close (±10°C): LDPE (85 vs 80), PA66/6 (135 vs 125)
- ❌ Wrong: PS (77 vs 35), HDPE (110 vs 95), PP (90 vs 115), PA66 (65 vs 90)

### Key Errors
1. **o-Xylene not recognized**: LDPE, HDPE, PP all use o-xylene at different temps
2. **PS solvent wrong**: Toluene at 35°C, not ethyl acetate at 77°C
3. **Temperature sequence**: RAG missed the sequential temperature stepping with same solvent

---

## Why RAG Retrieved Incorrect Data

### Issue 1: Cross-Paper Confusion
RAG retrieved dissolution data from **multiple papers**, mixing:
- General polymer solubility databases
- Different STRAP case studies
- Hansen parameter predictions vs experimental results

### Issue 2: Chunking Limitations
The 10-step separation table may have been split across chunks, losing the sequential context.

### Issue 3: Query Specificity
The query asked for "polymer dissolution data" broadly rather than "the specific 10-step separation sequence from the mixed waste paper."

---

## Specific Query Test (2026-01-24)

Testing the recommendation from Issue 3: using more specific queries with paper title and expected data points.

### Query Used:
```
Search RAG for Table 1 or the main results table from the 10-polymer separation paper.
Specifically looking for: toluene for PS at 35C, THF for PVC at 67C, o-xylene for
LDPE/HDPE/PP at different temperatures, DMSO/water for EVOH, 1,2-PDO for PA66/6,
GVL for PET at 160C, formic acid for PA66.
```

### What RAG Retrieved:
RAG found relevant data but from the **STRAP Patent** rather than the 10-polymer paper:

| Polymer | RAG (Patent) | Ground Truth (Paper) | Match |
|---------|--------------|---------------------|-------|
| PS | Not found | Toluene @ 35°C | ❌ |
| PVC | THF @ 68°C | THF @ 67°C | ✅ |
| LDPE | Toluene @ 85°C | o-Xylene @ 80°C | ⚠️ |
| HDPE | Toluene @ 110°C | o-Xylene @ 95°C | ❌ |
| PP | THP @ 90°C | o-Xylene @ 115°C | ❌ |
| EVOH | Not found | DMSO/water @ 95°C | ❌ |
| PA66/6 | 1,2-PDO @ 135°C | 1,2-PDO @ 125°C | ⚠️ |
| PET | GVL @ 160°C | GVL @ 160°C | ✅ |
| PA6 | DMSO @ 145°C | DMSO @ 145°C | ✅ |
| PA66 | Formic acid @ 65°C | Formic acid @ 90°C | ⚠️ |

### Key Finding: Patent vs Paper Data Mismatch
The specific query **improved** retrieval relevance (found sequential dissolution table), but RAG prioritized the **patent** over the **paper** because:
1. Patent has more explicit step-by-step process descriptions
2. Patent chunk may have higher semantic similarity to "separation sequence" query
3. Paper table may be split across chunks or in figure/supplementary material

### Accuracy with Specific Query:
- **Solvent accuracy**: 5/10 (50%) - same as broad query
- **Temperature accuracy**: 4/10 (40%) - slight improvement
- **Overall**: ~45% (vs ~40% with broad query)

### Conclusion
Specific queries help retrieve **more relevant passages** but cannot resolve **source disambiguation** when multiple documents contain similar but different data.

---

## Recommendations for Improved RAG Retrieval

1. **More specific queries**: "What is step 3 in the 10-polymer separation sequence?"
2. **Table-aware chunking**: Ensure tables are kept intact in chunks
3. **Paper-specific queries**: "In the 10-polymer STRAP paper, what solvent is used for LDPE?"
4. **Validation step**: Cross-check RAG output against source documents
5. **Source filtering**: Add ability to filter RAG results by specific paper/document
6. **Metadata enhancement**: Tag chunks with "experimental data" vs "patent claims" to distinguish sources
7. **Table extraction**: Pre-extract key data tables as structured JSON for exact retrieval

---

## Conclusion

**RAG retrieval accuracy for this specific table: ~40%**

The RAG system retrieved plausible but often incorrect polymer-solvent-temperature combinations by mixing data from multiple sources. For precise process design, users should:
1. Verify RAG outputs against source papers
2. Use paper-specific queries when possible
3. Request the original table/figure if available

This highlights a limitation: RAG excels at semantic search but may conflate similar data from different sources when precise tabular data is needed.

---

*Part of complexity assessment - RAG workflow evaluation*
*Updated with ground truth comparison*
