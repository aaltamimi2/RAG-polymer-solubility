# Ingestion Quality Report (WITH LLM ENRICHMENT)

**Paper:** Pigment Removal from Reverse-Printed Laminated Flexible Films by Solvent-Targeted Recovery and Precipitation

**Date:** 2026-01-23
**Knowledge Base:** STRAP-CORE

---

## Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Chunks Created | 189 | Good coverage |
| Chunks Indexed | 176 | 93% indexed |
| LLM API Calls | 22 | Full enrichment |
| Figure Interpretations | 9 | All captioned figures analyzed |
| Parent Contexts | 13 | Hierarchical enrichment |
| Avg Chunk Size | 307 tokens | Rich context |

---

## LLM Enrichment Details

### Figure Interpretation (9 LLM calls)
All 9 captioned figures were analyzed by Gemini 2.0 Flash:

| Figure | Type | Key Data Extracted |
|--------|------|-------------------|
| Fig 1 | Process Flowchart | STRAP process: dissolution → hot filtration → hot adsorption → precipitation → drying → extrusion |
| Fig 2 | Photographs | PE recovery from PIW: toluene/heptane/dodecane comparison; yellow cast film issue |
| Fig 3 | Composite Photos | LDPE films with adhesive/ink: 8 formulations, before/after STRAP, filter mesh residues |
| Fig 4 | Chemical Structures | Dye solubility: Red 57:1, Orange 5/13, Yellow 1/12/110, Blue 15:4, Green 7, Violet 19/23 |
| Fig 5 | UV-Vis + Plots | Yellow 12 dissolution: absorbance vs concentration/time; COSMO-RS solubility predictions |
| Fig 6 | Scatter Plots | Mechanical deliquoring: pressure vs solvent retention; solvent retention vs yellowness |
| Fig 7 | Photos + Plot | AC adsorption: Yellow 12/Yellow 1 removal; 100% removal at 0.8-1.0 g/L AC loading |
| Fig 8 | Bar Chart + Photos | Hot adsorption: 600mg AC + 18h → YI -6.5 (best); UV-Vis correlation with yellowness |
| Fig 9 | Composite | PIW 2 processing: vacuum vs piston filtration; colorant ppm vs Yellow Index |

**Interpretation Quality: 9/10** - Detailed analysis with quantitative extraction

### Hierarchical Contextual Enrichment (13 LLM calls)
- 13 parent (section-level) chunks enriched with semantic context
- 176 child (paragraph-level) chunks inherit parent context
- **Cost savings: 93.1%** (13 LLM calls vs 189 if all chunks enriched individually)

---

## Key Quantitative Data Extracted

### Pigment Solubility Data
| Pigment | Solvent | Temperature | Solubility |
|---------|---------|-------------|------------|
| Yellow 12 | Dodecane | 110°C | ~0.06 g/L |
| Yellow 12 | Toluene | 110°C | ~1.0 g/L |
| Yellow 1 | Toluene | 110°C | ~2.5 g/L |
| Yellow 1 | Toluene | 140°C | ~4.7 g/L |

### Mechanical Deliquoring Results
| Method | Pressure | Solvent/PE Ratio |
|--------|----------|------------------|
| Vacuum | 0 bar | ~3.0 g/g |
| Piston | 1 bar | ~2.4 g/g |
| Piston | 6.5 bar | ~1.3 g/g |

### Activated Carbon Adsorption
| Dye | AC Loading | Removal Efficiency |
|-----|------------|-------------------|
| Yellow 12 | 0.8 g/L | ~98% |
| Yellow 12 | 1.0 g/L | ~100% |
| Yellow 1 | 4.0 g/L | ~98% |
| Yellow 1 | 6.0 g/L | ~99% |

### Hot Adsorption Treatment
| Treatment | AC Amount | Time | PE YI |
|-----------|-----------|------|-------|
| V4 (control) | None | - | -3.0 |
| V1 | 300 mg | 4 hr | -5.1 |
| V3 | 600 mg | 4 hr | -5.3 |
| V2 | 600 mg | 18 hr | -6.5 |

---

## Chunking Quality

```
Total Chunks:    189 (13 parents, 176 children)
Indexed:         176 (0 duplicates)
Avg Chunk Size:  307 tokens (enriched)

Split Methods:
  - Semantic:    140 chunks (79%)  ← Excellent semantic coherence
  - Sentence:    34 chunks (19%)
  - None:        2 chunks (1%)
```

---

## Sample Figure Interpretation

**Figure 1: STRAP Process Flowchart**

> "The flowchart depicts a multi-step process starting with plastic feed and ending with purified resin in the form of plastic pellets. The process involves dissolution, hot filtration, optional hot adsorption, precipitation and mechanical separation, drying, and extrusion. Solvent recovery and cleaning are integrated into the process."

**Key Process Steps:**
1. Plastic feed (high color) → Dissolution
2. Hot Filtration → removes insoluble polymers
3. Hot Adsorption (optional) → removes colorants
4. Precipitation → polymer recovery
5. Drying → solvent removal
6. Extrusion → purified resin pellets (no color)

---

## Knowledge Base Status

```
Knowledge Base:  STRAP-CORE
Total Papers:    6 (after this ingestion)
Total Chunks:    1063
Collection:      kb_strap_core
Status:          ready
```

### Papers in STRAP-CORE
1. Recycling of multilayer plastic packaging materials (STRAP original)
2. Solvent Based Plastic Recycling Review
3. Recycling of Single-Use Multilayer Plastics for Biomanufacturing
4. Screening Green Solvents for Multilayer Plastic Film Recycling
5. **Pigment Removal from Reverse-Printed Laminated Flexible Films** ← NEW

---

## Quality Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Completeness** | 9/10 | 14 pages, 9 figures processed |
| **Chunking Quality** | 9/10 | 79% semantic splitting, 307 avg tokens |
| **Figure Interpretation** | 9/10 | All 9 figures analyzed with quantitative data |
| **Contextual Enrichment** | 9/10 | 13 parent contexts with 93.1% cost savings |
| **Deduplication** | 10/10 | No duplicates |
| **Overall** | **9.2/10** | Excellent enriched ingestion |

---

## Test Queries

Recommended queries to verify enrichment quality:

```
✓ "How does activated carbon remove yellow pigments from PE?"
  Expected: Yellow 12 removed at 0.8-1.0 g/L AC loading; Yellow 1 needs ~4-6 g/L

✓ "What is the effect of mechanical deliquoring on PE yellowness?"
  Expected: Piston at 6.5 bar reduces solvent/PE to 1.3 g/g; lower solvent = less yellow

✓ "What solvents are used for pigment removal in STRAP?"
  Expected: Dodecane, toluene, heptane; toluene has higher dye solubility

✓ "What is the optimal hot adsorption treatment for color removal?"
  Expected: 600 mg AC for 18 hours gives YI of -6.5 (best result)
```

---

## Files Generated

- **Interpretations:** `./rag_figures/Pigment removal from reverse-printed.../interpretations.json`
- **Ingestion Log:** `./rag_data/ingestion_log_STRAP-CORE.json`
- **Chunk Store:** `./rag_data/chunk_store_v2.pkl`

---

*Report generated automatically after LLM-enriched ingestion*
*Model: gemini-2.0-flash (figures + contextual enrichment)*
