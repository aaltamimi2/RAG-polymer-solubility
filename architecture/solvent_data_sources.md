# Solvent Price & GWP Data Sources

## Web Search Benchmark (5 solvents with known ground truth)

### Price — Web vs. Ground Truth

| Solvent | Ground Truth | Web Result | Deviation | Sources |
|---------|-------------|------------|-----------|---------|
| Heptane | $1.42/kg | $1.421/kg | <0.1% | ChemAnalyst, IMARC |
| Toluene | $0.82/kg | $0.80–0.83/kg | 1–2% | IMARC, ECHEMI, ChemAnalyst |
| Xylene | $0.84/kg | $0.843/kg | <0.5% | ChemAnalyst, IMARC |
| Ethylene Glycol | $0.53/kg | $0.50–0.53/kg | 0–6% | ChemAnalyst, ECHEMI, Intratec |
| Propylene Glycol | $1.53/kg | $1.49–1.58/kg | 3–5% | ChemAnalyst, IMARC |

**Verdict:** 5/5 within 6%. Web search is reliable for price. Main caveat: 15–25% seasonal volatility.

### GWP — Web vs. Ground Truth

| Solvent | Ground Truth | Web Result | Deviation | Found? |
|---------|-------------|------------|-----------|--------|
| Heptane | 0.897 | ~0.92 | ~3% | Barely (1 non-citable source) |
| Toluene | 1.61 | 1.47 | −8.7% | 1 source (Winnipeg emissions doc) |
| Xylene | 1.52 | N/A | — | No (only alternative-route values 2.2–3.3) |
| Ethylene Glycol | 2.70 | 1.28–4.52 | Wide range | Route-dependent, no match |
| Propylene Glycol | 5.16 | 6.34 | +23% | 1 source (CarbonCloud, opaque method) |

**Verdict:** 1/5 within 10%. ecoinvent/GaBi values are paywalled; open sources lack methodological transparency.

## Central Data Sources

### Price

| Source | Chemicals | API | Free | Notes |
|--------|-----------|-----|------|-------|
| **Intratec** | 225 | REST/JSON | No | Best commercial API (monthly, USD/MT) |
| **ICIS** | 180+ | REST | No ($$$) | Gold standard, enterprise pricing |
| ChemAnalyst | 250+ | No | No | No API, subscription |
| ECHEMI | 200 | No | No | China-focused |
| **FRED/BLS PPI** | ~50 categories | REST | **Yes** | Index only (relative, not $/kg) |
| **USITC DataWeb** | All traded | REST | **Yes** | Derived $/kg from import records (approx.) |

No free API provides absolute $/kg for industrial solvents. Best free workaround: USITC import-value-derived prices + FRED PPI for trend adjustment.

### GWP

| Source | Chemicals | API | Free | Notes |
|--------|-----------|-----|------|-------|
| **ecoinvent** | 15,000+ | No | No (~3K €/yr) | Gold standard, requires LCA software |
| **Carbon Minds** | 1,500+ | No | No (5K €/yr) | Best chemical-specific LCA DB |
| **Climatiq** | ecoinvent-backed | REST/JSON | 250 calls/mo free | Best API option; paid tier likely needed for chemicals |
| GaBi/Sphera | 2,300+ | No | No (5–15K €/yr) | Expensive, proprietary |
| **GREET** | ~5–10 | No | **Yes** | Only fuels + BTX aromatics |
| openLCA + USLCI | ~10–20 | Python | **Yes** | Limited solvent coverage |

No free central source for solvent GWP. ecoinvent and Carbon Minds dominate but are paywalled. Climatiq's REST API is the most promising programmatic path if its paid tier includes chemical production factors.

## Implications for Our Implementation

| Strategy | Justified? | Why |
|----------|-----------|-----|
| Price: curated DB + web search fallback | **Yes** | Web search reproduces ground truth within 0–6% via ChemAnalyst/IMARC snippets |
| GWP: curated DB only | **Yes** | Web search fails 80%+ of the time; paywalled LCA databases are the real source |
| GWP: class-average fallback | **Yes** | More honest than a potentially 2× wrong web-scraped point estimate |

The 26-solvent `_SOLVENT_DB` in `solvent_lookup.py` (16 BioSTEAM high-confidence + 10 estimated) is a pragmatic substitute for a $3–5K/yr ecoinvent or Carbon Minds license.
