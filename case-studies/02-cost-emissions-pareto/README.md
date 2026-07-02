# Case Study 02 — Cost vs Emissions / Circularity Pareto Landscapes

**Question.** For the STRAP waste-management superstructure (an MINLP that picks
dissolution solvents, wash stages, and a residual-stream technology), what does
the trade-off between total cost and environmental objectives actually look
like — and why did so many earlier Pareto runs return a single "broken" point?

## The finding: why single-point "broken" frontiers appeared

Reproducing the v9 symptom in v10 (`run_waste_management_pareto`, live SCIP
solver) showed the collapse has **three distinct causes**, only one of which is
a real defect:

1. **Genuine degeneracy (not a bug).** Under Scenario A the technology/emission
   parameters make the minimum-cost design also the minimum-emissions design,
   so the cost-vs-emissions frontier is legitimately a single point. The
   *landscape* of all feasible designs is still rich (18 points).

2. **All-or-nothing stage-3 (a modelling limit).** `model.py`'s
   `one_third_tech` constraint (`sum(z[k]) == 1`) sends the entire residual
   stream to a single technology, so the raw cost-vs-emissions trade-off
   resolves to the technology corners rather than a smooth blend.

3. **A cost-basis inconsistency in the engine's frontier (a real defect).**
   The engine's native `points[]` frontier is built from separate min-cost /
   min-objective *anchor* solves. Those anchors can return a design with
   `capital_cost = 0` and a total cost ~27× below every design that actually
   builds recovery capacity (e.g. `$92k` vs `$2.5–6.5M`). Such a phantom point
   dominates the whole space and collapses the reported frontier to one point,
   even though the 18-point landscape is on a consistent `$2.5M+` basis. A
   `capital_cost = 0` design that also reports near-zero emissions is
   physically inconsistent — no recovery plant cannot yield low emissions.

## The fix used here

**Recompute the frontier from the landscape, on the landscape's consistent
cost basis** (`_nondominated` in `reproduce.py`), instead of trusting the
engine's anchor-based `points[]`. This drops the phantom point and, crucially,
recovers the *true, richer* frontier:

| Panel | Engine native frontier | Landscape-consistent frontier |
|---|---|---|
| Scenario A · cost vs emissions | 1 | 1 (genuinely degenerate) |
| **Scenario B · cost vs emissions** | 2 (incl. phantom) | **3 real points** |
| Scenario B · cost vs circularity (≥1 wash) | 3 | 3 (already consistent) |

`figures/pareto_landscapes.png` plots all 18 feasible designs (grey) with the
recomputed non-dominated frontier (red diamonds) for each case. Scenario B
cost-vs-emissions now shows the full cheap-dirty → costly-clean trade-off
($2.5M @ 27k t CO₂e → $3.6M @ 21k → $5.8M @ 5k); circularity shows more cost
buying more recovered-material circularity.

## Recommended engine follow-up (not applied here)

The case study fixes the *visualization/analysis* by recomputing from the
landscape. The **engine itself** (`src/strap/tools/waste_optimization.py`)
should be hardened so its native `points[]` cannot include a
cost-basis-inconsistent anchor: tighten `_row_has_usable_economics` (currently
requires ≥2 of {CAPEX, OPEX, GWP} to be zero before rejecting — a
`CAPEX=0, OPEX>0` phantom passes today), and/or derive the reported frontier by
dominance-filtering the landscape so `points[] ⊆ landscape_points` by
construction. This is left as a separate change because it touches the shipped
MINLP contract and the 84 `test_waste_optimization.py` tests.

## Reproduce

```bash
# replay committed solver output (no solver needed):
python case-studies/02-cost-emissions-pareto/reproduce.py
# re-solve live with SCIP (workbook-backed TEA, no BioSTEAM/API):
python case-studies/02-cost-emissions-pareto/reproduce.py --live
```

## Files

```
02-cost-emissions-pareto/
├── README.md
├── reproduce.py                       replay/--live re-solve → figures/
├── data/
│   ├── A_emissions_degenerate.json    live SCIP output, Scenario A
│   ├── B_emissions_two_corner.json    live SCIP output, Scenario B emissions
│   └── B_circularity_rich.json        live SCIP output, Scenario B circularity
└── figures/pareto_landscapes.png/.pdf
```
