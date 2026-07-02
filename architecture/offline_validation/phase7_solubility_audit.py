"""Phase 7: solubility screening feature audit — engines, flexibility,
interpolation rigor, visuals, and handoff integrity. Zero model calls;
executes the real tools (DuckDB + interpolation + matplotlib).

Sections:
  A  coverage         every polymer x solvent pair with coefficients predicts
  B  query shapes     1xN, Nx1, NxM subsets, ceilings, precipitation ordering
  C  interpolation    range clamp/extension/cap semantics, point-vs-range
                      consistency, SQL-vs-interpolation agreement, error paths
  D  visuals          real PNG rendering for screening plots (size + validity)
  E  handoff          solubility payload fields survive adaptation downstream
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from strap.testing_utils import block_model_access

OUT = _ROOT / "architecture" / "test_results" / "subagent_validation_offline_20260701"

CHECKS: list[tuple[str, bool, str]] = []


def check(section: str, ok: bool, detail: str) -> None:
    CHECKS.append((section, bool(ok), detail))


def _parse(raw) -> dict:
    doc = json.loads(raw) if isinstance(raw, str) else raw
    return doc if isinstance(doc, dict) else {}


def run() -> None:
    random.seed(20260701)
    with block_model_access():
        from strap.database import get_connection
        from strap.solubility import (
            get_available_pairs,
            get_available_polymers,
            get_available_solvents,
            get_available_solvents_for_polymer,
        )
        from strap.tools.interpolation import predict_solubility, predict_solubility_range
        from strap.tools.advanced_separation import (
            find_differential_precipitation_solvents,
            rank_solvents_for_separation,
        )
        from strap.tools.visualization import plot_solubility_vs_temperature

        # ---------------- A: coverage ----------------
        polymers = get_available_polymers()
        solvents = get_available_solvents()
        pairs = get_available_pairs()
        check("A_coverage", len(polymers) >= 10, f"{len(polymers)} polymers available")
        check("A_coverage", len(solvents) >= 30, f"{len(solvents)} solvents available")
        check("A_coverage", len(pairs) >= 300, f"{len(pairs)} fitted pairs")

        sample_pairs = random.sample(sorted(pairs), 25)
        failures = []
        for polymer, solvent in sample_pairs:
            doc = _parse(predict_solubility(polymer, solvent, 80.0))
            data = doc.get("data") or {}
            value = data.get("solubility_pct", data.get("predicted_solubility"))
            if data.get("error") or value is None:
                failures.append(f"{polymer}/{solvent}: {str(data.get('error') or 'no value')[:60]}")
        check("A_coverage", not failures,
              f"25 random pairs predict at 80C ({len(failures)} failures: {failures[:3]})")

        aliases = ["xylene", "THF", "DCM", "dmso", "o-xylene", "toluene", "dodecane"]
        alias_failures = []
        for alias in aliases:
            doc = _parse(predict_solubility("LDPE", alias, 80.0))
            if (doc.get("data") or {}).get("error"):
                alias_failures.append(alias)
        check("A_coverage", not alias_failures, f"alias resolution {aliases} ({alias_failures} failed)")

        # ---------------- B: query shapes ----------------
        # (1) one polymer x N solvents
        shape_failures = []
        for solvent in ["toluene", "1,2-dimethylbenzene", "dodecane"]:
            doc = _parse(predict_solubility_range("LDPE", solvent, t_start_c=25, t_end_c=120))
            if not (doc.get("data") or {}).get("predictions"):
                shape_failures.append(solvent)
        check("B_shapes", not shape_failures, f"1 polymer x 3 solvents range tables ({shape_failures})")

        # (2) N polymers x one solvent
        shape_failures = []
        for polymer in ["LDPE", "HDPE", "PP", "PS"]:
            doc = _parse(predict_solubility_range(polymer, "toluene", t_start_c=25, t_end_c=120))
            if not (doc.get("data") or {}).get("predictions"):
                shape_failures.append(polymer)
        check("B_shapes", not shape_failures, f"4 polymers x toluene range tables ({shape_failures})")

        # (3) per-polymer candidate listing (multi-solvent discovery)
        ldpe_solvents = get_available_solvents_for_polymer("LDPE")
        check("B_shapes", len(ldpe_solvents) >= 25, f"LDPE candidate listing has {len(ldpe_solvents)} solvents")

        # (4) processing conditions: precipitation ordering, both directions
        doc = _parse(find_differential_precipitation_solvents("HDPE", "LDPE"))
        display = doc.get("display") or ""
        check("B_shapes", "HDPE @" in display and "LDPE @" in display,
              "differential precipitation HDPE-before-LDPE returns ranked solvents")
        doc = _parse(find_differential_precipitation_solvents("LDPE", "HDPE"))
        check("B_shapes", "solvent" in (doc.get("display") or "").lower(),
              "reverse ordering LDPE-before-HDPE also answers")

        # (5) selectivity at a fixed condition
        doc = _parse(rank_solvents_for_separation("LDPE", "PET", temperature=100.0))
        check("B_shapes", not doc.get("error"),
              f"selectivity ranking LDPE vs PET at 100C ({str(doc.get('error'))[:60]})")

        # ---------------- C: interpolation rigor ----------------
        conn = get_connection()
        sql_min, sql_max, n_sql = conn.execute(
            "SELECT MIN(temperature___c_), MAX(temperature___c_), COUNT(*) "
            "FROM all_solvents_ldpe WHERE LOWER(solvent) LIKE '%toluene%'"
        ).fetchone()

        def temps_of(doc):
            return [row["temperature_c"] for row in (doc.get("data") or {}).get("predictions", [])]

        doc90 = _parse(predict_solubility_range("LDPE", "toluene", t_start_c=25, t_end_c=90))
        check("C_interp", max(temps_of(doc90) or [0]) == 90.0,
              f"request to 90C clamps curve at 90 (SQL data reaches {sql_max})")

        doc160 = _parse(predict_solubility_range("LDPE", "toluene", t_start_c=25, t_end_c=160))
        check("C_interp", max(temps_of(doc160) or [0]) == 160.0, "request to 160C reaches 160")
        check("C_interp", (doc160.get("data") or {}).get("extrapolated_points") in (0, [], None),
              "no extrapolation label inside fitted range")

        doc200 = _parse(predict_solubility_range("LDPE", "toluene", t_start_c=25, t_end_c=200))
        d200 = doc200.get("data") or {}
        check("C_interp", max(temps_of(doc200) or [0]) == 200.0
              and (d200.get("extrapolated_points") or 0) > 0,
              f"beyond fitted range extends to 200 with {d200.get('extrapolated_points')} labeled extrapolations")

        doc250 = _parse(predict_solubility_range("LDPE", "toluene", t_start_c=25, t_end_c=250))
        d250 = doc250.get("data") or {}
        check("C_interp", d250.get("range_was_capped") is True and max(temps_of(doc250) or [0]) <= 200.0,
              "250C request capped at hard sensitivity limit with flag")

        # point vs range consistency at 80C
        point = _parse(predict_solubility("LDPE", "toluene", 80.0))
        point_value = (point.get("data") or {}).get("solubility_pct")
        range_rows = {row["temperature_c"]: row for row in (doc160.get("data") or {}).get("predictions", [])}
        range_value = (range_rows.get(80.0) or {}).get("solubility_pct")
        consistent = (
            point_value is not None and range_value is not None
            and abs(float(point_value) - float(range_value)) < 1e-6
        )
        check("C_interp", consistent, f"point (80C={point_value}) matches range row ({range_value})")

        # SQL vs interpolation agreement across sampled pairs at SQL temperatures
        disagreements = []
        sampled = random.sample(sorted(pairs), 12)
        for polymer, solvent in sampled:
            table = {
                "LDPE": "all_solvents_ldpe", "HDPE": "all_solvents_hdpe", "PP": "all_solvents_pp_lmw",
                "PS": "all_solvents_ps", "PVC": "all_solvents_pvc", "EVOH": "all_solvents_evoh",
                "PET": "all_solvents_pet", "PC": "all_solvents_pc", "NYLON66": "all_solvents_nylon66",
            }.get(polymer)
            if table is None:
                continue
            rows = conn.execute(
                f"SELECT temperature___c_, solubility____ FROM {table} "
                f"WHERE LOWER(solvent) = LOWER(?) ORDER BY temperature___c_",
                [solvent],
            ).fetchall()
            if len(rows) < 3:
                continue
            errors = []
            for temp, sql_value in rows[:: max(1, len(rows) // 4)]:
                doc = _parse(predict_solubility(polymer, solvent, float(temp)))
                pred = (doc.get("data") or {}).get("solubility_pct")
                if pred is None:
                    continue
                errors.append(abs(float(pred) - float(sql_value)))
            if errors and (sum(errors) / len(errors)) > 5.0:  # >5 percentage points mean error
                disagreements.append(f"{polymer}/{solvent}: mean|err|={sum(errors)/len(errors):.1f}pp")
        check("C_interp", not disagreements,
              f"interpolation tracks SQL data within 5pp mean error on sampled pairs ({disagreements[:3]})")

        # error paths
        bad = _parse(predict_solubility("LDPE", "unobtainium", 80.0))
        bad_data = bad.get("data") or {}
        check("C_interp", bad_data.get("error") and bad_data.get("error_code"),
              f"unknown solvent yields structured error ({bad_data.get('error_code')})")
        bad = _parse(predict_solubility_range("LDPE", "toluene", t_start_c=150, t_end_c=90))
        check("C_interp", bool(bad.get("error") or (bad.get("data") or {}).get("predictions") == []
              or bad.get("display")), "inverted range handled without crash")

        # ---------------- F: engine consistency (no fragmentation) ----------------
        from strap.solubility import get_entry

        inconsistent = []
        anomalous_pairs = []
        for polymer in polymers:
            for solvent in solvents:
                entry = get_entry(polymer, solvent)
                if entry and (entry.get("category") == "anomalous"
                              or (entry.get("r_squared") or 1.0) < 0.98):
                    anomalous_pairs.append((polymer, solvent))
        for polymer, solvent in anomalous_pairs:
            point_doc = _parse(predict_solubility(polymer, solvent, 80.0))
            range_doc = _parse(predict_solubility_range(polymer, solvent, t_start_c=25, t_end_c=120))
            point_ok = not (point_doc.get("data") or {}).get("error")
            range_ok = not (range_doc.get("data") or {}).get("error")
            if point_ok != range_ok:
                inconsistent.append(f"{polymer}/{solvent}: point={point_ok} range={range_ok}")
        check("F_consistency", not inconsistent,
              f"{len(anomalous_pairs)} anomalous/low-R2 pairs behave identically in point and "
              f"range tools (SQL fallback engine) ({inconsistent[:3]})")

        # ---------------- D: visuals ----------------
        plot_dir = Path(tempfile.mkdtemp(prefix="phase7_plots_"))
        doc = _parse(plot_solubility_vs_temperature(
            "", "", "", "", "",
            polymers="LDPE,HDPE,PP", solvents="toluene,dodecane",
            temperature_min=25, temperature_max=120,
            output_dir=str(plot_dir),
        ))
        data = doc.get("data") or {}
        produced = [Path(p) for p in ([data.get("plot_filepath")] if data.get("plot_filepath") else [])]
        ok_files = [p for p in produced if p.exists() and p.stat().st_size > 20_000]
        check("D_visuals", bool(ok_files),
              f"3 polymers x 2 solvents screening plot rendered ({[p.name for p in produced]}, "
              f"{[p.stat().st_size if p.exists() else 0 for p in produced]} bytes)")

        doc = _parse(plot_solubility_vs_temperature(
            "", "", "", "", "",
            polymers="EVOH", solvents="dmso",
            temperature_min=25, temperature_max=180,
            annotate_model_limits=True,
            output_dir=str(plot_dir),
        ))
        data = doc.get("data") or {}
        produced = [Path(p) for p in ([data.get("plot_filepath")] if data.get("plot_filepath") else [])]
        check("D_visuals", any(p.exists() and p.stat().st_size > 15_000 for p in produced),
              "extrapolation-annotated single-pair plot rendered")

        # ---------------- E: handoff integrity ----------------
        from strap.handoff_store import cleanup_handoff_scope, initialize_handoff_scope, store_agent_result
        from strap.handoffs import build_handoff_for_consumer

        payload = {
            "agent": "separation-engineer", "schema_version": "1.0",
            "polymers": ["HDPE", "LDPE"],
            "best_sequence": ["HDPE", "LDPE"],
            "steps": [
                {"step": 1, "polymer": "HDPE", "solvent": "n-heptane", "temperature_c": 85.0, "selectivity_pct": 78.0},
                {"step": 2, "polymer": "LDPE", "solvent": "n-heptane", "temperature_c": 50.0, "selectivity_pct": 70.0},
            ],
            "solvent_mapping": {"HDPE": "n-heptane", "LDPE": "n-heptane"},
            "polymer_solvent_candidates": {
                "HDPE": [{"rank": 1, "solvent": "n-heptane", "temperature_c": 85.0, "selectivity_pct": 78.0},
                          {"rank": 2, "solvent": "cyclohexane", "temperature_c": 80.0, "selectivity_pct": 74.0}],
                "LDPE": [{"rank": 1, "solvent": "n-heptane", "temperature_c": 50.0, "selectivity_pct": 70.0}],
            },
            "top_k_sequences": [{"rank": 1, "sequence": ["HDPE", "LDPE"], "min_selectivity": 70.0,
                                 "solvent_mapping": {"HDPE": "n-heptane", "LDPE": "n-heptane"}}],
        }
        scratch = Path(tempfile.mkdtemp(prefix="phase7_handoff_"))
        initialize_handoff_scope(user_query="precipitate HDPE before LDPE then optimize", artifact_root=scratch)
        try:
            source = store_agent_result(producer="separation-engineer", payload=payload, source_tool_call_id="t1")
            check("E_handoff", source.status == "ok", f"solubility-rich payload stores ok ({source.status})")
            for consumer, must_carry in [
                ("optimization-engineer", ["candidate_pairs", "feed_composition"]),
                ("biosteam-analyst", []),
                ("visualization-specialist", []),
            ]:
                derived = build_handoff_for_consumer(consumer=consumer,
                                                     source_handoff_id=source.handoff_id,
                                                     producer="separation-engineer")
                flat = json.dumps(derived.payload)
                carried = all(key in flat for key in must_carry)
                temp_preserved = ("85.0" in flat or "85" in flat) if consumer == "optimization-engineer" else True
                check("E_handoff", derived.status == "ok" and carried and temp_preserved,
                      f"sep->{consumer}: status={derived.status}, prompt={len(derived.task_prompt or '')}c, "
                      f"temps preserved={temp_preserved}")
        finally:
            cleanup_handoff_scope()

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {
        "summary": {"checks": len(CHECKS), "passed": passed, "failed": len(CHECKS) - passed},
        "checks": [{"section": s, "ok": ok, "detail": d} for s, ok, d in CHECKS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase7_solubility_audit.json").write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc["summary"], indent=2))
    for section, ok, detail in CHECKS:
        print(f"[{'ok' if ok else 'FAIL'}] {section:<12} {detail[:150]}")


if __name__ == "__main__":
    run()
