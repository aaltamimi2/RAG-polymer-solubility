#!/usr/bin/env python
"""Case Study #2 Extended — Multi-Polymer Sequence Pareto Front.

Evaluates predefined separation sequences (from the DP optimizer or manual
specification) by running BioSTEAM TEA/LCA for each step, then computing
total MSP and GWP per sequence and finding the Pareto front across sequences.

Each sequence defines an ordered series of dissolution steps — polymer,
solvent, and dissolution temperature are specified per step. BioSTEAM runs
each step independently with the correct parameters.

Input format (JSON):
    [
      {
        "label": "Seq #1",
        "steps": [
          {"polymer": "PS",  "solvent": "dimethylsulfoxide", "temperature_c": 135},
          {"polymer": "PVC", "solvent": "dimethylformamide",  "temperature_c": 80},
          ...
        ]
      },
      ...
    ]

Solvent names use the solubility-DB keys (lowercase, no spaces) and are
automatically resolved to BioSTEAM-compatible names via the solvent registry.

Usage:
    # Evaluate sequences from JSON
    python run_multi_polymer_pareto.py --sequences sequences.json

    # Use the DP-optimal sequence from pareto_frontier.json
    python run_multi_polymer_pareto.py \
        --from-pareto-frontier dp_lattice_sweep/pareto_frontier.json

    # Quick test with built-in example sequences
    python run_multi_polymer_pareto.py --example
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ARCH_DIR = _THIS_DIR.parent.parent
_ROOT_DIR = _ARCH_DIR.parent
sys.path.insert(0, str(_ROOT_DIR / "src"))
_V8_SRC = Path("/home/aaltamimi2/langchain-STRAP-v8/src")
if _V8_SRC.is_dir():
    sys.path.insert(0, str(_V8_SRC))

from strap.vendor.biosteam_runner import (  # noqa: E402
    build_batch_configs,
    run_batch_simulations,
    _SOLVENT_DEFAULTS,
    _curated_lookup,
)
from strap.solvent_registry import resolve_to_biosteam  # noqa: E402

# Reuse helpers from trace20
_T20_DIR = _THIS_DIR.parent / "trace20-cs2-pareto"
sys.path.insert(0, str(_T20_DIR))
from cs2_pareto_figures import (  # noqa: E402
    classify_solvent,
    compute_pareto_front,
    normalize_result,
    _is_finite,
)


# ── Solvent name resolution ──────────────────────────────────────────

def _resolve_solvent_name(interp_key: str) -> str:
    """Resolve solubility-DB key to BioSTEAM-compatible name.

    Falls back to title-casing if the registry has no mapping.
    """
    bst = resolve_to_biosteam(interp_key)
    if bst:
        return bst
    # Common manual fallbacks
    _MANUAL = {
        "glycol": "Ethylene Glycol",
        "propyleneglycol": "Propylene Glycol",
        "diethyleneglycol": "Diethylene glycol",
        "dimethylsulfoxide": "Dimethyl sulfoxide",
        "dimethylformamide": "N,N-Dimethylformamide",
        "diphenylether": "Diphenyl ether",
        "dodecane": "Dodecane",
        "cyclohexanol": "Cyclohexanol",
        "1,2-dimethylbenzene": "o-Xylene",
        "toluene": "Toluene",
        "xylene": "Xylene",
        "ethylacetate": "Ethyl Acetate",
        "acetonitrile": "Acetonitrile",
        "ethanol": "Ethanol",
        "nheptane": "n-Heptane",
        "oleylalcohol": "Oleyl Alcohol",
        "propylene carbonate": "Propylene Carbonate",
        "propylenecarbonate": "Propylene Carbonate",
        "2,3-dihydropyran": "2,3-Dihydropyran",
        "cyclohexanone": "Cyclohexanone",
        "dmso": "Dimethyl sulfoxide",
        "dmf": "N,N-Dimethylformamide",
    }
    key = interp_key.lower().strip()
    if key in _MANUAL:
        return _MANUAL[key]
    return interp_key


def _get_solvent_price(bst_name: str) -> float:
    """Look up solvent price from defaults or curated CSV."""
    defaults = _SOLVENT_DEFAULTS.get(bst_name)
    if defaults:
        return defaults[0]
    curated = _curated_lookup(bst_name)
    if curated and curated.get("price") is not None:
        return curated["price"]
    return 1.50  # fallback


# ── Helpers ──────────────────────────────────────────────────────────

def _abbreviate_solvent(name: str) -> str:
    """Short label for solvent names in figures."""
    abbrevs = {
        "Toluene": "Tol", "Xylene": "Xyl", "Dodecane": "Dod",
        "Ethylene Glycol": "EG", "Propylene Glycol": "PG",
        "Diethylene glycol": "DEG", "Dimethyl sulfoxide": "DMSO",
        "N,N-Dimethylformamide": "DMF", "Diphenyl ether": "DPE",
        "Ethanol": "EtOH", "Isopropanol": "iPrOH",
        "Ethyl Acetate": "EtOAc", "Acetonitrile": "ACN",
        "Cyclohexanol": "CyOH", "Cyclohexanone": "CyO",
        "o-Xylene": "oXyl", "p-Xylene": "pXyl",
        "Oleyl Alcohol": "OleOH", "n-Heptane": "Hep",
        "Propylene Carbonate": "PC", "2,3-Dihydropyran": "DHP",
    }
    return abbrevs.get(name, name[:5])


# ── Sequence loading ─────────────────────────────────────────────────

def load_sequences_json(path: Path) -> list[dict]:
    """Load sequences from JSON.

    Expected format: list of {"label": str, "steps": [{"polymer", "solvent",
    "temperature_c"}, ...]}.
    """
    with open(path) as f:
        data = json.load(f)

    sequences = data if isinstance(data, list) else data.get("sequences", [data])

    for seq in sequences:
        for step in seq.get("steps", []):
            step["bst_solvent"] = _resolve_solvent_name(step["solvent"])
    return sequences


def load_from_pareto_frontier(path: Path) -> list[dict]:
    """Load sequences from pareto_frontier.json (DP optimizer output).

    The frontier JSON has one sequence and multiple pareto_points, each
    with different temperature configurations. We create one sequence per
    Pareto point.
    """
    with open(path) as f:
        data = json.load(f)

    base_sequence = data["sequence"]  # [{step, polymer, solvent, dp_temp, dp_selectivity}]
    pareto_points = data.get("pareto_points", [])

    sequences = []

    # The DP-optimal sequence (using dp_temp values)
    dp_steps = []
    for s in base_sequence:
        dp_steps.append({
            "polymer": s["polymer"],
            "solvent": s["solvent"],
            "temperature_c": s["dp_temp"],
            "selectivity": s.get("dp_selectivity", 0),
            "bst_solvent": _resolve_solvent_name(s["solvent"]),
        })
    sequences.append({
        "label": "DP-optimal",
        "steps": dp_steps,
        "source": "dp_optimal",
    })

    # Each Pareto point is a temperature variant of the same sequence
    for pi, pp in enumerate(pareto_points):
        steps = []
        step_temps = pp.get("step_temps", [])
        step_sels = pp.get("step_sels", [])
        step_msps = pp.get("step_msps", [])

        for si, base_step in enumerate(base_sequence):
            temp = step_temps[si] if si < len(step_temps) else base_step["dp_temp"]
            sel = step_sels[si] if si < len(step_sels) else 0
            steps.append({
                "polymer": base_step["polymer"],
                "solvent": base_step["solvent"],
                "temperature_c": temp,
                "selectivity": sel,
                "bst_solvent": _resolve_solvent_name(base_step["solvent"]),
                "cached_msp": step_msps[si] if si < len(step_msps) else None,
            })
        sequences.append({
            "label": f"Pareto-{pi+1} (min_sel={pp.get('min_sel', 0):.1f})",
            "steps": steps,
            "source": "pareto_frontier",
            "min_selectivity": pp.get("min_sel"),
            "cached_sum_msp": pp.get("sum_msp"),
            "cached_avg_msp": pp.get("avg_msp"),
        })

    return sequences


def _build_example_sequences() -> list[dict]:
    """Built-in example sequences for quick testing (3 polymers, 2 sequences)."""
    return [
        {
            "label": "Seq A: EG/DMSO/DMF",
            "steps": [
                {"polymer": "PE",  "solvent": "glycol",              "temperature_c": 120},
                {"polymer": "PET", "solvent": "dimethylsulfoxide",   "temperature_c": 135},
                {"polymer": "EVOH", "solvent": "dimethylformamide",  "temperature_c": 100},
            ],
        },
        {
            "label": "Seq B: Tol/EtOAc/PC",
            "steps": [
                {"polymer": "PE",  "solvent": "toluene",               "temperature_c": 110},
                {"polymer": "PET", "solvent": "ethylacetate",          "temperature_c": 80},
                {"polymer": "EVOH", "solvent": "propylenecarbonate",   "temperature_c": 120},
            ],
        },
        {
            "label": "Seq C: Dod/PG/DMSO",
            "steps": [
                {"polymer": "PE",  "solvent": "dodecane",              "temperature_c": 130},
                {"polymer": "PET", "solvent": "propyleneglycol",       "temperature_c": 160},
                {"polymer": "EVOH", "solvent": "dimethylsulfoxide",    "temperature_c": 135},
            ],
        },
    ]


# ── BioSTEAM execution ──────────────────────────────────────────────

def run_sequences(
    sequences: list[dict],
    energy_case: str = "C1",
    output_dir: Path | None = None,
) -> dict:
    """Run BioSTEAM for each step of each sequence.

    Deduplicates configs: if two sequences share the same
    (solvent, polymer, temperature) triple, only one sim is run.

    Returns structured results dict.
    """
    print("=" * 60)
    print("Multi-Polymer Sequence Evaluation")
    print("=" * 60)
    print(f"  Sequences: {len(sequences)}")
    total_steps = sum(len(s.get('steps', [])) for s in sequences)
    print(f"  Total steps: {total_steps}")

    # Resolve solvent names
    for seq in sequences:
        for step in seq.get("steps", []):
            if "bst_solvent" not in step:
                step["bst_solvent"] = _resolve_solvent_name(step["solvent"])

    # Deduplicate: key = (polymer, bst_solvent, temperature_c)
    unique_configs: dict[tuple, dict] = {}
    for seq in sequences:
        for step in seq["steps"]:
            key = (step["polymer"], step["bst_solvent"], step["temperature_c"])
            if key not in unique_configs:
                bst_name = step["bst_solvent"]
                price = _get_solvent_price(bst_name)
                unique_configs[key] = {
                    "solvent": bst_name,
                    "target_plastic": step["polymer"],
                    "energy_case": energy_case,
                    "dissolution_temperature_c": step["temperature_c"],
                    "solvent_price": price,
                    "precipitation_temperature_c": 25.0,
                    "solvent_loss_pct": 0.01,
                    "feedstock_distance_km": 0.0,
                    "processing_capacity": 20_000,
                }

    config_keys = list(unique_configs.keys())
    configs = [unique_configs[k] for k in config_keys]
    print(f"  Unique (polymer, solvent, temp) configs: {len(configs)}")
    print(f"  Estimated time: ~{len(configs) * 12 // 3 // 60} min\n")

    # Print config summary
    for k, cfg in zip(config_keys, configs):
        poly, solv, temp = k
        print(f"    {poly:>8}/{solv:<25} @ {temp:>5.0f}°C  (${cfg['solvent_price']:.2f}/kg)")

    # Run BioSTEAM
    print()
    t0 = time.time()
    raw_results = run_batch_simulations(configs, max_parallel=3, timeout_per_sim=180)
    elapsed = time.time() - t0

    # Map results back by key
    results_by_key: dict[tuple, dict] = {}
    n_ok, n_fail = 0, 0
    for i, raw in enumerate(raw_results):
        key = config_keys[i]
        if raw.get("success"):
            nr = normalize_result(raw)
            nr["tier"] = classify_solvent(nr["solvent"])
            nr["polymer"] = key[0]
            nr["temperature_c"] = key[2]
            results_by_key[key] = nr
            n_ok += 1
        else:
            n_fail += 1
            err = raw.get("error", "unknown")[:80]
            print(f"  FAIL: {key[0]}/{key[1]} @ {key[2]}°C — {err}")
            results_by_key[key] = {
                "success": False,
                "polymer": key[0],
                "solvent": key[1],
                "temperature_c": key[2],
                "error": raw.get("error", "unknown"),
            }

    print(f"\n  Batch complete: {n_ok}/{len(configs)} succeeded, "
          f"{n_fail} failed, {elapsed:.0f}s elapsed")

    # Assemble per-sequence results
    sequence_results = []
    for seq in sequences:
        steps_out = []
        total_msp = 0.0
        total_gwp = 0.0
        all_ok = True

        for step in seq["steps"]:
            key = (step["polymer"], step["bst_solvent"], step["temperature_c"])
            result = results_by_key.get(key, {})

            if result.get("success", True) and "msp" in result:
                msp = result["msp"]
                gwp = result["gwp"]
                steps_out.append({
                    "polymer": step["polymer"],
                    "solvent_interp": step["solvent"],
                    "solvent_bst": step["bst_solvent"],
                    "temperature_c": step["temperature_c"],
                    "msp": msp,
                    "gwp": gwp,
                    "selectivity": step.get("selectivity"),
                })
                total_msp += msp
                total_gwp += gwp
            else:
                all_ok = False
                steps_out.append({
                    "polymer": step["polymer"],
                    "solvent_interp": step["solvent"],
                    "solvent_bst": step["bst_solvent"],
                    "temperature_c": step["temperature_c"],
                    "msp": None,
                    "gwp": None,
                    "error": result.get("error", "sim failed"),
                })

        n_steps = len(seq["steps"])
        seq_result = {
            "label": seq.get("label", "unnamed"),
            "steps": steps_out,
            "n_steps": n_steps,
            "all_succeeded": all_ok,
            "total_msp": round(total_msp, 4) if all_ok else None,
            "total_gwp": round(total_gwp, 4) if all_ok else None,
            "avg_msp": round(total_msp / n_steps, 4) if all_ok and n_steps else None,
            "avg_gwp": round(total_gwp / n_steps, 4) if all_ok and n_steps else None,
            "min_selectivity": seq.get("min_selectivity"),
        }
        sequence_results.append(seq_result)

    # Find Pareto front across sequences (minimize avg_msp, minimize avg_gwp)
    valid_seqs = [s for s in sequence_results
                  if s["all_succeeded"] and s["avg_msp"] is not None]

    if valid_seqs:
        pareto_pts = compute_pareto_front(
            [{"msp": s["avg_msp"], "gwp": s["avg_gwp"], "label": s["label"]}
             for s in valid_seqs]
        )
        pareto_labels = {p["label"] for p in pareto_pts}
        for s in sequence_results:
            s["is_pareto"] = s["label"] in pareto_labels
        n_pareto = len(pareto_labels)
    else:
        for s in sequence_results:
            s["is_pareto"] = False
        n_pareto = 0

    # Print summary
    print(f"\n  Sequence Results:")
    print(f"  {'Label':<40}  {'Steps':>5}  {'Avg MSP':>8}  {'Avg GWP':>8}  Pareto?")
    for s in sorted(sequence_results, key=lambda x: x["avg_msp"] or 999):
        flag = " *" if s.get("is_pareto") else ""
        if s["avg_msp"] is not None:
            print(f"  {s['label']:<40}  {s['n_steps']:>5}  "
                  f"${s['avg_msp']:.3f}  {s['avg_gwp']:.2f}{flag}")
        else:
            print(f"  {s['label']:<40}  {s['n_steps']:>5}  {'FAIL':>8}  {'FAIL':>8}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "phase": "sequence-pareto",
        "timestamp": timestamp,
        "n_sequences": len(sequences),
        "n_unique_configs": len(configs),
        "n_succeeded": n_ok,
        "n_failed": n_fail,
        "n_pareto": n_pareto,
        "elapsed_s": round(elapsed, 1),
        "sequences": sequence_results,
        "pareto_sequences": [s for s in sequence_results if s.get("is_pareto")],
    }

    if output_dir:
        out_path = output_dir / f"sequence_pareto_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved: {out_path}")

    return output


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Polymer Sequence Pareto Front"
    )
    parser.add_argument(
        "--sequences", default=None, metavar="PATH",
        help="Path to sequences JSON file",
    )
    parser.add_argument(
        "--from-pareto-frontier", default=None, metavar="PATH",
        help="Load sequences from pareto_frontier.json (DP optimizer output)",
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Run built-in example sequences (3 polymers, 3 sequences)",
    )
    parser.add_argument(
        "--energy-case", default="C1",
        help="Energy case (default: C1)",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: this script's directory)",
    )
    args = parser.parse_args()

    if not args.sequences and not args.from_pareto_frontier and not args.example:
        parser.error("Provide --sequences, --from-pareto-frontier, or --example")

    output_dir = Path(args.output_dir) if args.output_dir else _THIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    if args.sequences:
        sequences = load_sequences_json(Path(args.sequences))
    elif args.from_pareto_frontier:
        sequences = load_from_pareto_frontier(Path(args.from_pareto_frontier))
    else:
        sequences = _build_example_sequences()

    # Resolve solvent names for any that don't have bst_solvent yet
    for seq in sequences:
        for step in seq.get("steps", []):
            if "bst_solvent" not in step:
                step["bst_solvent"] = _resolve_solvent_name(step["solvent"])

    result = run_sequences(sequences, energy_case=args.energy_case,
                           output_dir=output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
