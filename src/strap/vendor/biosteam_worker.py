"""
BioSTEAM Subprocess Worker — STRAP Process Simulation
======================================================

Standalone script that runs BioSTEAM BaselineSTRAPProcess simulations
in an isolated subprocess. Receives simulation parameters as JSON via
sys.argv[1] and writes results as JSON to stdout.

Usage:
    python biosteam_worker.py '{"solvent":"Toluene","target_plastic":"PE"}'
    python biosteam_worker.py '{"solvent":"Toluene","target_plastic":"PS"}'
    python biosteam_worker.py '{"solvent":"Dichloromethane","target_plastic":"PC"}'

Input/Output contracts are documented in the calling module.
All diagnostic output goes to stderr; only the final JSON goes to stdout.
"""

import sys
import json
import time
import warnings
import traceback
from types import SimpleNamespace

# Suppress all warnings (BioSTEAM / thermosteam emit many RuntimeWarnings)
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Energy-case mapping
# ---------------------------------------------------------------------------
_ENERGY_CASES = {
    "C1": {"facilities": True, "turbogenerator": True},   # CHP
    "C2": {"facilities": False, "turbogenerator": False},  # Grid + AMCOR
    "C3": {"facilities": True, "turbogenerator": False},   # Grid + Boiler
}

# Map polymer names to BaselineSTRAPProcess target_plastic values.
# BioSTEAM only registers PEoligomer, EVOHoligomer, and PColigomer in its
# chemical DB.  LDPE/HDPE use the same PE process model; PET uses the PE
# model with different solvents (the dissolution/precipitation equipment is
# identical).
#
# PE-proxy polymers (PS, PP, PVC):
#   thermosteam has no native PS/PP/PVC oligomer.  These polymers are mapped
#   to "PE" so that BioSTEAM can run the dissolution/precipitation flowsheet.
#   Economics and LCA results are approximate — treat as order-of-magnitude
#   estimates only.  Use polymer-specific solvents (see biosteam_runner.py)
#   to improve physical fidelity of the operating conditions.
#
# PC (polycarbonate):
#   thermosteam ships a native PColigomer, so PC maps to itself.
_TARGET_PLASTIC_MAP = {
    "LDPE": "PE",
    "HDPE": "PE",
    "PET":  "PE",
    # ── PE-proxy polymers (approximate economics / LCA) ──────────────────
    "PS":   "PE",   # polystyrene — no native oligomer; PE flowsheet used
    "PP":   "PE",   # polypropylene — no native oligomer; PE flowsheet used
    "PVC":  "PE",   # poly(vinyl chloride) — no native oligomer; PE flowsheet used
    "PC":   "PE",   # polycarbonate — PE flowsheet proxy
    "EVOH": "PE",   # ethylene vinyl alcohol — PE flowsheet proxy
    # ── PE-proxy polymers (approximate economics / LCA) ────────────────
    "NYLON6":  "PE",   # polyamide 6
    "NYLON66": "PE",   # polyamide 66
    "PA6":     "PE",   # alias for Nylon6
    "PA66":    "PE",   # alias for Nylon66
    "PES":     "PE",   # polyethersulfone
}


class _NullBoiler:
    """Placeholder for facility-free cases where STRAP expects a boiler alias."""

    def __init__(self):
        self.ins = [None]
        self.natural_gas_price = 0.0
        self.design_results = {}
        self.blowdown_water = SimpleNamespace(imass={"Water": 0.0})


def _patch_baseline_process_compat(process_cls):
    """Patch older STRAP process models that assume ``self.B`` always exists.

    In current STRAP reference code, ``create_model()`` unconditionally aliases
    ``self.BT = self.B`` whenever ``turbogenerator=False``. That breaks C2-style
    scenarios where ``facilities=False`` and no boiler object is created at all.
    Some versions also expose the non-turbogenerator unit as ``BT`` rather than
    ``B``. This wrapper normalizes both cases without changing the scenario.
    """

    if getattr(process_cls, "_strap_worker_compat_patched", False):
        return process_cls

    original_create_model = process_cls.create_model

    def create_model_with_boiler_compat(self, *args, **kwargs):
        scenario = self.scenario
        if not scenario.turbogenerator and not hasattr(self, "B"):
            if hasattr(self, "BT"):
                self.B = self.BT
            else:
                self.B = _NullBoiler()
        result = original_create_model(self, *args, **kwargs)
        if not hasattr(self, "BT") and hasattr(self, "B"):
            self.BT = self.B
        return result

    process_cls.create_model = create_model_with_boiler_compat
    process_cls._strap_worker_compat_patched = True
    return process_cls


def _safe_call(fn, default=None):
    """Call *fn*(); return *default* on any exception."""
    try:
        val = fn()
        # Guard against NaN / Inf which are not valid JSON
        if val != val:  # NaN check
            return default
        return val
    except Exception:
        return default


def _run(config: dict) -> dict:
    """Execute a single BioSTEAM simulation and return the result dict."""

    t0 = time.time()

    solvent = config["solvent"]
    target_plastic_orig = config["target_plastic"]
    target_plastic = _TARGET_PLASTIC_MAP.get(target_plastic_orig.upper(), target_plastic_orig)
    energy_case = config.get("energy_case", "C1").upper()

    if energy_case not in _ENERGY_CASES:
        raise ValueError(
            f"Unknown energy_case '{energy_case}'. Must be one of C1, C2, C3."
        )

    facilities = _ENERGY_CASES[energy_case]["facilities"]
    turbogenerator = _ENERGY_CASES[energy_case]["turbogenerator"]

    # ------------------------------------------------------------------
    # Import BioSTEAM inside the function (subprocess starts fresh)
    # ------------------------------------------------------------------
    from plastics.strap import BaselineSTRAPProcess  # noqa: E402
    from plastics import strap as _strap_pkg  # noqa: E402
    BaselineSTRAPProcess = _patch_baseline_process_compat(BaselineSTRAPProcess)

    # Add HCl to chemical property package (required for chlorinated solvents,
    # harmless for others). Matches reference notebook convention.
    try:
        _strap_pkg.STRAP_chemicals_outline.append('HCl')
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Build Scenario and process model
    # ------------------------------------------------------------------
    scenario = BaselineSTRAPProcess.Scenario(
        solvent=solvent,
        target_plastic=target_plastic,
        target_plastic_percent=config.get("target_plastic_percent", 60),
        processing_capacity=config.get("processing_capacity", 20000),
        sell_leftover_plastic=config.get("sell_leftover_plastic", False),
        burn_leftover_plastic=config.get("burn_leftover_plastic", False),
        facilities=facilities,
        turbogenerator=turbogenerator,
        precipitation_temperature_format="constant",
    )
    pm = BaselineSTRAPProcess(scenario=scenario)

    # ------------------------------------------------------------------
    # Remove shredder (U1) and storage tank (T1) per notebook convention
    # ------------------------------------------------------------------
    try:
        pm.T1.disconnect(join_ends=True)
        pm.U1.disconnect(join_ends=True)
        pm.system.update_configuration(
            units=[u for u in pm.system.units if u not in [pm.T1, pm.U1]]
        )
    except Exception:
        pass  # some configurations may not have T1/U1

    # ------------------------------------------------------------------
    # Operating parameters
    # ------------------------------------------------------------------
    pm.tea.labor_cost = config.get("labor_cost", 120000)

    if "solvent_price" in config:
        pm.set_solvent_price(config["solvent_price"])

    pm.set_feedstock_distance(config.get("feedstock_distance_km", 0))

    # solvent_loss_pct is given as percent (e.g. 0.01 means 0.01%).
    # BioSTEAM expects a fraction, so divide by 100.
    pm.set_solvent_loss(config.get("solvent_loss_pct", 0.01) / 100)

    if "dissolution_temperature_c" in config:
        pm.set_dissolution_temperature(
            config["dissolution_temperature_c"] + 273.15
        )

    pm.set_precipitation_temperature(
        config.get("precipitation_temperature_c", 25) + 273.15
    )

    pm.set_dissolution_capacity(config.get("dissolution_capacity", 3))

    # ------------------------------------------------------------------
    # LCA characterisation factors (optional)
    # ------------------------------------------------------------------
    if "lca_cfs" in config:
        cfs = config["lca_cfs"]

        # Natural-gas CFs only exist when facilities are present (C1, C3)
        if facilities:
            try:
                pm.natural_gas.set_CF(
                    "GWP", cfs.get("natural_gas_gwp", 3.841)
                )
                pm.natural_gas.set_CF(
                    "HTC", cfs.get("natural_gas_htc", 2.474e-7)
                )
                pm.natural_gas.set_CF(
                    "HTNC", cfs.get("natural_gas_htnc", 1.196e-7)
                )
                pm.natural_gas.set_CF(
                    "ETOX", cfs.get("natural_gas_etox", 6.68)
                )
            except Exception:
                print(
                    "[biosteam_worker] warning: could not set natural-gas CFs",
                    file=sys.stderr,
                )

        # Solvent CFs (base offsets included)
        try:
            pm.solvent.set_CF(
                "GWP", cfs.get("solvent_gwp", 0) + 0.1563
            )
            pm.solvent.set_CF(
                "HTC", cfs.get("solvent_htc", 0) + 3.56126e-10
            )
            pm.solvent.set_CF(
                "HTNC", cfs.get("solvent_htnc", 0) + 8.0464e-9
            )
            pm.solvent.set_CF(
                "ETOX", cfs.get("solvent_etox", 0) + 0.00900
            )
        except Exception:
            print(
                "[biosteam_worker] warning: could not set solvent CFs",
                file=sys.stderr,
            )

        # Grid electricity CFs (C2 and C3 only — passed by runner)
        if "electricity_gwp" in cfs:
            import biosteam as bst
            try:
                bst.settings.set_electricity_CF(
                    "GWP", cfs["electricity_gwp"], basis="MJ"
                )
                bst.settings.set_electricity_CF(
                    "HTC", cfs["electricity_htc"], basis="MJ"
                )
                bst.settings.set_electricity_CF(
                    "HTNC", cfs["electricity_htnc"], basis="MJ"
                )
                bst.settings.set_electricity_CF(
                    "ETOX", cfs["electricity_etox"], basis="MJ"
                )
            except Exception:
                print(
                    "[biosteam_worker] warning: could not set electricity CFs",
                    file=sys.stderr,
                )

        # Water CFs
        for water_attr in ("makeup_water", "cooling_tower_makeup_water"):
            try:
                ws = getattr(pm, water_attr)
                ws.set_CF("GWP", cfs.get("water_gwp", 0.000127))
                ws.set_CF("HTC", cfs.get("water_htc", 1.40e-10))
                ws.set_CF("HTNC", cfs.get("water_htnc", 7.96e-11))
                ws.set_CF("ETOX", cfs.get("water_etox", 0.00538))
            except Exception:
                print(
                    f"[biosteam_worker] warning: could not set CFs on {water_attr}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    pm.system.simulate()

    # ------------------------------------------------------------------
    # Extract TEA results
    # ------------------------------------------------------------------
    msp = _safe_call(pm.MSP)
    # BioSTEAM exposes the TEA object as .tea (lowercase) on the process model;
    # `pm.system.TEA` is not a valid path and was raising AttributeError inside
    # _safe_call, which swallowed the exception and returned None. That made
    # every sim look like "success but zero economics", which then tripped the
    # zero-metric row guards downstream. Matches the v8 reference at
    # plastics-master-3/plastics/strap/process_model.py:952.
    tci = _safe_call(lambda: pm.tea.TCI)
    aoc = _safe_call(lambda: pm.tea.AOC)

    # ------------------------------------------------------------------
    # Extract LCA results
    # ------------------------------------------------------------------
    gwp = _safe_call(pm.GWP)
    htc = _safe_call(pm.HTC)
    htnc = _safe_call(pm.HTNC)
    etox = _safe_call(pm.ETOX)

    # ------------------------------------------------------------------
    # Water consumption (CT exists for all cases; BT only when facilities=True)
    # ------------------------------------------------------------------
    water_consumed = None
    water_circulated = None
    try:
        ct = pm.CT
        ct_blowdown = ct.blowdown_water.imass["Water"] / 1000
        ct_evap = ct.evaporation_water.imass["Water"] / 1000
        ct_cooling = ct.cooling_water.imass["Water"] / 1000

        if facilities:
            # C1 or C3: include BT contributions
            bt_blowdown = pm.BT.blowdown_water.imass["Water"] / 1000
            consumed_m3hr = ct_blowdown + ct_evap + bt_blowdown
            bt_flow_rate = pm.BT.design_results["Flow rate"] / 1000
            circulated_m3hr = bt_flow_rate + ct_cooling
        else:
            # C2: CT only, no BT
            consumed_m3hr = ct_blowdown + ct_evap
            circulated_m3hr = ct_cooling

        annual_factor = 24 * pm.tea.operating_days
        water_consumed = consumed_m3hr * annual_factor
        water_circulated = circulated_m3hr * annual_factor
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Resolve resin stream (shared by energy normalisation + waste diverted)
    # ------------------------------------------------------------------
    resin_stream = None
    for resin_attr in (f"{target_plastic}_resin", "PE_resin", "resin"):
        try:
            resin_stream = getattr(pm, resin_attr)
            if resin_stream.F_mass > 0:
                break
        except Exception:
            continue
    resin_F_mass = resin_stream.F_mass if resin_stream else None  # kg/hr

    # ------------------------------------------------------------------
    # Energy metrics (MJ/kg of product)
    # ------------------------------------------------------------------
    elec_raw = _safe_call(
        lambda: pm.system.get_electricity_consumption() * 3.6  # MJ/hr
    )
    heating_raw = _safe_call(
        lambda: pm.system.get_heating_duty() * 0.001  # MJ/hr
    )
    cooling_raw = _safe_call(
        lambda: pm.system.get_cooling_duty() * 0.001  # MJ/hr
    )

    # Normalise to MJ per kg resin product
    if resin_F_mass and resin_F_mass > 0:
        elec = elec_raw / resin_F_mass if elec_raw is not None else None
        heating = heating_raw / resin_F_mass if heating_raw is not None else None
        cooling = cooling_raw / resin_F_mass if cooling_raw is not None else None
    else:
        elec = heating = cooling = None

    total_energy = None
    if elec is not None and heating is not None and cooling is not None:
        total_energy = elec + heating + cooling

    # ------------------------------------------------------------------
    # Waste streams
    # ------------------------------------------------------------------
    waste_generated = _safe_call(
        lambda: pm.spent_activated_carbon.F_mass * pm.tea.operating_hours
    )
    waste_diverted = None
    if resin_stream is not None:
        waste_diverted = _safe_call(
            lambda: resin_stream.F_mass * pm.tea.operating_hours
        )

    # ------------------------------------------------------------------
    # Unit operations count
    # ------------------------------------------------------------------
    n_units = _safe_call(lambda: len(pm.system.units))

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    runtime = round(time.time() - t0, 3)

    result = {
        "success": True,
        "solvent": solvent,
        "target_plastic": target_plastic_orig,
        "energy_case": energy_case,
        "tea": {
            "msp_usd_per_kg": msp,
            "tci_usd": tci,
            "aoc_usd_per_yr": aoc,
        },
        "lca": {
            "gwp_kg_co2e_per_kg": gwp,
            "htc_ctuh_per_kg": htc,
            "htnc_ctuh_per_kg": htnc,
            "etox_ctue_per_kg": etox,
        },
        "operations": {
            "water_consumed_m3_yr": water_consumed,
            "water_circulated_m3_yr": water_circulated,
            "electricity_consumed_mj_per_kg": elec,
            "heating_duty_mj_per_kg": heating,
            "cooling_duty_mj_per_kg": cooling,
            "total_energy_mj_per_kg": total_energy,
            # Grid electricity intensity (C2/C3 only; C1 uses on-site CHP)
            "electricity_intensity_mj_per_kg": (
                elec if energy_case in ("C2", "C3") else None
            ),
            "waste_generated_kg_yr": waste_generated,
            "waste_diverted_kg_yr": waste_diverted,
            "unit_operations": n_units,
        },
        "runtime_seconds": runtime,
    }

    return result


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(
                json.dumps({
                    "success": False,
                    "error": "No JSON config provided via sys.argv[1]",
                    "error_type": "UsageError",
                })
            )
            sys.exit(1)

        config = json.loads(sys.argv[1])

        # Validate required fields
        for key in ("solvent", "target_plastic"):
            if key not in config:
                print(
                    json.dumps({
                        "success": False,
                        "error": f"Missing required field: '{key}'",
                        "error_type": "ValueError",
                    })
                )
                sys.exit(1)

        result = _run(config)
        print(json.dumps(result))

    except Exception as exc:
        # Always produce valid JSON, even on crash
        _cfg = config if "config" in locals() else {}
        error_result = {
            "success": False,
            "solvent": _cfg.get("solvent", "unknown") if isinstance(_cfg, dict) else "unknown",
            "target_plastic": _cfg.get("target_plastic", "unknown") if isinstance(_cfg, dict) else "unknown",
            "energy_case": _cfg.get("energy_case", "C1") if isinstance(_cfg, dict) else "C1",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        # Full traceback to stderr for debugging
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps(error_result))
        sys.exit(1)
