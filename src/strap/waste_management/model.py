"""
Pyomo model for multi-objective optimisation of multilayer plastic waste
management via a three-stage technology superstructure.

Translates the Julia/JuMP model from PIW Paper Dec 21 w Transportation.ipynb.
"""

import pyomo.environ as pyo
from strap.waste_management.data_loader import (
    I_SET, J_SET, K_SET, OTHERTECH,
    POLYMERS as DEFAULT_POLYMERS,
    ALL_SOLVENTS as DEFAULT_ALL_SOLVENTS,
    S_PE as DEFAULT_S_PE,
    S_EV1 as DEFAULT_S_EV1,
    S_EV2 as DEFAULT_S_EV2,
    SOLVENTS_BY_STAGE_POLYMER as DEFAULT_SOLVENTS_BY_STAGE_POLYMER,
)


def _max_strap_metric(strap, metric, wash, polymer, solvents):
    return max(
        (float(strap[metric].get((wash, polymer, solvent), 0.0)) for solvent in solvents),
        default=0.0,
    )


def estimate_metric_upper_bound(
    strap,
    other,
    metric,
    feed,
    *,
    solvents_by_stage_polymer=None,
    othertech=None,
):
    """Return a conservative but data-driven upper bound for a process metric."""
    stage_map = solvents_by_stage_polymer or DEFAULT_SOLVENTS_BY_STAGE_POLYMER
    strap_upper = sum(
        _max_strap_metric(strap, metric, wash, polymer, solvents)
        for wash, polymer_map in stage_map.items()
        for polymer, solvents in polymer_map.items()
    )
    othertech = list(othertech or OTHERTECH)
    other_upper = 3.0 * float(feed) * max(
        (float(other[metric].get(tech, 0.0)) for tech in othertech),
        default=0.0,
    )
    return strap_upper + other_upper


def build_model(data, config):
    """
    Build and return the Pyomo ConcreteModel.

    Parameters
    ----------
    data : dict
        Output of data_loader.load_all_data().
    config : dict
        Model parameters:
            Feed          : total flow (ton/yr), default 8000
            PE_f, PET_f, N6_f, EV_f : composition fractions
            Cpe, Cevoh, Cwte        : product prices ($/ton)
            UB_energy, UB_ghg, UB_withdrawal, UB_waste : upper bounds
            distances     : dict tech_key -> distance (miles)
            fc_t, vc_t    : transport fixed/variable cost
            p_strap       : STRAP capacity fraction
            ce_weights    : dict of circularity weights (energy, ghg, water, waste, subs)
    """
    strap = data["strap"]
    other = data["other"]
    sets = data.get("sets", {})

    POLYMERS = list(sets.get("P", DEFAULT_POLYMERS))
    ALL_SOLVENTS = list(sets.get("S", DEFAULT_ALL_SOLVENTS))
    S_PE = list(sets.get("S_PE", DEFAULT_S_PE))
    S_EV1 = list(sets.get("S_EV1", DEFAULT_S_EV1))
    S_EV2 = list(sets.get("S_EV2", DEFAULT_S_EV2))
    SOLVENTS_BY_STAGE_POLYMER = {
        wash: {
            polymer: list(solvents)
            for polymer, solvents in polymer_map.items()
        }
        for wash, polymer_map in (
            sets.get("S_BY_STAGE_POLYMER", DEFAULT_SOLVENTS_BY_STAGE_POLYMER)
        ).items()
    }
    I_RUNTIME = list(sets.get("I", I_SET))
    J_RUNTIME = list(sets.get("J", J_SET))
    K_RUNTIME = list(sets.get("K", K_SET))
    OTHERTECH_RUNTIME = list(sets.get("othertech", OTHERTECH))

    Feed  = config.get("Feed", 8000)
    BIG_M = config.get("BIG_M", 100000)

    polymer_fractions = {
        str(polymer): float(value)
        for polymer, value in (
            config.get("polymer_fractions")
            or {
                "PE": config.get("PE_f", 0.6),
                "PET": config.get("PET_f", 0.2),
                "N6": config.get("N6_f", 0.1),
                "EVOH": config.get("EV_f", 0.1),
            }
        ).items()
    }
    product_values_per_ton = {
        str(polymer): float(value)
        for polymer, value in (
            config.get("polymer_market_values_per_ton")
            or {
                "PE": config.get("Cpe", 1173),
                "EVOH": config.get("Cevoh", 8100),
                "PET": config.get("Cpet", 0.0),
            }
        ).items()
    }
    polymer_recovery_yields = {
        str(polymer): float(value)
        for polymer, value in (
            config.get("polymer_recovery_yields")
            or {}
        ).items()
    }
    gas_stage2_addon_capex_by_polymer = {
        str(polymer): float(value)
        for polymer, value in (
            config.get("gas_stage2_addon_capex_by_polymer")
            or {
                "PE": 1.996874495e6,
                "EVOH": 3.248331031e6,
            }
        ).items()
    }

    UB_energy     = config.get("UB_energy", 6.26e7)
    UB_ghg        = config.get("UB_ghg", 21303.35985408156)
    UB_withdrawal = config.get("UB_withdrawal", 14468.80855)
    UB_waste      = config.get("UB_waste", 1.92e6)

    dist = config.get("distances", {
        "strap": 0, "lf": 9.2, "we": 151, "py": 1034,
        "gas_er": 0, "gas_h2": 2036, "gas_h2cc": 2036,
    })
    fc_t = config.get("fc_t", 3.01)
    vc_t = config.get("vc_t", 0.07)

    # Gasification product economics
    products_heat      = config.get("products_heat", 583.33)
    products_elec      = config.get("products_electricity", 724.3693)
    price_heat         = config.get("price_heat", 0.13)
    price_elec         = config.get("price_elec", 0.0996)
    Cgas_er = products_heat * price_heat + products_elec * price_elec
    Cgas_pw = config.get("Cgas_pw", 110)

    # CE weights
    W_energy = config.get("ce_weights", {}).get("energy", 0.20)
    W_ghg    = config.get("ce_weights", {}).get("ghg", 0.20)
    W_water  = config.get("ce_weights", {}).get("water", 0.20)
    W_waste  = config.get("ce_weights", {}).get("waste", 0.20)
    W_subs   = config.get("ce_weights", {}).get("subs", 0.20)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    m = pyo.ConcreteModel("PIW_WasteManagement")
    object.__setattr__(m, "_strap_feed", float(Feed))
    object.__setattr__(m, "_strap_polymer_fractions", dict(polymer_fractions))
    object.__setattr__(m, "_strap_polymer_recovery_yields", dict(polymer_recovery_yields))
    stage1_solvents_by_polymer = {
        polymer: list((SOLVENTS_BY_STAGE_POLYMER.get("Wash 1") or {}).get(polymer, []))
        for polymer in POLYMERS
    }
    stage2_solvents_by_polymer = {
        polymer: list((SOLVENTS_BY_STAGE_POLYMER.get("Wash 2") or {}).get(polymer, []))
        for polymer in POLYMERS
    }

    def _polymer_fraction(polymer: str) -> float:
        return float(polymer_fractions.get(polymer, 0.0))

    def _product_value(polymer: str) -> float:
        return float(product_values_per_ton.get(polymer, 0.0))

    def _recovery_yield(polymer: str) -> float:
        return float(polymer_recovery_yields.get(polymer, 0.97))

    def _has_stage1(tech: str) -> bool:
        return tech in I_RUNTIME

    def _has_stage2(tech: str) -> bool:
        return tech in J_RUNTIME

    def _has_stage3(tech: str) -> bool:
        return tech in K_RUNTIME

    def _stage1_var(tech: str):
        return m.x[tech] if _has_stage1(tech) else 0.0

    def _stage2_var(tech: str):
        return m.y[tech] if _has_stage2(tech) else 0.0

    def _stage3_var(tech: str):
        return m.z[tech] if _has_stage3(tech) else 0.0

    def _stage1_flow(tech: str):
        return m.F[tech] if _has_stage1(tech) else 0.0

    def _stage2_flow(tech: str):
        return m.L[tech] if _has_stage2(tech) else 0.0

    def _stage3_flow(tech: str):
        return m.N[tech] if _has_stage3(tech) else 0.0

    # --- Index sets ---
    m.I = pyo.Set(initialize=I_RUNTIME)
    m.J = pyo.Set(initialize=J_RUNTIME)
    m.K = pyo.Set(initialize=K_RUNTIME)
    m.OT = pyo.Set(initialize=OTHERTECH_RUNTIME)
    m.P = pyo.Set(initialize=POLYMERS)
    m.S = pyo.Set(initialize=ALL_SOLVENTS)
    m.S_PE = pyo.Set(initialize=S_PE)
    m.S_EV1 = pyo.Set(initialize=S_EV1)
    m.S_EV2 = pyo.Set(initialize=S_EV2)
    m.W = pyo.Set(initialize=["Wash 1", "Wash 2"])

    # --- Decision variables ---
    # Stage 1
    m.F = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.x = pyo.Var(m.I, within=pyo.Binary)

    # STRAP 1 polymer removal & solvent selection
    m.R = pyo.Var(m.P, within=pyo.NonNegativeReals)
    m.a = pyo.Var(m.P, m.S, within=pyo.Binary)

    # Leftover from STRAP 1
    m.LF = pyo.Var(within=pyo.NonNegativeReals)

    # Stage 2
    m.L = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.J, within=pyo.Binary)
    m.p_gas = pyo.Var(within=pyo.Binary)  # gasification capex selector 1
    m.t_gas = pyo.Var(within=pyo.Binary)  # gasification capex selector 2

    # STRAP 2 polymer removal & solvent selection
    m.b = pyo.Var(m.P, m.S, within=pyo.Binary)
    m.T = pyo.Var(m.P, within=pyo.NonNegativeReals)
    m.LSS = pyo.Var(m.P, m.S, within=pyo.NonNegativeReals)

    # Leftover from STRAP 2
    m.LS = pyo.Var(within=pyo.NonNegativeReals)

    # Stage 3
    m.z = pyo.Var(m.K, within=pyo.Binary)
    m.N = pyo.Var(m.K, within=pyo.NonNegativeReals)

    # Sales
    m.Sales = pyo.Var(within=pyo.NonNegativeReals)

    # Metrics
    m.E     = pyo.Var(within=pyo.NonNegativeReals)   # total energy
    m.RE    = pyo.Var(within=pyo.NonNegativeReals)   # renewable energy
    m.D_ghg = pyo.Var(within=pyo.NonNegativeReals)   # direct GHG
    m.I_ghg = pyo.Var(within=pyo.NonNegativeReals)   # indirect GHG
    m.W_w   = pyo.Var(within=pyo.NonNegativeReals)   # water withdrawal
    m.W_r   = pyo.Var(within=pyo.NonNegativeReals)   # water recycled
    m.W_g   = pyo.Var(within=pyo.NonNegativeReals)   # waste generated
    m.W_d   = pyo.Var(within=pyo.NonNegativeReals)   # waste disposal

    # Circularity sub-indices
    m.Econsumed  = pyo.Var(within=pyo.NonNegativeReals)
    m.Erenewable = pyo.Var(within=pyo.NonNegativeReals)
    m.z_energy   = pyo.Var(within=pyo.Binary)
    m.z_renew    = pyo.Var(within=pyo.Binary)

    m.z_emissions  = pyo.Var(within=pyo.Binary)
    m.Em_generated = pyo.Var(within=pyo.NonNegativeReals)

    m.z_water      = pyo.Var(within=pyo.Binary)
    m.W_withdrawal = pyo.Var(within=pyo.NonNegativeReals)
    m.W_recycled   = pyo.Var(within=pyo.NonNegativeReals)

    m.z_waste    = pyo.Var(within=pyo.Binary)
    m.W_generated = pyo.Var(within=pyo.NonNegativeReals)
    m.z_disposal = pyo.Var(within=pyo.Binary)
    m.W_diverted = pyo.Var(within=pyo.NonNegativeReals)

    # Scores
    m.E_score     = pyo.Var(within=pyo.NonNegativeReals)
    m.GHG_score   = pyo.Var(within=pyo.NonNegativeReals)
    m.Water_score = pyo.Var(within=pyo.NonNegativeReals)
    m.Waste_score = pyo.Var(within=pyo.NonNegativeReals)
    m.Subs_score  = pyo.Var(within=pyo.NonNegativeReals)

    m.TotalEmissions     = pyo.Var(within=pyo.NonNegativeReals)
    m.CEoverall          = pyo.Var(within=pyo.Reals)
    m.OperationalCost    = pyo.Var(within=pyo.Reals)
    m.CapitalCost        = pyo.Var(within=pyo.NonNegativeReals)
    m.TC                 = pyo.Var(m.K, within=pyo.NonNegativeReals)
    m.TransportationCost = pyo.Var(within=pyo.NonNegativeReals)

    # -----------------------------------------------------------------------
    # Helper to get strap data safely
    # -----------------------------------------------------------------------
    def sd(metric, w, p, s):
        return strap[metric].get((w, p, s), 0.0)

    def od(metric, t):
        return other[metric].get(t, 0.0)

    ghg_upper = max(
        UB_ghg,
        estimate_metric_upper_bound(
            strap, other, "direct_ghg", Feed, solvents_by_stage_polymer=SOLVENTS_BY_STAGE_POLYMER, othertech=OTHERTECH_RUNTIME
        )
        + estimate_metric_upper_bound(
            strap, other, "indirect_ghg", Feed, solvents_by_stage_polymer=SOLVENTS_BY_STAGE_POLYMER, othertech=OTHERTECH_RUNTIME
        ),
    )
    water_upper = max(
        UB_withdrawal,
        estimate_metric_upper_bound(
            strap, other, "water_with", Feed, solvents_by_stage_polymer=SOLVENTS_BY_STAGE_POLYMER, othertech=OTHERTECH_RUNTIME
        ),
    )
    waste_upper = max(
        UB_waste,
        estimate_metric_upper_bound(
            strap, other, "waste", Feed, solvents_by_stage_polymer=SOLVENTS_BY_STAGE_POLYMER, othertech=OTHERTECH_RUNTIME
        ),
    )
    energy_upper = max(
        UB_energy,
        estimate_metric_upper_bound(
            strap, other, "total_energy", Feed, solvents_by_stage_polymer=SOLVENTS_BY_STAGE_POLYMER, othertech=OTHERTECH_RUNTIME
        ),
    )
    renewable_upper = max(
        energy_upper,
        estimate_metric_upper_bound(
            strap, other, "renewable", Feed, solvents_by_stage_polymer=SOLVENTS_BY_STAGE_POLYMER, othertech=OTHERTECH_RUNTIME
        ),
    )

    energy_score_slack = max(1.0, energy_upper / UB_energy)
    ghg_score_slack = max(1.0, ghg_upper / UB_ghg)
    water_score_slack = max(1.0, water_upper / UB_withdrawal)
    waste_score_slack = max(1.0, waste_upper / UB_waste)

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Stage 1 flow and technology selection
    # -----------------------------------------------------------------------
    m.flow_def = pyo.Constraint(m.I,
        rule=lambda m, i: m.F[i] == Feed * m.x[i])
    m.no_pyrolysis_stage1 = pyo.Constraint(
        rule=lambda m: (m.x["py"] == 0) if _has_stage1("py") else pyo.Constraint.Feasible
    )
    m.one_first_tech = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 1)

    # STRAP 1 polymer removal
    m.R_def = pyo.Constraint(
        m.P,
        rule=lambda m, p: m.R[p] == Feed * _polymer_fraction(p) * sum(m.a[p, s] for s in ALL_SOLVENTS)
    )

    m.invalid_stage1_pairs = pyo.ConstraintList()
    m.invalid_stage2_pairs = pyo.ConstraintList()
    for polymer in POLYMERS:
        allowed_stage1 = set(stage1_solvents_by_polymer.get(polymer, []))
        allowed_stage2 = set(stage2_solvents_by_polymer.get(polymer, []))
        for solvent in ALL_SOLVENTS:
            if solvent not in allowed_stage1:
                m.invalid_stage1_pairs.add(m.a[polymer, solvent] == 0)
            if solvent not in allowed_stage2:
                m.invalid_stage2_pairs.add(m.b[polymer, solvent] == 0)

    # -----------------------------------------------------------------------
    # CONSTRAINTS – STRAP linkage and Stage 2/3 flow
    # -----------------------------------------------------------------------
    m.a_links_st1 = pyo.Constraint(expr=
        sum(m.a[p, s] for p in POLYMERS for s in ALL_SOLVENTS) == m.x["st1"])

    m.leftover_def = pyo.Constraint(expr=
        m.LF == m.F["st1"] - sum(m.R[p] for p in POLYMERS))

    # Stage 2 flow with Big-M
    m.L_ub = pyo.Constraint(m.J,
        rule=lambda m, j: m.L[j] <= BIG_M * m.y[j])
    m.L_lb = pyo.Constraint(m.J,
        rule=lambda m, j: m.L[j] >= m.LF - BIG_M * (1 - m.y[j]))
    m.L_ub2 = pyo.Constraint(m.J,
        rule=lambda m, j: m.L[j] <= m.LF)

    m.one_second_tech = pyo.Constraint(expr=
        sum(m.y[j] for j in m.J) == m.x["st1"])
    m.b_links_st2 = pyo.Constraint(expr=
        sum(m.b[p, s] for p in POLYMERS for s in ALL_SOLVENTS) == m.y["st2"])

    # STRAP 2 polymer removal
    m.T_def = pyo.Constraint(
        m.P,
        rule=lambda m, p: m.T[p] == Feed * _polymer_fraction(p) * sum(m.b[p, s] for s in ALL_SOLVENTS)
    )

    m.LS_def = pyo.Constraint(expr=
        m.LS == m.L["st2"] - sum(m.T[p] for p in POLYMERS))

    # Stage 3
    m.one_third_tech = pyo.Constraint(expr=
        sum(m.z[k] for k in m.K) == m.y["st2"])

    m.N_ub = pyo.Constraint(m.K,
        rule=lambda m, k: m.N[k] <= BIG_M * m.z[k])
    m.N_lb = pyo.Constraint(m.K,
        rule=lambda m, k: m.N[k] >= m.LS - BIG_M * (1 - m.z[k]))
    m.N_ub2 = pyo.Constraint(m.K,
        rule=lambda m, k: m.N[k] <= m.LS)
    m.no_pyrolysis_stage3 = pyo.Constraint(
        rule=lambda m: (m.N["py"] == 0) if _has_stage3("py") else pyo.Constraint.Feasible
    )

    # Each polymer removed at most once across all washes
    m.polymer_once = pyo.Constraint(m.P,
        rule=lambda m, p: (
            pyo.Constraint.Feasible
            if not ALL_SOLVENTS
            else (
                sum(m.a[p, s] for s in ALL_SOLVENTS)
                + sum(m.b[p, s] for s in ALL_SOLVENTS) <= 1
            )
        ))

    # LSS linearisation for operational cost of STRAP 2
    m.LSS_lb = pyo.Constraint(m.P, m.S,
        rule=lambda m, p, s: m.LSS[p, s] >= m.L["st2"] - 100000 * (1 - m.b[p, s]))
    m.LSS_ub = pyo.Constraint(m.P, m.S,
        rule=lambda m, p, s: m.LSS[p, s] <= m.L["st2"])
    m.LSS_ub2 = pyo.Constraint(m.P, m.S,
        rule=lambda m, p, s: m.LSS[p, s] <= 10000 * m.b[p, s])

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Environmental metrics (energy, GHG, water, waste)
    # -----------------------------------------------------------------------
    def _metric_expr(m, metric):
        """Sum of STRAP + other-tech contributions for a given metric."""
        return (
            sum(
                sd(metric, "Wash 1", polymer, solvent) * m.a[polymer, solvent]
                for polymer in POLYMERS
                for solvent in stage1_solvents_by_polymer.get(polymer, [])
            )
            + sum(
                sd(metric, "Wash 2", polymer, solvent) * m.b[polymer, solvent]
                for polymer in POLYMERS
                for solvent in stage2_solvents_by_polymer.get(polymer, [])
            )
            + sum(
                od(metric, t) * (_stage1_flow(t) + _stage2_flow(t) + _stage3_flow(t))
                for t in OTHERTECH_RUNTIME
            )
        )

    m.E_def = pyo.Constraint(expr=m.E == _metric_expr(m, "total_energy"))
    m.RE_def = pyo.Constraint(expr=m.RE == _metric_expr(m, "renewable"))
    m.D_ghg_def = pyo.Constraint(expr=m.D_ghg == _metric_expr(m, "direct_ghg"))
    m.I_ghg_def = pyo.Constraint(expr=m.I_ghg == _metric_expr(m, "indirect_ghg"))
    m.W_w_def = pyo.Constraint(expr=m.W_w == _metric_expr(m, "water_with"))
    m.W_r_def = pyo.Constraint(expr=m.W_r == _metric_expr(m, "water_recyc"))
    m.W_g_def = pyo.Constraint(expr=m.W_g == _metric_expr(m, "waste"))
    m.W_d_def = pyo.Constraint(expr=m.W_d == _metric_expr(m, "disposal"))

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Circularity indicator: Energy score
    # -----------------------------------------------------------------------
    small_d = 1e-3

    # Energy consumed sub-index
    # Score maxes at 1.0
    m.econ_bound = pyo.Constraint(expr=m.Econsumed <= 1.0)
    m.e_ub = pyo.Constraint(expr=m.E <= UB_energy + (1 - m.z_energy) * max(0.0, energy_upper - UB_energy))
    m.e_lb = pyo.Constraint(expr=m.E >= UB_energy - m.z_energy * UB_energy)
    m.econ_ub = pyo.Constraint(expr=
        m.Econsumed <= 1 - m.E / UB_energy + energy_score_slack * (1 - m.z_energy))
    m.econ_lb = pyo.Constraint(expr=
        m.Econsumed >= 1 - m.E / UB_energy - energy_score_slack * (1 - m.z_energy))
    m.econ_bin = pyo.Constraint(expr=m.Econsumed <= m.z_energy)

    # Renewable energy sub-index
    m.erenew_bound = pyo.Constraint(expr=m.Erenewable <= 1.0)
    m.erenew_ub = pyo.Constraint(expr=m.E <= (1 - m.z_renew) * energy_upper)
    m.erenew_lb = pyo.Constraint(expr=m.E >= m.RE - m.z_renew * renewable_upper + small_d)
    m.erenew_prod_ub = pyo.Constraint(expr=
        m.E * m.Erenewable <= m.RE + m.z_renew * energy_upper)
    m.erenew_prod_lb = pyo.Constraint(expr=
        m.E * m.Erenewable >= m.RE - m.z_renew * renewable_upper)
    m.erenew_val_ub = pyo.Constraint(expr=
        m.Erenewable <= 1 + (1 - m.z_renew))
    m.erenew_val_lb = pyo.Constraint(expr=
        m.Erenewable >= 1 - (1 - m.z_renew))

    m.E_score_def = pyo.Constraint(expr=
        m.E_score == 0.6 * m.Econsumed + 0.4 * m.Erenewable)

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Circularity indicator: GHG score
    # -----------------------------------------------------------------------
    m.emgen_bound = pyo.Constraint(expr=m.Em_generated <= 1.0)
    m.em_ub = pyo.Constraint(expr=
        m.D_ghg + m.I_ghg <= UB_ghg + (1 - m.z_emissions) * max(0.0, ghg_upper - UB_ghg))
    m.em_lb = pyo.Constraint(expr=
        m.D_ghg + m.I_ghg >= UB_ghg - m.z_emissions * UB_ghg)
    m.emgen_ub = pyo.Constraint(expr=
        m.Em_generated <= 1 - (m.D_ghg + m.I_ghg) / UB_ghg + ghg_score_slack * (1 - m.z_emissions))
    m.emgen_lb = pyo.Constraint(expr=
        m.Em_generated >= 1 - (m.D_ghg + m.I_ghg) / UB_ghg - ghg_score_slack * (1 - m.z_emissions))
    m.emgen_bin = pyo.Constraint(expr=m.Em_generated <= m.z_emissions)

    m.GHG_score_def = pyo.Constraint(expr=m.GHG_score == m.Em_generated)

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Circularity indicator: Water score
    # -----------------------------------------------------------------------
    m.wwithdraw_bound = pyo.Constraint(expr=m.W_withdrawal <= 1.0)
    m.ww_ub = pyo.Constraint(expr=
        m.W_w <= UB_withdrawal + (1 - m.z_water) * max(0.0, water_upper - UB_withdrawal))
    m.ww_lb = pyo.Constraint(expr=
        m.W_w >= UB_withdrawal - m.z_water * UB_withdrawal)
    m.wwithdraw_ub = pyo.Constraint(expr=
        m.W_withdrawal <= 1 - m.W_w / UB_withdrawal + water_score_slack * (1 - m.z_water))
    m.wwithdraw_lb = pyo.Constraint(expr=
        m.W_withdrawal >= 1 - m.W_w / UB_withdrawal - water_score_slack * (1 - m.z_water))
    m.wwithdraw_bin = pyo.Constraint(expr=m.W_withdrawal <= m.z_water)

    # Water recycled (bilinear: W_recycled * W_w <= W_r)
    m.wrecyc_cap = pyo.Constraint(expr=m.W_recycled <= 1.0)
    m.wrecyc_prod = pyo.Constraint(expr=m.W_recycled * m.W_w <= m.W_r)

    m.Water_score_def = pyo.Constraint(expr=
        m.Water_score == 0.5 * m.W_withdrawal + 0.5 * m.W_recycled)

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Circularity indicator: Waste score
    # -----------------------------------------------------------------------
    M_d = 1.0

    m.wgen_bound = pyo.Constraint(expr=m.W_generated <= 1.0)
    m.wg_ub = pyo.Constraint(expr=
        m.W_g <= UB_waste + (1 - m.z_waste) * max(0.0, waste_upper - UB_waste))
    m.wg_lb = pyo.Constraint(expr=
        m.W_g >= UB_waste - m.z_waste * UB_waste)
    m.wgen_ub = pyo.Constraint(expr=
        m.W_generated <= 1 - m.W_g / UB_waste + waste_score_slack * (1 - m.z_waste))
    m.wgen_lb = pyo.Constraint(expr=
        m.W_generated >= 1 - m.W_g / UB_waste - waste_score_slack * (1 - m.z_waste))
    m.wgen_bin = pyo.Constraint(expr=m.W_generated <= m.z_waste)

    # Waste disposal / diversion
    m.wdiv_bound = pyo.Constraint(expr=m.W_diverted <= 1.0)
    m.wd_ub = pyo.Constraint(expr=m.W_d <= (1 - m.z_disposal) * M_d * Feed)
    m.wd_lb = pyo.Constraint(expr=m.W_d >= small_d - m.z_disposal * M_d * Feed)
    m.wdiv_ub1 = pyo.Constraint(expr=m.W_diverted <= 1 + M_d * (1 - m.z_disposal))
    m.wdiv_lb1 = pyo.Constraint(expr=m.W_diverted >= 1 - M_d * (1 - m.z_disposal))
    m.wdiv_ub2 = pyo.Constraint(expr=
        m.W_diverted <= 1 - m.W_d / Feed + M_d * m.z_disposal)
    m.wdiv_lb2 = pyo.Constraint(expr=
        m.W_diverted >= 1 - m.W_d / Feed - M_d * m.z_disposal)

    m.Waste_score_def = pyo.Constraint(expr=
        m.Waste_score == 0.5 * m.W_diverted + 0.5 * m.W_generated)

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Substitutability score
    # -----------------------------------------------------------------------
    m.Subs_score_def = pyo.Constraint(expr=
        m.Subs_score == m.Sales * (1.0 / 1.61e7))

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Transportation costs
    # -----------------------------------------------------------------------
    m.TC_def = pyo.Constraint(m.K,
        rule=lambda m, k: m.TC[k] == (m.F[k] + m.L[k] + m.N[k]) * (dist[k] * vc_t + fc_t))

    m.TransCost_def = pyo.Constraint(expr=
        m.TransportationCost == sum(m.TC[k] for k in m.K)
        + (dist["strap"] * vc_t + fc_t) * (
            Feed * sum(m.a[p, s] for p in POLYMERS for s in ALL_SOLVENTS)
            + sum(m.LSS[p, s] for p in POLYMERS for s in ALL_SOLVENTS)
        ))

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Capital cost
    # -----------------------------------------------------------------------
    m.gas_stage2_selector = pyo.Var(m.P, within=pyo.Binary)
    m.legacy_p_gas_zero = pyo.Constraint(expr=m.p_gas == 0)
    m.legacy_t_gas_zero = pyo.Constraint(expr=m.t_gas == 0)
    m.gas_stage2_selector_ub = pyo.ConstraintList()
    m.gas_stage2_selector_lb = pyo.ConstraintList()
    m.gas_stage2_selector_zero = pyo.ConstraintList()
    for polymer in POLYMERS:
        addon = float(gas_stage2_addon_capex_by_polymer.get(polymer, 0.0))
        if addon <= 0:
            m.gas_stage2_selector_zero.add(m.gas_stage2_selector[polymer] == 0)
            continue
        m.gas_stage2_selector_ub.add(m.gas_stage2_selector[polymer] <= _stage2_var("gas_er"))
        m.gas_stage2_selector_lb.add(
            m.gas_stage2_selector[polymer]
            >= _stage2_var("gas_er") + sum(m.a[polymer, s] for s in ALL_SOLVENTS) - 1
        )

    m.CapCost_def = pyo.Constraint(expr=
        m.CapitalCost == (
            sum(
                m.a[polymer, solvent] * sd("capex", "Wash 1", polymer, solvent)
                for polymer in POLYMERS
                for solvent in stage1_solvents_by_polymer.get(polymer, [])
            )
            + sum(
                m.b[polymer, solvent] * sd("capex", "Wash 2", polymer, solvent)
                for polymer in POLYMERS
                for solvent in stage2_solvents_by_polymer.get(polymer, [])
            )
            + _stage1_var("py") * od("capex", "py")
            + _stage1_var("gas_er") * od("capex", "gas_er")
            + sum(
                m.gas_stage2_selector[polymer] * float(gas_stage2_addon_capex_by_polymer.get(polymer, 0.0))
                for polymer in POLYMERS
            )
            + _stage3_var("gas_er") * 1.680302711e6
        ))

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Operational cost
    # -----------------------------------------------------------------------
    m.OpCost_def = pyo.Constraint(expr=
        m.OperationalCost == (
            sum(
                m.a[polymer, solvent] * sd("opex", "Wash 1", polymer, solvent)
                for polymer in POLYMERS
                for solvent in stage1_solvents_by_polymer.get(polymer, [])
            )
            + sum(
                od("opex", t) * (_stage1_flow(t) + _stage2_flow(t) + _stage3_flow(t))
                for t in OTHERTECH_RUNTIME
            )
            + sum(
                m.b[polymer, solvent] * sd("opex", "Wash 2", polymer, solvent)
                for polymer in POLYMERS
                for solvent in stage2_solvents_by_polymer.get(polymer, [])
            )
        ))

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Sales revenue
    # -----------------------------------------------------------------------
    m.Sales_def = pyo.Constraint(expr=
        m.Sales == (
            sum(
                Feed * _polymer_fraction(polymer) * _recovery_yield(polymer) * _product_value(polymer)
                * sum(m.a[polymer, solvent] for solvent in stage1_solvents_by_polymer.get(polymer, []))
                for polymer in POLYMERS
            )
            + Cgas_pw * _stage1_flow("we") + Cgas_er * _stage1_flow("gas_er")
            + Cgas_pw * _stage1_flow("gas_h2") + Cgas_pw * _stage1_flow("gas_h2cc")
            + Cgas_pw * _stage2_flow("we") + Cgas_er * _stage2_flow("gas_er")
            + Cgas_pw * _stage2_flow("gas_h2") + Cgas_pw * _stage2_flow("gas_h2cc")
            + sum(
                Feed * _polymer_fraction(polymer) * _recovery_yield(polymer) * _product_value(polymer)
                * sum(m.b[polymer, solvent] for solvent in stage2_solvents_by_polymer.get(polymer, []))
                for polymer in POLYMERS
            )
            + Cgas_pw * _stage3_flow("we") + Cgas_er * _stage3_flow("gas_er")
            + Cgas_pw * _stage3_flow("gas_h2") + Cgas_pw * _stage3_flow("gas_h2cc")
        ))

    # -----------------------------------------------------------------------
    # EXPRESSIONS – Profit, total emissions, CE overall
    # -----------------------------------------------------------------------
    m.Profit = pyo.Expression(expr=
        m.Sales - (m.CapitalCost + m.OperationalCost + m.TransportationCost))

    m.TotalEmissions_def = pyo.Constraint(expr=
        m.TotalEmissions == (
            sum(
                m.a[polymer, solvent] * sd("gwp", "Wash 1", polymer, solvent)
                for polymer in POLYMERS
                for solvent in stage1_solvents_by_polymer.get(polymer, [])
            )
            + sum(
                od("gwp", t) * (_stage1_flow(t) + _stage2_flow(t) + _stage3_flow(t))
                for t in OTHERTECH_RUNTIME
            )
            + sum(
                m.b[polymer, solvent] * sd("gwp", "Wash 2", polymer, solvent)
                for polymer in POLYMERS
                for solvent in stage2_solvents_by_polymer.get(polymer, [])
            )
        ))

    m.CE_def = pyo.Constraint(expr=
        m.CEoverall == 1e6 * (
            W_energy * m.E_score
            + W_ghg * m.GHG_score
            + W_water * m.Water_score
            + W_waste * m.Waste_score
            + W_subs * m.Subs_score
        ))

    return m
