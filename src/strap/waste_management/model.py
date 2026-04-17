"""
Pyomo model for multi-objective optimisation of multilayer plastic waste
management via a three-stage technology superstructure.

Translates the Julia/JuMP model from PIW Paper Dec 21 w Transportation.ipynb.
"""

import pyomo.environ as pyo
from strap.waste_management.data_loader import (
    I_SET, J_SET, K_SET, OTHERTECH, POLYMERS, ALL_SOLVENTS,
    S_PE, S_EV1, S_EV2,
)


def _max_strap_metric(strap, metric, wash, polymer, solvents):
    return max(
        (float(strap[metric].get((wash, polymer, solvent), 0.0)) for solvent in solvents),
        default=0.0,
    )


def estimate_metric_upper_bound(strap, other, metric, feed):
    """Return a conservative but data-driven upper bound for a process metric."""
    strap_upper = (
        _max_strap_metric(strap, metric, "Wash 1", "PE", S_PE)
        + _max_strap_metric(strap, metric, "Wash 1", "EVOH", S_EV1)
        + _max_strap_metric(strap, metric, "Wash 2", "PE", S_PE)
        + _max_strap_metric(strap, metric, "Wash 2", "EVOH", S_EV2)
    )
    other_upper = 3.0 * float(feed) * max(
        (float(other[metric].get(tech, 0.0)) for tech in OTHERTECH),
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

    Feed  = config.get("Feed", 8000)
    PE_f  = config.get("PE_f", 0.6)
    PET_f = config.get("PET_f", 0.2)
    N6_f  = config.get("N6_f", 0.1)
    EV_f  = config.get("EV_f", 0.1)
    BIG_M = config.get("BIG_M", 100000)

    Cpe   = config.get("Cpe", 1173)
    Cevoh = config.get("Cevoh", 8100)
    Cwte  = config.get("Cwte", 259.57)

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

    Cap1 = Feed - Feed * EV_f   # capacity wash 2 if EVOH removed first
    Cap2 = Feed - Feed * PE_f   # capacity wash 2 if PE removed first

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

    # --- Index sets ---
    m.I = pyo.Set(initialize=I_SET)
    m.J = pyo.Set(initialize=J_SET)
    m.K = pyo.Set(initialize=K_SET)
    m.OT = pyo.Set(initialize=OTHERTECH)
    m.P = pyo.Set(initialize=POLYMERS)
    m.S = pyo.Set(initialize=ALL_SOLVENTS)
    m.S_PE = pyo.Set(initialize=S_PE)
    m.S_EV1 = pyo.Set(initialize=S_EV1)
    m.S_EV2 = pyo.Set(initialize=S_EV2)

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

    energy_upper = max(UB_energy, estimate_metric_upper_bound(strap, other, "total_energy", Feed))
    renewable_upper = max(energy_upper, estimate_metric_upper_bound(strap, other, "renewable", Feed))
    ghg_upper = max(UB_ghg, estimate_metric_upper_bound(strap, other, "direct_ghg", Feed) + estimate_metric_upper_bound(strap, other, "indirect_ghg", Feed))
    water_upper = max(UB_withdrawal, estimate_metric_upper_bound(strap, other, "water_with", Feed))
    waste_upper = max(UB_waste, estimate_metric_upper_bound(strap, other, "waste", Feed))

    energy_score_slack = max(1.0, energy_upper / UB_energy)
    ghg_score_slack = max(1.0, ghg_upper / UB_ghg)
    water_score_slack = max(1.0, water_upper / UB_withdrawal)
    waste_score_slack = max(1.0, waste_upper / UB_waste)

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Stage 1 flow and technology selection
    # -----------------------------------------------------------------------
    m.flow_def = pyo.Constraint(m.I,
        rule=lambda m, i: m.F[i] == Feed * m.x[i])
    m.no_pyrolysis_stage1 = pyo.Constraint(expr=m.x["py"] == 0)
    m.one_first_tech = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 1)

    # STRAP 1 polymer removal
    m.R_PE = pyo.Constraint(expr=
        m.R["PE"] == Feed * PE_f * sum(m.a["PE", s] for s in ALL_SOLVENTS))
    m.R_EVOH = pyo.Constraint(expr=
        m.R["EVOH"] == Feed * EV_f * sum(m.a["EVOH", s] for s in ALL_SOLVENTS))

    # Incompatible solvent-polymer pairings
    m.evoh_no_pe_solv_a = pyo.Constraint(m.S_PE,
        rule=lambda m, s: m.a["EVOH", s] == 0)
    m.evoh_no_pe_solv_b = pyo.Constraint(m.S_PE,
        rule=lambda m, s: m.b["EVOH", s] == 0)
    m.evoh_no_pe_solv_lss = pyo.Constraint(m.S_PE,
        rule=lambda m, s: m.LSS["EVOH", s] == 0)

    # PE cannot use EVOH solvents
    ev_all = list(set(S_EV1) | set(S_EV2))
    m.S_EV_all = pyo.Set(initialize=ev_all)
    m.pe_no_ev_solv_a = pyo.Constraint(m.S_EV_all,
        rule=lambda m, s: m.a["PE", s] == 0)
    m.pe_no_ev_solv_b = pyo.Constraint(m.S_EV_all,
        rule=lambda m, s: m.b["PE", s] == 0)
    m.pe_no_ev_solv_lss = pyo.Constraint(m.S_EV_all,
        rule=lambda m, s: m.LSS["PE", s] == 0)

    # Wash 1 EVOH can only use S_EV1 solvents (not S_EV2 only solvents)
    s_ev2_only = list(set(S_EV2) - set(S_EV1))
    m.S_EV2_only = pyo.Set(initialize=s_ev2_only)
    m.evoh_wash1_restriction = pyo.Constraint(m.S_EV2_only,
        rule=lambda m, s: m.a["EVOH", s] == 0)

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
        sum(m.y[j] for j in J_SET) == m.x["st1"])
    m.b_links_st2 = pyo.Constraint(expr=
        sum(m.b[p, s] for p in POLYMERS for s in ALL_SOLVENTS) == m.y["st2"])

    # STRAP 2 polymer removal
    m.T_PE = pyo.Constraint(expr=
        m.T["PE"] == Feed * PE_f * sum(m.b["PE", s] for s in ALL_SOLVENTS))
    m.T_EVOH = pyo.Constraint(expr=
        m.T["EVOH"] == Feed * EV_f * sum(m.b["EVOH", s] for s in ALL_SOLVENTS))

    m.LS_def = pyo.Constraint(expr=
        m.LS == m.L["st2"] - sum(m.T[p] for p in POLYMERS))

    # Stage 3
    m.one_third_tech = pyo.Constraint(expr=
        sum(m.z[k] for k in K_SET) == m.y["st2"])

    m.N_ub = pyo.Constraint(m.K,
        rule=lambda m, k: m.N[k] <= BIG_M * m.z[k])
    m.N_lb = pyo.Constraint(m.K,
        rule=lambda m, k: m.N[k] >= m.LS - BIG_M * (1 - m.z[k]))
    m.N_ub2 = pyo.Constraint(m.K,
        rule=lambda m, k: m.N[k] <= m.LS)
    m.no_pyrolysis_stage3 = pyo.Constraint(expr=m.N["py"] == 0)

    # Each polymer removed at most once across all washes
    m.polymer_once = pyo.Constraint(m.P,
        rule=lambda m, p: (
            sum(m.a[p, s] for s in ALL_SOLVENTS)
            + sum(m.b[p, s] for s in ALL_SOLVENTS) <= 1
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
            sum(sd(metric, "Wash 1", "PE", s) * m.a["PE", s] for s in S_PE)
            + sum(sd(metric, "Wash 1", "EVOH", s) * m.a["EVOH", s] for s in S_EV1)
            + sum(sd(metric, "Wash 2", "PE", s) * m.b["PE", s] for s in S_PE)
            + sum(sd(metric, "Wash 2", "EVOH", s) * m.b["EVOH", s] for s in S_EV2)
            + sum(od(metric, t) * (m.F[t] + m.L[t] + m.N[t]) for t in OTHERTECH)
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
        m.TransportationCost == sum(m.TC[k] for k in K_SET)
        + (dist["strap"] * vc_t + fc_t) * (
            Feed * sum(m.a[p, s] for p in POLYMERS for s in ALL_SOLVENTS)
            + Cap1 * sum(m.b["PE", s] for s in S_PE)
            + Cap2 * sum(m.b["EVOH", s] for s in S_EV2)
        ))

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Capital cost
    # -----------------------------------------------------------------------
    m.p_gas_ub = pyo.Constraint(expr=m.p_gas <= m.y["gas_er"])
    m.t_gas_ub = pyo.Constraint(expr=m.t_gas <= m.y["gas_er"])
    m.pt_excl = pyo.Constraint(expr=m.p_gas + m.t_gas <= 1)
    m.p_gas_lb = pyo.Constraint(expr=
        m.p_gas >= m.y["gas_er"] + sum(m.a["PE", s] for s in S_PE) - 1)
    m.t_gas_lb = pyo.Constraint(expr=
        m.t_gas >= m.y["gas_er"] + sum(m.a["EVOH", s] for s in S_EV1) - 1)

    m.CapCost_def = pyo.Constraint(expr=
        m.CapitalCost == (
            sum(m.a["PE", s] * sd("capex", "Wash 1", "PE", s) for s in S_PE)
            + sum(m.a["EVOH", s] * sd("capex", "Wash 1", "EVOH", s) for s in S_EV1)
            + sum(m.b["PE", s] * sd("capex", "Wash 2", "PE", s) for s in S_PE)
            + sum(m.b["EVOH", s] * sd("capex", "Wash 2", "EVOH", s) for s in S_EV2)
            + m.x["py"] * od("capex", "py")
            + m.x["gas_er"] * od("capex", "gas_er")
            + m.p_gas * 1.996874495e6
            + m.t_gas * 3.248331031e6
            + m.z["gas_er"] * 1.680302711e6
        ))

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Operational cost
    # -----------------------------------------------------------------------
    m.OpCost_def = pyo.Constraint(expr=
        m.OperationalCost == (
            sum(m.a["PE", s] * sd("opex", "Wash 1", "PE", s) for s in S_PE)
            + sum(m.a["EVOH", s] * sd("opex", "Wash 1", "EVOH", s) for s in S_EV1)
            + sum(od("opex", t) * (m.F[t] + m.L[t] + m.N[t]) for t in OTHERTECH)
            + sum(m.b["PE", s] * sd("opex", "Wash 2", "PE", s) for s in S_PE)
            + sum(m.b["EVOH", s] * sd("opex", "Wash 2", "EVOH", s) for s in S_EV2)
        ))

    # -----------------------------------------------------------------------
    # CONSTRAINTS – Sales revenue
    # -----------------------------------------------------------------------
    m.Sales_def = pyo.Constraint(expr=
        m.Sales == (
            Feed * PE_f * 0.97 * Cpe * sum(m.a["PE", s] for s in S_PE)
            + Feed * EV_f * Cevoh * 0.97 * sum(m.a["EVOH", s] for s in S_EV1)
            + Cgas_pw * m.F["we"] + Cgas_er * m.F["gas_er"]
            + Cgas_pw * m.F["gas_h2"] + Cgas_pw * m.F["gas_h2cc"]
            + Cgas_pw * m.L["we"] + Cgas_er * m.L["gas_er"]
            + Cgas_pw * m.L["gas_h2"] + Cgas_pw * m.L["gas_h2cc"]
            + Feed * PE_f * Cpe * 0.97 * sum(m.b["PE", s] for s in S_PE)
            + Feed * EV_f * Cevoh * 0.97 * sum(m.b["EVOH", s] for s in S_EV2)
            + Cgas_pw * m.N["we"] + Cgas_er * m.N["gas_er"]
            + Cgas_pw * m.N["gas_h2"] + Cgas_pw * m.N["gas_h2cc"]
        ))

    # -----------------------------------------------------------------------
    # EXPRESSIONS – Profit, total emissions, CE overall
    # -----------------------------------------------------------------------
    m.Profit = pyo.Expression(expr=
        m.Sales - (m.CapitalCost + m.OperationalCost + m.TransportationCost))

    m.TotalEmissions_def = pyo.Constraint(expr=
        m.TotalEmissions == (
            sum(m.a["PE", s] * sd("gwp", "Wash 1", "PE", s) for s in S_PE)
            + sum(m.a["EVOH", s] * sd("gwp", "Wash 1", "EVOH", s) for s in S_EV1)
            + sum(od("gwp", t) * (m.F[t] + m.L[t] + m.N[t]) for t in OTHERTECH)
            + sum(m.b["PE", s] * sd("gwp", "Wash 2", "PE", s) for s in S_PE)
            + sum(m.b["EVOH", s] * sd("gwp", "Wash 2", "EVOH", s) for s in S_EV2)
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
