"""
Solver routines for the PIW waste management optimisation model.

Supports:
  - Single-objective optimisation (max profit, min emissions, max circularity)
  - Epsilon-constraint method for Pareto front generation
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
from strap.waste_management.data_loader import I_SET, J_SET, K_SET


# ---------------------------------------------------------------------------
# Solver selection helpers
# ---------------------------------------------------------------------------

import shutil

# Homebrew SCIP path (ARM Mac default; fallback to PATH lookup)
_SCIP_EXECUTABLE = shutil.which("scip") or "/opt/homebrew/bin/scip"

# Gurobi default options: NumericFocus=3 prevents failure on large-coeff models
_GUROBI_DEFAULTS = {"NumericFocus": 3, "NonConvex": 2}


def summarize_constraint_residuals(m, abs_tol=1e-7, rel_tol=1e-9):
    """Return the largest post-solve residuals on active constraints."""
    violations = []

    for constraint in m.component_data_objects(pyo.Constraint, active=True):
        lower = pyo.value(constraint.lower) if constraint.has_lb() else None
        body = pyo.value(constraint.body)
        upper = pyo.value(constraint.upper) if constraint.has_ub() else None

        bound_scale = max(
            abs(body),
            abs(lower) if lower is not None else 0.0,
            abs(upper) if upper is not None else 0.0,
            1.0,
        )
        tolerance = abs_tol + rel_tol * bound_scale
        residual = 0.0

        if lower is not None and body < lower:
            residual = max(residual, lower - body)
        if upper is not None and body > upper:
            residual = max(residual, body - upper)

        if residual > tolerance:
            violations.append({
                "constraint": constraint.name,
                "residual": residual,
                "tolerance": tolerance,
                "lower": lower,
                "body": body,
                "upper": upper,
            })

    violations.sort(key=lambda item: item["residual"], reverse=True)
    return {
        "max_residual": violations[0]["residual"] if violations else 0.0,
        "count": len(violations),
        "violations": violations,
    }


def _get_solver(solver_name="gurobi", options=None):
    """
    Return a Pyomo SolverFactory instance.
    Falls back through gurobi -> scip -> glpk if unavailable.
    """
    candidates: list[str] = []
    for name in [solver_name, "gurobi", "scip", "glpk"]:
        if not name or name in candidates:
            continue
        candidates.append(name)
    for name in candidates:
        if name == "scip":
            solver = pyo.SolverFactory("scip", executable=_SCIP_EXECUTABLE)
        else:
            solver = pyo.SolverFactory(name)
        if solver.available():
            # Merge caller options on top of defaults
            merged = dict(_GUROBI_DEFAULTS) if name == "gurobi" else {}
            if options:
                merged.update(options)
            for k, v in merged.items():
                solver.options[k] = v
            return solver
    raise RuntimeError(
        f"No suitable solver found. Tried: {candidates}. "
        "Install Gurobi, SCIP (brew install scip), or GLPK.")


_ACCEPTABLE_TERMINATIONS = {
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.locallyOptimal,
    pyo.TerminationCondition.feasible,
}


def _is_acceptable_termination(status) -> bool:
    return status in _ACCEPTABLE_TERMINATIONS


def _set_objective(m, sense, penalty_weight=1e-7):
    """
    Attach an objective to the model.

    sense : str
        "max_profit"      -> Max Profit + penalty * W_recycled
        "min_emissions"   -> Min TotalEmissions - penalty * W_recycled
        "max_circularity" -> Max CEoverall + penalty * W_recycled
        "min_total_cost"  -> Min (Capital + Operating + Transport) - penalty * W_recycled
    """
    # Deactivate any existing objective
    for obj in m.component_objects(pyo.Objective, active=True):
        obj.deactivate()

    if sense == "max_profit":
        m.obj = pyo.Objective(
            expr=m.Profit + penalty_weight * m.W_recycled,
            sense=pyo.maximize)
    elif sense == "min_emissions":
        m.obj = pyo.Objective(
            expr=m.TotalEmissions - penalty_weight * m.W_recycled,
            sense=pyo.minimize)
    elif sense == "max_circularity":
        m.obj = pyo.Objective(
            expr=m.CEoverall + penalty_weight * m.W_recycled,
            sense=pyo.maximize)
    elif sense == "min_total_cost":
        m.obj = pyo.Objective(
            expr=(m.CapitalCost + m.OperationalCost + m.TransportationCost) - penalty_weight * m.W_recycled,
            sense=pyo.minimize)
    else:
        raise ValueError(f"Unknown objective sense: {sense}")


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def extract_results(m):
    """Extract key results from a solved model into a dict."""
    def _val(var):
        try:
            return pyo.value(var)
        except Exception:
            return None

    polymers = list(m.P)
    all_solvents = list(m.S)

    # Identify selected technologies
    stage1_set = list(m.I)
    stage2_set = list(m.J)
    stage3_set = list(m.K)

    x_sel = [i for i in stage1_set if _val(m.x[i]) is not None and _val(m.x[i]) > 0.5]
    y_sel = [j for j in stage2_set if _val(m.y[j]) is not None and _val(m.y[j]) > 0.5]
    z_sel = [k for k in stage3_set if _val(m.z[k]) is not None and _val(m.z[k]) > 0.5]

    wash1_sel = [
        f"{p}-{s}" for p in polymers for s in all_solvents
        if _val(m.a[p, s]) is not None and _val(m.a[p, s]) > 0.5
    ]
    wash2_sel = [
        f"{p}-{s}" for p in polymers for s in all_solvents
        if _val(m.b[p, s]) is not None and _val(m.b[p, s]) > 0.5
    ]

    return {
        "profit": _val(m.Profit),
        "emissions": _val(m.TotalEmissions),
        "CE": _val(m.CEoverall),
        "sales": _val(m.Sales),
        "capital_cost": _val(m.CapitalCost),
        "operational_cost": _val(m.OperationalCost),
        "transportation_cost": _val(m.TransportationCost),
        "total_cost": sum(
            value or 0.0
            for value in (
                _val(m.CapitalCost),
                _val(m.OperationalCost),
                _val(m.TransportationCost),
            )
        ),
        "stage1_tech": x_sel,
        "stage2_tech": y_sel,
        "stage3_tech": z_sel,
        "wash1_selection": wash1_sel,
        "wash2_selection": wash2_sel,
        "E_score": _val(m.E_score),
        "GHG_score": _val(m.GHG_score),
        "Water_score": _val(m.Water_score),
        "Waste_score": _val(m.Waste_score),
        "Subs_score": _val(m.Subs_score),
    }


def print_results(res, label=""):
    """Pretty-print optimisation results."""
    if label:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

    print(f"  Profit [USD]:      {res['profit']:,.2f}")
    print(f"  Emissions [tCO2]:  {res['emissions']:,.4f}")
    print(f"  CE Score:          {res['CE']:,.6f}")
    print(f"  Sales:             {res['sales']:,.2f}")
    print(f"  Capital Cost:      {res['capital_cost']:,.2f}")
    print(f"  Operational Cost:  {res['operational_cost']:,.2f}")
    print(f"  Transport Cost:    {res['transportation_cost']:,.2f}")
    print(f"  Total Cost:        {res['total_cost']:,.2f}")
    print(f"  Stage 1:           {res['stage1_tech']}")
    print(f"  Stage 2:           {res['stage2_tech']}")
    print(f"  Stage 3:           {res['stage3_tech']}")
    print(f"  STRAP Wash 1:      {res['wash1_selection']}")
    print(f"  STRAP Wash 2:      {res['wash2_selection']}")
    print(f"  E_score:  {res['E_score']:.6f}  |  GHG_score: {res['GHG_score']:.6f}")
    print(f"  Water_score: {res['Water_score']:.6f}  |  Waste_score: {res['Waste_score']:.6f}")
    print(f"  Subs_score:  {res['Subs_score']:.6f}")


def _load_verified_result(m, result, status, *, label=""):
    """Load a solver result only if its active constraints satisfy tolerance."""
    if _is_acceptable_termination(status):
        if len(result.solution) > 0:
            m.solutions.load_from(result)
        residuals = summarize_constraint_residuals(m)
        if residuals["count"]:
            top = residuals["violations"][0]
            print(
                "[WARN] Rejecting post-solve solution for "
                f"{label}: {top['constraint']} residual={top['residual']:.6g} "
                f"tolerance={top['tolerance']:.6g}"
            )
            return None
        res = extract_results(m)
        res["solver_feasibility"] = residuals
        return res

    if len(result.solution) > 0 and str(getattr(result.solution[0], 'status', '')).split('.')[-1] in ("feasible", "locallyOptimal"):
        m.solutions.load_from(result)
        residuals = summarize_constraint_residuals(m)
        if residuals["count"]:
            top = residuals["violations"][0]
            print(
                "[WARN] Rejecting non-optimal post-solve solution for "
                f"{label}: {top['constraint']} residual={top['residual']:.6g} "
                f"tolerance={top['tolerance']:.6g}"
            )
            return None
        res = extract_results(m)
        res["solver_feasibility"] = residuals
        print(f"[WARN] Solver returned non-optimal but feasible solution for {label}: {status}")
        return res

    return None


# ---------------------------------------------------------------------------
# Single-objective solve
# ---------------------------------------------------------------------------

def solve_single(m, sense, solver_name="gurobi", solver_options=None):
    """
    Solve the model for a single objective.

    Parameters
    ----------
    m : ConcreteModel  (from model.build_model)
    sense : str
        "max_profit", "min_emissions", or "max_circularity"
    solver_name : str
    solver_options : dict or None

    Returns
    -------
    dict of results  (or None if infeasible)
    """
    _set_objective(m, sense)
    solver = _get_solver(solver_name, solver_options)

    result = solver.solve(m, tee=True, load_solutions=False)
    status = result.solver.termination_condition

    res = _load_verified_result(m, result, status, label=sense)
    if res is not None:
        label_prefix = "Single Objective"
        if status not in _ACCEPTABLE_TERMINATIONS:
            label_prefix += " (non-optimal)"
        print_results(res, label=f"{label_prefix}: {sense}")
        return res
    else:
        print(f"[WARN] Solver status for {sense}: {status} — no feasible solution found")
        return None


def _append_pareto_row(rows, epsilon, res):
    rows.append({
        "epsilon": epsilon,
        "profit": res["profit"],
        "emissions": res["emissions"],
        "CE": res["CE"],
        "total_cost": res["total_cost"],
        "capital_cost": res["capital_cost"],
        "operational_cost": res["operational_cost"],
        "transportation_cost": res["transportation_cost"],
        "stage1": res["stage1_tech"],
        "stage2": res["stage2_tech"],
        "stage3": res["stage3_tech"],
        "wash1": res["wash1_selection"],
        "wash2": res["wash2_selection"],
    })


# ---------------------------------------------------------------------------
# Epsilon-constraint Pareto front
# ---------------------------------------------------------------------------

def pareto_profit_vs_emissions(m, emission_ideal, emission_nonideal,
                                n_points=50, solver_name="gurobi",
                                solver_options=None):
    """
    Epsilon-constraint: max Profit  s.t.  TotalEmissions <= epsilon.
    Sweeps epsilon from emission_nonideal down to emission_ideal.

    Returns a DataFrame with columns:
        epsilon, profit, emissions, CE, stage1, stage2, stage3, wash1, wash2
    """
    solver = _get_solver(solver_name, solver_options)
    epsilons = np.linspace(emission_nonideal, emission_ideal, n_points)
    rows = []

    # Add a mutable parameter for the epsilon bound
    m.eps_em = pyo.Param(mutable=True, initialize=emission_nonideal)
    m.eps_em_con = pyo.Constraint(expr=m.TotalEmissions <= m.eps_em)
    _set_objective(m, "max_profit", penalty_weight=1e-5)

    for eps in epsilons:
        m.eps_em.set_value(eps)
        result = solver.solve(m, tee=False)
        status = result.solver.termination_condition

        res = _load_verified_result(m, result, status, label=f"pareto_profit_vs_emissions@epsilon={eps:.6g}")
        if res is not None:
            _append_pareto_row(rows, eps, res)
        else:
            print(f"  epsilon={eps:.2f} -> {status}")

    # Clean up
    m.del_component(m.eps_em_con)
    m.del_component(m.eps_em)

    df = pd.DataFrame(rows)
    print(f"\nPareto (Profit vs Emissions): {len(df)} feasible points")
    return df


def pareto_profit_vs_ce(m, ce_nonideal, ce_ideal,
                         n_points=50, solver_name="gurobi",
                         solver_options=None):
    """
    Epsilon-constraint: max Profit  s.t.  CEoverall >= epsilon.
    Sweeps epsilon from ce_nonideal up to ce_ideal.

    Returns a DataFrame.
    """
    solver = _get_solver(solver_name, solver_options)
    epsilons = np.linspace(ce_nonideal, ce_ideal, n_points)
    rows = []

    m.eps_ce = pyo.Param(mutable=True, initialize=ce_nonideal)
    m.eps_ce_con = pyo.Constraint(expr=m.CEoverall >= m.eps_ce)
    _set_objective(m, "max_profit", penalty_weight=1e-5)

    for eps in epsilons:
        m.eps_ce.set_value(eps)
        result = solver.solve(m, tee=False)
        status = result.solver.termination_condition

        res = _load_verified_result(m, result, status, label=f"pareto_profit_vs_ce@epsilon={eps:.6g}")
        if res is not None:
            _append_pareto_row(rows, eps, res)
        else:
            print(f"  epsilon={eps:.2f} -> {status}")

    m.del_component(m.eps_ce_con)
    m.del_component(m.eps_ce)

    df = pd.DataFrame(rows)
    print(f"\nPareto (Profit vs CE): {len(df)} feasible points")
    return df


def pareto_emissions_vs_ce(m, ce_nonideal, ce_ideal,
                            n_points=50, solver_name="gurobi",
                            solver_options=None):
    """
    Epsilon-constraint: min TotalEmissions  s.t.  CEoverall >= epsilon.
    Sweeps epsilon from ce_nonideal up to ce_ideal.

    Returns a DataFrame.
    """
    solver = _get_solver(solver_name, solver_options)
    epsilons = np.linspace(ce_nonideal, ce_ideal, n_points)
    rows = []

    m.eps_ce2 = pyo.Param(mutable=True, initialize=ce_nonideal)
    m.eps_ce2_con = pyo.Constraint(expr=m.CEoverall >= m.eps_ce2)
    _set_objective(m, "min_emissions", penalty_weight=1e-5)

    for eps in epsilons:
        m.eps_ce2.set_value(eps)
        result = solver.solve(m, tee=False)
        status = result.solver.termination_condition

        res = _load_verified_result(m, result, status, label=f"pareto_emissions_vs_ce@epsilon={eps:.6g}")
        if res is not None:
            _append_pareto_row(rows, eps, res)
        else:
            print(f"  epsilon={eps:.2f} -> {status}")

    m.del_component(m.eps_ce2_con)
    m.del_component(m.eps_ce2)

    df = pd.DataFrame(rows)
    print(f"\nPareto (Emissions vs CE): {len(df)} feasible points")
    return df


def pareto_cost_vs_emissions(
    m,
    emission_ideal,
    emission_nonideal,
    n_points=50,
    solver_name="gurobi",
    solver_options=None,
):
    """
    Epsilon-constraint: min TotalCost  s.t.  TotalEmissions <= epsilon.
    Sweeps epsilon from emission_nonideal down to emission_ideal.
    """
    solver = _get_solver(solver_name, solver_options)
    epsilons = np.linspace(emission_nonideal, emission_ideal, n_points)
    rows = []

    m.eps_em_cost = pyo.Param(mutable=True, initialize=emission_nonideal)
    m.eps_em_cost_con = pyo.Constraint(expr=m.TotalEmissions <= m.eps_em_cost)
    _set_objective(m, "min_total_cost", penalty_weight=1e-5)

    for eps in epsilons:
        m.eps_em_cost.set_value(eps)
        result = solver.solve(m, tee=False)
        status = result.solver.termination_condition

        res = _load_verified_result(m, result, status, label=f"pareto_cost_vs_emissions@epsilon={eps:.6g}")
        if res is not None:
            _append_pareto_row(rows, eps, res)
        else:
            print(f"  epsilon={eps:.2f} -> {status}")

    m.del_component(m.eps_em_cost_con)
    m.del_component(m.eps_em_cost)

    df = pd.DataFrame(rows)
    print(f"\nPareto (Cost vs Emissions): {len(df)} feasible points")
    return df


def pareto_cost_vs_ce(
    m,
    ce_nonideal,
    ce_ideal,
    n_points=50,
    solver_name="gurobi",
    solver_options=None,
):
    """
    Epsilon-constraint: min TotalCost  s.t.  CEoverall >= epsilon.
    Sweeps epsilon from ce_nonideal up to ce_ideal.
    """
    solver = _get_solver(solver_name, solver_options)
    epsilons = np.linspace(ce_nonideal, ce_ideal, n_points)
    rows = []

    m.eps_ce_cost = pyo.Param(mutable=True, initialize=ce_nonideal)
    m.eps_ce_cost_con = pyo.Constraint(expr=m.CEoverall >= m.eps_ce_cost)
    _set_objective(m, "min_total_cost", penalty_weight=1e-5)

    for eps in epsilons:
        m.eps_ce_cost.set_value(eps)
        result = solver.solve(m, tee=False)
        status = result.solver.termination_condition

        res = _load_verified_result(m, result, status, label=f"pareto_cost_vs_ce@epsilon={eps:.6g}")
        if res is not None:
            _append_pareto_row(rows, eps, res)
        else:
            print(f"  epsilon={eps:.2f} -> {status}")

    m.del_component(m.eps_ce_cost_con)
    m.del_component(m.eps_ce_cost)

    df = pd.DataFrame(rows)
    print(f"\nPareto (Cost vs CE): {len(df)} feasible points")
    return df
