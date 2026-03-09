"""Greedy and multi-scheme display builders for sequence planning tools."""

from __future__ import annotations

import math
from typing import Any


def build_greedy_planning_display(
    *,
    polymer_list: list[str],
    temperature: float,
    evaluations: list[dict[str, Any]],
    sequence: list[str],
    steps: list[dict[str, Any]],
) -> str:
    """Render the greedy planning workflow as markdown."""
    n_polymers = len(polymer_list)
    output = [
        "# Greedy Separation Planning\n",
        f"**Polymers:** {', '.join(polymer_list)}",
        f"**Count:** {n_polymers} polymers",
        f"**Algorithm:** Greedy (O(n^2) ~ {n_polymers**2} evaluations)",
        f"**vs Exhaustive:** {n_polymers}! = {math.factorial(n_polymers):,} permutations avoided",
        f"**Temperature:** {temperature} C\n",
        "## Algorithm Explanation\n",
        "At each step, we select the polymer that can be **most selectively** separated",
        "from all remaining polymers. This greedy approach finds a good (not necessarily optimal)",
        "sequence efficiently.\n",
        "## Step-by-Step Greedy Selection\n",
    ]

    for snapshot in evaluations:
        output.append(
            f"### Step {snapshot['step']}: Evaluating {len(snapshot['remaining'])} candidates\n"
        )
        output.append(f"**Remaining mixture:** {{{', '.join(snapshot['remaining'])}}}\n")
        output.append("| Polymer | Best Solvent | Selectivity |")
        output.append("|---------|--------------|-------------|")
        for candidate in snapshot["candidates"]:
            sel_str = (
                f"{candidate['selectivity']:.1f}%"
                if candidate["selectivity"] > -900
                else "N/A"
            )
            output.append(
                f"| {candidate['polymer']} | {candidate['solvent']} | {sel_str} |"
            )
        output.append("")

        selected = snapshot["selected"]
        if selected["selectivity"] > -900:
            output.append(
                f"**Selected: {selected['polymer']}** with {selected['solvent']} "
                f"(selectivity: {selected['selectivity']:.1f}%)\n"
            )
        else:
            output.append(
                f"**Selected: {selected['polymer']}** (no solubility data available)\n"
            )

    output.append(f"### Step {len(steps) + 1}: {sequence[-1]} is isolated\n")
    output.append("---\n")
    output.append("## Greedy Separation Sequence Summary\n")
    output.append(f"**Optimized Sequence:** {' -> '.join(sequence)}\n")
    output.append("### Step-by-Step Protocol\n")
    output.append("| Step | Separate | Using Solvent | Selectivity |")
    output.append("|------|----------|---------------|-------------|")

    valid_steps = [step for step in steps if step["selectivity"] > -900]
    for step in steps:
        sel_str = f"{step['selectivity']:.1f}%" if step["selectivity"] > -900 else "N/A"
        output.append(
            f"| {step['step']} | {step['target']} | {step['solvent']} | {sel_str} |"
        )
    output.append(f"| {len(steps) + 1} | {sequence[-1]} | (isolated) | done |")
    output.append("")

    if valid_steps:
        min_sel = min(step["selectivity"] for step in valid_steps)
        avg_sel = sum(step["selectivity"] for step in valid_steps) / len(valid_steps)
        unique_solvents = len(
            {step["solvent"] for step in valid_steps if step["solvent"] != "N/A"}
        )

        output.append("### Metrics\n")
        output.append(f"- **Minimum selectivity:** {min_sel:.1f}%")
        output.append(f"- **Average selectivity:** {avg_sel:.1f}%")
        output.append(f"- **Unique solvents needed:** {unique_solvents}")
        output.append(
            f"- **Evaluations performed:** ~{n_polymers * (n_polymers + 1) // 2}"
        )

    output.append("\n---\n")
    output.append(
        "*Note: Greedy algorithm finds a good sequence efficiently but may not be globally optimal.*"
    )
    output.append(
        "*For <=3 polymers, exhaustive search is used to find the true optimum.*"
    )
    return "\n".join(output)


def build_multi_scheme_display(
    *,
    polymer_list: list[str],
    temperature: float,
    n_variants: int,
    schemes: list[dict[str, Any]],
) -> str:
    """Render the compact multi-scheme planning comparison."""
    output = [
        f"MULTI-SCHEME SEPARATION: {','.join(polymer_list)} @ {temperature}C",
        f"Polymers: {len(polymer_list)} | Greedy O(n^2) | {len(schemes)} schemes ({n_variants} variants/type)\n",
    ]

    for scheme in schemes:
        output.append(f"== {scheme['tag']}: {scheme['name']} ==")
        output.append(f"Seq: {' > '.join(scheme['seq'])}")
        output.append(
            f"Min/Avg sel: {scheme['min_sel']:.1f}% / {scheme['avg_sel']:.1f}% | "
            f"Solvents: {scheme['n_solv']}"
        )
        output.append("Stp|Target  |Solvent       |Sel% |T(C) |GSK |LogP")
        output.append("---|--------|--------------|-----|-----|----|----")
        for step in scheme["steps"]:
            if step["solvent"] == "-":
                output.append(
                    f" {step['step']} |{step['target']:<8}|(isolated)    | done|  -  | -  | -"
                )
                continue
            sel_s = (
                f"{step['sel']:.0f}"
                if step["sel"] is not None and step["sel"] > -900
                else "?"
            )
            t_s = f"{step['temp']:.0f}" if step.get("temp") is not None else "-"
            gs_s = f"{step['gsk']:.1f}" if step.get("gsk") is not None else "-"
            lp_s = f"{step['logp']:.1f}" if step.get("logp") is not None else "-"
            output.append(
                f" {step['step']} |{step['target']:<8}|{step['solvent']:<14}|"
                f"{sel_s:>5}|{t_s:>5}|{gs_s:>4}|{lp_s:>4}"
            )
        output.append("")

    output.append("== COMPARISON ==")
    output.append("Scheme|MinSel|AvgSel|#Solvents|Bottleneck")
    output.append("------|------|------|---------|----------")
    for scheme in schemes:
        valid_steps = [
            step for step in scheme["steps"] if step["sel"] is not None and step["sel"] > -900
        ]
        if valid_steps:
            bottleneck = min(valid_steps, key=lambda step: step["sel"])
            output.append(
                f"{scheme['tag']:6}|{scheme['min_sel']:5.0f}%|{scheme['avg_sel']:5.0f}%|"
                f"{scheme['n_solv']:9}|Step {bottleneck['step']}:{bottleneck['target']}"
            )
        else:
            output.append(f"{scheme['tag']:6}|    ?|    ?|{scheme['n_solv']:9}|N/A")

    output.append("")
    output.append("SEL=max separation reliability, SAFE=regulatory/green, NRG=lowest operating cost.")
    return "\n".join(output)
