"""Reusable guardrail directive and repair message builders."""

from __future__ import annotations


def iteration_limit_message() -> str:
    return "[LIMIT] Max iterations reached. Synthesize your answer now."


def token_budget_message() -> str:
    return "[LIMIT] Token budget exceeded. Synthesize your answer now."


def tool_budget_suffix() -> str:
    return (
        "\n\n[LIMIT] Tool call budget exhausted. Synthesize your "
        "findings into a clear, complete answer NOW. Do NOT call "
        "any more tools."
    )


def tool_budget_repair_message(agent_name: str | None = None) -> str:
    agent_text = f" for `{agent_name}`" if agent_name else ""
    return (
        "Your previous response hit the tool-call limit"
        f"{agent_text} before producing a usable final synthesis. "
        "Rewrite the full final answer now using only the tool results already in the conversation. "
        "Do not call any more tools. Do not say you will run additional analyses later. "
        "State any remaining uncertainty explicitly, then end with exactly one valid "
        "<STRUCTURED_RESULT> block."
    )


def structured_result_repair_message(detail: str = "") -> str:
    return (
        "Your previous response attempted to finalize without a valid "
        "<STRUCTURED_RESULT> JSON block. Rewrite the full final answer now. "
        "Do not call any more tools. Keep the substantive analysis, but end "
        "with exactly one valid <STRUCTURED_RESULT> block that matches your "
        f"schema.{detail}"
    )


def separation_feasibility_repair_message(detail: str) -> str:
    return (
        "Your previous separation recommendation is not physically consistent "
        f"at 1 atm. {detail}. Rewrite the full final answer now without calling "
        "additional tools. If every candidate route requires operating at or "
        "above a solvent boiling point, say that no feasible atmospheric-pressure "
        "sequence exists within the user's constraint. If any requested polymer "
        "is unsupported, state clearly that conclusions apply only to the supported "
        "subset, include `supported_polymers` and `unsupported_polymers` in the "
        "<STRUCTURED_RESULT>, and do not describe a residue as pure or purified "
        "unless you can rule out the unsupported polymer from that phase. Do not "
        "assert whether an unsupported polymer dissolves, remains solid, or "
        "precipitates without additional data. Do not present an infeasible route "
        "as the best or optimal executable sequence. End with exactly one valid "
        "<STRUCTURED_RESULT> block."
    )


def separation_analysis_coverage_repair_message(detail: str) -> str:
    return (
        "Your previous separation answer finalized without running a substantive "
        f"separation analysis tool. {detail}. Call a real separation-analysis tool now "
        "for the supported subset, such as `plan_sequential_separation`, "
        "`find_optimal_separation_sequence`, `find_optimal_separation_conditions`, "
        "`analyze_selective_solubility_enhanced`, or an atmospheric-feasibility tool. "
        "Do not stop after only listing supported polymers or database coverage. "
        "If no supported subset can be analyzed, state that explicitly with a valid "
        "<STRUCTURED_RESULT> block."
    )


def synthesis_directive() -> str:
    return (
        "\n\n[NOTE] A comprehensive analysis tool has returned results. "
        "Your next response should usually be the final synthesis. Do not call "
        "more tools unless a critical verification gap remains. End your answer "
        "with exactly one valid <STRUCTURED_RESULT> JSON block."
    )


def visualization_tool_directive(required_tool: str) -> str:
    return (
        f"\n\n[REQUIRED TOOL]\nUse `{required_tool}` for this task. "
        "Do not call any other plotting tool unless the task description explicitly changes."
    )


def visualization_required_tool_repair_message(required_tool: str) -> str:
    return (
        "Your previous response attempted to finalize without calling the required "
        f"visualization tool `{required_tool}`. Call `{required_tool}` now using the "
        "source handoff or payload provided in the task. Do not invent plot paths and "
        "do not write the final synthesis until the plotting tool has returned."
    )


def separation_support_directive(
    requested: list[str],
    supported: list[str],
    unsupported: list[str],
) -> str:
    return (
        "\n\n[SUPPORT COVERAGE]\n"
        f"Requested polymers inferred from the task: {', '.join(requested)}.\n"
        f"Supported by local interpolation data: {', '.join(supported) if supported else 'none'}.\n"
        f"Unsupported polymers: {', '.join(unsupported)}.\n"
        "Do not assert phase behavior for unsupported polymers. "
        "If you provide a partial route, scope it explicitly to the supported subset. "
        "Include `supported_polymers` and `unsupported_polymers` in the final "
        "<STRUCTURED_RESULT>. Do not imply purity for any phase that could still "
        "contain an unsupported polymer."
    )


def separation_temperature_bound_directive(max_temp_c: float) -> str:
    return (
        "\n\n[TEMPERATURE LIMIT]\n"
        f"The user supplied an upper temperature bound of {max_temp_c:.1f}C. "
        "Treat that as a ceiling only, not the default operating point. "
        "In the final answer, report the actual recommended temperature for each step. "
        "If a chosen solvent boils near or below that bound, explicitly state that the "
        "step must run below that solvent's boiling point at 1 atm. Do not imply that "
        f"the process runs at {max_temp_c:.1f}C unless the solvent remains liquid there."
    )


def duplicate_biosteam_batch_message(
    *,
    energy_case: str,
    allocation_method: str,
    overlap: list[str],
    fresh: list[str],
) -> str:
    next_step = (
        f"Run only the non-overlapping solvents instead: {', '.join(fresh)}."
        if fresh
        else "Reuse the earlier successful batch and synthesize instead of rerunning it."
    )
    return (
        "Duplicate BioSTEAM batch blocked: a previous successful "
        "`run_biosteam_multi_polymer` call already covered overlapping "
        f"solvents under energy case {energy_case} "
        f"and allocation `{allocation_method}`. "
        f"Overlapping solvents: {', '.join(overlap)}. {next_step}"
    )


def visualization_tool_block_message(required_tool: str, tool_name: str) -> str:
    return (
        "Visualization directive blocked: the active task explicitly "
        f"requires `{required_tool}`. Do not call `{tool_name}`; "
        f"use `{required_tool}` instead."
    )


def late_separation_todo_message() -> str:
    return (
        "Separation todo rewriting is blocked once domain analysis has started. "
        "Continue from the existing separation tool results and synthesize instead of "
        "calling `write_todos` again."
    )


def budget_final_synthesis_directive(reason: str) -> str:
    return (
        f"\n\n[LIMIT] {reason} This is your FINAL model call and tools are disabled. "
        "Synthesize your complete answer NOW from the tool results already gathered. "
        "If your contract requires a <STRUCTURED_RESULT> block, include it, populated "
        "only from data you actually obtained. Do not mention the budget; just answer."
    )
