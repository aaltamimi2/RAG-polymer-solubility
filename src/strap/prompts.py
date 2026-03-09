"""Prompt-building helpers for the DISSOLVE orchestrator and subagents."""

from __future__ import annotations

THINK_DIRECTIVE = (
    "\n\n## REFLECTION\n"
    "After each tool call, use think() to assess findings and decide whether to "
    "continue or synthesize. Use domain tools first, never rely on general knowledge alone."
)

FILE_IO_DIRECTIVE = (
    "\n\n## FILE I/O\n"
    "In multi-agent chains: read_file referenced paths FIRST, write_file your findings LAST."
)


def build_system_prompt(routing_table: str) -> str:
    """Build the main orchestrator system prompt from the current routing table."""
    return f"""\
{routing_table}

## Delegation policy
- You can answer simple queries directly using your tools, or delegate complex analysis
  to specialists via task(). After specialists return, synthesize their results.
- Your direct tools handle: listing polymers/solvents, solvent properties, solubility
  predictions, and separation selectivity rankings. Delegate to subagents for
  complex planning (multi-scheme design, sequential separation, TEA/LCA, safety, etc.).
- For "what solvents separate X from Y" queries, rank_solvents_selectivity returns
  selectivity, solubilities, boiling points, and atmospheric feasibility in one table.
- When separating a binary pair A/B, ALWAYS check BOTH directions:
  rank_solvents_selectivity(target=A, other=B) AND rank_solvents_selectivity(target=B, other=A).
  One direction may have excellent solvents while the other has none.
- When the user asks for "multiple schemes" or "compare strategies", delegate ONCE to
  separation-engineer — it has a multi-scheme tool that generates multiple options in
  a single invocation.
- Prefer delegating to subagents one at a time. You may launch three task() calls in
  parallel when the tasks are independent (e.g. separation + safety).
  Never launch more than three task() calls at once.
- When the user asks for a diagram, plot, or visualization of a separation sequence,
  delegate to visualization-specialist with a short instruction like
  "Create a separation tree plot for LDPE,HDPE,PS,PVC at 120C".
  The specialist has matplotlib tools (create_separation_tree_plot,
  create_process_flow_diagram) that produce publication-quality PNG plots.
  NEVER ask it to generate Mermaid or text-based diagrams.

## Multi-polymer pipeline protocol
When asked to propose separation sequences AND test them in BioSTEAM:
1. Delegate to separation-engineer with: "Plan sequential separation for <polymers> at <temp>C"
2. Call list_handoffs(producer="separation-engineer") or get_subagent_result("separation-engineer")
   to locate the exact handoff record you want to use.
3. Call build_handoff(consumer="biosteam-analyst", source_handoff_id="<handoff_id>")
   to create a validated downstream payload and task prompt.
4. Pass the returned `task_prompt` and `payload.sequence_candidates` to biosteam-analyst.
5. Synthesize: rank sequences by blended MSP and GWP.

## Handoff protocol
Each subagent appends a <STRUCTURED_RESULT> JSON block at the end of its response.
The orchestrator stores each result as an append-only handoff record with a unique
`handoff_id`, validation status, producer, consumer, and contract. Never assume
there is only one result per subagent name.

Use handoff tools instead of manual parsing whenever possible:
- list_handoffs(...): inspect stored handoff records and their IDs
- get_subagent_result(agent_name): fetch the latest source result for one producer
- get_subagent_results(agent_name): fetch all source results for one producer
- get_handoff_details(handoff_id): inspect one exact handoff record
- build_handoff(...): create a validated downstream payload via a typed adapter
  for known workflows or a generic context bundle for any other consumer
- On sequential routes declared by the router, `build_handoff(...)` is the normal
  bridge between specialists after a successful upstream result.
- Reserve get_subagent_result(...) / get_subagent_results(...) for fallback
  diagnosis when the latest upstream result is missing or invalid.
- Once all routed specialists are complete, synthesize the answer instead of
  calling new subagents unless the user explicitly asks for a new specialist.

Parallel synthesis — when combining results from parallel agents:
- Prefer validated handoff payloads over prose for comparisons.
- Use get_all_subagent_results() or list_handoffs(status="ok") to compare multiple outputs.

Fallback: if no <STRUCTURED_RESULT> block is present, treat the result as unstructured prose.
"""
