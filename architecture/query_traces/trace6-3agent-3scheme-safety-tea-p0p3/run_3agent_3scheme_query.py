#!/usr/bin/env python
"""Run 3-scheme 9-polymer query with sequential safety + TEA analysis.

Routing: separation-engineer -> tea-lca-analyst -> safety-analyst (sequential)
Patches: increased tool budgets for all 3 subagents, synthesis injection off
for separation-engineer.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langsmith import Client as LangSmithClient
from strap import agent as agent_module

QUERY = (
    "Find the optimal separation sequence for a mixed polymer waste stream "
    "containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, and PET. "
    "Use selective dissolution at atmospheric pressure. "
    "Propose THREE different sets of solvents and conditions for this "
    "9-polymer dissolution scheme. Each set must use a different combination "
    "of solvents or temperatures. "
    "Then run a safety assessment on the solvents in each scheme to determine "
    "which scheme uses the safest solvents overall. "
    "Finally, run a techno-economic analysis on each scheme to compare "
    "operating costs and determine the most cost-effective process."
)

SUBAGENT_OVERRIDES = {
    "separation-engineer": {"max_tool_calls": 20, "synthesis_tools": set()},
    "tea-lca-analyst": {"max_tool_calls": 15},
}


def main():
    agent = agent_module.create_dissolve_agent(
        subagent_overrides=SUBAGENT_OVERRIDES,
    )
    for name, overrides in SUBAGENT_OVERRIDES.items():
        print(f"Override {name}: {overrides}")

    print(f"Query: {QUERY}\n")
    print("Running agent (recursion_limit=250)...\n")

    t0 = time.time()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": QUERY}]},
        {"recursion_limit": 250},
    )
    elapsed = time.time() - t0

    # Extract the last AI message
    answer = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    print(f"\n{'='*60}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Messages: {len(result['messages'])}")
    print(f"{'='*60}")
    print(f"\nFinal Answer:\n{answer[:3000]}")
    if len(answer) > 3000:
        print(f"\n... [truncated, full answer is {len(answer)} chars]")

    # LangSmith
    ls = LangSmithClient()
    print(f"\nCheck LangSmith project 'strap-agent' for the trace.")

    # Diagnostics
    msg_types = {}
    tool_calls_count = 0
    for msg in result["messages"]:
        t = getattr(msg, "type", "unknown")
        msg_types[t] = msg_types.get(t, 0) + 1
        if t == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls_count += len(msg.tool_calls)

    print(f"\nMessage breakdown: {msg_types}")
    print(f"Total tool calls: {tool_calls_count}")


if __name__ == "__main__":
    main()
