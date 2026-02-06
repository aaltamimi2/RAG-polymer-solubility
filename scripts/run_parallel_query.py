#!/usr/bin/env python
"""Run a parallel-subagent query and capture the LangSmith trace ID."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langsmith import Client as LangSmithClient
from strap.agent import create_dissolve_agent

QUERY = (
    "Find the safest solvents and the lowest operating cost solvents "
    "relevant to LDPE and EVOH. Run the safety assessment and cost analysis "
    "in parallel, then combine results to determine which solvents could "
    "potentially treat one polymer differently from the other."
)

def main():
    agent = create_dissolve_agent()

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
            answer = msg.content
            break

    print(f"\n{'='*60}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Messages: {len(result['messages'])}")
    print(f"{'='*60}")
    print(f"\nFinal Answer:\n{answer}")

    # Try to extract LangSmith run ID from the last run
    ls = LangSmithClient()
    print(f"\nCheck LangSmith project '{ls._get_optional_tenant_id() or 'strap-agent'}' for the trace.")
    print("Look for the most recent run in: https://smith.langchain.com")

    # Count message types for diagnostics
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
