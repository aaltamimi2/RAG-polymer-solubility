"""Run separation sequence queries for 8 and 9 polymers, capture full output."""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

logging.disable(logging.CRITICAL)

from strap.agent import create_dissolve_agent, _extract_text

print("Loading agent...", flush=True)
agent = create_dissolve_agent()
print("Agent loaded.\n", flush=True)

queries = {
    8: (
        "Design a sequential separation sequence for a mixed plastic stream "
        "containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66 at 120C. "
        "Show the step-by-step decision tree with solvents and selectivities."
    ),
    9: (
        "Design a sequential separation sequence for a mixed plastic stream "
        "containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, PET at 120C. "
        "Show the step-by-step decision tree with solvents and selectivities."
    ),
}

results = {}
for n, query in queries.items():
    print(f"{'='*60}")
    print(f"Running {n}-polymer separation sequence...")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"recursion_limit": 150},
        )
        elapsed = time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR ({elapsed:.1f}s): {e}\n")
        results[n] = {"error": str(e), "time_s": elapsed}
        continue

    # Extract final answer
    answer = None
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            answer = _extract_text(msg.content)
            break

    # Count tool calls
    tool_names = []
    for msg in result["messages"]:
        if msg.type == "ai":
            for tc in getattr(msg, "tool_calls", None) or []:
                tool_names.append(tc.get("name", "?"))

    results[n] = {
        "query": query,
        "answer": answer,
        "time_s": elapsed,
        "n_messages": len(result["messages"]),
        "tool_names": tool_names,
    }

    print(f"\n--- Answer ({elapsed:.1f}s, {len(tool_names)} tool calls) ---")
    print(answer or "(no answer)")
    print(f"\nTools used: {tool_names}\n")

# Save results
out = Path(__file__).parent / "separation_sequences_8_9.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out}")
