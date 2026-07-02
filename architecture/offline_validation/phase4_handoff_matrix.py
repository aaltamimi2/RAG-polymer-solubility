"""Phase 4: drive the REAL handoff machinery over every capability edge.

For each planning-graph capability edge (producer -> consumer):
  1. store the producer's own YAML STRUCTURED_RESULT example via
     store_agent_result (same validation path as production)
  2. build the derived consumer handoff via build_handoff_for_consumer
     (typed adapter when registered, generic fallback otherwise)
  3. assert status, prompt quality bounds, payload integrity

Also exercises the multi-source join (separation + optimization -> viz).
Zero model calls.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from strap.handoff_adapters import _ADAPTERS
from strap.handoff_store import cleanup_handoff_scope, initialize_handoff_scope, store_agent_result
from strap.handoffs import build_handoff_for_consumer, build_multi_source_handoff_for_consumer
from strap.planning_graph import build_planning_graph
from strap.subagent_config import load_subagent_specs
from strap.testing_utils import block_model_access

OUT = _ROOT / "architecture" / "test_results" / "subagent_validation_offline_20260701"
MAX_PROMPT_CHARS = 8000

# Minimal contract-passing payloads for producers whose prompts lack examples.
FALLBACK_PAYLOADS: dict[str, dict] = {
    "rag-analyst": {
        "agent": "rag-analyst", "schema_version": "1.0",
        "question": "What do indexed docs say about EVOH?",
        "answer": "Indexed sources describe EVOH as a barrier polymer.",
        "citations": [{"source": "paper1.pdf", "chunk": 3}],
    },
    "visualization-specialist": {
        "agent": "visualization-specialist", "schema_version": "1.0",
        "plot_type": "solubility_curve", "plot_paths": ["./plots/example.png"], "format": "png",
    },
}


def producer_payloads() -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for spec in load_subagent_specs():
        prompt = spec.get("system_prompt", "") or ""
        match = re.search(r"<STRUCTURED_RESULT>\s*(\{.*?\})\s*</STRUCTURED_RESULT>", prompt, re.DOTALL)
        if match:
            try:
                payloads[spec["name"]] = json.loads(match.group(1))
                continue
            except json.JSONDecodeError:
                pass
        if spec["name"] in FALLBACK_PAYLOADS:
            payloads[spec["name"]] = FALLBACK_PAYLOADS[spec["name"]]
    return payloads


def run() -> dict:
    graph = build_planning_graph()
    payloads = producer_payloads()
    rows: list[dict] = []
    problems: list[str] = []

    with block_model_access():
        for edge in graph.capability_edges:
            payload = payloads.get(edge.producer)
            if payload is None:
                problems.append(f"no example payload for producer {edge.producer}")
                continue
            scratch = Path(tempfile.mkdtemp(prefix="phase4_"))
            initialize_handoff_scope(user_query=f"validation edge {edge.producer}->{edge.consumer}",
                                     artifact_root=scratch)
            try:
                source = store_agent_result(producer=edge.producer, payload=payload,
                                            source_tool_call_id="tc_val")
                derived = build_handoff_for_consumer(
                    consumer=edge.consumer,
                    source_handoff_id=source.handoff_id,
                    producer=edge.producer,
                )
                prompt_text = derived.task_prompt or ""
                row = {
                    "edge": f"{edge.producer} -> {edge.consumer}",
                    "declared_artifacts": list(edge.artifacts),
                    "typed_adapter": (edge.producer, edge.consumer) in _ADAPTERS,
                    "source_status": source.status,
                    "derived_contract": derived.contract,
                    "derived_status": derived.status,
                    "prompt_chars": len(prompt_text),
                    "payload_chars": len(json.dumps(derived.payload)),
                    "payload_keys": sorted(derived.payload.keys())[:12],
                }
                if source.status != "ok":
                    problems.append(f"{row['edge']}: producer example stored with status={source.status}")
                if derived.status not in {"ok", "ready"}:
                    problems.append(f"{row['edge']}: derived handoff status={derived.status}")
                if not prompt_text.strip():
                    problems.append(f"{row['edge']}: empty consumer task_prompt")
                if len(prompt_text) > MAX_PROMPT_CHARS:
                    problems.append(f"{row['edge']}: task_prompt {len(prompt_text)} chars exceeds {MAX_PROMPT_CHARS}")
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{edge.producer} -> {edge.consumer}: RAISED {type(exc).__name__}: {exc}")
            finally:
                cleanup_handoff_scope()

        # Multi-source join: separation + optimization -> visualization
        scratch = Path(tempfile.mkdtemp(prefix="phase4_join_"))
        initialize_handoff_scope(user_query="validation join sep+opt->viz", artifact_root=scratch)
        try:
            sep = store_agent_result(producer="separation-engineer",
                                     payload=payloads["separation-engineer"], source_tool_call_id="t1")
            opt = store_agent_result(producer="optimization-engineer",
                                     payload=payloads["optimization-engineer"], source_tool_call_id="t2")
            joined = build_multi_source_handoff_for_consumer(
                consumer="visualization-specialist",
                source_handoff_ids=[sep.handoff_id, opt.handoff_id],
            )
            rows.append({
                "edge": "separation+optimization -> visualization (JOIN)",
                "typed_adapter": True,
                "derived_contract": joined.contract,
                "derived_status": joined.status,
                "prompt_chars": len(joined.task_prompt or ""),
                "payload_chars": len(json.dumps(joined.payload)),
                "payload_keys": sorted(joined.payload.keys())[:12],
            })
            if joined.status not in {"ok", "ready"}:
                problems.append(f"join handoff status={joined.status}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"multi-source join RAISED {type(exc).__name__}: {exc}")
        finally:
            cleanup_handoff_scope()

    result = {
        "summary": {
            "edges_tested": len(rows),
            "problems": len(problems),
            "typed": sum(1 for r in rows if r.get("typed_adapter")),
            "generic": sum(1 for r in rows if not r.get("typed_adapter")),
            "max_prompt_chars": max((r["prompt_chars"] for r in rows), default=0),
            "max_payload_chars": max((r["payload_chars"] for r in rows), default=0),
        },
        "problems": problems,
        "edges": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase4_handoff_matrix.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    for problem in problems:
        print("PROBLEM:", problem[:200])
    for row in rows:
        print(f"  {row['edge']:<58} typed={str(row.get('typed_adapter')):<5} "
              f"status={row.get('derived_status'):<8} prompt={row['prompt_chars']:>5}c "
              f"payload={row['payload_chars']:>6}c contract={row.get('derived_contract')}")
    return result


if __name__ == "__main__":
    run()
