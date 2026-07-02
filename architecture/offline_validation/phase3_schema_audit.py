"""Phase 3: subagent schema audit — tools, prompts, contracts, budgets.

Checks, for every configured subagent (no model calls):
  1. tool_groups all resolve in the agent's tool-group registry (silent-skip risk)
  2. guardrails.synthesis_tools / free_tools are subsets of granted tools
  3. tool names cited in the system prompt are actually granted to that subagent
  4. the <STRUCTURED_RESULT> example in the prompt parses and passes the
     handoff store's validate_agent_payload for that producer
  5. planning contracts: every consumed artifact has a producer; every
     produced artifact has a consumer; every capability edge has a typed
     adapter or an explicit generic fallback
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from strap.agent import _TOOL_GROUP_REGISTRY, _resolve_tools
from strap.handoff_adapters import _ADAPTERS
from strap.handoff_store import normalize_agent_payload, validate_agent_payload
from strap.planning_graph import GENERIC_CONTEXT_ARTIFACT, build_planning_graph
from strap.subagent_config import load_subagent_specs
from strap.testing_utils import block_model_access

OUT = _ROOT / "architecture" / "test_results" / "subagent_validation_offline_20260701"


def _tool_name(tool) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))


def run() -> dict:
    findings: list[dict] = []
    per_agent: dict[str, dict] = {}

    with block_model_access():
        specs = load_subagent_specs()
        graph = build_planning_graph()

        all_tool_names: set[str] = set()
        for group, getter in _TOOL_GROUP_REGISTRY.items():
            try:
                all_tool_names.update(_tool_name(t) for t in getter())
            except Exception as exc:  # noqa: BLE001
                findings.append({"severity": "error", "agent": None,
                                 "check": "tool_group_registry",
                                 "detail": f"group {group!r} getter raised {exc}"})

        for spec in specs:
            name = spec["name"]
            info: dict = {"tool_groups": spec.get("tool_groups", [])}

            # 1. tool groups resolve
            missing_groups = [g for g in spec.get("tool_groups", []) if g not in _TOOL_GROUP_REGISTRY]
            if missing_groups:
                findings.append({"severity": "error", "agent": name, "check": "tool_groups",
                                 "detail": f"unknown tool_groups silently skipped: {missing_groups}"})
            granted = {_tool_name(t) for t in _resolve_tools(spec.get("tool_groups", []))}
            info["granted_tools"] = sorted(granted)
            if not granted:
                findings.append({"severity": "error", "agent": name, "check": "tool_groups",
                                 "detail": "no tools granted at all"})

            # 2. guardrail tool lists are subsets
            guard = spec.get("guardrails") or {}
            for field in ("synthesis_tools", "free_tools"):
                extra = [t for t in guard.get(field, []) if t not in granted]
                if extra:
                    findings.append({"severity": "error", "agent": name, "check": f"guardrails.{field}",
                                     "detail": f"names not in granted toolset: {extra}"})
            info["budgets"] = {k: guard.get(k) for k in
                              ("max_tool_calls", "token_budget", "truncate_tool_results_after")}

            # 3. tools cited in the system prompt exist and are granted
            prompt = spec.get("system_prompt", "") or ""
            cited = {tok for tok in re.findall(r"\b[a-z][a-z0-9_]{5,}\b", prompt)
                     if tok in all_tool_names}
            ungranted = sorted(cited - granted)
            if ungranted:
                findings.append({"severity": "error", "agent": name, "check": "prompt_tools",
                                 "detail": f"prompt cites tools the agent does not have: {ungranted}"})
            info["prompt_cited_tools"] = sorted(cited)

            # 4. structured-result example validates against the store schema
            match = re.search(r"<STRUCTURED_RESULT>\s*(\{.*?\})\s*</STRUCTURED_RESULT>", prompt, re.DOTALL)
            if match:
                try:
                    example = json.loads(match.group(1))
                    errors = validate_agent_payload(name, normalize_agent_payload(name, example))
                    if errors:
                        findings.append({"severity": "warn", "agent": name, "check": "structured_example",
                                         "detail": f"example payload fails store validation: {errors}"})
                except json.JSONDecodeError as exc:
                    findings.append({"severity": "error", "agent": name, "check": "structured_example",
                                     "detail": f"example JSON does not parse: {exc}"})
            else:
                findings.append({"severity": "info", "agent": name, "check": "structured_example",
                                 "detail": "no STRUCTURED_RESULT example block in prompt"})

            per_agent[name] = info

        # 5. planning-contract consistency + adapter coverage
        producers_by_artifact: dict[str, set[str]] = {}
        consumers_by_artifact: dict[str, set[str]] = {}
        for node in graph.nodes.values():
            for artifact in node.produces:
                producers_by_artifact.setdefault(artifact, set()).add(node.name)
            for artifact in node.consumes:
                if artifact != GENERIC_CONTEXT_ARTIFACT:
                    consumers_by_artifact.setdefault(artifact, set()).add(node.name)

        for artifact, consumers in sorted(consumers_by_artifact.items()):
            if artifact not in producers_by_artifact:
                findings.append({"severity": "error", "agent": None, "check": "planning_contracts",
                                 "detail": f"artifact {artifact} consumed by {sorted(consumers)} but produced by no one"})
        for artifact, producers in sorted(producers_by_artifact.items()):
            if artifact not in consumers_by_artifact:
                findings.append({"severity": "info", "agent": None, "check": "planning_contracts",
                                 "detail": f"artifact {artifact} produced by {sorted(producers)} has no declared consumer (terminal deliverable?)"})

        edge_report = []
        for edge in graph.capability_edges:
            has_typed = (edge.producer, edge.consumer) in _ADAPTERS
            edge_report.append({
                "producer": edge.producer, "consumer": edge.consumer,
                "artifacts": list(edge.artifacts), "typed_adapter": has_typed,
            })
            if not has_typed:
                findings.append({"severity": "warn", "agent": edge.consumer, "check": "adapter_coverage",
                                 "detail": f"capability edge {edge.producer} -> {edge.consumer} "
                                           f"({', '.join(edge.artifacts)}) has NO typed adapter; "
                                           "falls back to generic context prompt"})

    result = {
        "summary": {
            "agents": len(per_agent),
            "errors": sum(1 for f in findings if f["severity"] == "error"),
            "warnings": sum(1 for f in findings if f["severity"] == "warn"),
            "infos": sum(1 for f in findings if f["severity"] == "info"),
            "capability_edges": len(edge_report),
            "typed_adapters": sum(1 for e in edge_report if e["typed_adapter"]),
        },
        "findings": findings,
        "capability_edges": edge_report,
        "per_agent": per_agent,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase3_schema_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    for f in findings:
        if f["severity"] in {"error", "warn"}:
            print(f"[{f['severity'].upper()}] {f['agent'] or '-'} :: {f['check']} :: {f['detail'][:160]}")
    return result


if __name__ == "__main__":
    run()
