"""Benchmark record helpers for comparing harnesses."""

from __future__ import annotations

from typing import Any

from .messages import ClaudeSdkTurnResult


def benchmark_record(
    *,
    query_id: str,
    harness: str,
    model_alias: str,
    model_id: str,
    result: ClaudeSdkTurnResult,
    latency_s: float,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "harness": harness,
        "model_alias": model_alias,
        "model_id": model_id,
        "success": result.ok,
        "result_subtype": result.result_subtype,
        "turns": result.num_turns,
        "tool_calls_mcp": list(result.mcp_tool_calls),
        "tool_calls_legacy": list(result.legacy_tool_calls),
        "subagents": [],
        "artifacts": result.additional_kwargs.get("strap_artifacts") or [],
        "cost_usd": result.total_cost_usd,
        "latency_s": latency_s,
        "failure_codes": [result.result_subtype] if not result.ok and result.result_subtype else [],
    }
