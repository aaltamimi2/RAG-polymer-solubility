#!/usr/bin/env python3
"""Run a query-bank seed plus scripted follow-ups through the real agent.

This is an operational harness, not a CI test. It preserves one conversation
thread so follow-up behavior can be evaluated as a user would experience it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_BANK = REPO_ROOT / "docs" / "subagent_query_bank-v1.xlsx"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "architecture" / "test_results"


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    return {
        "type": getattr(message, "type", message.__class__.__name__),
        "content": _extract_text(getattr(message, "content", "")),
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
    }


def _latest_ai_message(messages: list[Any]) -> dict[str, Any]:
    for message in reversed(messages):
        item = _message_to_dict(message)
        if item.get("type") in {"ai", "AIMessage"} or item.get("role") == "assistant":
            return item
    return {}


def _extract_messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            return messages
    messages = getattr(result, "messages", None)
    return messages if isinstance(messages, list) else []


def _read_query_bank_row(path: Path, sheet: str, row: int) -> str:
    df = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    if row < 1 or row > len(df):
        raise ValueError(f"row must be 1-{len(df)} for sheet {sheet!r}")
    values = [str(value).strip() for value in df.iloc[row - 1].tolist() if str(value).strip() and str(value) != "nan"]
    candidates = [value for value in values if len(value.split()) >= 4]
    if not candidates:
        raise ValueError(f"could not find query-like text in sheet {sheet!r}, row {row}")
    return max(candidates, key=len)


def _load_turns(args: argparse.Namespace) -> list[str]:
    if args.turns_json:
        data = json.loads(Path(args.turns_json).read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("--turns-json must contain a JSON string list")
        return data

    seed = args.query or _read_query_bank_row(Path(args.query_bank), args.sheet, args.row)
    return [seed, *args.followup]


def _turn_summary(turn: dict[str, Any]) -> str:
    kwargs = turn.get("assistant", {}).get("additional_kwargs") or {}
    content = turn.get("assistant", {}).get("content") or ""
    origin = kwargs.get("strap_origin", "legacy_or_model")
    status = kwargs.get("strap_typed_runtime_status") or kwargs.get("strap_tool_name") or ""
    progress = kwargs.get("strap_runtime_progress") if isinstance(kwargs.get("strap_runtime_progress"), dict) else {}
    paths = progress.get("produced_artifact_paths") if isinstance(progress.get("produced_artifact_paths"), list) else []
    line = f"- Turn {turn['turn']}: origin={origin}"
    if status:
        line += f", status/tool={status}"
    if paths:
        line += f", paths={paths}"
    line += f"\n  User: {turn['user']}"
    line += f"\n  Assistant preview: {content[:240].replace(chr(10), ' ')}"
    return line


def run_chat(turns: list[str], *, thread_id: str) -> list[dict[str, Any]]:
    from strap.agent import create_dissolve_agent

    agent = create_dissolve_agent(enable_persistence=True)
    messages: list[Any] = []
    transcript: list[dict[str, Any]] = []
    config = {"configurable": {"thread_id": thread_id}}

    for index, user_text in enumerate(turns, start=1):
        messages.append({"role": "user", "content": user_text})
        result = agent.invoke({"messages": messages}, config=config)
        messages = _extract_messages(result)
        assistant = _latest_ai_message(messages)
        transcript.append(
            {
                "turn": index,
                "user": user_text,
                "assistant": assistant,
            }
        )
    return transcript


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-bank", default=str(DEFAULT_QUERY_BANK))
    parser.add_argument("--sheet", default="08 statistics-ml")
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--query", help="Seed query override; skips workbook row lookup.")
    parser.add_argument("--followup", action="append", default=[], help="Follow-up turn. May be repeated.")
    parser.add_argument("--turns-json", help="JSON file containing the complete ordered turn list.")
    parser.add_argument("--mode", default="enforce_selected", help="DISSOLVE_TYPED_PLANNER mode.")
    parser.add_argument("--thread-id", default=f"query-bank-chat-{uuid.uuid4()}")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(argv)

    os.environ["DISSOLVE_TYPED_PLANNER"] = args.mode
    turns = _load_turns(args)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / f"query_bank_chat_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript = run_chat(turns, thread_id=args.thread_id)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": args.thread_id,
        "planner_mode": args.mode,
        "query_bank": str(Path(args.query_bank)),
        "sheet": args.sheet,
        "row": args.row,
    }
    payload = {"metadata": metadata, "turns": transcript}
    (output_dir / "transcript.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = ["# Query Bank Chat Harness Summary", "", json.dumps(metadata, indent=2), ""]
    summary_lines.extend(_turn_summary(turn) for turn in transcript)
    (output_dir / "summary.md").write_text("\n\n".join(summary_lines), encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
