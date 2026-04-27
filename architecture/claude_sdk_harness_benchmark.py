"""Small harness-comparison benchmark entrypoint for Claude SDK migration.

This is intentionally lightweight: it runs a prompt list against the optional
Claude SDK runner and writes JSONL records. Live Claude calls require
``RUN_LIVE_CLAUDE_SDK=1`` plus ``ANTHROPIC_API_KEY``; direct fast-path turns can
run without either.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from strap.claude_sdk_harness.benchmarks import benchmark_record  # noqa: E402
from strap.claude_sdk_harness.models import resolve_claude_model_selection  # noqa: E402
from strap.claude_sdk_harness.runner import ClaudeSdkRunner  # noqa: E402
from strap.session_state import (  # noqa: E402
    load_session_context,
    save_session_context,
    update_session_context_from_direct_metadata,
    update_session_context_from_text,
)

DEFAULT_PROMPTS = [
    "what are good solvents for dissolving EVOH",
    "what is the solubility of EVOH in DMF from room temp to 80C",
    "plot it up to 90C and save to docs",
    "where was that plot saved?",
]


def _read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=None, help="Text file with one prompt per line.")
    parser.add_argument("--output", type=Path, default=Path("architecture/test_results/claude_sdk_harness_benchmark.jsonl"))
    parser.add_argument("--session", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if os.getenv("RUN_LIVE_CLAUDE_SDK") != "1":
        os.environ.pop("ANTHROPIC_API_KEY", None)

    selection = resolve_claude_model_selection(args.model)
    thread_id = args.session or f"claude-sdk-bench-{uuid.uuid4().hex[:8]}"
    runner = ClaudeSdkRunner(
        thread_id=thread_id,
        model_alias=selection.alias,
        sdk_model=selection.sdk_model,
        cwd=REPO_ROOT,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for index, prompt in enumerate(_read_prompts(args.prompts), start=1):
            context = load_session_context(thread_id)
            context = update_session_context_from_text(context, prompt, role="user")
            t0 = time.perf_counter()
            result = runner.run_turn(prompt, session_context=context)
            latency = time.perf_counter() - t0
            context = update_session_context_from_text(context, result.content, role="assistant")
            context = update_session_context_from_direct_metadata(context, result.additional_kwargs)
            save_session_context(thread_id, context)
            fh.write(
                json.dumps(
                    benchmark_record(
                        query_id=f"turn-{index}",
                        harness="claude_sdk",
                        model_alias=selection.alias,
                        model_id=selection.provider_model_id,
                        result=result,
                        latency_s=latency,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
