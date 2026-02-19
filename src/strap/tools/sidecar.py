"""Per-session sidecar file tools for high-bandwidth sequential subagent handoffs.

Scholar-researcher and patent-researcher write structured JSON findings here.
Rag-analyst reads them at the start of execution to avoid token-overloading
the plain-string ToolMessage channel.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level slot — populated by set_scratch_dir() called from agent.py
_SCRATCH_DIR: Optional[Path] = None

_KEY_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')


def _get_scratch_dir() -> Path:
    if _SCRATCH_DIR is None:
        raise RuntimeError(
            "Sidecar tools require a scratch directory. "
            "Call set_scratch_dir() before using write_sidecar / read_sidecar."
        )
    return _SCRATCH_DIR


def set_scratch_dir(path: Path) -> None:
    global _SCRATCH_DIR
    path.mkdir(parents=True, exist_ok=True)
    _SCRATCH_DIR = path
    logger.info("Sidecar scratch directory: %s", path)


def write_sidecar(key: str, data: str) -> str:
    """Write structured JSON findings to the session sidecar directory.

    Use this BEFORE synthesizing your final answer so that downstream agents
    (rag-analyst) can read your full structured findings without relying on
    the truncated ToolMessage channel.

    Args:
        key: Filename stem identifying the data (e.g., "scholar_findings",
             "patent_findings"). Only letters, digits, hyphens, underscores. Max 64 chars.
        data: A JSON string representing your structured findings.

    Returns:
        Confirmation message with the path and size written.
    """
    if not _KEY_RE.match(key):
        return f"Error: key '{key}' is invalid. Use only letters, digits, hyphens, underscores (max 64 chars)."
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        return f"Error: data is not valid JSON: {e}"
    scratch = _get_scratch_dir()
    target = scratch / f"{key}.json"
    try:
        target.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"Sidecar written: {target} ({target.stat().st_size} bytes)"
    except OSError as e:
        return f"Error writing sidecar: {e}"


def read_sidecar(key: str) -> str:
    """Read structured JSON findings from the session sidecar directory.

    Call this at the START of your execution to check for upstream agent context.

    Args:
        key: Filename stem to read (e.g., "scholar_findings", "patent_findings").
             Use "list" as the key to see all available sidecar files.

    Returns:
        JSON string of the stored findings, or a message if not found.
    """
    scratch = _get_scratch_dir()
    if key == "list":
        files = sorted(scratch.glob("*.json"))
        if not files:
            return "No sidecar files available in this session."
        return "Available sidecar keys:\n" + "\n".join(f.stem for f in files)
    target = scratch / f"{key}.json"
    if not target.exists():
        available = [f.stem for f in sorted(scratch.glob("*.json"))]
        return f"No sidecar file found for key '{key}'. Available: {available}"
    try:
        content = target.read_text(encoding="utf-8")
        size_kb = len(content) / 1024
        return f"Sidecar '{key}' ({size_kb:.1f} KB):\n{content}"
    except OSError as e:
        return f"Error reading sidecar: {e}"
