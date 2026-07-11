"""Persistent agent memory — markdown fact files mirroring Claude Code's layout.

Layout (default root ``~/.dissolve/memory``, override ``DISSOLVE_MEMORY_DIR``)::

    memory/
      MEMORY.md          # human-browsable index, one line per memory
      <slug>.md          # one fact per file:  frontmatter + markdown body

Fact file format::

    ---
    name: biosteam-py311-blocker
    description: BioSTEAM cannot run under py3.11 on the dev box
    metadata:
      type: project
    ---

    <the fact body, markdown>

Design decisions (deliberately mirroring Claude Code's memory logic):

- **Markdown, human-editable.** Memories are reviewable prose, not rows. Users
  can edit or delete a fact file by hand and the system keeps working.
- **Index in context, bodies on demand.** Only the index (name + one-line
  description per memory) is injected into the system prompt each session;
  the agent reads a fact file with ``read_file`` when the index line looks
  relevant. Context cost stays O(#memories), not O(total content).
- **The index is derived, never trusted.** The in-context index is rebuilt
  from fact-file frontmatter every load, and MEMORY.md is rewritten whenever
  it drifts — a forgotten index update can never hide a memory.
- **Typed facts.** ``user`` (who the user is), ``feedback`` (durable guidance
  on how to work), ``project`` (ongoing goals/constraints not derivable from
  data), ``reference`` (pointers to external resources).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_TYPES = ("user", "feedback", "project", "reference")
_INDEX_FILENAME = "MEMORY.md"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def memory_root() -> Path:
    """Memory directory (created on first use)."""
    root = Path(os.getenv("DISSOLVE_MEMORY_DIR") or Path.home() / ".dissolve" / "memory")
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class MemoryRecord:
    name: str
    description: str
    memory_type: str
    path: Path

    def index_line(self) -> str:
        return f"- [{self.name}]({self.name}.md) — {self.description} ({self.memory_type})"


def _parse_frontmatter(text: str) -> dict[str, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("type:"):
            fields["type"] = stripped.split(":", 1)[1].strip()
        elif ":" in stripped and not stripped.startswith("metadata"):
            key, value = stripped.split(":", 1)
            fields.setdefault(key.strip(), value.strip())
    return fields


def list_memories(root: Path | None = None) -> list[MemoryRecord]:
    root = root or memory_root()
    records: list[MemoryRecord] = []
    for path in sorted(root.glob("*.md")):
        if path.name == _INDEX_FILENAME:
            continue
        try:
            fields = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        name = fields.get("name") or path.stem
        memory_type = fields.get("type") or "project"
        if memory_type not in MEMORY_TYPES:
            memory_type = "project"
        records.append(
            MemoryRecord(
                name=name,
                description=fields.get("description") or "(no description)",
                memory_type=memory_type,
                path=path,
            )
        )
    return records


def sync_index_file(root: Path | None = None) -> Path:
    """Rewrite MEMORY.md from fact-file frontmatter (deterministic)."""
    root = root or memory_root()
    records = list_memories(root)
    lines = ["# Memory index", ""]
    lines.extend(record.index_line() for record in records)
    if not records:
        lines.append("(no memories saved yet)")
    index_path = root / _INDEX_FILENAME
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def render_memory_context(root: Path | None = None) -> str:
    """The system-prompt block: index + recall instructions + save policy."""
    root = root or memory_root()
    records = list_memories(root)
    if records:
        index_body = "\n".join(record.index_line() for record in records)
    else:
        index_body = "(no memories saved yet)"
    return (
        "\n\n<dissolve_memory>\n"
        f"Persistent memory directory: {root}\n"
        "Memory index (one line per saved memory):\n"
        f"{index_body}\n\n"
        "Recall: when an index line is relevant to the current request, read the full "
        f"memory with read_file('{root}/<name>.md') before answering.\n"
        "Save: when the user states durable facts about themselves (user), gives "
        "lasting guidance or corrections on how to work (feedback), shares ongoing "
        "goals/constraints not derivable from the data (project), or points at "
        "external resources (reference), call save_memory. Update the existing "
        "memory instead of creating a near-duplicate; call delete_memory when a "
        "memory turns out to be wrong. Do not save transient conversation details, "
        "anything derivable from the database/tools, or one-off numbers from this "
        "session. Convert relative dates to absolute dates before saving.\n"
        "</dissolve_memory>"
    )


def _validate_slug(name: str) -> str | None:
    slug = (name or "").strip().lower()
    if not _SLUG_RE.match(slug):
        return None
    return slug


def save_memory_record(
    name: str,
    description: str,
    memory_type: str,
    content: str,
    root: Path | None = None,
) -> str:
    """Write (or update) one memory fact file and sync the index."""
    root = root or memory_root()
    slug = _validate_slug(name)
    if slug is None:
        return (
            "Error: name must be a short kebab-case slug (lowercase letters, digits, "
            "hyphens; 2-64 chars), e.g. 'prefers-msp-reporting'."
        )
    memory_type = (memory_type or "").strip().lower()
    if memory_type not in MEMORY_TYPES:
        return f"Error: type must be one of {', '.join(MEMORY_TYPES)}."
    description = " ".join((description or "").split())
    if not description:
        return "Error: a one-line description is required (it becomes the index line)."
    body = (content or "").strip()
    if not body:
        return "Error: memory content is empty; nothing to save."

    path = root / f"{slug}.md"
    existed = path.exists()
    path.write_text(
        "---\n"
        f"name: {slug}\n"
        f"description: {description}\n"
        "metadata:\n"
        f"  type: {memory_type}\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    sync_index_file(root)
    action = "Updated" if existed else "Saved"
    logger.info("memory_store: %s memory '%s' (%s)", action.lower(), slug, memory_type)
    return f"{action} memory '{slug}' ({memory_type}) at {path}. Index refreshed."


def delete_memory_record(name: str, root: Path | None = None) -> str:
    root = root or memory_root()
    slug = _validate_slug(name)
    if slug is None:
        return "Error: invalid memory name."
    path = root / f"{slug}.md"
    if not path.exists():
        return f"No memory named '{slug}' exists."
    path.unlink()
    sync_index_file(root)
    logger.info("memory_store: deleted memory '%s'", slug)
    return f"Deleted memory '{slug}'. Index refreshed."


# ---------------------------------------------------------------------------
# Agent-facing tools
# ---------------------------------------------------------------------------

def save_memory(name: str, description: str, memory_type: str, content: str) -> str:
    """Save a durable memory the agent should recall in future sessions.

    WHEN TO USE:
    - The user states lasting facts about themselves or their preferences (type 'user')
    - The user gives durable guidance/corrections on how to work (type 'feedback')
    - Ongoing project goals or constraints not derivable from the data (type 'project')
    - Pointers to external resources like papers, dashboards, tickets (type 'reference')

    Do NOT save transient conversation details or values derivable from the tools.
    If a similar memory exists (check the memory index), reuse its exact name to
    update it instead of creating a duplicate.

    Args:
        name: short kebab-case slug, e.g. 'prefers-msp-reporting'
        description: one line shown in the memory index
        memory_type: one of 'user', 'feedback', 'project', 'reference'
        content: the memory body (markdown). For feedback/project include why it
            matters and how to apply it.
    """
    return save_memory_record(name, description, memory_type, content)


def delete_memory(name: str) -> str:
    """Delete a saved memory that is wrong or obsolete.

    Args:
        name: the memory's kebab-case slug as shown in the memory index
    """
    return delete_memory_record(name)


def get_memory_tools() -> list:
    return [save_memory, delete_memory]


# ---------------------------------------------------------------------------
# Middleware: inject the index + policy into the orchestrator system prompt
# ---------------------------------------------------------------------------

from langchain.agents.middleware.types import AgentMiddleware  # noqa: E402


class DissolveMemoryMiddleware(AgentMiddleware):
    """Injects the persistent-memory index into the main agent's system prompt.

    The index is re-rendered on every model call, so a memory saved earlier in
    the same turn is visible to the next model call. Applies to the
    orchestrator only; specialists receive memory context through their task
    descriptions when the orchestrator judges it relevant.
    """

    def __init__(self, root: Path | None = None) -> None:
        super().__init__()
        self._root = root

    def _inject(self, request):
        if request.system_message is None:
            return request
        from deepagents.middleware._utils import append_to_system_message

        block = render_memory_context(self._root)
        return request.override(
            system_message=append_to_system_message(request.system_message, block)
        )

    def wrap_model_call(self, request, handler):
        return handler(self._inject(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._inject(request))
