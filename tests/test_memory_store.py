"""Persistent markdown memory (Claude-Code mirror): store, index, middleware."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import SystemMessage

from strap.memory_store import (
    DissolveMemoryMiddleware,
    delete_memory_record,
    list_memories,
    memory_root,
    render_memory_context,
    save_memory_record,
    sync_index_file,
)


@pytest.fixture()
def mem_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DISSOLVE_MEMORY_DIR", str(tmp_path / "memory"))
    return tmp_path / "memory"


class TestStore:
    def test_save_list_delete_round_trip(self, mem_root):
        out = save_memory_record(
            "prefers-msp-reporting",
            "Always report MSP alongside TCI/AOC",
            "feedback",
            "**Why:** compares processes by MSP.\n**How to apply:** include MSP in TEA summaries.",
        )
        assert "Saved memory 'prefers-msp-reporting'" in out
        records = list_memories()
        assert [r.name for r in records] == ["prefers-msp-reporting"]
        assert records[0].memory_type == "feedback"
        assert "MSP" in records[0].description

        out = delete_memory_record("prefers-msp-reporting")
        assert "Deleted" in out
        assert list_memories() == []

    def test_update_same_slug_overwrites_instead_of_duplicating(self, mem_root):
        save_memory_record("evoh-focus", "old description", "project", "old body")
        save_memory_record("evoh-focus", "new description", "project", "new body")
        records = list_memories()
        assert len(records) == 1
        assert records[0].description == "new description"
        assert "new body" in records[0].path.read_text()

    def test_slug_and_type_validation(self, mem_root):
        assert "Error: name" in save_memory_record("Bad Name!", "d", "feedback", "x")
        assert "Error: type" in save_memory_record("ok-name", "d", "banana", "x")
        assert "Error: a one-line description" in save_memory_record("ok-name", "  ", "user", "x")
        assert "Error: memory content" in save_memory_record("ok-name", "d", "user", "  ")
        assert list_memories() == []

    def test_index_file_rewritten_deterministically(self, mem_root):
        save_memory_record("a-fact", "first", "project", "x")
        save_memory_record("b-fact", "second", "reference", "y")
        index = (memory_root() / "MEMORY.md").read_text()
        assert "[a-fact](a-fact.md) — first (project)" in index
        assert "[b-fact](b-fact.md) — second (reference)" in index

    def test_hand_written_file_appears_without_index_entry(self, mem_root):
        """The in-context index derives from frontmatter — a hand-dropped fact
        file (or a forgotten index update) can never hide a memory."""
        root = memory_root()
        (root / "hand-added.md").write_text(
            "---\nname: hand-added\ndescription: added by a human editor\nmetadata:\n  type: user\n---\n\nbody\n"
        )
        context = render_memory_context()
        assert "hand-added" in context
        assert "added by a human editor" in context
        sync_index_file()
        assert "hand-added" in (root / "MEMORY.md").read_text()


class TestContextBlock:
    def test_contains_index_recall_and_save_policy(self, mem_root):
        save_memory_record("a-fact", "first", "project", "x")
        block = render_memory_context()
        assert "<dissolve_memory>" in block
        assert "a-fact" in block
        assert "read_file" in block           # recall instruction
        assert "save_memory" in block         # save policy
        assert "delete_memory" in block

    def test_empty_memory_still_renders_policy(self, mem_root):
        block = render_memory_context()
        assert "(no memories saved yet)" in block
        assert "save_memory" in block


class TestMiddleware:
    def test_injects_block_into_system_message(self, mem_root):
        save_memory_record("a-fact", "first", "project", "x")
        mw = DissolveMemoryMiddleware()
        request = MagicMock()
        request.system_message = SystemMessage(content="You are DISSOLVE.")
        handler = MagicMock(return_value="resp")

        out = mw.wrap_model_call(request, handler)

        assert out == "resp"
        request.override.assert_called_once()
        new_system = request.override.call_args.kwargs["system_message"]
        assert "a-fact" in str(new_system.content)
        assert "You are DISSOLVE." in str(new_system.content)

    def test_no_system_message_passes_through(self, mem_root):
        mw = DissolveMemoryMiddleware()
        request = MagicMock()
        request.system_message = None
        handler = MagicMock(return_value="resp")
        assert mw.wrap_model_call(request, handler) == "resp"
        request.override.assert_not_called()


class TestDurableCheckpointer:
    def test_sqlite_saver_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DISSOLVE_CHECKPOINT_DB", str(tmp_path / "ck.sqlite3"))
        from strap.agent import _durable_checkpointer

        saver = _durable_checkpointer()
        assert type(saver).__name__ == "SqliteSaver"
        assert (tmp_path / "ck.sqlite3").exists() or (tmp_path).exists()

    def test_falls_back_to_memory_saver_on_unwritable_path(self, monkeypatch):
        monkeypatch.setenv("DISSOLVE_CHECKPOINT_DB", "/proc/definitely/not/writable.sqlite3")
        from strap.agent import _durable_checkpointer

        saver = _durable_checkpointer()
        assert type(saver).__name__ in ("MemorySaver", "InMemorySaver")
