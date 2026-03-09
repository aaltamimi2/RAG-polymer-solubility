"""Tests for scoped, versioned sidecar artifact tools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def scoped_root():
    with tempfile.TemporaryDirectory(prefix="test_sidecar_") as tmpdir:
        from strap.handoffs import initialize_handoff_scope
        from strap.tools.sidecar import set_scratch_dir

        root = Path(tmpdir)
        set_scratch_dir(root)
        initialize_handoff_scope(
            run_id="run-a",
            thread_id="thread-a",
            invocation_id="inv-a",
        )
        yield root


class TestWriteSidecar:
    def test_writes_valid_json(self, scoped_root):
        from strap.tools.sidecar import write_sidecar

        result = json.loads(write_sidecar("scholar_findings", json.dumps({"papers": []})))
        assert result["ok"] is True
        assert result["version"] == 1
        assert Path(result["path"]).exists()

    def test_rejects_invalid_key(self, scoped_root):
        from strap.tools.sidecar import write_sidecar

        result = json.loads(write_sidecar("../etc/passwd", '{"bad": true}'))
        assert result["ok"] is False

    def test_rejects_invalid_json(self, scoped_root):
        from strap.tools.sidecar import write_sidecar

        result = json.loads(write_sidecar("test", "not json"))
        assert result["ok"] is False

    def test_versioned_writes_do_not_overwrite(self, scoped_root):
        from strap.tools.sidecar import read_sidecar, write_sidecar

        first = json.loads(write_sidecar("scholar_findings", '{"v": 1}'))
        second = json.loads(write_sidecar("scholar_findings", '{"v": 2}'))
        listing = json.loads(read_sidecar("list"))
        latest = json.loads(read_sidecar("scholar_findings"))

        assert first["version"] == 1
        assert second["version"] == 2
        assert len(listing["keys"]["scholar_findings"]) == 2
        assert latest["data"]["v"] == 2


class TestReadSidecar:
    def test_list_available_keys(self, scoped_root):
        from strap.tools.sidecar import read_sidecar, write_sidecar

        write_sidecar("alpha", '{"a": 1}')
        write_sidecar("beta", '{"b": 2}')
        result = json.loads(read_sidecar("list"))

        assert result["ok"] is True
        assert "alpha" in result["keys"]
        assert "beta" in result["keys"]

    def test_missing_key_shows_available(self, scoped_root):
        from strap.tools.sidecar import read_sidecar, write_sidecar

        write_sidecar("existing", '{"x": 1}')
        result = json.loads(read_sidecar("nonexistent"))

        assert result["ok"] is False
        assert "existing" in result["details"]["available_keys"]

    def test_scopes_are_isolated(self, scoped_root):
        from strap.handoffs import initialize_handoff_scope
        from strap.tools.sidecar import read_sidecar, write_sidecar

        write_sidecar("shared", '{"scope": "a"}')

        initialize_handoff_scope(
            run_id="run-b",
            thread_id="thread-b",
            invocation_id="inv-b",
        )
        result = json.loads(read_sidecar("shared"))
        assert result["ok"] is False

        write_sidecar("shared", '{"scope": "b"}')
        latest_b = json.loads(read_sidecar("shared"))
        assert latest_b["data"]["scope"] == "b"

        initialize_handoff_scope(
            run_id="run-a",
            thread_id="thread-a",
            invocation_id="inv-a-2",
        )
        latest_a = json.loads(read_sidecar("shared"))
        assert latest_a["data"]["scope"] == "a"

    def test_different_scopes_can_use_different_roots(self, scoped_root):
        from strap.handoffs import initialize_handoff_scope
        from strap.tools.sidecar import read_sidecar, write_sidecar

        with tempfile.TemporaryDirectory(prefix="test_sidecar_other_") as other_tmpdir:
            other_root = Path(other_tmpdir)

            initialize_handoff_scope(
                run_id="run-a",
                thread_id="thread-a",
                invocation_id="inv-a-root",
                artifact_root=scoped_root,
            )
            path_a = json.loads(write_sidecar("shared", '{"root": "a"}'))["path"]

            initialize_handoff_scope(
                run_id="run-b-root",
                thread_id="thread-b-root",
                invocation_id="inv-b-root",
                artifact_root=other_root,
            )
            path_b = json.loads(write_sidecar("shared", '{"root": "b"}'))["path"]
            latest_b = json.loads(read_sidecar("shared"))

            assert path_a.startswith(str(scoped_root))
            assert path_b.startswith(str(other_root))
            assert latest_b["data"]["root"] == "b"
