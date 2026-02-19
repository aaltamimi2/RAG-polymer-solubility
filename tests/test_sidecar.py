"""Tests for sidecar file tools."""
import json
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def scratch_dir():
    """Create a temporary scratch directory."""
    with tempfile.TemporaryDirectory(prefix="test_sidecar_") as d:
        from strap.tools.sidecar import set_scratch_dir
        set_scratch_dir(Path(d))
        yield Path(d)


class TestWriteSidecar:
    def test_writes_valid_json(self, scratch_dir):
        from strap.tools.sidecar import write_sidecar
        data = json.dumps({"papers": [{"title": "Test"}]})
        result = write_sidecar("scholar_findings", data)
        assert "Sidecar written" in result
        assert (scratch_dir / "scholar_findings.json").exists()

    def test_rejects_invalid_key(self, scratch_dir):
        from strap.tools.sidecar import write_sidecar
        result = write_sidecar("../etc/passwd", '{"bad": true}')
        assert "Error" in result or "invalid" in result

    def test_rejects_invalid_json(self, scratch_dir):
        from strap.tools.sidecar import write_sidecar
        result = write_sidecar("test", "not json")
        assert "Error" in result and "JSON" in result

    def test_key_length_limit(self, scratch_dir):
        from strap.tools.sidecar import write_sidecar
        # Key regex allows 1-64 chars; 65 chars should be rejected
        result = write_sidecar("a" * 65, '{"ok": true}')
        assert "invalid" in result.lower() or "Error" in result

    def test_valid_key_at_max_length(self, scratch_dir):
        """A 64-character key should be accepted (boundary case)."""
        from strap.tools.sidecar import write_sidecar
        key = "a" * 64
        result = write_sidecar(key, '{"x": 1}')
        assert "Sidecar written" in result
        assert (scratch_dir / f"{key}.json").exists()

    def test_written_file_is_valid_json(self, scratch_dir):
        """The file on disk should be valid JSON matching the input data."""
        from strap.tools.sidecar import write_sidecar
        data = {"alpha": 1, "beta": [1, 2, 3]}
        write_sidecar("mykey", json.dumps(data))
        on_disk = json.loads((scratch_dir / "mykey.json").read_text())
        assert on_disk == data

    def test_key_with_hyphen_and_underscore(self, scratch_dir):
        """Hyphens and underscores are valid in keys."""
        from strap.tools.sidecar import write_sidecar
        result = write_sidecar("my-key_001", '{"ok": true}')
        assert "Sidecar written" in result

    def test_key_with_space_rejected(self, scratch_dir):
        """Spaces are not allowed in keys."""
        from strap.tools.sidecar import write_sidecar
        result = write_sidecar("bad key", '{"ok": true}')
        assert "Error" in result or "invalid" in result.lower()

    def test_overwrites_existing_file(self, scratch_dir):
        """Writing twice to the same key should overwrite."""
        from strap.tools.sidecar import write_sidecar, read_sidecar
        write_sidecar("overwrite_test", '{"v": 1}')
        write_sidecar("overwrite_test", '{"v": 2}')
        content = read_sidecar("overwrite_test")
        assert '"v": 2' in content


class TestReadSidecar:
    def test_reads_written_file(self, scratch_dir):
        from strap.tools.sidecar import write_sidecar, read_sidecar
        write_sidecar("test_key", '{"value": 42}')
        result = read_sidecar("test_key")
        assert "42" in result
        assert "KB" in result

    def test_list_available_keys(self, scratch_dir):
        from strap.tools.sidecar import write_sidecar, read_sidecar
        write_sidecar("alpha", '{"a": 1}')
        write_sidecar("beta", '{"b": 2}')
        result = read_sidecar("list")
        assert "alpha" in result
        assert "beta" in result

    def test_missing_key_shows_available(self, scratch_dir):
        from strap.tools.sidecar import write_sidecar, read_sidecar
        write_sidecar("existing", '{"x": 1}')
        result = read_sidecar("nonexistent")
        assert "No sidecar file" in result
        assert "existing" in result

    def test_empty_list(self, scratch_dir):
        from strap.tools.sidecar import read_sidecar
        result = read_sidecar("list")
        assert "No sidecar files" in result

    def test_read_returns_original_data(self, scratch_dir):
        """Data read back should match what was written."""
        from strap.tools.sidecar import write_sidecar, read_sidecar
        original = {"nested": {"key": "value"}, "numbers": [1, 2, 3]}
        write_sidecar("data_test", json.dumps(original))
        result = read_sidecar("data_test")
        # The result string contains the JSON content
        assert '"nested"' in result
        assert '"value"' in result

    def test_size_reported_in_kb(self, scratch_dir):
        """The read result should include a KB size indicator."""
        from strap.tools.sidecar import write_sidecar, read_sidecar
        write_sidecar("size_check", json.dumps({"data": "x" * 100}))
        result = read_sidecar("size_check")
        assert "KB" in result
