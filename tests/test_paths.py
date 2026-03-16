from __future__ import annotations

from pathlib import Path

from strap import paths


def test_runtime_paths_respect_environment(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("ML_MODEL_DIR", str(models_dir))
    paths.get_data_dir.cache_clear()
    paths.get_models_dir.cache_clear()

    try:
        assert paths.get_data_dir() == data_dir.resolve()
        assert paths.get_models_dir() == models_dir.resolve()
        assert paths.get_data_path("foo.json") == data_dir.resolve() / "foo.json"
        assert paths.get_models_path("bar.pkl") == models_dir.resolve() / "bar.pkl"
    finally:
        paths.get_data_dir.cache_clear()
        paths.get_models_dir.cache_clear()


def test_runtime_data_path_resolves_existing_repo_assets() -> None:
    data_dir = paths.get_data_dir()
    assert data_dir.exists()
    assert paths.get_data_path("solubility_coefficients.json").exists()
