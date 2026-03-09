"""Shared loader for subagent specs and execution-pair configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIR = Path(__file__).parent
_DEFAULT_CONFIG_PATH = _PACKAGE_DIR / "subagents.yaml"


def _read_yaml(path: Path) -> Any:
    with open(path) as handle:
        return yaml.safe_load(handle)


def _normalize_legacy_bundle(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return {
            "subagents": list(data.get("subagents", [])),
            "execution_pairs": dict(data.get("execution_pairs", {})),
        }
    if isinstance(data, list):
        return {"subagents": list(data), "execution_pairs": {}}
    raise ValueError("Subagent config must be a YAML mapping or list.")


def _load_manifest_bundle(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    base_dir = manifest_path.parent
    subagent_files = manifest.get("subagent_files", [])
    execution_pairs_file = manifest.get("execution_pairs_file")

    subagents: list[dict[str, Any]] = []
    for rel_path in subagent_files:
        spec_path = (base_dir / rel_path).resolve()
        spec = _read_yaml(spec_path)
        if not isinstance(spec, dict):
            raise ValueError(f"Subagent spec must be a mapping: {spec_path}")
        subagents.append(spec)

    execution_pairs: dict[str, Any] = {}
    if execution_pairs_file:
        exec_path = (base_dir / execution_pairs_file).resolve()
        loaded = _read_yaml(exec_path) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"execution_pairs must be a mapping: {exec_path}")
        execution_pairs = loaded

    return {"subagents": subagents, "execution_pairs": execution_pairs}


def load_subagent_bundle(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load subagent specs plus execution-pair config.

    Supported formats:
    - legacy single YAML file with ``subagents`` / ``execution_pairs``
    - legacy flat YAML list of subagents
    - manifest file with ``subagent_files`` + ``execution_pairs_file``
    - config directory containing ``execution_pairs.yaml`` and ``subagents/*.yaml``
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

    if path.is_dir():
        exec_pairs_path = path / "execution_pairs.yaml"
        execution_pairs = _read_yaml(exec_pairs_path) if exec_pairs_path.exists() else {}
        subagent_dir = path / "subagents"
        subagents = [
            _read_yaml(spec_path)
            for spec_path in sorted(subagent_dir.glob("*.yaml"))
        ]
        return {
            "subagents": [spec for spec in subagents if isinstance(spec, dict)],
            "execution_pairs": execution_pairs if isinstance(execution_pairs, dict) else {},
        }

    data = _read_yaml(path)
    if isinstance(data, dict) and "subagent_files" in data:
        return _load_manifest_bundle(path, data)
    return _normalize_legacy_bundle(data)


def load_subagent_specs(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load only subagent specs."""
    bundle = load_subagent_bundle(config_path)
    return list(bundle.get("subagents", []))


def load_execution_pairs(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load only execution-pair config."""
    bundle = load_subagent_bundle(config_path)
    return dict(bundle.get("execution_pairs", {}))


def load_routing_configuration(
    config_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], set[frozenset[str]], set[frozenset[str]], dict[tuple[str, str], None]]:
    """Build routing rules and execution-pair lookup tables from config."""
    bundle = load_subagent_bundle(config_path)
    specs = bundle.get("subagents", [])
    execution_pairs = bundle.get("execution_pairs", {})

    routing_rules: list[dict[str, Any]] = []
    for spec in specs:
        routing = spec.get("routing")
        if not routing:
            continue
        routing_rules.append({
            "subagent": spec["name"],
            "priority": routing.get("priority", 999),
            "description": spec.get("description", "").strip(),
            "phrases": list(routing.get("phrases", [])),
            "high_stems": list(routing.get("high_stems", [])),
            "low_stems": list(routing.get("low_stems", [])),
            "negatives": list(routing.get("negatives", [])),
        })
    routing_rules.sort(key=lambda rule: rule["priority"])

    parallel_pairs = {
        frozenset(pair)
        for pair in execution_pairs.get("parallel", [])
    }
    parallel_3way = {
        frozenset(group)
        for group in execution_pairs.get("parallel_3way", [])
    }
    sequential_pairs = {
        (pair[0], pair[1]): None
        for pair in execution_pairs.get("sequential", [])
    }
    return routing_rules, parallel_pairs, parallel_3way, sequential_pairs
