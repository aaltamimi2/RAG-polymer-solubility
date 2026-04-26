"""Persistence for typed planner runtime diagnostics."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from strap.planning.compiler import CompileResult
from strap.planning.config import PlannerConfig
from strap.planning.models import ExecutionLedger, PlanningModel, RequestPlan
from strap.planning.runtime_paths import (
    RUNTIME_ARTIFACT_SCHEMA_VERSION,
    atomic_write_json,
    normalize_runtime_path,
    resolve_runtime_output_dir,
    slugify_run_component,
    unique_child_path,
)


class RuntimeArtifactManifest(PlanningModel):
    schema_version: Literal["1.0"] = RUNTIME_ARTIFACT_SCHEMA_VERSION
    run_id: str
    run_dir: str
    files: dict[str, str]
    produced_file_copies: dict[str, str] = Field(default_factory=dict)
    created_at: str


def _run_dir(root: Path, run_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return unique_child_path(root, f"{timestamp}_{slugify_run_component(run_id)}")


def _copy_existing_artifact_files(ledger: ExecutionLedger | None, run_dir: Path) -> dict[str, str]:
    if ledger is None:
        return {}
    copied: dict[str, str] = {}
    produced_dir = run_dir / "produced_files"
    seen_sources: set[str] = set()
    for artifact in ledger.artifacts:
        for raw_path in artifact.output_paths:
            source = Path(normalize_runtime_path(raw_path))
            source_key = str(source)
            if source_key in seen_sources:
                continue
            if not source.exists() or not source.is_file():
                continue
            destination = unique_child_path(produced_dir, source.name)
            shutil.copy2(source, destination)
            copied[source_key] = str(destination)
            seen_sources.add(source_key)
    return copied


def persist_runtime_artifacts(
    *,
    query: str,
    compile_result: CompileResult,
    config: PlannerConfig,
    plan: RequestPlan | None = None,
    ledger: ExecutionLedger | None = None,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> RuntimeArtifactManifest:
    """Persist a stable, readable typed-runtime diagnostic bundle."""
    root = resolve_runtime_output_dir(output_root)
    effective_run_id = run_id or (ledger.run_id if ledger else plan.plan_id if plan else "compile_failure")
    run_dir = _run_dir(root, effective_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "schema_version": RUNTIME_ARTIFACT_SCHEMA_VERSION,
        "query": query,
        "run_id": effective_run_id,
        "planner_mode": config.mode,
        "selected_enforcement_artifacts": sorted(config.selected_enforcement_artifacts),
        "selected_enforcement_workflows": sorted(config.selected_enforcement_workflows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    files = {
        "request": str(run_dir / "request.json"),
        "compile_result": str(run_dir / "compile_result.json"),
        "plan": str(run_dir / "plan.json"),
        "ledger": str(run_dir / "ledger.json"),
        "artifacts": str(run_dir / "artifacts.json"),
    }
    atomic_write_json(Path(files["request"]), request_payload)
    atomic_write_json(Path(files["compile_result"]), compile_result.model_dump(mode="json"))
    atomic_write_json(Path(files["plan"]), plan.model_dump(mode="json") if plan else None)
    atomic_write_json(Path(files["ledger"]), ledger.model_dump(mode="json") if ledger else None)
    atomic_write_json(
        Path(files["artifacts"]),
        [artifact.model_dump(mode="json") for artifact in ledger.artifacts] if ledger else [],
    )

    copied = _copy_existing_artifact_files(ledger, run_dir)
    manifest = RuntimeArtifactManifest(
        run_id=effective_run_id,
        run_dir=str(run_dir),
        files=files,
        produced_file_copies=copied,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = run_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    manifest.files["manifest"] = str(manifest_path)
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest
