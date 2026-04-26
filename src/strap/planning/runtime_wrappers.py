"""Runtime callable wrappers for the typed executor bridge."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from strap.planning.executor import StepCallable, StepCallableResult
from strap.planning.models import ArtifactFrame, ExecutionLedger, PlanStep
from strap.planning.runtime_paths import normalize_runtime_path


RawRuntimeCallable = Callable[..., Any]


def _required_artifact_types(step: PlanStep) -> list[str]:
    return [
        artifact.artifact_type
        for contract in step.output_contracts
        for artifact in contract.artifact_contracts
        if artifact.required
    ]


def _parse_json_like(raw: Any) -> Any:
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return raw
    return raw


def _collect_paths(raw: Any) -> list[str]:
    """Collect file-like paths from normalized legacy outputs."""
    raw = _parse_json_like(raw)
    paths: list[str] = []

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, list | tuple | set | frozenset):
            for item in value:
                visit(item, key)
            return
        if not isinstance(value, str):
            return
        normalized_key = (key or "").lower()
        if normalized_key in {
            "path",
            "paths",
            "filepath",
            "file_path",
            "local_path",
            "output_path",
            "output_paths",
            "plot_path",
            "plot_paths",
            "artifact",
            "artifacts",
            "sidecar_path",
        } or value.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".html", ".json", ".csv", ".xlsx")):
            paths.append(normalize_runtime_path(value))

    visit(raw)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def artifact_frames_from_contracts(
    step: PlanStep,
    *,
    raw_output: Any = None,
    path_overrides: Mapping[str, list[str]] | None = None,
    artifact_types: Iterable[str] | None = None,
) -> list[ArtifactFrame]:
    """Create artifact frames from the step's declared output contracts."""
    output_paths = _collect_paths(raw_output)
    path_overrides = path_overrides or {}
    allowed = set(artifact_types) if artifact_types is not None else None
    artifacts: list[ArtifactFrame] = []
    for artifact_type in _required_artifact_types(step):
        if allowed is not None and artifact_type not in allowed:
            continue
        paths = path_overrides.get(artifact_type, output_paths)
        artifacts.append(
            ArtifactFrame(
                artifact_id=f"{step.step_id}:{artifact_type}",
                artifact_type=artifact_type,
                source_step_id=step.step_id,
                output_paths=list(paths),
                validation_summary={"normalized_by": "typed_runtime_wrapper"},
            )
        )
    return artifacts


def wrap_legacy_callable(
    callable_func: RawRuntimeCallable,
    *,
    artifact_types: Iterable[str] | None = None,
) -> StepCallable:
    """Wrap a legacy callable so the executor only sees StepCallableResult.

    This generic wrapper is deliberately conservative: it does not mint
    artifact frames from a step's contracts unless the wrapper author explicitly
    declares which artifact types the legacy output evidences. Production
    selected-enforcement wrappers should be artifact-specific.
    """
    explicit_artifact_types = set(artifact_types or [])

    def wrapped(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        raw = callable_func(**step.tool_args_template)
        if isinstance(raw, StepCallableResult):
            return raw
        artifacts = (
            artifact_frames_from_contracts(step, raw_output=raw, artifact_types=explicit_artifact_types)
            if explicit_artifact_types
            else []
        )
        return StepCallableResult(
            success=True,
            artifacts=artifacts,
            data={
                "legacy_result_type": type(raw).__name__,
                "artifact_types_declared": sorted(explicit_artifact_types),
            },
        )

    return wrapped


def make_static_artifact_wrapper(
    *,
    output_paths: Mapping[str, list[str]] | None = None,
    success: bool = True,
) -> StepCallable:
    """Build a deterministic wrapper for tests and dry-run harnesses.

    This helper intentionally mints all declared artifact frames and should not
    be used as a production wrapper for selected enforcement.
    """

    def wrapped(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        return StepCallableResult(
            success=success,
            artifacts=artifact_frames_from_contracts(step, path_overrides=output_paths or {}),
            data={"callable_name": step.allowed_tools[0] if step.allowed_tools else step.step_id},
            error=None if success else "static wrapper configured to fail",
        )

    return wrapped


def get_runtime_callable_registry(
    *,
    wrappers: Mapping[str, StepCallable] | None = None,
    legacy_callables: Mapping[str, RawRuntimeCallable | tuple[RawRuntimeCallable, Iterable[str]]] | None = None,
) -> dict[str, StepCallable]:
    """Return the formal runtime callable registry.

    The default registry is intentionally empty in PR 5. Runtime callers must
    provide explicit wrappers so selected enforcement cannot accidentally route
    arbitrary legacy JSON into the typed executor.
    """
    registry: dict[str, StepCallable] = {}
    if legacy_callables:
        for name, legacy in legacy_callables.items():
            if isinstance(legacy, tuple):
                callable_func, artifact_types = legacy
                registry[name] = wrap_legacy_callable(callable_func, artifact_types=artifact_types)
            else:
                registry[name] = wrap_legacy_callable(legacy)
    if wrappers:
        registry.update(dict(wrappers))
    return registry
