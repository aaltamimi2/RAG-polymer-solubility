"""Claude model registry for the Claude Agent SDK harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict


class ClaudeModelSpec(TypedDict):
    label: str
    model: str
    provider: str
    env_var: str


@dataclass(frozen=True)
class ClaudeModelSelection:
    alias: str
    sdk_model: str
    provider_model_id: str
    spec: ClaudeModelSpec
    previous_alias: str | None = None
    notice: str | None = None


DEFAULT_CLAUDE_MODEL_ALIAS = "claude-sonnet"
DEFAULT_CLAUDE_SDK_MODEL = "claude-sonnet-4-6"
_CLAUDE_ALIASES = ("claude-sonnet", "claude")


def _normalize_sdk_model(model: str) -> str:
    value = str(model or "").strip()
    if value.startswith("anthropic:"):
        return value.split(":", 1)[1]
    return value


def _provider_model_id(model: str) -> str:
    sdk_model = _normalize_sdk_model(model)
    return f"anthropic:{sdk_model}" if sdk_model else ""


def claude_model_registry() -> dict[str, ClaudeModelSpec]:
    """Return Claude-compatible model aliases for SDK mode."""
    model = (
        os.getenv("DISSOLVE_CLAUDE_SONNET_MODEL")
        or os.getenv("DISSOLVE_CLAUDE_SDK_MODEL")
        or DEFAULT_CLAUDE_SDK_MODEL
    )
    provider_model = _provider_model_id(model)
    spec: ClaudeModelSpec = {
        "label": "Claude Sonnet",
        "model": provider_model,
        "provider": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
    }
    return {alias: dict(spec) for alias in _CLAUDE_ALIASES}


def is_claude_model_alias(value: str | None) -> bool:
    if not value:
        return False
    requested = value.strip()
    registry = claude_model_registry()
    if requested in registry:
        return True
    provider_model = _provider_model_id(requested)
    return any(provider_model == spec["model"] for spec in registry.values())


def resolve_claude_model_selection(raw_model: str | None = None) -> ClaudeModelSelection:
    """Resolve a CLI model request for Claude SDK mode.

    Non-Claude aliases are replaced by the default Claude alias. This keeps
    persisted Gemini selections from leaking into the Claude SDK runner.
    """
    registry = claude_model_registry()
    requested = (raw_model or os.getenv("DISSOLVE_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL_ALIAS).strip()
    if not requested:
        requested = DEFAULT_CLAUDE_MODEL_ALIAS

    if requested in registry:
        spec = registry[requested]
        return ClaudeModelSelection(
            alias=requested,
            sdk_model=_normalize_sdk_model(spec["model"]),
            provider_model_id=spec["model"],
            spec=spec,
        )

    provider_model = _provider_model_id(requested)
    for alias, spec in registry.items():
        if provider_model == spec["model"]:
            return ClaudeModelSelection(
                alias=alias,
                sdk_model=_normalize_sdk_model(spec["model"]),
                provider_model_id=spec["model"],
                spec=spec,
            )

    default_spec = registry[DEFAULT_CLAUDE_MODEL_ALIAS]
    return ClaudeModelSelection(
        alias=DEFAULT_CLAUDE_MODEL_ALIAS,
        sdk_model=_normalize_sdk_model(default_spec["model"]),
        provider_model_id=default_spec["model"],
        spec=default_spec,
        previous_alias=requested,
        notice=(
            f"Model '{requested}' is not Claude-compatible for claude_sdk; "
            f"using {DEFAULT_CLAUDE_MODEL_ALIAS} instead."
        ),
    )
