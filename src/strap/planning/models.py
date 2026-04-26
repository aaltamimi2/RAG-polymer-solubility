"""Pydantic models for the typed planning/execution harness.

These models are intentionally runtime-neutral: they define the contract shape
and deterministic validation rules, but they do not alter orchestration.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PlanMode = Literal[
    "direct_tool",
    "single_agent",
    "single_tool_or_specialist",
    "planned_workflow",
    "clarification_required",
    "unsupported",
]

IntentFamily = Literal[
    "separation",
    "safety",
    "biosteam_tea_lca",
    "optimization",
    "visualization",
    "statistics_ml",
    "research",
    "contaminant_removal",
    "mixed_workflow",
]

PlanRole = Literal[
    "separation-engineer",
    "safety-analyst",
    "biosteam-analyst",
    "scholar-researcher",
    "patent-researcher",
    "rag-analyst",
    "visualization-specialist",
    "statistics-ml",
    "contaminant-removal-analyst",
    "optimization-engineer",
    "handoff_adapter",
    "direct_tool",
]

ExecutionKind = Literal["tool", "subagent", "handoff_adapter", "synthesis"]
OutputFormat = Literal["json", "png", "csv", "xlsx", "markdown"]
PathPolicy = Literal["required", "optional", "forbidden"]
FallbackPolicy = Literal["clarify", "typed_failure", "best_effort_disclosed"]
CallableKind = Literal["tool", "subagent", "adapter"]


class PlanningModel(BaseModel):
    """Base config for strict planning models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CountConstraint(PlanningModel):
    min: int = 1
    max: int | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "CountConstraint":
        if self.min < 0:
            raise ValueError("count min must be non-negative")
        if self.max is not None and self.max < self.min:
            raise ValueError("count max must be greater than or equal to min")
        return self


class PlanAssumption(PlanningModel):
    key: str
    value: Any
    source: Literal["user", "session", "default", "inferred"] = "inferred"
    rationale: str | None = None


class MissingInput(PlanningModel):
    name: str
    reason: str
    required_for_step_id: str | None = None


class DataRequirement(PlanningModel):
    field_path: str
    required: bool = True
    expected_value: Any | None = None
    comparator: Literal["exists", "equals", "min", "max", "contains"] = "exists"


class TextRequirement(PlanningModel):
    requirement: str
    required: bool = True


class ArtifactContract(PlanningModel):
    artifact_type: str
    required: bool = True
    count: CountConstraint = Field(default_factory=CountConstraint)
    entities: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    output_formats: list[OutputFormat] = Field(default_factory=list)
    path_policy: PathPolicy = "optional"
    forbidden_artifact_types: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)


class InputContract(PlanningModel):
    artifact_type: str
    source_step_id: str | None = None
    required: bool = True
    entities: dict[str, Any] = Field(default_factory=dict)
    validation_checks: list[str] = Field(default_factory=list)


class OutputContract(PlanningModel):
    contract_id: str
    required: bool = True
    artifact_contracts: list[ArtifactContract] = Field(default_factory=list)
    data_requirements: list[DataRequirement] = Field(default_factory=list)
    text_requirements: list[TextRequirement] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_contract(self) -> "OutputContract":
        if not (
            self.artifact_contracts
            or self.data_requirements
            or self.text_requirements
            or self.validation_checks
        ):
            raise ValueError("OutputContract must define at least one requirement or check")
        return self


class RetryPolicy(PlanningModel):
    max_attempts: int = 2
    retry_on: list[str] = Field(
        default_factory=lambda: ["contract_failed", "tool_arg_invalid", "missing_artifact"]
    )
    allow_recompile: bool = False

    @field_validator("max_attempts")
    @classmethod
    def _validate_max_attempts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_attempts must be non-negative")
        return value


class StepBudget(PlanningModel):
    max_tool_calls: int | None = None
    max_seconds: float | None = None
    max_tokens: int | None = None

    @model_validator(mode="after")
    def _validate_budget(self) -> "StepBudget":
        for name in ("max_tool_calls", "max_seconds", "max_tokens"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        return self


class PlanStep(PlanningModel):
    step_id: str
    label: str
    role: PlanRole
    execution_kind: ExecutionKind
    allowed_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    input_contracts: list[InputContract] = Field(default_factory=list)
    output_contracts: list[OutputContract] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    tool_args_template: dict[str, Any] = Field(default_factory=dict)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    budget: StepBudget = Field(default_factory=StepBudget)
    allow_tool_choice: bool = False

    @field_validator("step_id", "label")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _validate_step(self) -> "PlanStep":
        if self.execution_kind == "tool" and not self.allow_tool_choice and len(self.allowed_tools) != 1:
            raise ValueError("tool steps must declare exactly one allowed tool")
        if self.execution_kind == "handoff_adapter" and len(self.allowed_tools) != 1:
            raise ValueError("handoff_adapter steps must declare exactly one adapter/tool")
        if self.execution_kind != "synthesis":
            if not self.output_contracts:
                raise ValueError("non-synthesis steps must declare at least one OutputContract")
            for contract in self.output_contracts:
                if not (contract.artifact_contracts or contract.data_requirements):
                    raise ValueError(
                        "non-synthesis OutputContract must include artifact_contracts or data_requirements"
                    )
        return self


class FinalResponseContract(PlanningModel):
    required_sections: list[str] = Field(default_factory=list)
    must_cite_artifacts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    require_paths: bool = False
    require_plan_status: bool = True


class RequestPlan(PlanningModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: str
    created_at: str
    compiler_version: str
    capability_registry_version: str
    planner_model_id: str | None = None
    user_query: str
    mode: PlanMode
    intent_family: IntentFamily
    complexity: Literal["simple", "moderate", "complex"]
    assumptions: list[PlanAssumption] = Field(default_factory=list)
    missing_inputs: list[MissingInput] = Field(default_factory=list)
    global_constraints: dict[str, Any] = Field(default_factory=dict)
    steps: list[PlanStep] = Field(default_factory=list)
    final_response_contract: FinalResponseContract = Field(default_factory=FinalResponseContract)
    fallback_policy: FallbackPolicy = "typed_failure"
    workflow_id: str | None = None
    unsupported_reason: str | None = None

    @field_validator(
        "plan_id",
        "created_at",
        "compiler_version",
        "capability_registry_version",
        "user_query",
    )
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _validate_plan(self) -> "RequestPlan":
        if self.mode == "direct_tool" and len(self.steps) > 1:
            raise ValueError("direct_tool plans may have at most one step")
        if self.mode in {"single_agent", "single_tool_or_specialist"} and not self.steps:
            raise ValueError(f"{self.mode} plans must contain at least one step")
        if self.mode in {"single_agent", "single_tool_or_specialist"}:
            has_enforceable_step = any(
                step.execution_kind != "synthesis" or step.output_contracts
                for step in self.steps
            )
            if not has_enforceable_step:
                raise ValueError(f"{self.mode} plans must contain at least one enforceable step")
        if self.mode == "planned_workflow" and len(self.steps) < 2:
            raise ValueError("planned_workflow plans must contain at least two steps")
        if self.mode == "clarification_required" and not self.missing_inputs:
            raise ValueError("clarification_required plans must include missing_inputs")
        if self.mode == "unsupported" and not self.unsupported_reason:
            raise ValueError("unsupported plans must include unsupported_reason")

        seen: set[str] = set()
        for index, step in enumerate(self.steps):
            if step.step_id in seen:
                raise ValueError(f"duplicate step_id: {step.step_id}")
            seen.add(step.step_id)
            prior_ids = {item.step_id for item in self.steps[:index]}
            for dep in step.depends_on:
                if dep not in seen and dep not in prior_ids:
                    raise ValueError(f"step {step.step_id} depends on unknown or future step {dep}")
                if dep == step.step_id:
                    raise ValueError(f"step {step.step_id} cannot depend on itself")
        return self


class CapabilitySpec(PlanningModel):
    capability_id: str
    owner: PlanRole
    callable_name: str
    callable_kind: CallableKind
    produces: list[str]
    consumes: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    rejects: list[str] = Field(default_factory=list)
    artifact_schema_versions: dict[str, str] = Field(default_factory=dict)
    supports_batch: bool = False
    supports_multislice: bool = False
    deterministic: bool = False
    legacy_unplanned: bool = False

    @model_validator(mode="after")
    def _validate_capability(self) -> "CapabilitySpec":
        if not self.produces:
            raise ValueError("CapabilitySpec must produce at least one artifact type")
        return self


class ArtifactFrame(PlanningModel):
    artifact_id: str
    artifact_type: str
    schema_version: str = "1.0"
    source_step_id: str | None = None
    source_handoff_ids: list[str] = Field(default_factory=list)
    output_paths: list[str] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    inputs_used: dict[str, Any] = Field(default_factory=dict)
    validation_summary: dict[str, Any] = Field(default_factory=dict)


class StepExecutionRecord(PlanningModel):
    step_id: str
    status: Literal["pending", "running", "succeeded", "failed", "skipped"]
    attempt: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    callable_name: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    verification_status: Literal["not_checked", "passed", "failed"] = "not_checked"
    failed_checks: list[str] = Field(default_factory=list)
    error: str | None = None


class ExecutionLedger(PlanningModel):
    plan_id: str
    run_id: str
    status: Literal["running", "succeeded", "failed", "partial"]
    started_at: str
    completed_at: str | None = None
    step_records: list[StepExecutionRecord] = Field(default_factory=list)
    artifacts: list[ArtifactFrame] = Field(default_factory=list)
    repairs: list[dict[str, Any]] = Field(default_factory=list)
    final_contract_status: dict[str, Any] = Field(default_factory=dict)


_NON_TRIVIAL_MARKERS = (
    "plot",
    "figure",
    "visual",
    "png",
    "xlsx",
    "workbook",
    "case study",
    "save",
    "handoff",
    "pass to",
    "exactly those",
    "provenance",
    "use only",
    "optimization",
    "optimize",
    "pareto",
    "frontier",
    "landscape",
    "tea",
    "lca",
    "biosteam",
    "safety",
    "citation",
    "contaminant",
    "separation",
    "dynamic-programming",
    "dynamic programming",
    "sweep",
    "batch",
    "compare",
    "multi-slice",
    "composition",
    "finally",
)


def is_non_trivial_request(query: str) -> bool:
    """Return whether a request requires typed planning by deterministic rules."""
    text = (query or "").lower()
    return any(marker in text for marker in _NON_TRIVIAL_MARKERS)
