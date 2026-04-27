# DISSOLVE Backend Codebase Review Spec

## Purpose
This document is the benchmark-facing backend walkthrough for DISSOLVE `v10-core`. It is migrated from the historical `langchain-STRAP-v9-contaminants` spec and updated for the typed planning/runtime work now present in this branch. It is intentionally scoped to code and runtime metadata that the packaged `dissolve` CLI and shared backend surfaces depend on.

The working assumption is strict: if a backend file is not named here, it should be treated as out of scope for benchmark package review. The inventory is explicit so that reviewers can compare the package boundary against a concrete checklist.

## Scope And Exclusions
Included scope is the packaged backend in `src/strap/`, the CLI/package manifest in `pyproject.toml`, the shared FastAPI surface in `app_server.py`, export helpers in `export_manager.py` and `output_models.py`, and top-level reporting services in `services/`. The v10 typed-planning architecture spec in `docs/v10_typed_planning_execution_harness_spec.md` and the query-bank workbook in `docs/subagent_query_bank-v1.xlsx` are included as review context.

Excluded scope is the frontend in `frontend/`, evaluation and architecture outputs in `architecture/`, experiments in `experiments/`, tests in `tests/`, helper scripts in `scripts/`, and generated case-study artifacts under `docs/case_studies/`. Raw datasets, model binaries, and workbooks are not reviewed file-by-file here, but they are called out where backend code depends on them.

## Reading Order
Read this backend from orchestration to scientific kernels. `pyproject.toml` and `src/strap/agent.py` define how the CLI boots. The direct fast path and typed runtime run before advisory legacy routing for selected workflows. The planning package compiles, validates, executes, verifies, and persists typed plans. Legacy routing, handoff, guardrail, and verifier files remain important fallback and specialist-control layers. The substrate modules resolve polymer, solvent, HSP, and asset data. Services and engines hold reusable computation. Tool modules expose those capabilities to the orchestrator and specialists. Vendor, optimization, and server/reporting files sit outside the core CLI loop but still matter for the shipped backend package.

## Tier 0: CLI, Typed Runtime, And Orchestration Spine
The CLI entrypoint begins in `pyproject.toml`, which declares the `strap-agent` package and binds the `dissolve` console script to `strap.agent:main`. `src/strap/__init__.py` marks the package boundary. `src/strap/agent.py` is the composition root: it initializes the chat model, builds the DeepAgents graph, wires middleware, loads subagent manifests, attaches `src/strap/AGENTS.md` as backend memory, exposes `src/strap/skills/`, and implements the REPL-style CLI entrypoint.

The current middleware order is the most important v10 change: `DirectToolFastPathMiddleware` runs first for deterministic simple queries, `TypedRuntimeMiddleware` then gets a chance to execute selected typed plans, legacy `RoutingMiddleware` remains the fallback advisory router, `OutputVerifierMiddleware` checks final synthesis, `StructuredResultExtractorMiddleware` captures subagent structured outputs, and `SubagentGuardMiddleware` caps budgets and tool behavior.

The typed runtime is owned by `src/strap/planning/`. `models.py` defines `RequestPlan`, `PlanStep`, contracts, artifact frames, execution records, and ledgers. `capability_registry.py` maps roles, tools, workflows, and artifact types. `extractors.py` deterministically pulls polymers, solvents, temperatures, compositions, requested artifacts, forbidden artifacts, save paths, and workflow markers from text. `compiler.py` produces compile-only `RequestPlan` objects. `validators.py` rejects malformed or unsupported plans. `guard.py` evaluates selected tool-call/final-synthesis enforcement. `executor.py` is the deterministic state machine. `runtime.py` bridges compile plus executor. `runtime_wrappers.py` defines normalized wrapper envelopes for tests and dry harnesses. `runtime_production_wrappers.py` contains evidence-based wrappers for selected real tools. `runtime_persistence.py` writes auditable runtime bundles. `runtime_paths.py` normalizes local, WSL, and UNC paths. `typed_runtime_integration.py` provides the middleware, result formatting, progress summaries, and selected-runtime entrypoint. `typed_runtime_context.py` and `typed_runtime_followups.py` reuse verified typed artifacts across turns without handing control back to prose memory. `frontier_formatting.py` renders compact Pareto/frontier tables, and `query_bank.py` turns the query-bank workbook into compile/evaluation expectations.

The legacy orchestration-control plane remains active for unselected, off, shadow, simple, or unsupported typed-runtime cases. `src/strap/direct_fast_path.py` handles deterministic one-tool requests. `src/strap/orchestrator_runtime.py` defines lightweight route decisions, artifact frames, and run ledgers used by fast paths and session metadata. `src/strap/session_state.py` persists compact CLI state and prior artifacts across turns. `src/strap/routing.py` injects advisory hints, detects partial workflows, and auto-builds downstream handoffs. `src/strap/routing_classifier.py` combines regex heuristics, query-context extraction, planning-graph metadata, and an LLM classifier to choose specialist sets. `src/strap/routing_progress.py`, `src/strap/routing_message_state.py`, and `src/strap/routing_handoff_state.py` infer what has already happened in routed workflows. `src/strap/routing_guards.py` provides fallback payloads, filesystem/tool restrictions, and route-safe response helpers. `src/strap/verifier.py` performs final synthesis checks. `src/strap/prompts.py` holds orchestrator-level delegation, file I/O, and reasoning directives.

The multi-agent glue lives in the handoff stack. `src/strap/result_extractor.py` intercepts `task()` results and extracts `<STRUCTURED_RESULT>` blocks. `src/strap/handoffs.py`, `src/strap/handoff_store.py`, `src/strap/handoff_models.py`, and `src/strap/handoff_adapters.py` define append-only storage, validation rules, JSON envelopes, and typed downstream adapters. `src/strap/traced_subagent_middleware.py` and `src/strap/langsmith_tracing.py` add observability and trace capture.

Configuration and planner metadata are shared by typed and legacy paths. `src/strap/query_context.py` normalizes polymers, solvents, contaminants, request labels, and feed metadata. `src/strap/planning_graph.py` converts subagent planning metadata into a capability graph. `src/strap/subagent_config.py` loads YAML manifests. `src/strap/subagents.yaml`, `src/strap/config/execution_pairs.yaml`, and the ten `src/strap/config/subagents/*.yaml` files define specialists, routing phrases, planning contracts, tool groups, prompts, and guardrail overrides.

## Tier 1: Scientific Runtime Substrate
Below orchestration is the shared data and model substrate. `src/strap/paths.py` resolves data and model directories. `src/strap/database.py` loads CSV-backed tables into DuckDB. `src/strap/solvent_registry.py` centralizes solvent aliases across interpolation, properties, safety, BioSTEAM, and plotting. `src/strap/hsp_registry.py` centralizes HSP category and solvent metadata. `src/strap/solubility.py` is the source of truth for temperature-dependent solubility, combining fitted coefficients, generated entries, fuzzy polymer/solvent resolution, and SQL fallback. `src/strap/models.py` and `src/strap/analysis.py` hold dataclasses, selectivity, ranking, and compatibility logic.

Supporting scientific modules include `src/strap/ml_assets.py` for ML polymer catalogs and HSP lookup assets, `src/strap/cosmo_interface.py` for COSMO-RS style calculations, and `src/strap/testing_utils.py` for deterministic package-local harness helpers.

The thermal-property branch lives in `src/strap/thermal_ml/`. It includes dataset construction, the polyBERT-based model, training, uncertainty estimation, Van Krevelen group-contribution estimates, and Tg lookup utilities.

## Tier 2: Services, Engines, And Tool Adapters
The service layer in `src/strap/services/` is the package reuse boundary. Separation services split sequence runtime search, analysis, plotting, payload construction, and display formatting. `visualization_service.py` centralizes plotting style and plot-path helpers. `precipitation_service.py` formats precipitation workflows. `tool_response_service.py` standardizes JSON envelopes. `solvent_safety_service.py` supports the safety-card workflow. `biosteam_service.py` holds BioSTEAM catalogs, request builders, and result extraction. `rag_service.py` wraps the vendored RAG system. The contaminant services normalize Zhou screening data and implement deterministic leaching and STRAP-style contaminant-removal decisions.

The lower-level engines in `src/strap/engines/` expose reusable algorithms outside the agent-facing tool surface: greedy and dynamic-programming sequence search, temperature optimization windows, precipitation analysis, and separation-specific visualization builders. The PIW optimization branch lives in `src/strap/waste_management/`, where workbook data is loaded, a Pyomo superstructure model is built, and single-objective/Pareto solves are run.

The agent-facing tool layer in `src/strap/tools/` is what specialists and typed production wrappers call. `__init__.py` is the lazy-loading registry. `_helpers.py` provides safe tool wrappers, output normalization, plot persistence, and common validation helpers. Data-access tools cover database query, listing, solvent properties, solvent lookup, and interpolation. Separation tools cover adaptive separation, advanced separation, sequence planning, sequence analysis, separation planning, separation visualization, precipitation, and general visualization. Safety and contaminant tools cover safety cards, GSK, PubChem, and contaminant removal. Process and optimization tools expose BioSTEAM TEA/LCA and waste-management optimization. Modeling tools cover ML prediction, thermal prediction, and statistics. Research and coordination tools cover literature, RAG, diagnostics, relevancy, sidecar artifacts, and reflection.

## Tier 3: Vendor Integrations And Auxiliary Backend Surfaces
The vendor directory in `src/strap/vendor/` contains active integrations and bundled provenance code. `rag.py` powers indexed retrieval. `biosteam_runner.py` and `biosteam_worker.py` isolate BioSTEAM simulations in subprocesses. `solubility_predictor.py` supplies the Hansen-parameter predictor. Async wrappers and external research clients support blocking database work, arXiv, SerpAPI Scholar, SerpAPI Patents, PatentsView, and Web of Science. The large provenance files `_agent_sql_source.py` and `_langchain_tools_source.py` document earlier monolithic implementations and migration ancestry.

Outside the packaged CLI hot path, `app_server.py` wraps the shared runtime in FastAPI, manages sessions, serves plots, exposes workflow previews, loads the agent, and integrates issue-reporting behavior. `export_manager.py` and `output_models.py` handle CSV export metadata. The top-level `services/` package powers issue reports, structured LLM diagnoses, codebase context loading, and GitHub automation.

## Data And Asset Dependencies
Several non-code assets are part of the backend runtime contract. DuckDB loading expects CSVs in `data/`. `src/strap/solubility.py` depends on coefficient JSON in the data directory. `src/strap/ml_assets.py` and `src/strap/hsp_registry.py` expect catalog and lookup JSON assets. The thermal stack depends on generated lookup files under `data/thermal_properties/`. BioSTEAM helpers depend on TEA/LCA solvent data. Waste-management optimization depends on `src/strap/waste_management/Data for model_Scenarios.xlsx`. The typed planning acceptance path depends on `docs/subagent_query_bank-v1.xlsx` for offline and CI-style checks.

## Complete Backend Inventory
The inventory below is intentionally explicit so that packaging or benchmark scoping can be audited against a concrete checklist.

### Entrypoints, Runtime Metadata, And Planner Configuration
- `pyproject.toml`: package manifest, dependencies, and `dissolve = "strap.agent:main"` console script.
- `src/strap/__init__.py`: shipped backend namespace marker.
- `src/strap/agent.py`: CLI bootstrap, DeepAgents graph assembly, middleware order, subagent loading, scratch-directory management, startup UX, and interactive REPL entrypoint.
- `src/strap/prompts.py`: orchestrator-level delegation, file I/O, reflection, and handoff instructions.
- `src/strap/AGENTS.md`: backend memory loaded into the DeepAgents runtime.
- `src/strap/subagents.yaml`: manifest pointing the runtime at subagent prompt/config files.
- `src/strap/config/execution_pairs.yaml`: declarative map of allowed parallel and sequential specialist combinations.
- `src/strap/config/subagents/01_separation-engineer.yaml`, `src/strap/config/subagents/02_safety-analyst.yaml`, `src/strap/config/subagents/03_biosteam-analyst.yaml`, `src/strap/config/subagents/04_scholar-researcher.yaml`, `src/strap/config/subagents/05_patent-researcher.yaml`, `src/strap/config/subagents/06_rag-analyst.yaml`, `src/strap/config/subagents/07_visualization-specialist.yaml`, `src/strap/config/subagents/08_statistics-ml.yaml`, `src/strap/config/subagents/09_contaminant-removal-analyst.yaml`, `src/strap/config/subagents/10_optimization-engineer.yaml`: per-specialist routing phrases, planning contracts, tool groups, guardrails, and prompts.
- `src/strap/skills/data-lookup/SKILL.md`, `src/strap/skills/multi-agent-workflow/SKILL.md`, `src/strap/skills/separation-design/SKILL.md`: runtime skill files available to DeepAgents.
- `docs/v10_typed_planning_execution_harness_spec.md`: architectural roadmap for typed planning and execution.
- `docs/subagent_query_bank-v1.xlsx`: query-bank acceptance and smoke-test seed workbook.
- `docs/phase0_findings.md`: early architecture findings retained as supporting context.

### Typed Planning And Selected Runtime
- `src/strap/planning/__init__.py`: planning package exports.
- `src/strap/planning/models.py`: Pydantic plan, step, contract, artifact, execution record, and ledger models.
- `src/strap/planning/capability_registry.py`: authoritative capability registry for roles, tools, workflows, and artifact types.
- `src/strap/planning/config.py`: `DISSOLVE_TYPED_PLANNER` mode parsing and selected enforcement configuration.
- `src/strap/planning/extractors.py`: deterministic fact, entity, path, composition, artifact, and workflow-marker extraction.
- `src/strap/planning/compiler.py`: compile-only request planner and provider-agnostic backend interface.
- `src/strap/planning/validators.py`: deterministic validation for compiled plans and malformed model/compiler outputs.
- `src/strap/planning/guard.py`: selected enforcement decisions for tool calls, provenance, and final synthesis sources.
- `src/strap/planning/executor.py`: deterministic step executor, dependency handling, contract verification, retry policy, and ledger construction.
- `src/strap/planning/runtime.py`: opt-in compile/execute bridge used by tests and production integration.
- `src/strap/planning/runtime_wrappers.py`: canonical callable-result envelope, normalization, and dry/test wrapper helpers.
- `src/strap/planning/runtime_production_wrappers.py`: evidence-based wrappers for selected production workflows.
- `src/strap/planning/runtime_persistence.py`: persistent runtime bundles for requests, compile results, plans, ledgers, artifacts, and manifests.
- `src/strap/planning/runtime_paths.py`: local, WSL, and UNC-safe output path normalization.
- `src/strap/planning/typed_runtime_integration.py`: DeepAgents middleware, selected-runtime entrypoint, typed success/failure formatting, progress summaries, and final-answer anchoring.
- `src/strap/planning/typed_runtime_context.py`: cross-turn typed artifact lookup and context hydration for selected follow-ups.
- `src/strap/planning/typed_runtime_followups.py`: deterministic answers to prior-artifact follow-up questions.
- `src/strap/planning/frontier_formatting.py`: compact formatting for Pareto frontier and landscape payloads.
- `src/strap/planning/query_bank.py`: query-bank loader and expected tool/artifact extraction.

### Legacy Routing, Guardrails, Handoffs, And Verification
- `src/strap/direct_fast_path.py`: deterministic simple-query bypass for known one-tool requests.
- `src/strap/orchestrator_runtime.py`: lightweight route decisions, artifact frames, and run ledgers for direct/runtime metadata.
- `src/strap/session_state.py`: compact persisted CLI session state and artifact metadata.
- `src/strap/routing.py`: top-level legacy routing middleware for hints, workflow resumption, and downstream auto-dispatch.
- `src/strap/routing_classifier.py`: keyword and LLM-assisted classification, goal inference, and workflow-rule planning.
- `src/strap/routing_progress.py`: progress-directive construction and routed-workflow guards.
- `src/strap/routing_message_state.py`: message-history extraction utilities for completed, failed, and active routed tasks.
- `src/strap/routing_handoff_state.py`: readiness checks for sequential handoff chains.
- `src/strap/routing_guards.py`: fallback response builders, route-safe tool filters, and workflow guard logic.
- `src/strap/verifier.py`: final-response verification for unsupported claims, missing caveats, and route-quality issues.
- `src/strap/guardrails.py`: subagent guardrail middleware for iteration, token, tool-budget, and synthesis control.
- `src/strap/guardrail_messages.py`: repair and directive message builders.
- `src/strap/guardrail_policy.py`: mutation and blocking policies applied by guardrails.
- `src/strap/guardrail_utils.py`: parsing, normalization, and extraction helpers shared across checks.
- `src/strap/guardrail_checks.py`: pure checks for structured results, temperature bounds, support scope, and selectivity overclaiming.
- `src/strap/result_extractor.py`: `<STRUCTURED_RESULT>` extraction, handoff browsing tools, and middleware binding results to execution scope.
- `src/strap/handoffs.py`: handoff construction, generic context bundling, and JSON-envelope utilities.
- `src/strap/handoff_store.py`: append-only scope storage, validation, and retrieval for handoff records.
- `src/strap/handoff_models.py`: dataclasses and artifact helpers for handoff records and scopes.
- `src/strap/handoff_adapters.py`: typed downstream payload builders.
- `src/strap/traced_subagent_middleware.py`: LangSmith-aware replacement for default DeepAgents subagent middleware.
- `src/strap/langsmith_tracing.py`: tracing enablement, run links, and captured subagent trace summaries.
- `src/strap/query_context.py`: extraction of polymers, solvents, contaminants, request labels, and feed metadata.
- `src/strap/planning_graph.py`: planning-node and edge construction from subagent metadata.
- `src/strap/subagent_config.py`: YAML loader and normalizer for runtime subagent/planning manifests.

### Shared Scientific And Asset Substrate
- `src/strap/paths.py`: centralized path resolution for data and model assets.
- `src/strap/database.py`: in-memory DuckDB bootstrap over CSV-backed data.
- `src/strap/solvent_registry.py`: canonical solvent alias map.
- `src/strap/hsp_registry.py`: HSP category, polymer, and solvent metadata registry.
- `src/strap/solubility.py`: temperature-dependent solubility source of truth with interpolation and SQL fallback.
- `src/strap/models.py`: lightweight dataclasses and enums used by analysis code.
- `src/strap/analysis.py`: selectivity, solvent ranking, and compatibility-matrix logic.
- `src/strap/ml_assets.py`: ML polymer catalog and HSP lookup utilities.
- `src/strap/cosmo_interface.py`: COSMO-RS style SLE/LLE calculations and uncertainty-aware runs.
- `src/strap/testing_utils.py`: deterministic package-local helpers for tests and harnesses.
- `src/strap/thermal_ml/__init__.py`, `src/strap/thermal_ml/dataset.py`, `src/strap/thermal_ml/model.py`, `src/strap/thermal_ml/train.py`, `src/strap/thermal_ml/uncertainty.py`, `src/strap/thermal_ml/group_contribution.py`, `src/strap/thermal_ml/tg_lookup.py`: thermal-property ML subsystem.

### Service Layer
- `src/strap/services/__init__.py`: service package marker.
- `src/strap/services/tool_response_service.py`: uniform JSON response helpers.
- `src/strap/services/advanced_separation_service.py`: shared reports, scoring, plotting, and formatting for advanced separation.
- `src/strap/services/sequence_runtime_service.py`: runtime search helpers for greedy and ranked sequence planning.
- `src/strap/services/sequence_analysis_runtime_service.py`: runtime helpers for integrated multi-step sequence analysis.
- `src/strap/services/sequence_analysis_service.py`: display builders and alternative-sequence selection.
- `src/strap/services/sequence_analysis_plot_service.py`: integrated sequence-analysis figure builders.
- `src/strap/services/sequence_planning_payload_service.py`: structured payload builders and serializers.
- `src/strap/services/sequence_planning_exhaustive_display_service.py`: exhaustive-planning displays.
- `src/strap/services/sequence_planning_greedy_display_service.py`: greedy and multi-scheme displays.
- `src/strap/services/sequence_planning_display_service.py`, `src/strap/services/sequence_planning_service.py`: compatibility re-export layers.
- `src/strap/services/visualization_service.py`: plotting style, validation, property lookup, and plot URL helpers.
- `src/strap/services/precipitation_service.py`: precipitation report and plot builders.
- `src/strap/services/solvent_safety_service.py`: safety-card/comparison data normalization and display support.
- `src/strap/services/biosteam_service.py`: BioSTEAM catalogs, request builders, result extraction, and utility logic.
- `src/strap/services/contaminant_data_service.py`: normalized access to Zhou contaminant-removal data.
- `src/strap/services/contaminant_screening_service.py`: deterministic leaching-mode and STRAP-style contaminant-removal screening.
- `src/strap/services/rag_service.py`: service shim over the vendored RAG runtime.

### Engines And Optimization Core
- `src/strap/engines/__init__.py`: compute-engine package marker.
- `src/strap/engines/separation.py`: greedy and dynamic-programming sequence search.
- `src/strap/engines/optimization.py`: temperature-window optimization utilities.
- `src/strap/engines/precipitation.py`: precipitation-point and atmospheric-feasibility analysis logic.
- `src/strap/engines/visualization.py`: reusable heatmap and process-diagram builders.
- `src/strap/waste_management/__init__.py`: PIW optimization package marker.
- `src/strap/waste_management/data_loader.py`: workbook-driven superstructure data ingestion.
- `src/strap/waste_management/model.py`: Pyomo model construction for waste-management optimization.
- `src/strap/waste_management/solver.py`: single-objective and Pareto solver routines plus result extraction.
- `src/strap/waste_management/Data for model_Scenarios.xlsx`: workbook dependency for the optimization stack.

### Agent-Facing Tool Surface
- `src/strap/tools/__init__.py`: lazy tool-group registry.
- `src/strap/tools/_helpers.py`: safe tool wrapper, plot persistence, result formatting, and shared validation.
- `src/strap/tools/reflection.py`, `src/strap/tools/sidecar.py`: reflection and sidecar-artifact utilities for multi-agent coordination.
- `src/strap/tools/database_query.py`, `src/strap/tools/listing.py`, `src/strap/tools/solvent_properties.py`, `src/strap/tools/solvent_lookup.py`, `src/strap/tools/interpolation.py`: direct data-access and lookup tools.
- `src/strap/tools/adaptive_separation.py`, `src/strap/tools/advanced_separation.py`, `src/strap/tools/sequence_planning_tools.py`, `src/strap/tools/sequence_analysis_tools.py`, `src/strap/tools/separation_planning_tools.py`, `src/strap/tools/separation_visualization_tools.py`, `src/strap/tools/precipitation_analysis.py`, `src/strap/tools/visualization.py`: separation-design, sequence, precipitation, and plotting tools.
- `src/strap/tools/safety_card.py`, `src/strap/tools/safety_gsk.py`, `src/strap/tools/safety_pubchem.py`, `src/strap/tools/contaminant_removal.py`: safety and contaminant tool surfaces.
- `src/strap/tools/biosteam_tea_lca.py`, `src/strap/tools/waste_optimization.py`: process-simulation and superstructure-optimization entrypoints.
- `src/strap/tools/ml_prediction.py`, `src/strap/tools/thermal_prediction.py`, `src/strap/tools/statistical.py`: HSP/ML, thermal-property, and statistics tools.
- `src/strap/tools/literature.py`, `src/strap/tools/rag_core.py`, `src/strap/tools/rag_diagnostics.py`, `src/strap/tools/relevancy.py`: research search, RAG retrieval, diagnostics, and relevancy scoring tools.

### Vendored Integrations And Provenance Sources
- `src/strap/vendor/__init__.py`: vendor package marker.
- `src/strap/vendor/rag.py`: vendored RAG runtime for chunking, indexing, search, and retrieval analysis.
- `src/strap/vendor/biosteam_runner.py`, `src/strap/vendor/biosteam_worker.py`: subprocess-isolated BioSTEAM execution pair.
- `src/strap/vendor/solubility_predictor.py`: Hansen-parameter ML predictor.
- `src/strap/vendor/async_db.py`, `src/strap/vendor/async_utils.py`: asynchronous wrappers for blocking work.
- `src/strap/vendor/arxiv_client.py`, `src/strap/vendor/serpapi_scholar.py`, `src/strap/vendor/serpapi_patents.py`, `src/strap/vendor/patentsview_client.py`, `src/strap/vendor/wos_client.py`: external literature and patent search clients.
- `src/strap/vendor/_agent_sql_source.py`, `src/strap/vendor/_langchain_tools_source.py`: historical source snapshots and migration ancestry.

### Top-Level Backend Surfaces Outside The CLI Hot Path
- `app_server.py`: FastAPI server wrapping the shared runtime, plots, workflow previews, sessions, and issue reports.
- `export_manager.py`, `output_models.py`: CSV export management and export metadata datamodels.
- `services/__init__.py`: issue-report subsystem package marker.
- `services/ai_diagnosis.py`: structured LLM diagnosis for submitted issue reports.
- `services/codebase_context.py`: file, tool, and endpoint context loader for issue reports.
- `services/github_pr.py`: GitHub issue and PR automation.
- `services/issue_reporter.py`: orchestration layer tying diagnosis, codebase context, and GitHub automation together.
