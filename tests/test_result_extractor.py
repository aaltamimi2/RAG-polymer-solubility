"""Tests for structured result extraction and handoff orchestration."""

from __future__ import annotations

import contextvars
import json
import threading
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage
from langgraph.types import Command


def _minimal_payloads() -> dict[str, dict]:
    return {
        "separation-engineer": {
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "PET"],
            "best_sequence": ["LDPE", "PET"],
            "steps": [{"step": 1, "polymer": "LDPE", "solvent": "Xylene", "temperature_c": 120.0}],
            "solvent_mapping": {"LDPE": "Xylene", "PET": "Toluene"},
            "top_k_sequences": [
                {
                    "rank": 1,
                    "sequence": ["LDPE", "PET"],
                    "min_selectivity": 72.1,
                    "solvent_mapping": {"LDPE": "Xylene", "PET": "Toluene"},
                }
            ],
        },
        "safety-analyst": {
            "agent": "safety-analyst",
            "schema_version": "1.0",
            "solvents_assessed": ["Xylene", "Toluene"],
            "gscore_results": [{"solvent": "Xylene", "g_score": 4.2}],
            "ghs_results": [{"solvent": "Xylene", "hazard_codes": ["H226"]}],
            "safest_solvent": "Toluene",
        },
        "biosteam-analyst": {
            "agent": "biosteam-analyst",
            "schema_version": "1.0",
            "target_plastic": "LDPE",
            "energy_case": "C1",
            "results": [{"scenario_label": "Xylene", "success": True, "tea": {"msp_usd_per_kg": 1.2}, "lca": {"gwp_kg_co2e_per_kg": 2.3}}],
            "n_simulations": 1,
            "n_failed": 0,
        },
        "scholar-researcher": {
            "agent": "scholar-researcher",
            "schema_version": "1.0",
            "query": "polymer dissolution",
            "n_results": 1,
            "papers": [{"title": "Paper A", "year": 2024, "url": "https://example.com/paper"}],
            "saved_to_rag": False,
        },
        "patent-researcher": {
            "agent": "patent-researcher",
            "schema_version": "1.0",
            "query": "polymer solvent patent",
            "n_results": 1,
            "patents": [{"patent_id": "US123", "title": "Patent A", "filing_date": "2024-01-01"}],
            "saved_to_rag": False,
        },
        "rag-analyst": {
            "agent": "rag-analyst",
            "schema_version": "1.0",
            "operation": "search",
            "query": "LDPE dissolution",
            "n_passages": 3,
            "top_sources": ["paper_a.pdf"],
            "top_score": 0.87,
            "kb_name": "user-library",
        },
        "visualization-specialist": {
            "agent": "visualization-specialist",
            "schema_version": "1.0",
            "plot_type": "comparison_dashboard",
            "plot_paths": ["/plots/example.png"],
            "format": "png",
        },
        "statistics-ml": {
            "agent": "statistics-ml",
            "schema_version": "1.0",
            "analysis_type": "summary",
            "table": "common_solvents_database",
            "summary": {"n": 2, "mean": 1.5},
            "plot_paths": ["/plots/stats.png"],
        },
        "contaminant-removal-analyst": {
            "agent": "contaminant-removal-analyst",
            "schema_version": "1.0",
            "mode": "leaching",
            "target_polymer": "PET",
            "contaminants": ["di-n-butyl phthalate (DBP)"],
            "supported_contaminants": ["di-n-butyl phthalate (DBP)"],
            "unsupported_contaminants": [],
            "candidate_solvents": [
                {
                    "solvent": "acetone",
                    "passes": True,
                    "operating_temperature_c": 55.0,
                    "contaminant_logd_min": 0.69,
                    "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                }
            ],
            "recommended_solvents": ["acetone"],
            "caveats": ["polymer swelling is proxy-inferred"],
        },
    }


def test_validate_contract_payload_accepts_optimization_stage_candidates():
    from strap.handoff_store import validate_contract_payload

    errors = validate_contract_payload(
        "optimization.stage_candidates.v1",
        {
            "schema_version": "1.0",
            "workflow_scope": "multi_stage",
            "route_id": "route-1",
            "constraint_mode": "ranked_soft",
            "fallback_policy": "broaden_disclosed",
            "operating_constraints": {"temperature_max_c": 120.0, "pressure": "atmospheric"},
            "stages": [
                {
                    "stage_id": "wash_1",
                    "stage_kind": "selective_dissolution",
                    "target_polymer": "PE",
                    "candidate_pairs": [{"polymer": "PE", "solvent": "Cyclohexane"}],
                }
            ],
            "candidate_pairs": [{"stage_id": "wash_1", "polymer": "PE", "solvent": "Cyclohexane"}],
            "polymer_solvent_filters": {"PE": ["Cyclohexane"]},
            "candidate_solvents": ["Cyclohexane"],
        },
    )

    assert errors == []


def test_validate_contract_payload_rejects_invalid_optimization_stage_candidates():
    from strap.handoff_store import validate_contract_payload

    errors = validate_contract_payload(
        "optimization.stage_candidates.v1",
        {
            "schema_version": "1.0",
            "workflow_scope": "unknown",
            "constraint_mode": "loose",
            "fallback_policy": "warn",
            "operating_constraints": [],
            "stages": [],
            "candidate_pairs": {},
            "polymer_solvent_filters": [],
            "candidate_solvents": {},
        },
    )

    assert any("workflow_scope must be one of" in error for error in errors)
    assert any("constraint_mode must be one of" in error for error in errors)
    assert any("fallback_policy must be one of" in error for error in errors)


def test_normalize_agent_payload_flattens_contaminant_comparison_modes():
    from strap.handoff_store import normalize_agent_payload

    payload = {
        "agent": "contaminant-removal-analyst",
        "schema_version": "1.0",
        "mode": "comparison",
        "target_polymer": "EVOH",
        "contaminants": ["di-n-butyl phthalate (DBP)"],
        "supported_contaminants": ["di-n-butyl phthalate (DBP)"],
        "unsupported_contaminants": [],
        "recommended_mode": "leaching",
        "recommended_solvents": {
            "leaching": ["toluene"],
            "strap_contaminant_removal": ["dimethyl sulfoxide"],
        },
        "leaching": {
            "recommended_solvents": ["toluene"],
            "candidate_solvents": [
                {
                    "solvent": "toluene",
                    "passes": True,
                    "contaminant_logd_min": 0.61,
                }
            ],
        },
        "strap_contaminant_removal": {
            "recommended_solvents": ["dimethyl sulfoxide"],
            "candidate_solvents": [
                {
                    "solvent": "dimethyl sulfoxide",
                    "passes": True,
                    "contaminant_logd_min": 0.31,
                }
            ],
        },
    }

    normalized = normalize_agent_payload("contaminant-removal-analyst", payload)

    assert normalized["recommended_solvents"] == ["toluene"]
    assert isinstance(normalized["modes"], dict)
    assert normalized["modes"]["leaching"]["recommended_solvents"] == ["toluene"]
    flattened = normalized["candidate_solvents"]
    assert len(flattened) == 2
    assert any(row["screen_mode"] == "leaching" for row in flattened)
    assert any(row["screen_mode"] == "strap_contaminant_removal" for row in flattened)


class TestExtractStructuredResult:
    def test_extracts_valid_json(self):
        from strap.result_extractor import _extract_structured_result

        text = (
            "Some prose.\n<STRUCTURED_RESULT>\n"
            '{"agent": "test", "value": 42}\n'
            "</STRUCTURED_RESULT>"
        )
        result = _extract_structured_result(text)
        assert result == {"agent": "test", "value": 42}

    def test_returns_none_on_no_block(self):
        from strap.result_extractor import _extract_structured_result

        result = _extract_structured_result("Just prose, no structured block.")
        assert result is None

    def test_returns_none_on_malformed_json(self):
        from strap.result_extractor import _extract_structured_result

        text = "<STRUCTURED_RESULT>\n{not valid json}\n</STRUCTURED_RESULT>"
        result = _extract_structured_result(text)
        assert result is None

    def test_extracts_json_inside_markdown_fence(self):
        from strap.result_extractor import _extract_structured_result

        text = (
            "<STRUCTURED_RESULT>\n```json\n"
            '{"agent": "test", "value": 7}\n'
            "```\n</STRUCTURED_RESULT>"
        )
        result = _extract_structured_result(text)
        assert result == {"agent": "test", "value": 7}

    def test_extracts_root_user_query_from_state_messages(self):
        from langchain_core.messages import HumanMessage

        from strap.result_extractor import StructuredResultExtractorMiddleware

        state = {
            "messages": [
                HumanMessage(
                    content="Find the best separation sequence, then create a selectivity heatmap."
                )
            ]
        }

        query = StructuredResultExtractorMiddleware._extract_root_user_query(state)

        assert query == "Find the best separation sequence, then create a selectivity heatmap."

    def test_extract_text_from_command_prefers_tool_message_with_structured_result(self):
        from strap.result_extractor import StructuredResultExtractorMiddleware

        result = Command(
            update={
                "messages": [
                    ToolMessage(content="Only prose here.", tool_call_id="tc-prose"),
                    ToolMessage(
                        content=(
                            "<STRUCTURED_RESULT>\n"
                            '{"agent": "statistics-ml", "schema_version": "1.0", "analysis_type": "summary", "summary": {"n": 1}}'
                            "\n</STRUCTURED_RESULT>"
                        ),
                        tool_call_id="tc-structured",
                    ),
                ]
            }
        )

        extracted = StructuredResultExtractorMiddleware._extract_text_from_result(result)

        assert extracted is not None
        assert "<STRUCTURED_RESULT>" in extracted

    def test_extract_text_from_command_falls_back_to_last_tool_message(self):
        from strap.result_extractor import StructuredResultExtractorMiddleware

        result = Command(
            update={
                "messages": [
                    ToolMessage(content="First tool output.", tool_call_id="tc-first"),
                    ToolMessage(content="Last tool output.", tool_call_id="tc-last"),
                ]
            }
        )

        extracted = StructuredResultExtractorMiddleware._extract_text_from_result(result)

        assert extracted == "Last tool output."


class TestHandoffMiddleware:
    def _make_request(
        self,
        subagent_type: str,
        tool_call_id: str = "tc1",
        description: str | None = None,
    ):
        request = MagicMock()
        request.tool_call = {
            "id": tool_call_id,
            "name": "task",
            "args": {"subagent_type": subagent_type},
        }
        if description is not None:
            request.tool_call["args"]["description"] = description
        return request

    def test_wrap_tool_call_stores_append_only_records(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware,
            get_subagent_result,
            get_subagent_results,
        )

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        first = ToolMessage(
            content=(
                "Done.\n<STRUCTURED_RESULT>\n"
                '{"agent": "safety-analyst", "schema_version": "1.0", '
                '"solvents_assessed": ["Water"], "gscore_results": [], '
                '"ghs_results": []}\n</STRUCTURED_RESULT>'
            ),
            tool_call_id="tc1",
        )
        second = ToolMessage(
            content=(
                "Done.\n<STRUCTURED_RESULT>\n"
                '{"agent": "safety-analyst", "schema_version": "1.0", '
                '"solvents_assessed": ["Acetone"], "gscore_results": [], '
                '"ghs_results": []}\n</STRUCTURED_RESULT>'
            ),
            tool_call_id="tc2",
        )

        mw.wrap_tool_call(self._make_request("safety-analyst", "tc1"), MagicMock(return_value=first))
        mw.wrap_tool_call(self._make_request("safety-analyst", "tc2"), MagicMock(return_value=second))

        latest = json.loads(get_subagent_result("safety-analyst"))
        all_records = json.loads(get_subagent_results("safety-analyst"))

        assert latest["ok"] is True
        assert latest["handoff"]["payload"]["solvents_assessed"] == ["Acetone"]
        assert len(all_records["handoffs"]) == 2
        assert all_records["handoffs"][0]["payload"]["solvents_assessed"] == ["Water"]
        assert all_records["handoffs"][1]["payload"]["solvents_assessed"] == ["Acetone"]

    def test_wrap_tool_call_persists_task_description_on_record(self):
        from strap.result_extractor import StructuredResultExtractorMiddleware, get_subagent_result

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        result = ToolMessage(
            content=(
                "Done.\n<STRUCTURED_RESULT>\n"
                '{"agent": "separation-engineer", "schema_version": "1.0", '
                '"polymers": ["PS", "PMMA"], "best_sequence": ["PS", "PMMA"], '
                '"steps": [], "solvent_mapping": {"PS": "THF"}, '
                '"top_k_sequences": [{"rank": 1, "sequence": ["PS", "PMMA"], "solvent_mapping": {"PS": "THF"}}]}\n'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-sep-desc",
        )

        mw.wrap_tool_call(
            self._make_request(
                "separation-engineer",
                "tc-sep-desc",
                description="Find the separation sequence and then create a selectivity heatmap.",
            ),
            MagicMock(return_value=result),
        )

        latest = json.loads(get_subagent_result("separation-engineer"))
        assert latest["handoff"]["task_prompt"] == "Find the separation sequence and then create a selectivity heatmap."

    def test_wrap_tool_call_stores_normalized_structured_result_payload(self):
        from strap.result_extractor import StructuredResultExtractorMiddleware, get_subagent_result

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        result = ToolMessage(
            content=(
                "Done.\n<STRUCTURED_RESULT>\n"
                "{"
                '"agent": "separation-engineer", '
                '"schema_version": "1.0", '
                '"polymers": ["LDPE", "EVOH", "PET"], '
                '"supported_polymers": ["LDPE", "EVOH", "PET"], '
                '"unsupported_polymers": ["EACHCANDIDATE", "THEOPTIMIZATION-ENGINEER"], '
                '"best_sequence": ["LDPE", "EVOH", "PET"], '
                '"steps": [], '
                '"solvent_mapping": {}, '
                '"top_k_sequences": [{"rank": 1, "sequence": ["LDPE", "EVOH", "PET"], "solvent_mapping": {}}]'
                "}\n"
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-sep-normalize",
        )

        returned = mw.wrap_tool_call(
            self._make_request("separation-engineer", "tc-sep-normalize"),
            MagicMock(return_value=result),
        )

        assert isinstance(returned, ToolMessage)

        latest = json.loads(get_subagent_result("separation-engineer"))
        assert latest["handoff"]["payload"]["unsupported_polymers"] == []

    def test_wrap_tool_call_stores_backfilled_separation_candidate_lists(self, monkeypatch):
        from langchain_core.messages import HumanMessage

        from strap.result_extractor import StructuredResultExtractorMiddleware, get_subagent_result

        def fake_augment(payload, *, scope_user_query=None):
            assert "top 8 solvent candidates per polymer" in scope_user_query
            enriched = dict(payload)
            enriched["polymer_solvent_candidates"] = {
                "LDPE": [{"rank": i, "solvent": f"s{i}"} for i in range(1, 9)],
                "EVOH": [{"rank": i, "solvent": f"e{i}"} for i in range(1, 9)],
                "PET": [{"rank": i, "solvent": f"p{i}"} for i in range(1, 9)],
            }
            enriched["candidate_backfill_warnings"] = ["filled"]
            return enriched

        monkeypatch.setattr("strap.result_extractor._augment_underfilled_polymer_solvent_candidates", fake_augment)

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(
            {
                "messages": [
                    HumanMessage(
                        content="For LDPE/EVOH/PET, propose the top 8 solvent candidates per polymer."
                    )
                ]
            },
            None,
        )

        result = ToolMessage(
            content=(
                "Done.\n<STRUCTURED_RESULT>\n"
                "{"
                '"agent": "separation-engineer", '
                '"schema_version": "1.0", '
                '"polymers": ["LDPE", "EVOH", "PET"], '
                '"supported_polymers": ["LDPE", "EVOH", "PET"], '
                '"unsupported_polymers": [], '
                '"best_sequence": ["LDPE", "EVOH", "PET"], '
                '"steps": [], '
                '"solvent_mapping": {}, '
                '"top_k_sequences": [{"rank": 1, "sequence": ["LDPE", "EVOH", "PET"], "solvent_mapping": {}}], '
                '"polymer_solvent_candidates": {"LDPE": [{"rank": 1, "solvent": "s1"}]}'
                "}\n"
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-sep-backfill",
        )

        returned = mw.wrap_tool_call(
            self._make_request("separation-engineer", "tc-sep-backfill"),
            MagicMock(return_value=result),
        )

        assert isinstance(returned, ToolMessage)

        latest = json.loads(get_subagent_result("separation-engineer"))
        candidates = latest["handoff"]["payload"]["polymer_solvent_candidates"]
        assert {polymer: len(entries) for polymer, entries in candidates.items()} == {
            "LDPE": 8,
            "EVOH": 8,
            "PET": 8,
        }

    def test_invalid_payload_is_stored_but_marked_invalid(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware,
            get_subagent_result,
            build_handoff,
        )

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        invalid = ToolMessage(
            content=(
                "<STRUCTURED_RESULT>\n"
                '{"agent": "separation-engineer", "schema_version": "1.0"}\n'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-invalid",
        )

        mw.wrap_tool_call(
            self._make_request("separation-engineer", "tc-invalid"),
            MagicMock(return_value=invalid),
        )

        record = json.loads(get_subagent_result("separation-engineer"))
        assert record["handoff"]["status"] == "invalid"
        assert record["handoff"]["validation_errors"]

        derived = json.loads(
            build_handoff(
                consumer="biosteam-analyst",
                producer="separation-engineer",
            )
        )
        assert derived["ok"] is False

    def test_missing_structured_result_is_stored_as_missing(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware,
            get_subagent_result,
        )

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        missing = ToolMessage(
            content="I ran the analysis but returned only prose.",
            tool_call_id="tc-missing",
        )

        mw.wrap_tool_call(
            self._make_request("statistics-ml", "tc-missing"),
            MagicMock(return_value=missing),
        )

        record = json.loads(get_subagent_result("statistics-ml"))
        assert record["handoff"]["status"] == "missing"
        assert record["handoff"]["source_tool_call_id"] == "tc-missing"
        assert record["handoff"]["payload"]["error_kind"] == "missing_structured_result"
        assert record["handoff"]["validation_errors"]

    def test_build_handoff_creates_valid_derived_record(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware,
            build_handoff,
            get_subagent_result,
        )

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        separation_result = ToolMessage(
            content=(
                "Done.\n<STRUCTURED_RESULT>\n"
                "{"
                '"agent": "separation-engineer", '
                '"schema_version": "1.0", '
                '"polymers": ["LDPE", "PET"], '
                '"best_sequence": ["LDPE", "PET"], '
                '"steps": [{"step": 1, "polymer": "LDPE", "solvent": "Xylene", "temperature_c": 120.0}], '
                '"solvent_mapping": {"LDPE": "Xylene", "PET": "Toluene"}, '
                '"top_k_sequences": ['
                '{"rank": 1, "sequence": ["LDPE", "PET"], '
                '"min_selectivity": 72.1, '
                '"solvent_mapping": {"LDPE": "Xylene", "PET": "Toluene"}}'
                "]"
                "}\n</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-sep",
        )

        mw.wrap_tool_call(
            self._make_request("separation-engineer", "tc-sep"),
            MagicMock(return_value=Command(update={"messages": [separation_result]})),
        )

        source = json.loads(get_subagent_result("separation-engineer"))
        source_id = source["handoff"]["handoff_id"]
        derived = json.loads(
            build_handoff(
                consumer="biosteam-analyst",
                source_handoff_id=source_id,
            )
        )

        assert derived["ok"] is True
        assert derived["handoff"]["consumer"] == "biosteam-analyst"
        assert derived["handoff"]["contract"] == "sequence_batch.v1"
        candidate = derived["handoff"]["payload"]["sequence_candidates"][0]
        assert candidate["polymers_json"] == [
            {"polymer": "LDPE", "solvent": "Xylene"},
            {"polymer": "PET", "solvent": "Toluene"},
        ]
        assert "Run multi-polymer BioSTEAM" in derived["handoff"]["task_prompt"]

    def test_build_handoff_falls_back_to_generic_when_typed_adapter_cannot_build_payload(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware,
            build_handoff,
        )

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        separation_result = ToolMessage(
            content=(
                "<STRUCTURED_RESULT>\n"
                "{"
                '"agent": "separation-engineer", '
                '"schema_version": "1.0", '
                '"polymers": ["LDPE", "PET"], '
                '"best_sequence": ["LDPE", "PET"], '
                '"steps": [], '
                '"solvent_mapping": {"LDPE": "Xylene", "PET": "Toluene"}, '
                '"top_k_sequences": ['
                '1'
                "]"
                "}\n</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-sep-empty",
        )

        mw.wrap_tool_call(
            self._make_request("separation-engineer", "tc-sep-empty"),
            MagicMock(return_value=Command(update={"messages": [separation_result]})),
        )

        derived = json.loads(
            build_handoff(
                consumer="biosteam-analyst",
                producer="separation-engineer",
            )
        )

        assert derived["ok"] is True
        assert (
            derived["handoff"]["contract"]
            == "separation-engineer.to.biosteam-analyst.context.v1"
        )
        assert derived["handoff"]["payload"]["source_producer"] == "separation-engineer"
        assert derived["handoff"]["payload"]["source_payload"]["top_k_sequences"] == [1]

    def test_list_handoffs_filters_by_consumer(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware,
            build_handoff,
            list_handoffs,
        )

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        text = (
            "<STRUCTURED_RESULT>\n"
            "{"
            '"agent": "statistics-ml", '
            '"schema_version": "1.0", '
            '"analysis_type": "summary", '
            '"table": "common_solvents_database", '
            '"summary": {"n": 1}'
            "}\n</STRUCTURED_RESULT>"
        )
        mw.wrap_tool_call(
            self._make_request("statistics-ml", "tc-stats"),
            MagicMock(return_value=ToolMessage(content=text, tool_call_id="tc-stats")),
        )

        build_handoff(consumer="visualization-specialist", producer="statistics-ml")
        records = json.loads(list_handoffs(consumer="visualization-specialist"))

        assert records["ok"] is True
        assert len(records["handoffs"]) == 1
        assert records["handoffs"][0]["consumer"] == "visualization-specialist"

    def test_build_handoff_uses_latest_record_for_repeated_producer(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware,
            build_handoff,
        )

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        first = ToolMessage(
            content=(
                "<STRUCTURED_RESULT>\n"
                "{"
                '"agent": "separation-engineer", '
                '"schema_version": "1.0", '
                '"polymers": ["A"], '
                '"best_sequence": ["A"], '
                '"steps": [], '
                '"solvent_mapping": {"A": "OldSolvent"}, '
                '"top_k_sequences": ['
                '{"rank": 1, "sequence": ["A"], "solvent_mapping": {"A": "OldSolvent"}}'
                "]"
                "}\n</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-old",
        )
        second = ToolMessage(
            content=(
                "<STRUCTURED_RESULT>\n"
                "{"
                '"agent": "separation-engineer", '
                '"schema_version": "1.0", '
                '"polymers": ["A"], '
                '"best_sequence": ["A"], '
                '"steps": [], '
                '"solvent_mapping": {"A": "NewSolvent"}, '
                '"top_k_sequences": ['
                '{"rank": 1, "sequence": ["A"], "solvent_mapping": {"A": "NewSolvent"}}'
                "]"
                "}\n</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-new",
        )

        mw.wrap_tool_call(
            self._make_request("separation-engineer", "tc-old"),
            MagicMock(return_value=first),
        )
        mw.wrap_tool_call(
            self._make_request("separation-engineer", "tc-new"),
            MagicMock(return_value=second),
        )

        derived = json.loads(
            build_handoff(
                consumer="biosteam-analyst",
                producer="separation-engineer",
            )
        )
        candidate = derived["handoff"]["payload"]["sequence_candidates"][0]
        assert candidate["polymers_json"] == [{"polymer": "A", "solvent": "NewSolvent"}]

    def test_build_handoff_preserves_visualization_intent_from_task_prompt(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-viz-intent",
            thread_id="thread-viz-intent",
            invocation_id="inv-viz-intent",
        )
        source = store_agent_result(
            producer="separation-engineer",
            payload={
                "agent": "separation-engineer",
                "schema_version": "1.0",
                "polymers": ["PS", "PMMA", "PET"],
                "best_sequence": ["PS", "PMMA", "PET"],
                "steps": [
                    {
                        "step": 1,
                        "polymer": "PS",
                        "solvent": "THF",
                        "temperature_c": 60.0,
                        "selectivity_pct": 20.0,
                    }
                ],
                "solvent_mapping": {"PS": "THF"},
                "top_k_sequences": [
                    {
                        "rank": 1,
                        "sequence": ["PS", "PMMA", "PET"],
                        "min_selectivity": 20.0,
                        "solvent_mapping": {"PS": "THF"},
                    }
                ],
                "top_solvents": ["THF", "Toluene"],
            },
            source_tool_call_id="tc-sep-viz-intent",
            task_prompt="Find the optimal separation sequence, then create a selectivity heatmap showing the results.",
        )

        derived = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                source_handoff_id=source.handoff_id,
            )
        )

        assert derived["ok"] is True
        handoff = derived["handoff"]
        assert handoff["payload"]["requested_plot_type"] == "selectivity_heatmap"
        assert handoff["payload"]["preferred_tool"] == "create_selectivity_heatmap"
        assert handoff["payload"]["suggested_solvents"] == ["THF", "Toluene"]
        assert "selectivity heatmap" in handoff["task_prompt"].lower()
        assert "Required tool: create_selectivity_heatmap" in handoff["task_prompt"]
        assert "Required call pattern: create_selectivity_heatmap(" in handoff["task_prompt"]

    def test_build_handoff_adapts_separation_to_contaminant_screen(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-sep-contaminant",
            thread_id="thread-sep-contaminant",
            invocation_id="inv-sep-contaminant",
            user_query="Plan a separation route, then screen those solvents for PFAS leaching from PET.",
        )
        source = store_agent_result(
            producer="separation-engineer",
            payload={
                "agent": "separation-engineer",
                "schema_version": "1.0",
                "polymers": ["PET", "PE"],
                "best_sequence": ["PET", "PE"],
                "steps": [{"step": 1, "polymer": "PET", "solvent": "acetone", "temperature_c": 55.0}],
                "solvent_mapping": {"PET": "acetone"},
                "top_solvents": ["acetone", "methanol"],
                "top_k_sequences": [{"rank": 1, "sequence": ["PET", "PE"], "solvent_mapping": {"PET": "acetone"}}],
            },
            source_tool_call_id="tc-sep-contaminant",
            task_prompt="Find the best separation and then evaluate leaching-mode PFAS removal.",
        )

        derived = json.loads(
            build_handoff(
                consumer="contaminant-removal-analyst",
                source_handoff_id=source.handoff_id,
            )
        )

        assert derived["ok"] is True
        assert derived["handoff"]["contract"] == "contaminant_screen.v1"
        assert derived["handoff"]["payload"]["candidate_solvents"] == ["acetone", "methanol"]
        assert derived["handoff"]["payload"]["suggested_mode"] == "leaching"

    def test_build_handoff_adapts_contaminant_to_separation(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-contaminant-sep",
            thread_id="thread-contaminant-sep",
            invocation_id="inv-contaminant-sep",
            user_query="Find a separation route that also removes phthalates from PET.",
        )
        source = store_agent_result(
            producer="contaminant-removal-analyst",
            payload=_minimal_payloads()["contaminant-removal-analyst"],
            source_tool_call_id="tc-contaminant-sep",
            task_prompt="Compare contaminant-removal modes and recommend solvents.",
        )

        derived = json.loads(
            build_handoff(
                consumer="separation-engineer",
                source_handoff_id=source.handoff_id,
            )
        )

        assert derived["ok"] is True
        assert derived["handoff"]["contract"] == "contaminant_guided_separation.v1"
        assert derived["handoff"]["payload"]["recommended_solvents"] == ["acetone"]
        assert "Refine the separation route" in derived["handoff"]["task_prompt"]

    def test_build_handoff_adapts_contaminant_to_biosteam(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-contaminant-bio",
            thread_id="thread-contaminant-bio",
            invocation_id="inv-contaminant-bio",
            user_query="Plan separation, screen phthalate removal, then run TEA on the best screened option.",
        )
        source = store_agent_result(
            producer="contaminant-removal-analyst",
            payload=_minimal_payloads()["contaminant-removal-analyst"],
            source_tool_call_id="tc-contaminant-bio",
            task_prompt="Compare contaminant-removal modes and recommend solvents.",
        )

        derived = json.loads(
            build_handoff(
                consumer="biosteam-analyst",
                source_handoff_id=source.handoff_id,
            )
        )

        assert derived["ok"] is True
        assert derived["handoff"]["contract"] == "contaminant_biosteam.v1"
        assert derived["handoff"]["payload"]["target_plastic"] == "PET"
        assert derived["handoff"]["payload"]["best_solvent"] == "acetone"

    def test_build_handoff_prefers_scope_user_query_for_visualization_intent(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-viz-user-query",
            thread_id="thread-viz-user-query",
            invocation_id="inv-viz-user-query",
            user_query="Find the optimal separation sequence, then create a selectivity heatmap showing the results.",
        )
        source = store_agent_result(
            producer="separation-engineer",
            payload={
                "agent": "separation-engineer",
                "schema_version": "1.0",
                "polymers": ["PS", "PMMA", "PET"],
                "best_sequence": ["PS", "PMMA", "PET"],
                "steps": [
                    {
                        "step": 1,
                        "polymer": "PS",
                        "solvent": "THF",
                        "temperature_c": 60.0,
                        "selectivity_pct": 20.0,
                    }
                ],
                "solvent_mapping": {"PS": "THF"},
                "top_k_sequences": [
                    {
                        "rank": 1,
                        "sequence": ["PS", "PMMA", "PET"],
                        "min_selectivity": 20.0,
                        "solvent_mapping": {"PS": "THF"},
                    }
                ],
                "top_solvents": ["THF", "Toluene"],
            },
            source_tool_call_id="tc-sep-viz-user-query",
            task_prompt="Find the optimal separation sequence for these polymers.",
        )

        derived = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                source_handoff_id=source.handoff_id,
            )
        )

        assert derived["ok"] is True
        handoff = derived["handoff"]
        assert handoff["payload"]["requested_plot_type"] == "selectivity_heatmap"
        assert handoff["payload"]["preferred_tool"] == "create_selectivity_heatmap"
        assert handoff["payload"]["source_user_query"].startswith(
            "Find the optimal separation sequence"
        )
        assert "Original user request:" in handoff["task_prompt"]
        assert "Required tool: create_selectivity_heatmap" in handoff["task_prompt"]

    def test_build_handoff_limits_visualization_to_supported_subset(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-viz-supported-subset",
            thread_id="thread-viz-supported-subset",
            invocation_id="inv-viz-supported-subset",
            user_query="Find the optimal separation sequence, then create a selectivity heatmap showing the results.",
        )
        source = store_agent_result(
            producer="separation-engineer",
            payload={
                "agent": "separation-engineer",
                "schema_version": "1.0",
                "polymers": ["PS", "PMMA", "PET"],
                "supported_polymers": ["PS", "PET"],
                "unsupported_polymers": ["PMMA"],
                "best_sequence": ["PS", "PET", "PMMA"],
                "steps": [
                    {
                        "step": 1,
                        "polymer": "PS",
                        "solvent": "THF",
                        "temperature_c": 65.0,
                        "selectivity_pct": 30.7,
                    }
                ],
                "solvent_mapping": {"PS": "THF"},
                "top_k_sequences": [
                    {
                        "rank": 1,
                        "sequence": ["PS", "PET", "PMMA"],
                        "min_selectivity": 30.7,
                        "solvent_mapping": {"PS": "THF"},
                    }
                ],
                "top_solvents": ["THF"],
            },
            source_tool_call_id="tc-sep-supported-subset",
            task_prompt="Find the optimal separation sequence for PS, PMMA, and PET, then create a selectivity heatmap.",
        )

        derived = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                source_handoff_id=source.handoff_id,
            )
        )

        assert derived["ok"] is True
        handoff = derived["handoff"]
        assert handoff["payload"]["plot_polymers"] == ["PS", "PET"]
        assert handoff["payload"]["unsupported_polymers"] == ["PMMA"]
        assert 'polymers="PS,PET"' in handoff["task_prompt"]
        assert "unsupported polymers: PMMA" in handoff["task_prompt"]

    def test_build_handoff_infers_supported_subset_from_local_coverage(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-viz-supported-local",
            thread_id="thread-viz-supported-local",
            invocation_id="inv-viz-supported-local",
            user_query="Create a selectivity heatmap for PS, PMMA, and PET.",
        )
        source = store_agent_result(
            producer="separation-engineer",
            payload={
                "agent": "separation-engineer",
                "schema_version": "1.0",
                "polymers": ["PS", "PMMA", "PET"],
                "best_sequence": ["PS", "PMMA", "PET"],
                "steps": [
                    {
                        "step": 1,
                        "polymer": "PS",
                        "solvent": "THF",
                        "temperature_c": 65.0,
                        "selectivity_pct": 30.7,
                    }
                ],
                "solvent_mapping": {"PS": "THF"},
                "top_k_sequences": [
                    {
                        "rank": 1,
                        "sequence": ["PS", "PMMA", "PET"],
                        "min_selectivity": 30.7,
                        "solvent_mapping": {"PS": "THF"},
                    }
                ],
                "top_solvents": ["THF"],
            },
            source_tool_call_id="tc-sep-supported-local",
            task_prompt="Find the optimal separation sequence for PS, PMMA, and PET, then create a selectivity heatmap.",
        )

        derived = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                source_handoff_id=source.handoff_id,
            )
        )

        assert derived["ok"] is True
        handoff = derived["handoff"]
        assert handoff["payload"]["plot_polymers"] == ["PS", "PET"]
        assert handoff["payload"]["unsupported_polymers"] == ["PMMA"]

    def test_parallel_writes_from_threads_keep_all_records(self):
        from strap.handoffs import get_latest_handoff, initialize_handoff_scope, list_handoff_records, store_agent_result

        scope = initialize_handoff_scope(
            run_id="run-parallel",
            thread_id="thread-parallel",
            invocation_id="inv-parallel",
        )

        def write_result(idx: int):
            from strap.handoffs import _scope_key

            _scope_key.set(scope.scope_id)
            store_agent_result(
                producer="safety-analyst",
                payload={
                    "agent": "safety-analyst",
                    "schema_version": "1.0",
                    "solvents_assessed": [f"S{idx}"],
                    "gscore_results": [],
                    "ghs_results": [],
                },
                source_tool_call_id=f"tc-{idx}",
            )

        threads = [threading.Thread(target=write_result, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        records = list_handoff_records(producer="safety-analyst")
        latest = get_latest_handoff(producer="safety-analyst")

        assert len(records) == 3
        assert latest is not None
        assert latest.payload["solvents_assessed"][0].startswith("S")

    def test_build_handoff_supports_all_current_subagent_pairs(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-all-pairs",
            thread_id="thread-all-pairs",
            invocation_id="inv-all-pairs",
        )
        payloads = _minimal_payloads()
        sources = {
            producer: store_agent_result(
                producer=producer,
                payload=payload,
                source_tool_call_id=f"tc-{producer}",
            )
            for producer, payload in payloads.items()
        }
        typed_contracts = {
            ("biosteam-analyst", "visualization-specialist"): "biosteam_plot.v1",
            ("contaminant-removal-analyst", "biosteam-analyst"): "contaminant_biosteam.v1",
            ("contaminant-removal-analyst", "separation-engineer"): "contaminant_guided_separation.v1",
            ("patent-researcher", "rag-analyst"): "patent_context.v1",
            ("scholar-researcher", "rag-analyst"): "literature_context.v1",
            ("separation-engineer", "biosteam-analyst"): "sequence_batch.v1",
            ("separation-engineer", "contaminant-removal-analyst"): "contaminant_screen.v1",
            ("separation-engineer", "visualization-specialist"): "separation_plot.v1",
            ("statistics-ml", "visualization-specialist"): "analysis_plot.v1",
        }

        for producer, source in sources.items():
            for consumer in payloads:
                derived = json.loads(
                    build_handoff(
                        consumer=consumer,
                        source_handoff_id=source.handoff_id,
                    )
                )
                assert derived["ok"] is True, (producer, consumer, derived)
                handoff = derived["handoff"]
                assert handoff["producer"] == producer
                assert handoff["consumer"] == consumer
                assert handoff["parent_handoff_id"] == source.handoff_id
                expected_contract = typed_contracts.get(
                    (producer, consumer),
                    f"{producer}.to.{consumer}.context.v1",
                )
                assert handoff["contract"] == expected_contract

    def test_build_handoff_uses_latest_source_result_not_latest_derived_record(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff, get_subagent_result

        initialize_handoff_scope(
            run_id="run-latest-source",
            thread_id="thread-latest-source",
            invocation_id="inv-latest-source",
        )
        source = store_agent_result(
            producer="statistics-ml",
            payload=_minimal_payloads()["statistics-ml"],
            source_tool_call_id="tc-stats-latest",
        )

        first = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                producer="statistics-ml",
            )
        )
        second = json.loads(
            build_handoff(
                consumer="rag-analyst",
                producer="statistics-ml",
            )
        )
        latest_result = json.loads(get_subagent_result("statistics-ml"))

        assert first["ok"] is True
        assert second["ok"] is True
        assert second["handoff"]["payload"]["source_handoff_id"] == source.handoff_id
        assert latest_result["handoff"]["handoff_id"] == source.handoff_id
        assert latest_result["handoff"]["contract"] == "statistics-ml.result.v1"

    def test_build_handoff_falls_back_to_latest_result_when_explicit_id_is_stale(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-stale-source-id",
            thread_id="thread-stale-source-id",
            invocation_id="inv-stale-source-id",
        )
        source = store_agent_result(
            producer="optimization-engineer",
            payload={
                "agent": "optimization-engineer",
                "schema_version": "1.0",
                "analysis_type": "pareto_slices",
                "x_metric": "total_cost",
                "y_metric": "circularity",
                "n_slices_requested": 2,
                "n_slices_solved": 2,
                "n_points_requested_per_slice": 100,
                "pareto_slices_payload_path": "/tmp/pareto_slices.json",
                "slices": [],
            },
            source_tool_call_id="tc-opt-stale-source-id",
        )

        derived = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                source_handoff_id="not-a-real-handoff",
                producer="optimization-engineer",
                strategy="latest",
            )
        )

        assert derived["ok"] is True
        assert derived["handoff"]["producer"] == "optimization-engineer"
        assert derived["handoff"]["consumer"] == "visualization-specialist"
        assert derived["handoff"]["parent_handoff_id"] == source.handoff_id
        assert derived["handoff"]["payload"]["source_handoff_id"] == source.handoff_id

    def test_build_handoff_can_chain_from_derived_handoff(self):
        from strap.handoffs import initialize_handoff_scope, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-chain",
            thread_id="thread-chain",
            invocation_id="inv-chain",
        )
        source = store_agent_result(
            producer="scholar-researcher",
            payload=_minimal_payloads()["scholar-researcher"],
            source_tool_call_id="tc-scholar-chain",
        )

        first = json.loads(
            build_handoff(
                consumer="rag-analyst",
                source_handoff_id=source.handoff_id,
            )
        )
        second = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                source_handoff_id=first["handoff"]["handoff_id"],
            )
        )

        assert first["ok"] is True
        assert first["handoff"]["contract"] == "literature_context.v1"
        assert second["ok"] is True
        assert second["handoff"]["contract"] == "scholar-researcher.to.visualization-specialist.context.v1"
        assert second["handoff"]["parent_handoff_id"] == first["handoff"]["handoff_id"]
        assert second["handoff"]["payload"]["source_handoff_id"] == first["handoff"]["handoff_id"]
        assert second["handoff"]["payload"]["source_contract"] == "literature_context.v1"

    def test_build_multi_source_handoff_for_consumer_stores_deduplicated_join_envelope(self):
        from strap.handoffs import (
            build_handoff_for_consumer,
            build_multi_source_handoff_for_consumer,
            initialize_handoff_scope,
            list_handoff_records,
            store_agent_result,
        )

        initialize_handoff_scope(
            run_id="run-multi-source",
            thread_id="thread-multi-source",
            invocation_id="inv-multi-source",
        )
        payloads = _minimal_payloads()
        scholar = store_agent_result(
            producer="scholar-researcher",
            payload=payloads["scholar-researcher"],
            source_tool_call_id="tc-scholar-multi",
        )
        patent = store_agent_result(
            producer="patent-researcher",
            payload=payloads["patent-researcher"],
            source_tool_call_id="tc-patent-multi",
        )
        literature = build_handoff_for_consumer(
            consumer="rag-analyst",
            source_handoff_id=scholar.handoff_id,
        )
        patents = build_handoff_for_consumer(
            consumer="rag-analyst",
            source_handoff_id=patent.handoff_id,
        )

        first = build_multi_source_handoff_for_consumer(
            consumer="rag-analyst",
            source_handoff_ids=[literature.handoff_id, patents.handoff_id],
        )
        second = build_multi_source_handoff_for_consumer(
            consumer="rag-analyst",
            source_handoff_ids=[patents.handoff_id, literature.handoff_id],
        )
        third = build_multi_source_handoff_for_consumer(
            consumer="rag-analyst",
            source_handoff_ids=[literature.handoff_id, patents.handoff_id],
        )
        records = list_handoff_records(producer="multi-source", consumer="rag-analyst")

        assert first.handoff_id == second.handoff_id
        assert second.handoff_id == third.handoff_id
        assert first.contract == "multi-source.to.rag-analyst.context.v1"
        assert first.parent_handoff_ids == first.payload["source_handoff_ids"]
        assert set(first.payload["source_handoff_ids"]) == {literature.handoff_id, patents.handoff_id}
        assert {item["producer"] for item in first.payload["source_handoffs"]} == {
            "scholar-researcher",
            "patent-researcher",
        }
        assert {item["contract"] for item in first.payload["source_handoffs"]} == {
            "literature_context.v1",
            "patent_context.v1",
        }
        assert all(item["handoff_id"] in first.payload["source_handoff_ids"] for item in first.payload["source_handoffs"])
        assert set(first.payload["producers"]) == {"scholar-researcher", "patent-researcher"}
        assert set(first.payload["contracts"]) == {"literature_context.v1", "patent_context.v1"}
        assert "Treat `payload.source_handoffs` as authoritative upstream context." in first.task_prompt
        assert literature.handoff_id in first.task_prompt
        assert patents.handoff_id in first.task_prompt
        assert "contract=literature_context.v1" in first.task_prompt
        assert "contract=patent_context.v1" in first.task_prompt
        assert len(records) == 1

    def test_rebinding_same_scope_preserves_records_for_later_handoffs(self):
        from strap.handoffs import initialize_handoff_scope, list_handoff_records, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-rebind",
            thread_id="thread-rebind",
            invocation_id="inv-one",
        )
        source = store_agent_result(
            producer="statistics-ml",
            payload=_minimal_payloads()["statistics-ml"],
            source_tool_call_id="tc-stats-rebind",
        )

        initialize_handoff_scope(
            run_id="run-rebind",
            thread_id="thread-rebind",
            invocation_id="inv-two",
        )
        derived = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                producer="statistics-ml",
            )
        )

        records = list_handoff_records(producer="statistics-ml")
        assert len(records) == 2
        assert records[0].handoff_id == source.handoff_id
        assert derived["ok"] is True
        assert derived["handoff"]["parent_handoff_id"] == source.handoff_id

    def test_wrap_tool_call_rebinds_original_scope_for_non_task_tools(self):
        from strap.result_extractor import StructuredResultExtractorMiddleware, build_handoff

        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        stats_result = ToolMessage(
            content=(
                "<STRUCTURED_RESULT>\n"
                "{"
                '"agent": "statistics-ml", '
                '"schema_version": "1.0", '
                '"analysis_type": "summary", '
                '"summary": {"n": 1}'
                "}\n</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc-stats-scope",
        )
        mw.wrap_tool_call(
            self._make_request("statistics-ml", "tc-stats-scope"),
            MagicMock(return_value=ToolMessage(content=stats_result.content, tool_call_id="tc-stats-scope")),
        )

        class _Request:
            def __init__(self):
                self.tool_call = {
                    "id": "bh-scope",
                    "name": "build_handoff",
                    "args": {"consumer": "visualization-specialist", "producer": "statistics-ml"},
                }

        def _handler(_request):
            return ToolMessage(
                content=build_handoff(
                    consumer="visualization-specialist",
                    producer="statistics-ml",
                ),
                tool_call_id="bh-scope",
            )

        result = contextvars.Context().run(lambda: mw.wrap_tool_call(_Request(), _handler))
        envelope = json.loads(result.content)

        assert envelope["ok"] is True
        assert envelope["handoff"]["consumer"] == "visualization-specialist"

    def test_repeated_same_source_handoffs_append_without_overwrite(self):
        from strap.handoffs import initialize_handoff_scope, list_handoff_records, store_agent_result
        from strap.result_extractor import build_handoff

        initialize_handoff_scope(
            run_id="run-repeat-derived",
            thread_id="thread-repeat-derived",
            invocation_id="inv-repeat-derived",
        )
        source = store_agent_result(
            producer="safety-analyst",
            payload=_minimal_payloads()["safety-analyst"],
            source_tool_call_id="tc-safety-repeat",
        )

        first = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                source_handoff_id=source.handoff_id,
            )
        )
        second = json.loads(
            build_handoff(
                consumer="visualization-specialist",
                source_handoff_id=source.handoff_id,
            )
        )
        derived_records = list_handoff_records(
            producer="safety-analyst",
            consumer="visualization-specialist",
        )

        assert first["ok"] is True
        assert second["ok"] is True
        assert first["handoff"]["handoff_id"] != second["handoff"]["handoff_id"]
        assert len(derived_records) == 2
