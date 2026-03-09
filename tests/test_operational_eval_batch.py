from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ARCH_DIR = Path(__file__).resolve().parent.parent / "architecture"
sys.path.insert(0, str(ARCH_DIR))

import architecture.test_harness as harness
import operational_eval_batch as batch


def test_suite_has_50_unique_cases_and_expected_categories():
    names = [case.name for case in batch.DEFAULT_SUITE]
    assert len(batch.DEFAULT_SUITE) == 50
    assert len(set(names)) == 50

    categories = {case.category for case in batch.DEFAULT_SUITE}
    assert categories == {
        "separation",
        "biosteam",
        "safety",
        "hsp",
        "sep-biosteam",
        "sep-safety",
    }


def test_hsp_suite_uses_statistics_ml_only_and_minimum_ml_counts():
    hsp_cases = [case for case in batch.DEFAULT_SUITE if case.category == "hsp"]
    assert len(hsp_cases) == 10
    for case in hsp_cases:
        assert case.expected_subagents == ["statistics-ml"]
        assert case.min_trace_counts["predict_solubility_ml"] >= 2


def test_contaminant_suite_has_expected_categories_and_routes():
    names = [case.name for case in batch.CONTAMINANT_SUITE]
    assert len(batch.CONTAMINANT_SUITE) == 6
    assert len(set(names)) == 6
    categories = {case.category for case in batch.CONTAMINANT_SUITE}
    assert categories == {"contaminant", "sep-contaminant"}
    contam_only = [case for case in batch.CONTAMINANT_SUITE if case.category == "contaminant"]
    assert len(contam_only) == 5
    for case in contam_only:
        assert case.expected_subagents == ["contaminant-removal-analyst"]
    seq_cases = [case for case in batch.CONTAMINANT_SUITE if case.category == "sep-contaminant"]
    assert len(seq_cases) == 1
    assert seq_cases[0].expected_subagents == ["separation-engineer", "contaminant-removal-analyst"]


def test_separation_trace_toolset_includes_selectivity_analysis_tools():
    assert "analyze_selective_solubility_enhanced" in batch.SEPARATION_TRACE_TOOLS
    assert "rank_solvents_for_separation" in batch.SEPARATION_TRACE_TOOLS


def test_eval_flags_missing_trace_and_wrong_route():
    case = batch.SUITE_BY_NAME["safety-toluene-dmso-thf"]
    fake = SimpleNamespace(
        error=None,
        full_answer="DMSO has the best G-score. PubChem hazards show THF is more hazardous.",
        actual_subagents=["statistics-ml"],
        thread_id="thread-1",
        run_id=None,
        trace_id=None,
        trace_summary=None,
        tool_names=[],
    )

    checks = batch._evaluate_case(case, fake)
    failed = {check.name for check in checks if not check.passed}
    assert "Executed subagent route matches expected set exactly" in failed
    assert "Harness persisted thread_id/run_id/trace_id" in failed
    assert "LangSmith trace summary captured" in failed


def test_eval_passes_minimal_good_hsp_case():
    case = batch.SUITE_BY_NAME["hsp-pe-tol-dmso-hex"]
    fake = SimpleNamespace(
        error=None,
        full_answer=(
            "Using Hansen solubility parameters, I compared PE with Toluene, DMSO, and Hexane. "
            "Hexane has the lowest RED, Toluene is also soluble, and DMSO is incompatible. "
            "RED ranking supports Hexane as the best match."
        ),
        actual_subagents=["statistics-ml"],
        thread_id="thread-1",
        run_id="run-1",
        trace_id="trace-1",
        trace_summary={
            "tool_names": ["predict_solubility_ml", "predict_solubility_ml", "predict_solubility_ml"],
            "child_errors": [],
        },
        tool_names=["task", "predict_solubility_ml"],
    )

    checks = batch._evaluate_case(case, fake)
    assert all(check.passed for check in checks), [check for check in checks if not check.passed]


def test_parse_category_timeouts_accepts_valid_mapping():
    parsed = batch._parse_category_timeouts("hsp=90,separation=150,sep-safety=210,contaminant=120")

    assert parsed == {
        "hsp": 90,
        "separation": 150,
        "sep-safety": 210,
        "contaminant": 120,
    }


def test_parse_category_timeouts_rejects_unknown_category():
    try:
        batch._parse_category_timeouts("unknown=30")
    except SystemExit as exc:
        assert "Unknown category" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for unknown category")


def test_resolve_case_timeout_prefers_category_override():
    case = batch.SUITE_BY_NAME["sepsafe-ps-over-pvc"]

    timeout = batch._resolve_case_timeout(
        case,
        timeout_s=120,
        category_timeouts={"sep-safety": 210},
    )

    assert timeout == 210


def test_resolve_case_timeout_falls_back_to_global_timeout():
    case = batch.SUITE_BY_NAME["tea-pe-toluene-c1"]

    timeout = batch._resolve_case_timeout(
        case,
        timeout_s=120,
        category_timeouts={"sep-safety": 210},
    )

    assert timeout == 120


def test_write_case_artifact_persists_query_answer_and_trace(tmp_path):
    result = batch.CaseResult(
        name="sep-ps-pvc-below-90",
        category="separation",
        query="Below 90C, can you separate PS from PVC?",
        pattern="single-agent",
        expected_subagents=["separation-engineer"],
        actual_subagents=["separation-engineer"],
        attempts=1,
        checks=[batch.Check(name="ok", passed=True, detail="detail")],
        wall_time_s=12.5,
        total_tokens=1234,
        tool_names=["task", "analyze_selective_solubility_enhanced"],
        thread_id="thread-1",
        run_id="run-1",
        trace_id="trace-1",
        trace_summary={"tool_names": ["task"], "child_errors": []},
        full_answer="Full answer text",
        answer_preview="Full answer text",
        final_answer_diagnostics={"last_ai_origin": "routing_single_specialist_prose"},
        error=None,
        raw_result_path="/tmp/trace.png",
    )

    batch._write_case_artifact(tmp_path, result)

    json_payload = (tmp_path / "sep-ps-pvc-below-90.json").read_text()
    markdown = (tmp_path / "sep-ps-pvc-below-90.md").read_text()

    assert "Below 90C, can you separate PS from PVC?" in json_payload
    assert "Full answer text" in json_payload
    assert '"trace_id": "trace-1"' in json_payload
    assert "## Query" in markdown
    assert "## Final Answer Diagnostics" in markdown
    assert "## Full Answer" in markdown
    assert "trace-1" in markdown


def test_recover_timeout_result_uses_snapshot_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        batch,
        "_extract_subagents_from_trace",
        lambda *args, **kwargs: ["separation-engineer"],
    )

    snapshot = harness.QueryResult(
        name="sep-evoh-ldpe-pet-film",
        query="Only do process design.",
        pattern="single-agent",
        expected_subagents=["separation-engineer"],
        actual_subagents=["separation-engineer"],
        wall_time_s=61.0,
        total_tokens=1234,
        input_tokens=1000,
        output_tokens=234,
        n_tool_calls=1,
        n_messages=4,
        tool_names=["task"],
        thread_id="thread-timeout",
        run_id=None,
        trace_id=None,
        full_answer="Recovered timeout answer.",
        answer_preview="Recovered timeout answer.",
        routing_match=True,
        timestamp="2026-03-07T00:00:00",
        error=None,
        trace_summary=None,
        final_answer_diagnostics={"last_ai_origin": "routing_single_specialist_prose"},
    )
    monkeypatch.setattr(batch, "load_timeout_snapshot", lambda thread_id: snapshot)

    recovered = batch._recover_timeout_result(
        case=batch.SUITE_BY_NAME["sep-evoh-ldpe-pet-film"],
        thread_id="thread-timeout",
        timeout_s=150,
        trace_info={
            "run_id": "run-1",
            "trace_id": "trace-1",
            "tool_names": ["plan_multiple_separation_schemes", "task"],
            "child_errors": [],
        },
        ls_client=SimpleNamespace(),
        project_name="strap-agent",
    )

    assert recovered.full_answer == "Recovered timeout answer."
    assert recovered.error is None
    assert recovered.run_id == "run-1"
    assert recovered.trace_id == "trace-1"
    assert recovered.actual_subagents == ["separation-engineer"]
    assert recovered.final_answer_diagnostics["timeout_recovered"] is True


def test_recover_timeout_result_uses_validated_trace_payload_when_snapshot_missing(monkeypatch):
    monkeypatch.setattr(batch, "load_timeout_snapshot", lambda thread_id: None)
    monkeypatch.setattr(
        batch,
        "_extract_subagents_from_trace",
        lambda *args, **kwargs: ["separation-engineer"],
    )
    monkeypatch.setattr(
        batch,
        "_extract_latest_task_output_from_trace",
        lambda *args, **kwargs: (
            "<STRUCTURED_RESULT>"
            '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET","PC"],'
            '"best_sequence":["PS","PC","PET"],'
            '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":105.0}],'
            '"solvent_mapping":{"PS":"Toluene"},'
            '"top_k_sequences":[{"sequence":["PS","PC","PET"],"score":1.0}]}'
            "</STRUCTURED_RESULT>"
        ),
    )

    recovered = batch._recover_timeout_result(
        case=batch.SUITE_BY_NAME["sep-evoh-ldpe-pet-film"],
        thread_id="thread-trace",
        timeout_s=150,
        trace_info={
            "run_id": "run-2",
            "trace_id": "trace-2",
            "tool_names": ["find_optimal_separation_sequence", "task"],
            "child_errors": [],
        },
        ls_client=SimpleNamespace(),
        project_name="strap-agent",
    )

    assert "Recommended separation sequence" in recovered.full_answer
    assert recovered.error is None
    assert recovered.run_id == "run-2"
    assert recovered.trace_id == "trace-2"
    assert recovered.final_answer_diagnostics["timeout_recovered_from_validated_trace_payload"] is True
    assert recovered.final_answer_diagnostics["last_ai_origin"] == "routing_single_specialist_separation_fallback"


def test_recover_timeout_result_supports_validated_contaminant_payload(monkeypatch):
    monkeypatch.setattr(batch, "load_timeout_snapshot", lambda thread_id: None)
    monkeypatch.setattr(
        batch,
        "_extract_subagents_from_trace",
        lambda *args, **kwargs: ["contaminant-removal-analyst"],
    )
    monkeypatch.setattr(
        batch,
        "_extract_latest_task_output_from_trace",
        lambda *args, **kwargs: (
            "<STRUCTURED_RESULT>"
            '{'
            '"agent":"contaminant-removal-analyst",'
            '"schema_version":"1.0",'
            '"mode":"comparison",'
            '"target_polymer":"EVOH",'
                '"other_polymers":["LDPE"],'
                '"contaminants":["di-n-butyl phthalate (DBP)"],'
                '"supported_contaminants":["di-n-butyl phthalate (DBP)"],'
                '"unsupported_contaminants":[],'
                '"candidate_solvents":[],'
                '"recommended_mode":"strap_contaminant_removal",'
                '"modes":{"strap_contaminant_removal":{"candidate_solvents":[{"solvent":"dimethyl sulfoxide","passes":true,"operating_temperature_c":90.0,"boiling_point_c":189.0,"contaminant_logd_min":0.31}]},'
                '"leaching":{"candidate_solvents":[]}},'
            '"recommended_solvents":{"leaching":[],"strap_contaminant_removal":["dimethyl sulfoxide"]},'
            '"caveats":["temperature-swing contaminant removal is screened using a 1 wt% polymer precipitation threshold"]'
            '}'
            "</STRUCTURED_RESULT>"
        ),
    )

    recovered = batch._recover_timeout_result(
        case=batch.SUITE_BY_NAME["contam-strap-evoh-dbp-pe"],
        thread_id="thread-contam",
        timeout_s=120,
        trace_info={
            "run_id": "run-contam",
            "trace_id": "trace-contam",
            "tool_names": ["compare_contaminant_removal_modes", "task"],
            "child_errors": [],
        },
        ls_client=SimpleNamespace(),
        project_name="strap-agent",
    )

    assert recovered.error is None
    assert "Recommended mode" in recovered.full_answer
    assert "temperature-swing STRAP contaminant removal" in recovered.full_answer
    assert "logD" in recovered.full_answer
    assert recovered.trace_id == "trace-contam"
    assert recovered.final_answer_diagnostics["timeout_recovered_from_validated_trace_payload"] is True
    assert recovered.final_answer_diagnostics["last_ai_origin"] == "timeout_validated_contaminant_payload"


def test_select_cases_supports_contaminant_suite():
    cases = batch._select_cases("contaminant", "contaminant", None, None, None)
    assert len(cases) == 5
    assert all(case.category == "contaminant" for case in cases)


def test_select_cases_supports_all_suite_by_case_name():
    cases = batch._select_cases("all", None, None, "tea-pe-toluene-c1,contam-leach-pet-dbp", None)
    assert [case.name for case in cases] == ["tea-pe-toluene-c1", "contam-leach-pet-dbp"]
