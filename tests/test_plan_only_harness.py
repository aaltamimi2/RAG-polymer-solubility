from __future__ import annotations

from architecture.plan_only_harness import build_plan_only_cases, run_plan_only_suite


def test_plan_only_harness_builds_exactly_100_cases():
    cases = build_plan_only_cases()

    assert len(cases) == 100
    assert {case.category for case in cases} == {
        "single-separation",
        "single-safety",
        "single-biosteam",
        "single-contaminant",
        "stats-viz",
        "seq-sep-viz",
        "seq-sep-biosteam",
        "seq-sep-contam-bio",
        "mixed-research",
        "parallel-scholar-patent",
    }


def test_plan_only_harness_validates_all_queries_without_model_access():
    summary = run_plan_only_suite()

    assert summary.total == 100
    assert summary.passed == 100
    assert summary.failed == 0
    assert sum(summary.blocked_model_call_attempts.values()) == 0
