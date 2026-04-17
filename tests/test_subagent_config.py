from __future__ import annotations

from pathlib import Path

import yaml

from strap.subagent_config import (
    load_execution_pairs,
    load_routing_configuration,
    load_subagent_bundle,
    load_subagent_specs,
)


def test_load_split_manifest_defaults():
    bundle = load_subagent_bundle()

    assert len(bundle["subagents"]) == 10
    assert "parallel" in bundle["execution_pairs"]
    assert "sequential" in bundle["execution_pairs"]
    assert bundle["subagents"][0]["name"] == "separation-engineer"
    assert any(spec["name"] == "optimization-engineer" for spec in bundle["subagents"])


def test_separation_engineer_prompt_includes_bounded_temperature_guidance():
    bundle = load_subagent_bundle()
    separation = next(item for item in bundle["subagents"] if item["name"] == "separation-engineer")
    prompt = separation["system_prompt"]

    assert "treat that as an upper limit" in prompt
    assert 'Do not conclude "not feasible"' in prompt
    assert "Do not present that route as the best or optimal executable sequence" in prompt
    assert "actual recommended operating temperatures" in prompt
    assert "stay below that boiling point at 1 atm" in prompt
    assert "narrow atmospheric-pressure operating margin" in prompt
    assert "conclusions apply only to the supported subset" in prompt
    assert "do not call that material pure, purified, or cleanly isolated" in prompt
    assert "`supported_polymers` and `unsupported_polymers` arrays" in prompt
    assert "Do not assert whether an unsupported polymer dissolves" in prompt
    assert "predicted/selectivity-based candidate" in prompt
    assert "experimental confirmation or a fuller feasibility check" in prompt


def test_statistics_ml_prompt_mentions_hsp_visualization_workflow():
    bundle = load_subagent_bundle()
    stats_ml = next(item for item in bundle["subagents"] if item["name"] == "statistics-ml")
    prompt = stats_ml["system_prompt"]
    description = stats_ml["description"]
    routing_phrases = stats_ml["routing"]["phrases"]

    assert "HSP radar" in description
    assert "predict_solubility_ml generates HSP visual artifacts" in prompt
    assert "show or plot the HSP radar" in prompt
    assert "return the generated artifact path(s)" in prompt
    assert "hsp radar" in routing_phrases
    assert "hansen sphere" in routing_phrases


def test_load_routing_configuration_from_split_manifest():
    routing_rules, parallel_pairs, parallel_3way, sequential_pairs = load_routing_configuration()

    assert routing_rules[0]["subagent"] == "separation-engineer"
    assert frozenset({"separation-engineer", "safety-analyst"}) in parallel_pairs
    assert frozenset({"separation-engineer", "safety-analyst", "biosteam-analyst"}) in parallel_3way
    assert ("separation-engineer", "biosteam-analyst") in sequential_pairs
    assert ("separation-engineer", "contaminant-removal-analyst") in sequential_pairs
    assert ("contaminant-removal-analyst", "separation-engineer") in sequential_pairs


def test_legacy_single_file_still_loads(tmp_path: Path):
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(yaml.safe_dump({
        "execution_pairs": {"sequential": [["alpha", "beta"]]},
        "subagents": [
            {
                "name": "alpha",
                "description": "alpha desc",
                "system_prompt": "prompt",
                "tool_groups": ["reflection"],
                "routing": {"priority": 2, "phrases": ["alpha"]},
            },
            {
                "name": "beta",
                "description": "beta desc",
                "system_prompt": "prompt",
                "tool_groups": ["reflection"],
                "routing": {"priority": 3, "phrases": ["beta"]},
            },
        ],
    }, sort_keys=False))

    specs = load_subagent_specs(legacy_path)
    execution_pairs = load_execution_pairs(legacy_path)
    routing_rules, _, _, sequential_pairs = load_routing_configuration(legacy_path)

    assert [spec["name"] for spec in specs] == ["alpha", "beta"]
    assert execution_pairs["sequential"] == [["alpha", "beta"]]
    assert [rule["subagent"] for rule in routing_rules] == ["alpha", "beta"]
    assert ("alpha", "beta") in sequential_pairs
