# Feature 4: Externalize Subagent Configs to YAML

## What It Does
Replaces the ~250-line hardcoded `_build_subagents()` function with a YAML configuration file (`subagents.yaml`) and a lightweight loader (~40 lines).

## Problem Solved
Previously, changing a subagent's description, system prompt, tool assignment, or guardrail settings required modifying Python code. Now, domain experts can edit `subagents.yaml` without touching any `.py` files.

## Files Created
- `src/strap/subagents.yaml` — All 8 subagent definitions (name, description, system_prompt, tool_groups, guardrails)

## Files Modified
- `src/strap/agent.py`:
  - Added `_TOOL_GROUP_REGISTRY` mapping (tool group name → getter function)
  - Added `_resolve_tools()` and `_resolve_guardrails()` helper functions
  - Replaced `_build_subagents()` with YAML-loading version
  - Added `yaml` import
- `pyproject.toml`:
  - Added `pyyaml` dependency
  - Added `force-include` for YAML/MD files in wheel builds

## YAML Format
```yaml
- name: safety-analyst
  description: >
    Safety and environmental specialist...
  system_prompt: |
    You are a chemical safety analyst...
  tool_groups:
    - safety_gsk        # resolved via _TOOL_GROUP_REGISTRY
    - safety_pubchem
    - reflection
  guardrails:
    max_tool_calls: 10  # optional, defaults shown
    free_tools: [think]
    synthesis_tools: []
```

## How to Cherry-Pick
```bash
cp src/strap/subagents.yaml <target>/src/strap/
# Apply agent.py diff: _TOOL_GROUP_REGISTRY, _resolve_tools, _resolve_guardrails, new _build_subagents
# Add pyyaml to dependencies
```

## Key Design Decisions
1. **Tool group registry**: Maps string names to getter functions, so YAML stays clean
2. **_THINK_DIRECTIVE auto-appended**: Every system_prompt gets the reflection protocol appended at load time — no duplication in YAML
3. **Guardrails as nested config**: Parsed into `SubagentGuardMiddleware` kwargs at load time
4. **Backward compatible**: `_build_subagents()` still returns `list[SubAgent]` with same signature
