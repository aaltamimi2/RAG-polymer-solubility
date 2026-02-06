# Feature 3: Progressive Skill Loading

## What It Does
Splits the monolithic SYSTEM_PROMPT into three tiers of context using the deepagents `memory` and `skills` primitives:

| Tier | Mechanism | Loaded | Content |
|------|-----------|--------|---------|
| Memory | `AGENTS.md` | Always | Agent identity, available data, guidelines |
| System prompt | `system_prompt=` | Always | Dynamic routing table (generated from code) |
| Skills | `skills/*/SKILL.md` | On-demand | Workflow instructions (multi-agent, data-lookup, separation-design) |

## Problem Solved
Previously, the full system prompt (~3K tokens) was always loaded, including guidelines that only apply to certain query types. Skills are loaded only when the LLM determines they're relevant, reducing context waste.

## Files Created
- `src/strap/AGENTS.md` — Agent identity (always loaded)
- `src/strap/skills/multi-agent-workflow/SKILL.md` — Multi-agent coordination workflow
- `src/strap/skills/data-lookup/SKILL.md` — Direct data query workflow
- `src/strap/skills/separation-design/SKILL.md` — Separation process design workflow

## Files Modified
- `src/strap/agent.py`:
  - Added `FilesystemBackend` import and `_PACKAGE_DIR`
  - Moved static identity to `AGENTS.md`, keeping only dynamic routing table in `system_prompt`
  - Updated `create_dissolve_agent()` with `memory=`, `skills=`, `backend=`

## How to Cherry-Pick
```bash
# Copy these files/directories:
cp src/strap/AGENTS.md <target>/src/strap/
cp -r src/strap/skills/ <target>/src/strap/
# Apply the agent.py diff (FilesystemBackend import + create_dissolve_agent changes)
```

## Key Design Decisions
1. **Routing table stays in system_prompt** — it's dynamically generated from Python code, can't be static markdown
2. **Backend root = package directory** — `FilesystemBackend(root_dir=str(_PACKAGE_DIR))` resolves paths relative to `src/strap/`
3. **Skills are orchestrator-level** — subagents don't use skills (they have their own system prompts)
