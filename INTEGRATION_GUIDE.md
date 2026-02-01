# Multi-Agent System Integration Guide

## Branch: `multiagent-1`

This branch contains consolidated multi-agent enhancements including:
- P0-P2 enhancements (review/revision loops, parallel execution, knowledge graph)
- 3-way collaboration (Separation → Literature → TEA)
- Gemini message ordering fixes
- Iteration counter fixes for handoff timing

---

## Part 1: P0-P2 Enhancements

### P0: Review/Revision Loop (Critical)
**Purpose:** Quality gate before TEA analysis - validates separation results and triggers retries if needed.

**Files:**
- `multi_agent_system.py:150-320` - `separation_reviewer_node()`, `SEPARATION_QUALITY_THRESHOLDS`
- `agent_schemas.py:560-607` - `ReviewerFeedback` schema

**Integration Points:**
- Graph routing: Insert `separation_reviewer` node between `collab_separation_agent` → `collab_tea_agent`
- State fields: Add `separation_retry_count`, `reviewer_feedback`, `retry_params` to `MultiAgentState`

### P1: Checkpointing, Parallel Execution, Supervisor
**Files:**
- `multi_agent_system.py:80-150` - `CheckpointerConfig` class
- `multi_agent_system.py:200-280` - `parallel_orchestrator_node()`, `supervisor_decision_node()`
- `agent_schemas.py:611-640` - `SupervisorDecision` schema

### P2: Cross-Session Store & Knowledge Graph
**Files:**
- `multi_agent_system.py:320-400` - `SessionStore`, `PolymerKnowledgeGraph` classes

---

## Part 2: 3-Way Collaboration

**Purpose:** Enable separation planning with literature verification and economic analysis.

**Flow:** Separation → Literature (STRAP-CORE) → TEA → Aggregator

**Router Detection:**
```python
# Triggers on queries with separation + literature + cost keywords
if has_separation_keyword and has_literature_keyword and has_cost_keyword:
    collaboration_specialists=["separation", "literature", "tea_lca"]
```

**Files:**
- `multi_agent_system.py` - Router, agent handoffs, aggregator output
- `test_literature_agent.py` - Test suite

---

## Part 3: Gemini Message Ordering Fixes

**File:** `agent_sql_final_1212_patched.py` (lines ~11554-11640)

**Problem:** `sanitize_messages_for_gemini()` was dropping ToolMessages as "orphaned" when searching backward for matching AIMessages stopped at HumanMessage boundaries.

**Solution:** Two-pass sanitization approach:
```python
# Pass 1: Collect all valid tool_call_ids from AIMessages
valid_tool_call_ids = set()
for msg in msgs:
    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.get('id'):
                valid_tool_call_ids.add(tc.get('id'))

# Pass 2: Build sanitized list with bridge HumanMessages where needed
```

**Key changes:**
- Remove early break on HumanMessage when searching for matching AIMessages
- Insert bridge `HumanMessage("Continue.")` when AIMessage with tool_calls follows another AIMessage
- Ensure messages end with user role for Gemini compatibility

---

## Part 4: Iteration Counter Fixes

**File:** `multi_agent_system.py` (MultiAgentState, agent nodes)

**Problem:** Handoff metrics were recorded on every iteration, not just the final pass.

**Solution:** Added iteration counters to `MultiAgentState`:
```python
sep_iteration_count: int = 0
tea_iteration_count: int = 0
```

Agent nodes now:
- Return `dict` (not `Command`) when tools are pending
- Only record handoff metrics on final pass
- Have max iteration limits (10) to prevent infinite loops

---

## Integration Commands

```bash
# From another worktree
git fetch origin
git merge origin/multiagent-1

# Or cherry-pick specific commits
git cherry-pick f97a9be  # 3-way collaboration
git cherry-pick 6a24b7f  # P0-P2 enhancements
git cherry-pick 3aa75a2  # LangChain tools
git cherry-pick e7f793f  # Integration fixes
```
