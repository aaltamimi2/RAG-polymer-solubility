# Deep Agents Examples — Architecture Analysis for DISSOLVE

Comparison of [langchain-ai/deepagents/examples](https://github.com/langchain-ai/deepagents/tree/master/examples) against the DISSOLVE multi-agent system.

---

## 1. content-builder-agent

**What it does:** Content writing agent (blog posts, LinkedIn, tweets) with cover image generation.

### Architecture Pattern: Filesystem-as-Configuration

Three filesystem primitives replace hardcoded Python:

| Primitive | File | Purpose |
|-----------|------|---------|
| **Memory** | `AGENTS.md` | Brand voice, style guide — always loaded |
| **Skills** | `skills/*/SKILL.md` | Task-specific workflows — loaded on demand |
| **Subagents** | `subagents.yaml` | Agent definitions externalized to YAML |

- `AGENTS.md` = persistent identity (equivalent to system prompt)
- `SKILL.md` files define structured workflows with checklists, output schemas, and tool-call templates (e.g., `task(subagent_type="researcher", description="...")`)
- Subagents defined in YAML with name, description, model, tools, and system_prompt fields
- `FilesystemBackend` provides `write_file`/`read_file` tools automatically

### Research-First Delegation

Every content creation task starts with `task(subagent_type="researcher", ...)` — the researcher saves findings to a file, then the orchestrator reads it before writing. This enforces **grounded generation**.

### Key Takeaway for DISSOLVE

DISSOLVE hardcodes all subagent definitions in `_build_subagents()` (~300 lines of Python) and system prompts as string literals. Externalizing to:
- `AGENTS.md` → orchestrator identity, domain description, delegation policy
- `skills/*.md` → per-workflow instructions (e.g., "separation-sequence-design", "safety-assessment", "tea-cost-analysis")
- `subagents.yaml` → subagent configs (name, tools, guardrail params, system prompt)

would make the system **easier to iterate on without touching Python code** and enable non-developers (domain experts) to modify agent behavior.

---

## 2. deep_research

**What it does:** Multi-step research agent with orchestrator → parallel sub-agents → synthesis.

### Architecture Pattern: Structured Research Loop

Five-step workflow enforced via prompt:
1. **Plan** — `write_todos` to decompose the research question
2. **Save** — `write_file("/research_request.md")` for traceability
3. **Delegate** — `task()` to sub-agents (1–3 parallel, max 3 rounds)
4. **Synthesize** — consolidate citations across sub-agents
5. **Write** — `write_file("/final_report.md")` with structured sections

### Think Tool for Reflection

```python
@tool
def think_tool(reflection: str) -> str:
    """Strategic reflection on research progress and decision-making."""
    return f"Reflection recorded: {reflection}"
```

A zero-side-effect tool that forces the LLM to pause and assess:
- What did I find? What's missing? Should I search more or stop?
- Used **after every search** — prevents blind tool-call chains

### Prompt-Embedded Stop Conditions

Hard limits wired directly into the sub-agent prompt (not just middleware):
- Simple queries: 2–3 search calls max
- Complex queries: 5 search calls max
- Stop when: 3+ relevant sources found, or last 2 searches returned similar info

This is **complementary** to middleware guardrails — the LLM self-regulates before the hard cap fires.

### Parallelism Strategy

```
DEFAULT: 1 sub-agent (most queries)
PARALLEL: Only for explicit comparisons or geographically separated aspects
LIMIT: max 3 concurrent, max 3 rounds
```

The key insight: **bias toward single sub-agent**. Parallelism is reserved for genuinely independent dimensions (Company A vs B, Region X vs Y). This avoids DISSOLVE's Trace 2 problem where parallelism inflated tokens 67% for minimal benefit.

### Citation Consolidation

Sub-agents cite inline `[1], [2], [3]`; orchestrator re-numbers across all sub-agents into a unified `### Sources` section. Prevents duplicate/conflicting citations in multi-agent synthesis.

### Key Takeaways for DISSOLVE

1. **Think tool** — DISSOLVE's safety-analyst answered entirely from parametric knowledge in Trace 2 (0 tool calls). A think tool would force self-assessment: "Am I using tools or guessing?"
2. **Prompt-level stop conditions** — DISSOLVE relies solely on middleware guardrails (`max_tool_calls=10`). Adding prompt-embedded heuristics ("stop after 3 successful DB queries") would make limits more nuanced and LLM-driven
3. **File-based intermediate state** — DISSOLVE passes context through task descriptions (string concatenation). Using `write_file`/`read_file` for inter-agent state would be cleaner for 3+ agent chains (Trace 5's sequential sep → tea → safety)
4. **Parallelism restraint** — Default to 1 sub-agent, only parallelize for genuinely independent dimensions. DISSOLVE's `PARALLEL_PAIRS` allows parallelism too eagerly (safety + tea-lca are often interdependent)

---

## 3. downloading_agents

**What it does:** Shows that agents can be distributed as zip files — just a folder with markdown.

### Architecture Pattern: Agents-as-Folders

```
.deepagents/
├── AGENTS.md           # Identity + instructions (always loaded)
└── skills/
    ├── blog-post/SKILL.md
    └── social-media/SKILL.md
```

No Python code. No configuration files. Just markdown defining behavior.

### Key Takeaway for DISSOLVE

This is the extreme version of the content-builder pattern. While DISSOLVE is too tool-heavy to go fully no-code, the principle applies: **domain logic should live in declarative files, not procedural code**. Separation of "what the agent knows" (markdown) from "what the agent can do" (Python tools) makes the system more maintainable.

---

## 4. ralph_mode

**What it does:** Autonomous looping — run the same task repeatedly with fresh context, using the filesystem as persistent memory.

### Architecture Pattern: Stateless Iteration

```python
while True:
    prompt = f"Iteration {i}. Your previous work is in the filesystem. Check what exists and keep building. TASK: {task}"
    await execute_task(prompt, agent, ...)
```

- Each iteration starts with **fresh context** (no conversation history)
- Git + filesystem = memory across iterations
- Declarative task ("Build X") not imperative ("First do A, then B")

### Key Takeaway for DISSOLVE

This pattern would benefit **iterative refinement workflows** that currently require multiple manual runs:
- "Propose 3 separation schemes" → (iteration 1)
- "Evaluate the cost of each scheme" → (iteration 2, reads previous output)
- "Optimize the cheapest scheme further" → (iteration 3)

Currently DISSOLVE handles this as a single 311-second, 628K-token invocation (Trace 5). A ralph-mode loop could break this into smaller, cheaper iterations with intermediate checkpoints, allowing human review between rounds.

The fresh-context-per-loop approach also **naturally prevents context window exhaustion** — a real problem for DISSOLVE's complex queries (1,573K tokens in Trace 4).

---

## 5. text-to-sql-agent

**What it does:** Natural language → SQL query agent with progressive skill disclosure.

### Architecture Pattern: Progressive Disclosure

```python
agent = create_deep_agent(
    memory=["./AGENTS.md"],     # Always loaded: identity, safety rules, guidelines
    skills=["./skills/"],       # On-demand: schema-exploration, query-writing
    tools=sql_tools,            # Always available: list_tables, get_schema, execute_query
    subagents=[],               # None needed for focused domain
)
```

- `AGENTS.md` contains **always-active** context: role, safety rules (READ-ONLY), general guidelines
- `skills/schema-exploration/SKILL.md` is loaded **only when exploring** database structure
- `skills/query-writing/SKILL.md` is loaded **only when writing** SQL queries
- The LLM decides when to load each skill based on the task

### Planning with write_todos

For complex queries, the agent creates a TODO list before executing:
1. Identify tables needed
2. Map relationships (foreign keys)
3. Plan JOIN structure
4. Execute and verify

### Safety via Identity

Safety constraints (no INSERT/UPDATE/DELETE/DROP) are embedded in the agent's identity (`AGENTS.md`), not enforced via middleware. The agent "believes" it's read-only.

### Key Takeaways for DISSOLVE

1. **Progressive disclosure** — DISSOLVE loads ALL 9 core tools + ALL subagent definitions into every orchestrator call. With 82 tools across 8 subagents, this means massive system prompts. Loading subagent-specific skills on-demand would reduce token waste significantly
2. **write_todos as first-class planning** — DISSOLVE's subagents sometimes jump straight into tool calls without planning. Requiring a `write_todos` step (like text-to-sql does for complex queries) would improve tool-call efficiency
3. **No-subagent option** — Not every query needs subagent delegation. DISSOLVE's orchestrator already has 9 core tools but the routing middleware sometimes delegates unnecessarily. The text-to-sql pattern shows that a well-equipped single agent with skills can handle focused domains without subagents

---

## Consolidated Recommendations for DISSOLVE

### High Impact — Adopt Now

| # | Feature | Source | DISSOLVE Gap | Implementation |
|---|---------|--------|--------------|----------------|
| 1 | **Think tool** | deep_research | Subagents make ungrounded claims (Trace 2: safety-analyst, 0 tools) | Add `think_tool` to every subagent; prompt requires reflection after each tool call |
| 2 | **Prompt-embedded stop conditions** | deep_research | Hard caps only (middleware), no soft LLM-driven heuristics | Add "stop when: 3+ DB queries succeeded, selectivity calculated, conditions identified" to subagent prompts |
| 3 | **Progressive skill loading** | text-to-sql, content-builder | All 8 subagent descriptions always in system prompt (~2K tokens each = 16K wasted) | Use `memory=` for core identity, `skills=` for subagent-specific workflows loaded on demand |

### Medium Impact — Next Iteration

| # | Feature | Source | DISSOLVE Gap | Implementation |
|---|---------|--------|--------------|----------------|
| 4 | **Externalize configs to markdown/YAML** | content-builder | `_build_subagents()` is 300+ lines of hardcoded Python | Move subagent definitions to `subagents.yaml`, system prompts to `AGENTS.md` |
| 5 | **File-based inter-agent state** | deep_research | Sequential chains pass context via string concatenation in task descriptions | Use `write_file`/`read_file` for intermediate results in 3+ agent chains |
| 6 | **Parallelism restraint** | deep_research | `PARALLEL_PAIRS` allows eager parallelism | Default to sequential; only parallelize for genuinely independent dimensions |

### Exploratory — Evaluate for Complex Workflows

| # | Feature | Source | DISSOLVE Gap | Implementation |
|---|---------|--------|--------------|----------------|
| 7 | **Ralph-mode iteration** | ralph_mode | Complex queries exhaust context (1,573K tokens, Trace 4) | Iterative loop with file checkpoints for multi-scheme queries |
| 8 | **Citation/source tracking** | deep_research | Orchestrator synthesis doesn't track which claims came from which tools/data | Inline citation format `[source: tool_name]` in subagent responses |
| 9 | **Research-first delegation** | content-builder | Some subagents skip tool use entirely | Enforce "research before answering" pattern in subagent prompts |

### Architecture Comparison Matrix

| Feature | DISSOLVE (Current) | deep_research | content-builder | text-to-sql |
|---------|-------------------|---------------|-----------------|-------------|
| Routing | Keyword scoring + middleware injection | N/A (single subagent type) | N/A (single subagent type) | N/A (no subagents) |
| Guardrails | Middleware (tool limits, token budget, synthesis injection, truncation) | Prompt-embedded limits + tool budgets | None explicit | Safety rules in identity |
| State passing | Task description strings | Files (`/research_request.md`, `/final_report.md`) | Files (research → content) | write_todos |
| Parallelism | PARALLEL_PAIRS frozenset lookup | Max 3 concurrent, bias toward single | None (single subagent) | N/A |
| Reflection | None | think_tool after every search | None | None |
| Config format | Python code | Python + prompt templates | Markdown + YAML | Markdown + skills |
| Skill loading | All-at-once | All-at-once | On-demand (SkillsMiddleware) | On-demand (SkillsMiddleware) |
| Tools | 82 across 8 subagents | 2 (tavily_search, think_tool) | 3 (web_search, generate_cover, generate_social_image) | 4 (SQL toolkit) |
