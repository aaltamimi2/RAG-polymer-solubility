# LLM-Based Semantic Routing Classifier

## What changed

We replaced brittle regex/keyword expansion with a single Gemini 3 Flash call that classifies user intent into 1–3 subagents before the orchestrator sees the query.

### Previous approach: more regex

When natural phrasings like "dissolve", "purify", "all possible sequences for separating" missed the separation-engineer, we patched `ROUTING_RULES` with wider regexes and extra low_stems:

```python
# v7 additions (now reverted)
r"sequenc\w*\s+(?:\w+\s+)*separat",
r"all\s+(?:possible\s+)?sequenc",
"mixed stream", "sequenc",  # low_stems
```

This is a losing game — every new user phrasing requires a new pattern, and broad stems cause false positives (e.g. "sequenc" matching DNA sequence queries).

### New approach: LLM classifier + keyword fallback

```
User query
    │
    ▼
┌─────────────────────────┐
│  Gemini 3 Flash classifier │  ~150ms, ~200 tokens
│  (SystemMessage with     │
│   subagent descriptions) │
└────────────┬────────────┘
             │
        JSON response:
        {"subagents": ["separation-engineer", "safety-analyst"],
         "confidence": "HIGH"}
             │
             ▼
    Map names → ROUTING_RULES entries
             │
             ▼
   _build_hint_from_matches()
             │
             ▼
   Advisory hint appended to system prompt
             │
             ▼
┌─────────────────────────┐
│     Orchestrator LLM     │  Gemini 2.5 Pro
│  (sees hint, decides     │
│   whether to delegate)   │
└─────────────────────────┘
```

If the Flash call fails or `classifier_model=None`, the keyword matcher runs as fallback — same behavior as before.

## Why three LLM layers?

```
          ┌──────────────────┐
  Query → │ 1. ROUTING       │ Gemini 3 Flash  (~0.2s, ~200 tok)
          │    classifier    │ "WHO should handle this?"
          └───────┬──────────┘
                  │ advisory hint
                  ▼
          ┌──────────────────┐
          │ 2. ORCHESTRATOR  │ Gemini 2.5 Pro  (main model)
          │    + subagents   │ "DO the work"
          └───────┬──────────┘
                  │ draft response
                  ▼
          ┌──────────────────┐
          │ 3. OUTPUT        │ Gemini 3 Flash  (~0.3s, ~300 tok)
          │    verifier      │ "IS this correct?"
          └──────────────────┘
```

**Why can't the orchestrator do all three?**

1. **Routing is a pre-decision.** The orchestrator's system prompt already contains the routing table, tool descriptions, and delegation policy — thousands of tokens of context. Adding "also classify this query's intent before you do anything" makes it do two cognitive tasks at once. In practice, the orchestrator often skips delegation and tries to answer directly with its own tools, especially for ambiguous phrasings. A dedicated classifier with a focused 350-token prompt reliably identifies intent before the orchestrator's attention is split across 15+ tools.

2. **Verification is a post-decision.** The orchestrator cannot objectively audit its own output in the same generation — it has already committed to its reasoning chain. A separate Flash call with a fresh context and an adversarial prompt ("find unsupported claims, contradictions, missing caveats") catches errors that self-reflection within the same context window misses. This is the same principle as code review: the author cannot effectively review their own work.

3. **Cost is negligible.** Both the classifier and verifier use Gemini 3 Flash, which is ~20x cheaper than the Pro orchestrator. Together they add ~$0.001 and ~0.5s per query — trivial compared to the orchestrator's multi-tool, multi-subagent runs that take 30–120s.

4. **Separation of concerns.** Each layer has one job with a minimal, focused prompt. The classifier doesn't need to understand tools or delegation syntax — just intent. The verifier doesn't need to understand routing — just scientific accuracy. This makes each component independently testable and updatable.

## Key design decisions

| Decision | Rationale |
|---|---|
| Reuse same Flash instance for classifier + verifier | No extra init cost, same API key/quota |
| Keywords as fallback, not replacement | LLM handles semantic understanding; keywords handle offline/failure mode |
| Classifier returns `list[dict]` not hint string | Both LLM and keyword paths feed into shared `_build_hint_from_matches()` |
| Class-based middleware (`RoutingMiddleware`) | Holds `classifier_model` reference; matches `OutputVerifierMiddleware` pattern |
| Reverted regex expansions | LLM handles "dissolve", "purify", "all possible sequences" semantically |
