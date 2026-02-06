# Data Lookup

Use this skill for simple data queries that don't need subagent delegation.

## When to Use Your Direct Tools

- "List available polymers/solvents" → list_available_polymers / list_available_solvents
- "What is the boiling point of X?" → get_solvent_properties
- "Rank solvents by LogP" → rank_solvents_by_property
- "Show me the schema" → list_tables + describe_table
- "Run this SQL" → query_database or validate_and_query

## Workflow

1. Identify which core tool answers the question directly
2. Call it ONCE with the right parameters
3. Format the result clearly and respond

Do NOT delegate to a subagent for simple lookups — that wastes time and tokens.
