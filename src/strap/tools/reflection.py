"""Think tool for subagent reflection and self-assessment.

A zero-side-effect tool that forces the LLM to pause after tool calls and
evaluate its progress: Are findings grounded in tool results? Should it
search more or synthesize? Prevents ungrounded claims from LLM parametric
knowledge.

Based on the deep_research pattern from langchain-ai/deepagents.
"""

from __future__ import annotations

from ._helpers import safe_tool_wrapper


@safe_tool_wrapper
def think(reflection: str) -> str:
    """Pause and reflect on your research progress before deciding next steps.

    Use this tool AFTER each domain tool call to assess your findings.
    This creates a deliberate checkpoint in your workflow for quality control.

    Your reflection MUST address:
    1. What concrete data did I just obtain from the tool? (cite specific numbers)
    2. Is this finding grounded in tool output, or am I relying on general knowledge?
    3. What critical information is still missing to answer the query?
    4. Should I call another tool, or do I have enough to synthesize a final answer?

    Args:
        reflection: Your detailed assessment of current findings, data gaps,
            and whether to continue tool use or synthesize.

    Returns:
        Confirmation that reflection was recorded.
    """
    return f"Reflection recorded. Continue with your next action based on this assessment."
