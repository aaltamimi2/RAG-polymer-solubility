"""Claude Agent SDK harness for DISSOLVE.

The package is intentionally import-safe without ``claude-agent-sdk`` installed.
Runtime SDK imports happen behind explicit guards in ``options`` and
``mcp_server`` so the default LangChain harness remains unaffected.
"""

from __future__ import annotations

from .models import ClaudeModelSelection, resolve_claude_model_selection
from .runner import ClaudeSdkRunner, ClaudeSdkTurnResult

__all__ = [
    "ClaudeModelSelection",
    "ClaudeSdkRunner",
    "ClaudeSdkTurnResult",
    "resolve_claude_model_selection",
]
