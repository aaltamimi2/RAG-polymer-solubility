"""Console entrypoint for running DISSOLVE through the Claude SDK harness."""

from __future__ import annotations

import sys


def _argv_with_claude_harness(argv: list[str]) -> list[str]:
    """Return argv with ``--harness claude_sdk`` inserted unless already set."""
    if "--harness" in argv:
        return argv
    return [argv[0], "--harness", "claude_sdk", *argv[1:]]


def main() -> None:
    """Run the shared DISSOLVE CLI with the Claude SDK harness selected."""
    from strap.agent import main as dissolve_main

    original_argv = sys.argv
    try:
        sys.argv = _argv_with_claude_harness(list(sys.argv))
        dissolve_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
