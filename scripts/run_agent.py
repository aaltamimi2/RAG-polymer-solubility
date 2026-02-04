#!/usr/bin/env python
"""Entry point to run the STRAP agent without pip install."""

import sys
from pathlib import Path

# Add src/ to path so strap package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from strap.agent import main  # noqa: E402

if __name__ == "__main__":
    main()
