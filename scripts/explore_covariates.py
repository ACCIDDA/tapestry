#!/usr/bin/env python3
"""Build or serve the local surveillance-data explorer from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influpaintx.explorer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
