# -*- coding: utf-8 -*-
"""Path setup for the R10-F (arm API + safety) tests.

Run:
    python -m pytest tests/r10_f -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT / "src", ROOT / "simulation", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))
