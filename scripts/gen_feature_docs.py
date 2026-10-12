"""Regenerate docs/features.md from the registry:  python scripts/gen_feature_docs.py"""

from pathlib import Path

import brepgraph  # noqa: F401  (registers built-ins)
from brepgraph.registry import describe

Path(__file__).parent.parent.joinpath("docs", "features.md").write_text(describe(), encoding="utf-8")
