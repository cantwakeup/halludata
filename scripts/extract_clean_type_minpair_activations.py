#!/usr/bin/env python3
"""Extract official-LLaVA activations for clean_type_minpair_v2 rows.

This is a thin clean-v2 entrypoint over extract_subtype_minpair_activations.py.
The underlying extractor already uses the official LLaVA loader and records
z_visual, z_fact_text, and z_counterfact_text for every row.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from extract_subtype_minpair_activations import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
