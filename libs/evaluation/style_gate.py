"""Compatibility shim: old single-module imports default to the game style gate."""
from __future__ import annotations

from libs.evaluation.game.style_gate import (  # noqa: F401
    check_style_gate,
    compute_style_features,
    tested_side_messages,
)
