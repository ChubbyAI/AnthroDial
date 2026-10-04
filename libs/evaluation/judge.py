"""Compatibility shim: old single-module imports default to the game judge."""
from __future__ import annotations

from libs.evaluation.core.judge import (  # noqa: F401
    BaseBenchmarkJudge,
    DimResult,
    HolisticScoreCard,
    L0Violation,
    TurnScoreCard,
)
from libs.evaluation.game.judge import GameBenchmarkJudge as BenchmarkJudge  # noqa: F401
