"""Offline evaluation: score existing dialogue JSON files saved by the chat
session (or by external sources sharing the same schema).

Each input file is expected to follow the schema produced by
``src.chat.session.save_dialogue``::

    {
      "persona1_id": "P001",
      "persona2_id": "P002",
      "scenario_id": "A0002_B0001_C0001",
      "dialogue": [...],
      "meta": {...}   # optional
    }

The evaluator loads the matching persona / scenario YAMLs (seed first, then
managed dir) and runs the same Judge used by ``eval_model``. Aggregated
results reuse :class:`ModelReport` so ``write_summary`` works on the report.
"""
from __future__ import annotations

import glob as glob_mod
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.core.config import REPO_ROOT
from libs.evaluation.eval_model import ModelReport, _aggregate
from libs.evaluation.light_judge import Judge

LOGGER = logging.getLogger(__name__)


def collect_paths(pattern: str) -> List[Path]:
    """Resolve a glob (relative or absolute) into a list of existing files."""
    p = Path(pattern)
    if not p.is_absolute():
        p = REPO_ROOT / pattern
    matches = sorted(glob_mod.glob(str(p)))
    return [Path(m) for m in matches if Path(m).is_file()]


def _load_dialogue_payload(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a dialogue JSON object")
    return data


def _resolve_ids(payload: Dict[str, Any]) -> Dict[str, str]:
    meta = payload.get("meta") or {}
    return {
        "persona1_id": str(payload.get("persona1_id") or meta.get("persona1_id") or "").strip(),
        "persona2_id": str(payload.get("persona2_id") or meta.get("persona2_id") or "").strip(),
        "scenario_id": str(payload.get("scenario_id") or meta.get("scenario_id") or "").strip(),
    }


def evaluate_dialogues(paths: List[Path],
                       label: str = "offline",
                       judge: Optional[Judge] = None) -> ModelReport:
    """Score a list of saved dialogue files. Returns one aggregated report."""
    judge = judge or Judge()
    cards = []
    for p in paths:
        try:
            payload = _load_dialogue_payload(p)
        except Exception as exc:
            LOGGER.warning("skip %s: %s", p, exc)
            continue
        ids = _resolve_ids(payload)
        if not (ids["persona1_id"] and ids["persona2_id"] and ids["scenario_id"]):
            LOGGER.warning("skip %s: missing persona/scenario ids", p)
            continue
        try:
            persona1 = Persona.load(ids["persona1_id"])
            persona2 = Persona.load(ids["persona2_id"])
            scenario = Scenario.load(ids["scenario_id"])
        except FileNotFoundError as exc:
            LOGGER.warning("skip %s: %s", p, exc)
            continue
        dialogue = payload.get("dialogue") or []
        if not isinstance(dialogue, list) or not dialogue:
            LOGGER.warning("skip %s: empty dialogue", p)
            continue
        # 尽量从对话 meta 里读实际生成该对话的 model，以便 trace 按被评模型分层。
        meta = payload.get("meta") or {}
        dialogue_model = meta.get("model") or label
        try:
            card = judge.score(dialogue=dialogue,
                               persona1=persona1,
                               persona2=persona2,
                               scenario=scenario,
                               tag=f"offline_{label}",
                               mode="offline",
                               model=dialogue_model)
        except Exception as exc:
            LOGGER.exception("judge failed on %s: %s", p, exc)
            continue
        cards.append(card)

    return _aggregate(label, cards)
