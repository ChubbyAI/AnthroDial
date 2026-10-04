"""Shared helpers for deterministic style-gate modules."""
from __future__ import annotations

import statistics
from typing import Any, Dict, List


def compute_style_features(messages: List[str]) -> Dict[str, float]:
    n = len(messages)
    if n == 0:
        return {}
    lens = [len(m) for m in messages]
    return {
        "n_messages": float(n),
        "median_len": float(statistics.median(lens)),
        "mean_len": round(sum(lens) / n, 2),
        "short_ratio": round(sum(1 for l in lens if l <= 5) / n, 4),
        "long_ratio": round(sum(1 for l in lens if l > 20) / n, 4),
    }


def tested_side_messages(dialogue: List[Dict[str, Any]], tested_id: str) -> List[str]:
    msgs: List[str] = []
    for turn in dialogue:
        if turn.get("role") != tested_id:
            continue
        resp = turn.get("response")
        if isinstance(resp, list):
            for r in resp:
                if isinstance(r, dict) and r.get("content"):
                    msgs.append(str(r["content"]))
        elif turn.get("content"):
            msgs.append(str(turn["content"]))
    return msgs
