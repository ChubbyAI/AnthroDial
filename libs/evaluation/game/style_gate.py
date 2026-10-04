"""Game-domain style gate.

Thresholds calibrated on 58 real game dialogues vs 192 Qwen3.6-35B-A3B
benchmark dialogues (2026-07): real pass 90%, machine pass 11%.
Key signal: humans send short fragments (median 5 chars, half <=5 chars,
almost no punctuation); LLMs send longer, punctuated, well-formed messages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from libs.evaluation.core.style_gate import compute_style_features, tested_side_messages

# 对话级风格门阈值（按被测角色的全部消息统计）
MAX_MEDIAN_LEN = 8       # 消息长度中位数（字符）
MAX_PUNCT_RATIO = 0.25   # 含正式标点（，。！；：）的消息占比
MIN_SHORT_RATIO = 0.15   # 短消息（<=5 字符）占比下限
MAX_LONG_RATIO = 0.15    # 长消息（>20 字符）占比上限

MIN_MESSAGES = 3         # 消息太少不启用风格门（视为通过）

_PUNCT = "，。！；："


def check_style_gate(messages: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Return (passed, {features, failed_rules})."""
    feats = compute_style_features(messages)
    n = len(messages)
    if n == 0:
        return True, {"features": feats, "failed_rules": [],
                      "note": "no messages, gate skipped"}
    if n < MIN_MESSAGES:
        return True, {"features": feats, "failed_rules": [],
                      "note": "too few messages, gate skipped"}

    punct_ratio = round(
        sum(1 for m in messages if any(c in m for c in _PUNCT)) / n, 4
    )
    feats["punct_ratio"] = punct_ratio

    failed: List[str] = []
    if feats["median_len"] > MAX_MEDIAN_LEN:
        failed.append(
            f"median_len {feats['median_len']:.0f} > {MAX_MEDIAN_LEN}（消息普遍过长，真人游戏私聊中位数约 5 字）")
    if punct_ratio > MAX_PUNCT_RATIO:
        failed.append(
            f"punct_ratio {punct_ratio:.2f} > {MAX_PUNCT_RATIO}（正式标点过多，真人几乎不打标点）")
    if feats["short_ratio"] < MIN_SHORT_RATIO:
        failed.append(
            f"short_ratio {feats['short_ratio']:.2f} < {MIN_SHORT_RATIO}（缺少 <=5 字的碎片短句）")
    if feats["long_ratio"] > MAX_LONG_RATIO:
        failed.append(
            f"long_ratio {feats['long_ratio']:.2f} > {MAX_LONG_RATIO}（>20 字长消息过多）")

    return not failed, {"features": feats, "failed_rules": failed}
