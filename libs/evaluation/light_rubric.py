"""Scoring rubric: mirrors the 6 principles in §3 of the labelling spec."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from libs.core.config import get_settings


@dataclass
class Dimension:
    id: str
    name: str
    description: str


# Default descriptions; can be overridden via configs if you want.
_DEFAULT_DESCRIPTIONS: Dict[str, str] = {
    "concise": "回复是否简短自然、像微信私聊；避免长段、说明文、客服话术。",
    "knowledge_bounded": "在人设的 weak_topics 范围内，回复是否承认不懂或克制表达，不装专家、不胡说。",
    "proactive": "是否在合适的时机主动推进话题：追问、共情、分享相关经历、提议或引出下一轮。",
    "wechat_feel": "是否有微信聊天的口语感和语气词使用，避免书面化连接词。",
    "segment": "是否合理分段回复（该拆则拆，该合则合），分段后每条独立可读。",
    "consistency": "回复是否与人设、场景、上下文一致；前后事实/人设不矛盾；不暴露 AI 身份。",
}


def load_dimensions() -> List[Dimension]:
    s = get_settings()
    raw = s.get("evaluation", "dimensions", default=[]) or []
    out: List[Dimension] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        did = str(item.get("id"))
        if not did:
            continue
        out.append(Dimension(
            id=did,
            name=str(item.get("name") or did),
            description=_DEFAULT_DESCRIPTIONS.get(did, ""),
        ))
    return out


def score_range() -> tuple:
    s = get_settings()
    return int(s.get("evaluation", "score_min", default=1)), \
        int(s.get("evaluation", "score_max", default=5))
