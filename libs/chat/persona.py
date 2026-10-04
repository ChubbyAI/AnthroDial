"""Persona card I/O.

A persona card is a YAML file matching the schema in §4.1 of
``docs/ChatBot标注要求.md``. We render it into a natural-language paragraph that
will be embedded into the agent's system prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from libs.core.config import get_settings


# Display labels for known fields. Anything else gets dumped verbatim.
FIELD_LABELS: List[tuple] = [
    ("nickname", "昵称"),
    ("gender", "性别"),
    ("age", "年龄"),
    ("hometown", "籍贯"),
    ("city", "现居城市"),
    ("education", "学历"),
    ("school", "毕业院校"),
    ("major", "专业"),
    ("occupation", "职业"),
    ("industry", "行业"),
    ("years_of_work", "工作年限"),
    ("family", "家庭结构"),
    ("relationship_status", "感情状态"),
    ("personality_keywords", "性格关键词"),
    ("social_style", "社交风格"),
    ("hobbies", "兴趣爱好"),
    ("schedule", "作息习惯"),
    ("food_preference", "饮食偏好"),
    ("values", "价值观倾向"),
    ("catchphrases", "口头禅/常用语气词"),
    ("good_topics", "擅长话题"),
    ("weak_topics", "不擅长的话题"),
    ("emotion_style", "情绪表达方式"),
    ("reply_habit", "回复习惯"),
    ("punctuation_style", "标点符号习惯"),
    ("mbti", "MBTI"),
]


@dataclass
class Persona:
    persona_id: str
    nickname: str
    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path | str) -> "Persona":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        pid = str(data.get("persona_id") or path.stem)
        nick = str(data.get("nickname") or pid)
        return cls(persona_id=pid, nickname=nick, raw=data)

    @classmethod
    def load(cls, persona_id: str) -> "Persona":
        s = get_settings()
        candidates = [
            s.path("personas_dir") / persona_id / "card.yaml",
        ]
        try:
            seeds_personas = s.path("seeds_dir") / "personas"
            candidates.append(seeds_personas / f"{persona_id}.yaml")
            candidates.append(seeds_personas / persona_id / "card.yaml")
        except KeyError:
            pass
        for c in candidates:
            if c.exists():
                return cls.from_yaml(c)
        raise FileNotFoundError(f"Persona {persona_id} not found in {candidates}")

    def save(self, target_dir: Optional[Path] = None) -> Path:
        s = get_settings()
        base = Path(target_dir) if target_dir else (s.path("personas_dir") / self.persona_id)
        base.mkdir(parents=True, exist_ok=True)
        out = base / "card.yaml"
        with open(out, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.raw, fh, allow_unicode=True, sort_keys=False)
        return out

    # ------------------------------------------------------------------
    def to_system_prompt(self) -> str:
        lines: List[str] = [
            f"你正在扮演一位真实的人，请彻底进入角色，不要暴露自己是 AI。",
            f"你的身份编号: {self.persona_id}",
            f"你的称呼: {self.nickname}",
            "",
            "## 你的人物设定",
        ]
        for key, label in FIELD_LABELS:
            if key in self.raw and self.raw[key] not in (None, "", []):
                value = self.raw[key]
                if isinstance(value, list):
                    value = "、".join(str(v) for v in value)
                lines.append(f"- {label}: {value}")
        # 兜底：把没在字段表里的字段也带上
        known = {k for k, _ in FIELD_LABELS} | {"persona_id", "nickname"}
        extras = {k: v for k, v in self.raw.items() if k not in known}
        if extras:
            lines.append("")
            lines.append("## 其它设定")
            for k, v in extras.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)
