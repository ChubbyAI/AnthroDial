"""Per-persona memory file (memory.md).

Layout (Markdown for LLM-friendly self-update):

    # P001 小岚 的记忆

    ## 关于 P002 阿杰
    - 2025-06-18 晚 10 点：他被领导临时加活、没吃饭。
    - 习惯：说话偏短、爱用"啊"。

    ## 关于 P003 ...

The file is parsed by splitting on level-2 headings of the form
"## 关于 <PARTNER_ID> ...". Updates are produced by an LLM call that takes the
current section + new dialogue and outputs a fresh bullet list.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from libs.core.config import get_settings
from libs.llm.client import LLMClient

if TYPE_CHECKING:
    from libs.chat.persona import Persona

LOGGER = logging.getLogger(__name__)

SECTION_RE = re.compile(r"^##\s*关于\s*([A-Za-z0-9_\-]+)", re.MULTILINE)


@dataclass
class Memory:
    persona_id: str
    nickname: str
    sections: Dict[str, List[str]]  # partner_id -> bullet list

    # ------------------------------------------------------------------
    @classmethod
    def _file_path(cls, persona_id: str) -> Path:
        return get_settings().path("personas_dir") / persona_id / "memory.md"

    @classmethod
    def load(cls, persona_id: str, nickname: str = "") -> "Memory":
        path = cls._file_path(persona_id)
        if not path.exists():
            return cls(persona_id=persona_id, nickname=nickname or persona_id, sections={})
        text = path.read_text(encoding="utf-8")
        sections = _parse_sections(text)
        return cls(persona_id=persona_id, nickname=nickname or persona_id, sections=sections)

    def save(self) -> Path:
        path = self._file_path(self.persona_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {self.persona_id} {self.nickname} 的记忆", ""]
        for partner_id in sorted(self.sections):
            bullets = self.sections[partner_id]
            if not bullets:
                continue
            lines.append(f"## 关于 {partner_id}")
            for b in bullets:
                b = b.strip()
                if not b:
                    continue
                if not b.startswith("-"):
                    b = f"- {b}"
                lines.append(b)
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    def section_for(self, partner_id: str) -> str:
        bullets = self.sections.get(partner_id, [])
        if not bullets:
            return f"(暂无与 {partner_id} 的过往记忆，这是第一次/几乎第一次聊。)"
        return "\n".join(b if b.startswith("-") else f"- {b}" for b in bullets)

    def update_after_session(
        self,
        partner_id: str,
        partner_nickname: str,
        dialogue: List[dict],
        llm: LLMClient,
        session_date: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Run an LLM call that merges this session's content into our section."""
        s = get_settings()
        model = model or s.model_for("memory_updater")
        temperature = s.temperature_for("memory_updater", 0.2) if temperature is None else temperature
        max_points = int(s.get("memory", "max_points_per_partner", default=12))

        # session_date 格式为 "YYYY-MM-DD"，由 session 传入。
        if session_date is None:
            from datetime import datetime
            session_date = datetime.now().strftime("%Y-%m-%d")

        existing = self.section_for(partner_id)
        transcript = _flatten_dialogue_for_prompt(dialogue)
        prompt = _build_memory_prompt(
            self_id=self.persona_id,
            self_nickname=self.nickname,
            partner_id=partner_id,
            partner_nickname=partner_nickname,
            existing_bullets=existing,
            transcript=transcript,
            max_points=max_points,
            session_date=session_date,
        )
        try:
            res = llm.chat(
                [{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
            )
            bullets = _parse_bullets(res.content, max_points=max_points)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("memory update failed for %s/%s: %s",
                           self.persona_id, partner_id, exc)
            bullets = self.sections.get(partner_id, [])

        if bullets:
            self.sections[partner_id] = bullets
            self.save()
        return bullets


# ---------------------------------------------------------------------------
def _parse_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)
            sections.setdefault(current, [])
            continue
        if line.startswith("# "):
            current = None
            continue
        if current is None:
            continue
        s = line.strip()
        if s.startswith("-"):
            sections[current].append(s.lstrip("- ").strip())
        elif s:
            sections[current].append(s)
    return sections


def _flatten_dialogue_for_prompt(dialogue: List[dict]) -> str:
    parts: List[str] = []
    for turn in dialogue:
        role = turn.get("role", "?")
        responses = turn.get("response") or []
        for msg in responses:
            content = msg.get("content", "").strip()
            if content:
                parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _build_memory_prompt(
    self_id: str,
    self_nickname: str,
    partner_id: str,
    partner_nickname: str,
    existing_bullets: str,
    transcript: str,
    max_points: int,
    session_date: str = "",
) -> str:
    return (
        f"你是 {self_id} {self_nickname}，需要把刚刚和 {partner_id} {partner_nickname} 的这次聊天，"
        f"整理成一份个人记忆。请用第一人称视角，从我的角度记录**对对方的长期认知和印象**。\n\n"
        f"本次对话发生日期：{session_date}\n\n"
        f"❗ 重要要求：\n"
        f"- 每条记忆必须以 `[日期]` 开头，格式为 `[YYYY-MM-DD]`。\n"
        f"  - 本次新增的信息用 `[{session_date}]` 标记。\n"
        f"  - 旧条目保留其原有日期（如果有的话）；没有日期的旧条目酌情补上估计日期或删除。\n"
        f"- 只记录**有长期价值的信息**：对方的性格印象、爱好、职业近况、关系状态、约定等。\n"
        f"- **不要记录对话流水账**（如'他说了晚安''我去洗漱了''他到地铁了'这类一次性事件）。\n"
        f"- **不要记录对话的结束语或道别**。\n"
        f"- 信息应该对下一次聊天有帮助（比如知道对方最近在忙什么、喜欢什么、需要关心什么）。\n"
        f"- 可以合并/去重旧条目；过时的一次性信息可以删除。\n\n"
        f"【我已有的与对方相关的记忆条目（可保留可改写可去重可删除）】\n{existing_bullets}\n\n"
        f"【本次聊天原文】\n{transcript}\n\n"
        f"请输出一个 Markdown bullet 列表 (每条以 '- [YYYY-MM-DD] ' 开头)，最多 {max_points} 条，"
        f"按重要性排序。条目要简短、概括性、有事实信息。不要输出任何额外说明或标题，"
        f"只输出 bullet 列表本身。"
    )


def _parse_bullets(text: str, max_points: int) -> List[str]:
    bullets: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("-") or s.startswith("*"):
            bullets.append(s.lstrip("-* ").strip())
    if not bullets:
        # fallback: treat each non-empty line as a bullet
        bullets = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bullets[:max_points]
