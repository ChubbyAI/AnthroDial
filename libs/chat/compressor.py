"""Long-context compressor.

When the running message list grows beyond ``trigger_token_threshold``, fold
the oldest messages (everything older than ``keep_recent_turns`` turns) into a
single summary message that gets prepended back to the history.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from libs.core.config import get_settings
from libs.llm.client import LLMClient

LOGGER = logging.getLogger(__name__)


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Cheap token estimate: ~ len(text) / 2 for Chinese-heavy content.

    We optionally use tiktoken when available; fall back to char heuristic.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        n = 0
        for m in messages:
            n += len(enc.encode(m.get("content", "") or ""))
            n += 4  # role/separator overhead
        return n
    except Exception:
        return sum(max(1, len(m.get("content", "")) // 2) for m in messages)


class Compressor:
    """Rolling summary compressor + conversation memory folder."""

    SUMMARY_PREFIX = "[早期对话摘要] "

    def __init__(self, llm: LLMClient, model: Optional[str] = None,
                 temperature: Optional[float] = None) -> None:
        s = get_settings()
        self.llm = llm
        self.model = model or s.model_for("compressor")
        self.temperature = (s.temperature_for("compressor", 0.2)
                            if temperature is None else temperature)
        self.fold_max_chars = int(s.get("compression", "fold_max_chars", default=300))

    # ------------------------------------------------------------------
    def fold_conversation(
        self,
        existing_memory: str,
        turns_to_fold: List[Dict[str, Any]],
    ) -> str:
        """Compress dialogue turns into a concise conversation memory summary.

        Used by the single-turn architecture: every N turns, fold the history
        into a compact memory block that gets injected into subsequent user
        snapshots.
        """
        transcript_lines: List[str] = []
        for turn in turns_to_fold:
            role = turn.get("role", "?")
            for msg in (turn.get("response") or []):
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                ts = (msg.get("timestamp") or "").strip()
                if ts:
                    transcript_lines.append(f"[{role}][{ts}] {content}")
                else:
                    transcript_lines.append(f"[{role}] {content}")
        if not transcript_lines:
            return existing_memory

        transcript = "\n".join(transcript_lines)
        user_msg = (
            f"请把下面的对话历史压缩成一段简洁中文摘要（不超过 {self.fold_max_chars} 字），"
            "保留：双方关键事实、情绪变化、约定、未结话题、话题走向，便于后续接着聊。"
            "只输出摘要正文，不要标题或额外说明。\n\n"
            + (f"【已有记忆】\n{existing_memory}\n\n" if existing_memory else "")
            + f"【待压缩对话】\n{transcript}"
        )
        try:
            res = self.llm.chat(
                [{"role": "user", "content": user_msg}],
                model=self.model,
                temperature=self.temperature,
            )
            return res.content.strip()
        except Exception as exc:
            LOGGER.warning("fold_conversation failed: %s", exc)
            if existing_memory:
                return existing_memory + "\n" + transcript[:200]
            return transcript[:400]

