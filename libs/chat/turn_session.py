"""一对一严格轮流 session（turn_taking baseline 机制）。

与 :class:`ChatSession` 的单草稿调度机制相比：

- **严格轮流**：一方把消息真正发出（append 进 dialogue）之后，另一方才会被
  叫去生成回复；不存在「双方并行决策 + 虚拟时间戳竞争」。
- **无草稿缓存**：``prev_draft`` 恒为 ``None``，因此没有抢发改写、没有连发
  （self-followup）、没有动态连发延迟表。
- 复用父类的全部安全语义与输出格式：
  - ``_decide``：空 content → ``alive=False`` + ``llm_failures``（降级终止）、
    复读防御、``min_send_interval_ms`` 下限钳制。
  - ``_emit`` / ``_wrap_opener`` / ``_dump_live``：turn 结构、live 落盘、
    meta 字段与主机制完全一致（额外多一个 ``generation_mode`` 标记，用于
    缓存区分与离线分析）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from libs.chat.session import ChatSession, SessionResult, _Speaker

LOGGER = logging.getLogger(__name__)


class TurnTakingSession(ChatSession):
    """你一句我一句的极简对话驱动器。只覆写 ``run()``。"""

    generation_mode = "turn_taking"

    #: 复读防御丢掉草稿后，同一方最多重新生成的次数；超过则结束对话。
    _MAX_REASK = 3

    def run(self) -> SessionResult:
        dialogue: List[Dict[str, Any]] = list(self.resume_dialogue)
        ended = False
        resumed_turns = len(dialogue)

        if self.live_path is None:
            self.live_path = self._default_live_path()
        _trace_dir = self.raw_trace_path or self._default_trace_dir()
        if not self.agent_a.trace_dir:
            self.agent_a.trace_dir = _trace_dir
        if not self.agent_b.trace_dir:
            self.agent_b.trace_dir = _trace_dir

        state_a = _Speaker(agent=self.agent_a)
        state_b = _Speaker(agent=self.agent_b)

        # 定位首个发言方。
        if dialogue:
            # resume：重放回调后，轮到最后一条消息的对方。
            for idx, t in enumerate(dialogue):
                self._notify(idx, t, dialogue[: idx + 1])
            if dialogue[-1].get("role") == self.agent_b.persona.persona_id:
                speaker, receiver = state_a, state_b
            else:
                speaker, receiver = state_b, state_a
        elif self.opener:
            # opener 视为 agent_a 的第 0 条（不调 LLM），轮到 agent_b 回应。
            opener_turn = self._wrap_opener(t_send_ms=0)
            dialogue.append(opener_turn)
            self._notify(0, opener_turn, dialogue)
            self._dump_live(dialogue, ended=False)
            speaker, receiver = state_b, state_a
        else:
            starter_id = self.scenario.first_speaker
            if starter_id == self.agent_b.persona.persona_id:
                speaker, receiver = state_b, state_a
            else:
                speaker, receiver = state_a, state_b

        # 主循环：轮到的一方生成 → 发出 → 换人。严格轮流。
        while self.max_turns <= 0 or len(dialogue) < self.max_turns:
            if not speaker.alive:
                ended = True
                break

            t_now = self._last_send_at(dialogue)
            # 复读防御可能把草稿丢掉：重问同一方（有界），防死循环。
            reask = 0
            while True:
                self._decide(speaker, dialogue, t_now=t_now, prev_draft=None)
                if speaker.has_pending or not speaker.alive:
                    break
                reask += 1
                if reask >= self._MAX_REASK:
                    break

            if not speaker.has_pending:
                # alive=False 是空 content 降级；否则是复读重问超限。
                ended = True
                if speaker.alive:
                    LOGGER.info(
                        "agent %s dropped drafts (repeat-defense) %d times; "
                        "ending turn-taking dialogue",
                        speaker.agent.persona.persona_id, reask,
                    )
                break

            t_send = int(speaker.pending.send_at)  # type: ignore[union-attr]
            self._emit(speaker, dialogue, t_send)
            speaker, receiver = receiver, speaker

        # 收尾：触发记忆更新（双方）。
        if self.enable_memory:
            session_date = self.virtual_start.strftime("%Y-%m-%d")
            for ag, partner in (
                (self.agent_a, self.agent_b),
                (self.agent_b, self.agent_a),
            ):
                try:
                    ag.write_memory(dialogue, partner.persona.nickname,
                                    session_date=session_date)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("memory update for %s failed: %s",
                                   ag.persona.persona_id, exc)

        if ended and self.llm_failures:
            termination_reason = "llm_failure"
        elif ended:
            termination_reason = "natural"
        else:
            termination_reason = "max_turns"

        meta = {
            "persona1_id": self.agent_a.persona.persona_id,
            "persona2_id": self.agent_b.persona.persona_id,
            "scenario_id": self.scenario.scenario_id,
            "model": self.agent_a.model,
            "session_tag": self.session_tag,
            "turns": len(dialogue),
            "ended_naturally": ended,
            "termination_reason": termination_reason,
            "fail_count": self.fail_count,
            "llm_failures": list(self.llm_failures),
            "interrupts": self.interrupt_log,
            "resumed_from_turns": resumed_turns,
            "virtual_duration_ms": self._last_send_at(dialogue),
            "virtual_start_time": self.virtual_start.isoformat(timespec="seconds"),
            "raw_trace_dir": (str(self.agent_a.trace_dir)
                              if self.agent_a.trace_dir else None),
            "generation_mode": self.generation_mode,
        }
        self._dump_live(dialogue, ended=ended, meta=meta)
        return SessionResult(dialogue=dialogue, meta=meta,
                             output_path=self.live_path)
