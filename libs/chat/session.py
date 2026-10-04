"""Two-agent dialogue session orchestrator (single-draft scheduling).

调度模型（极简版）
==================

每个 agent **任意时刻只持有一条"未发送草稿"**——绝不累积。
两边角色**并行**思考，各自维护：

- ``alive: bool``      —— 是否仍可参与（仅在 agent 返回空内容时变 False）
- ``pending: _Pending`` —— 当前待发草稿；可能为 None

调度循环：

1. 双方的 pending 各有一个绝对发送时刻 ``send_at``（=决策时虚拟时刻 + pre_send_delay_ms）。
2. 选 ``send_at`` 较小者把消息真正发出，append 到 dialogue。
3. 发出后：
   - **发送方**：基于"我刚发了什么"重新决策**新一条**草稿（prev_draft = None）。
   - **接收方**：在虚拟时刻 ``send_at`` 看到了对方新消息——
     - 如果它**之前已有 pending**：把这条旧草稿喂给 LLM，让它综合（保留/修改/换一句）后给新草稿。
     - 如果它**没 pending**（刚被唤醒）：直接基于历史决策一条草稿。
4. 对话在达到 max_turns 或 max_duration_ms 时硬截断。
5. 双方都无 pending → 自然结束（仅异常情况）。

第一轮：A 主动 decide 一次（B 此刻 pending=None）；A 发出后 B 才会被叫去决策。
Opener：opener 已是 A 的第 0 条 turn，发完后 A、B 都被叫去决策。

输出文件即时写入 ``outputs/dialogues/<model>/<scenario>/<P1>_<P2>_<tag>.json``。
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from libs.chat.agent import ChatAgent
from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.core.config import get_settings, slug, scenario_path
from libs.llm.client import LLMClient

LOGGER = logging.getLogger(__name__)

TurnSupervisor = Callable[[Dict[str, Any], List[Dict[str, Any]]], Optional[str]]
TurnCallback = Callable[[int, Dict[str, Any]], None]
AsyncSupervisor = Callable[[int, Dict[str, Any], List[Dict[str, Any]]], None]

# 安全阀：同一发言者尾部最多连发多少条没收到对方回应；超过则软停本轮。
# 仅作为 session.max_consecutive_same_speaker 未配置时的兼容默认值。
MAX_CONSECUTIVE_SAME_SPEAKER = 3


def _default_on_turn(idx: int, turn: Dict[str, Any]) -> None:
    """[turn N | T+12.3s | 22:01:35] role=Pxx | text"""
    role = turn.get("role", "?")
    text = (turn.get("content") or "").replace("\n", " ")
    if not text:
        text = " / ".join(
            (m.get("content") or "").replace("\n", " ")
            for m in (turn.get("response") or [])
        )
    t_ms = turn.get("t_send_ms")
    if t_ms is None:
        t_ms = turn.get("t_start_ms")
    vt = turn.get("virtual_start_time", "")
    bits = [f"turn {idx:02d}"]
    if isinstance(t_ms, (int, float)):
        bits.append(f"T+{t_ms / 1000:7.1f}s")
    if vt:
        bits.append(vt)
    print(f"[{' | '.join(bits)}] {role} | {text}", flush=True)


@dataclass
class SessionResult:
    dialogue: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[Path] = None


@dataclass
class _Pending:
    """单 agent 的待发草稿。"""
    content: str
    send_at: float                # 绝对虚拟毫秒
    pre_send_delay_ms: int        # 决策时给的延迟（trace 用）
    decided_at: int               # 决策时的虚拟毫秒
    send_timestamp: str = ""      # 虚拟时钟发出时刻的字符串 (= _vt(send_at))
    degraded: bool = False
    revised_from: Optional[str] = None  # 综合修改前的旧草稿内容（trace 用）


@dataclass
class _Speaker:
    """单 agent 的调度态。"""
    agent: ChatAgent
    alive: bool = True
    pending: Optional[_Pending] = None

    @property
    def has_pending(self) -> bool:
        return self.pending is not None


class ChatSession:
    """Drives two ChatAgents through a multi-turn conversation."""

    def __init__(
        self,
        agent_a: ChatAgent,
        agent_b: ChatAgent,
        scenario: Scenario,
        max_turns: Optional[int] = None,
        supervisor: Optional[TurnSupervisor] = None,
        max_retries_per_turn: int = 0,
        opener: Optional[str] = None,
        live_path: Optional[Path] = None,
        on_turn: Optional[TurnCallback] = _default_on_turn,
        resume_dialogue: Optional[List[Dict[str, Any]]] = None,
        async_supervisor: Optional[AsyncSupervisor] = None,
        raw_trace_path: Optional[Path] = None,
        session_tag: Optional[str] = None,
        enable_memory: bool = True,
        min_duration_ms: Optional[int] = None,  # deprecated, kept for API compat
    ) -> None:
        s = get_settings()
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.scenario = scenario
        self.session_tag: str = session_tag or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.max_turns = int(max_turns or s.get("session", "max_turns", default=50))
        self.supervisor = supervisor
        self.max_retries_per_turn = max_retries_per_turn
        self.opener = opener
        self.live_path = Path(live_path) if live_path else None
        self.on_turn = on_turn
        self.resume_dialogue: List[Dict[str, Any]] = list(resume_dialogue or [])
        self.async_supervisor = async_supervisor
        self._rng = random.Random(0)
        self.raw_trace_path = raw_trace_path
        self.fail_count = 0
        self.interrupt_log: List[Dict[str, Any]] = []
        self.llm_failures: List[str] = []
        self.virtual_start: datetime = scenario.virtual_start_time
        # 连发安全阀阈值：优先读 session.max_consecutive_same_speaker，未配置则回退到 MAX_CONSECUTIVE_SAME_SPEAKER。
        self.enable_memory: bool = enable_memory
        self.max_consecutive_same_speaker: int = int(
            s.get(
                "session", "max_consecutive_same_speaker",
                default=MAX_CONSECUTIVE_SAME_SPEAKER,
            )
        )
        # 最小发送间隔（ms）：同一人任意两条消息之间的绝对下限。
        self.min_send_interval_ms: int = int(
            s.get("session", "min_send_interval_ms", default=3000)
        )
        # 连续发言动态延迟调度表（ms）：第 n 条连发的最小延迟。
        # 索引含义：[0]=第2条连发(首次 self-followup), [1]=第3条, ...
        # 超出列表长度时取最后一个值。
        raw_schedule = s.get("session", "consecutive_delay_schedule_ms", default=None)
        if isinstance(raw_schedule, list) and raw_schedule:
            self.consecutive_delay_schedule: List[int] = [int(x) for x in raw_schedule]
        else:
            # 默认调度表：逐步递增，最后趋向「不再主动追发」
            self.consecutive_delay_schedule = [
                3000, 3000, 5000, 20000, 60000,
                180000, 600000, 3600000, 43200000, 86400000,
            ]

    # ------------------------------------------------------------------
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
        states = (state_a, state_b)

        # 初始化：决定第一波 pending。
        if dialogue:
            for idx, t in enumerate(dialogue):
                self._notify(idx, t, dialogue[: idx + 1])
            last_t = self._last_send_at(dialogue)
            for st in states:
                if st.alive:
                    self._decide(st, dialogue, t_now=last_t, prev_draft=None)
        elif self.opener:
            # 强制 opener：A 第 0 条直接发出（不调 LLM）。
            opener_turn = self._wrap_opener(t_send_ms=0)
            dialogue.append(opener_turn)
            self._notify(0, opener_turn, dialogue)
            self._dump_live(dialogue, ended=False)
            # 双方都决策（A 决策下一条；B 看到 A 的 opener 决策回应）。
            for st in states:
                if st.alive:
                    self._decide(st, dialogue, t_now=0, prev_draft=None)
        else:
            # 无 opener：按场景卡 first_speaker 选中开场方主动决策。
            # 另一方暂不决策（它还没看到任何消息）。
            starter_id = self.scenario.first_speaker
            if starter_id == self.agent_b.persona.persona_id:
                starter = state_b
            else:
                starter = state_a
            self._decide(starter, dialogue, t_now=0, prev_draft=None)

        # 主循环。max_turns <= 0 表示不限制轮次。
        while self.max_turns <= 0 or len(dialogue) < self.max_turns:
            candidates = [st for st in states if st.alive and st.has_pending]
            if not candidates:
                ended = True
                break

            # tie：双方草稿同一虚拟时刻 → 物理顺序随机，互不参考各自这一条。
            if (len(candidates) == 2
                    and candidates[0].pending.send_at  # type: ignore[union-attr]
                    == candidates[1].pending.send_at):  # type: ignore[union-attr]
                t_now = int(candidates[0].pending.send_at)  # type: ignore[union-attr]
                pair = list(candidates)
                self._rng.shuffle(pair)
                if self.max_turns > 0:
                    pair = pair[: self.max_turns - len(dialogue)]
                for st in pair:
                    self._emit(st, dialogue, t_now)
                if self.max_turns <= 0 or len(dialogue) < self.max_turns:
                    # tie 后双方各自基于最新 dialogue（含两条同时刻消息）重新决策。
                    for st in states:
                        if st.alive:
                            self._decide(st, dialogue, t_now=t_now, prev_draft=None)
                    self._enforce_consecutive_safety(dialogue, states)
                continue

            # 选 send_at 较小者发出。
            candidates.sort(key=lambda s: s.pending.send_at)  # type: ignore[union-attr]
            sender = candidates[0]
            receiver = state_a if sender is state_b else state_b

            t_now = int(sender.pending.send_at)  # type: ignore[union-attr]
            self._emit(sender, dialogue, t_now)

            # 发送方：仍 alive 则重新决策新一条草稿。
            if sender.alive:
                self._decide(sender, dialogue, t_now=t_now, prev_draft=None)

            # 接收方：在 t_now 看到对方新消息。
            if receiver.alive:
                prev = None
                if receiver.has_pending:
                    p = receiver.pending  # type: ignore[union-attr]
                    prev = (p.content, p.send_timestamp or self._vt(int(p.send_at)))
                receiver.pending = None
                self._decide(receiver, dialogue, t_now=t_now, prev_draft=prev)

            self._enforce_consecutive_safety(dialogue, states)

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
        }
        self._dump_live(dialogue, ended=ended, meta=meta)
        return SessionResult(dialogue=dialogue, meta=meta,
                             output_path=self.live_path)

    # ------------------------------------------------------------------
    def _vt(self, t_ms: int) -> str:
        """虚拟毫秒 → 人类可读壁钟（基于场景起始时间）。

        输出毫秒精度 ``YYYY-MM-DD HH:MM:SS.mmm``。同一虚拟秒内发出的多条
        消息在毫秒级别仍可区分排序。
        """
        dt = self.virtual_start + timedelta(milliseconds=t_ms)
        # µs 袪到 ms（3 位）
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

    # ------------------------------------------------------------------
    def _decide(
        self,
        state: _Speaker,
        history: List[Dict[str, Any]],
        *,
        t_now: int,
        prev_draft: Optional[Tuple[str, str]],
    ) -> None:
        """让 state.agent 在虚拟时刻 t_now 做一次决策，结果写入 state.pending。

        prev_draft: 综合修改时传入旧草稿 (content, send_timestamp_str)。
        对话系统永远不会主动结束，仅由 max_turns 硬截断。
        content 为空视为输出失败，在 agent 层已通过 retry 机制处理。
        """
        payload = self._produce_turn(
            state, history, t_now=t_now,
            virtual_clock=self._vt(t_now),
            prev_draft=prev_draft,
        )

        content = str(payload.get("content") or "")
        delay = max(0, int(payload.get("pre_send_delay_ms") or 0))
        send_ts = str(payload.get("send_timestamp") or "")
        degraded = bool(payload.get("_degraded"))

        if not content:
            # content 为空：agent retry 全部失败后的降级情况。
            # 标记 agent 为死亡（alive=False），对话将因无可用 speaker 而终止。
            state.pending = None
            state.alive = False
            pid = state.agent.persona.persona_id
            self.llm_failures.append(pid)
            LOGGER.warning(
                "agent %s returned empty content after retries; "
                "marking dead (dialogue will terminate)",
                pid,
            )
            return

        # 复读防御：若新草稿与该方在 dialogue 中最近一条同内容，则清掉（本轮不发）。
        # 场景：LLM 在 self-followup（对方回复中）状态下偶尔复读上一条 assistant 内容。
        # 跳过被「对方抢先发了」推上去的草稿场景（prev_draft 不为 None）不查：
        # 那种场景下「保留原话」是合法选择。
        if prev_draft is None and self._is_repeat_of_last_self(content, history,
                                                               state.agent.persona.persona_id):
            LOGGER.info(
                "agent %s draft repeats its own last message; dropping pending",
                state.agent.persona.persona_id,
            )
            state.pending = None
            return

        send_at = float(t_now) + float(delay)
        if not send_ts:
            send_ts = self._vt(int(send_at))

        state.pending = _Pending(
            content=content,
            send_at=send_at,
            pre_send_delay_ms=delay,
            decided_at=t_now,
            send_timestamp=send_ts,
            degraded=degraded,
            revised_from=(prev_draft[0] if prev_draft is not None else None),
        )

        # ── 动态延迟调度：根据连续发言次数强制最小间隔 ──
        self._enforce_delay_schedule(state, history)

    # ------------------------------------------------------------------
    def _emit(
        self,
        state: _Speaker,
        dialogue: List[Dict[str, Any]],
        t_send_ms: int,
    ) -> Dict[str, Any]:
        """把 state.pending 落地为一条 turn，append 到 dialogue。"""
        p = state.pending
        assert p is not None, "_emit called without pending"
        ts = self._vt(t_send_ms)
        turn: Dict[str, Any] = {
            "role": state.agent.persona.persona_id,
            "response": [{
                "content": p.content,
                "timestamp": ts,
                "t_emit_ms": t_send_ms,
            }],
            "content": p.content,
            "pre_send_delay_ms": p.pre_send_delay_ms,
            "send_timestamp": ts,
            "t_send_ms": t_send_ms,
            "t_start_ms": t_send_ms,
            "t_end_ms": t_send_ms,
            "virtual_start_time": ts,
            "_decided_at_ms": p.decided_at,
            "_degraded": p.degraded,
        }
        if p.revised_from is not None and p.revised_from != p.content:
            turn["_revised_from"] = p.revised_from
        dialogue.append(turn)
        self._notify(len(dialogue) - 1, turn, dialogue)
        self._dump_live(dialogue, ended=False)
        state.pending = None
        mark_emitted = getattr(state.agent, "mark_last_trace_emitted", None)
        if callable(mark_emitted):
            mark_emitted(len(dialogue) - 1)
        return turn

    # ------------------------------------------------------------------
    def _enforce_consecutive_safety(
        self,
        dialogue: List[Dict[str, Any]],
        states: Tuple[_Speaker, _Speaker],
    ) -> None:
        """尾部连发同一人 ≥ self.max_consecutive_same_speaker 条 → **软停**该方。

        仅清掉其刚决策出的 pending（本轮不主动接话），alive 保持 True。
        这样当对方下一次真的发言时，该方作为 receiver 仍会被唤醒重新决策。
        避免「一时倒豆子”误判为「独角戏」后永久 dead 导致对话突然终止。
        """
        if not dialogue:
            return
        tail_role = dialogue[-1].get("role")
        n = self._tail_consecutive_count(dialogue, tail_role)
        if n < self.max_consecutive_same_speaker:
            return
        for st in states:
            if st.agent.persona.persona_id == tail_role and st.alive:
                LOGGER.info(
                    "agent %s consecutive >= %d; suspending pending (alive kept)",
                    tail_role, self.max_consecutive_same_speaker,
                )
                st.pending = None

    # ------------------------------------------------------------------
    @staticmethod
    def _tail_consecutive_count(dialogue: List[Dict[str, Any]], role: Any) -> int:
        n = 0
        for t in reversed(dialogue):
            if t.get("role") == role:
                n += 1
            else:
                break
        return n

    # ------------------------------------------------------------------
    def _enforce_delay_schedule(
        self,
        state: _Speaker,
        dialogue: List[Dict[str, Any]],
    ) -> None:
        """根据连续发言次数，强制 pending 的 send_at 不低于调度表规定的最小延迟。

        逻辑：
        1. 统计 dialogue 尾部该角色的连续消息数 n_consecutive。
        2. 若 n_consecutive >= 1（即即将发的是第 n+1 条连发），
           查表 consecutive_delay_schedule[n_consecutive - 1] 作为最小延迟。
        3. 同时强制 min_send_interval_ms 下限。
        4. 若当前 delay 不足，将 send_at 向后推迟。
        """
        if state.pending is None:
            return

        self_role = state.agent.persona.persona_id
        n_consecutive = self._tail_consecutive_count(dialogue, self_role)

        # 基础下限：min_send_interval_ms
        required_delay = self.min_send_interval_ms

        # 连发调度表：n_consecutive >= 1 表示已有 n 条在尾部，即将发的是第 n+1 条
        if n_consecutive >= 1:
            schedule = self.consecutive_delay_schedule
            idx = n_consecutive - 1  # 0-based: [0]=第2条, [1]=第3条...
            if idx < len(schedule):
                schedule_min = schedule[idx]
            else:
                # 超出表长度取最后一个值（通常是非常大的数）
                schedule_min = schedule[-1]
            required_delay = max(required_delay, schedule_min)

        # 如果当前 pending 的延迟已满足，不做任何修改
        current_delay = int(state.pending.send_at - state.pending.decided_at)
        if current_delay >= required_delay:
            return

        # 强制推迟 send_at
        new_send_at = float(state.pending.decided_at) + float(required_delay)
        state.pending.send_at = new_send_at
        state.pending.pre_send_delay_ms = required_delay
        state.pending.send_timestamp = self._vt(int(new_send_at))
        LOGGER.debug(
            "agent %s consecutive=%d; delay enforced %dms -> %dms",
            self_role, n_consecutive, current_delay, required_delay,
        )

    @staticmethod
    def _is_repeat_of_last_self(content: str,
                                dialogue: List[Dict[str, Any]],
                                self_role: Any) -> bool:
        """新草稿 content 是否与 dialogue 中该 role 最近一条 content 完全一致。"""
        norm = (content or "").strip()
        if not norm:
            return False
        for t in reversed(dialogue):
            if t.get("role") != self_role:
                continue
            prev = (t.get("content") or "").strip()
            return prev == norm
        return False
    
    @staticmethod
    def _last_send_at(dialogue: List[Dict[str, Any]]) -> int:
        for t in reversed(dialogue):
            ts = t.get("t_send_ms")
            if ts is None:
                ts = t.get("t_end_ms") or t.get("t_start_ms")
            if isinstance(ts, (int, float)):
                return int(ts)
        return 0

    # ------------------------------------------------------------------
    def _wrap_opener(self, t_send_ms: int) -> Dict[str, Any]:
        ts = self._vt(t_send_ms)
        return {
            "role": self.agent_a.persona.persona_id,
            "response": [{
                "content": self.opener,
                "timestamp": ts,
                "t_emit_ms": t_send_ms,
            }],
            "content": self.opener,
            "pre_send_delay_ms": 0,
            "t_send_ms": t_send_ms,
            "t_start_ms": t_send_ms,
            "t_end_ms": t_send_ms,
            "virtual_start_time": ts,
            "is_opener": True,
        }

    # ------------------------------------------------------------------
    def _notify(self, idx: int, turn: Dict[str, Any],
                history: List[Dict[str, Any]]) -> None:
        if self.on_turn:
            try:
                self.on_turn(idx, turn)
            except Exception:  # pragma: no cover
                pass
        if self.async_supervisor:
            try:
                self.async_supervisor(idx, turn, history)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("async_supervisor hook raised: %s", exc)

    # ------------------------------------------------------------------
    def _produce_turn(
        self,
        state: _Speaker,
        history: List[Dict[str, Any]],
        *,
        t_now: int,
        virtual_clock: str,
        prev_draft: Optional[Tuple[str, int]],
    ) -> Dict[str, Any]:
        """调一次 agent.reply（必要时按 supervisor 重试）。"""
        last_reason: Optional[str] = None
        agent = state.agent
        for attempt in range(self.max_retries_per_turn + 1):
            agent.extra_instructions = (
                f"上一轮被监督打回，原因：{last_reason}\n请在本轮严格修正。"
                if last_reason else ""
            )
            payload = agent.reply(
                history,
                current_virtual_time=virtual_clock,
                prev_draft=prev_draft,
            )
            if self.supervisor is None:
                return payload
            reason = self.supervisor(payload, history)
            if reason is None:
                return payload
            self.interrupt_log.append({
                "agent": agent.persona.persona_id,
                "attempt": attempt + 1,
                "reason": reason,
                "turn": payload,
                "at_t_ms": t_now,
            })
            last_reason = reason
        self.fail_count += 1
        agent.extra_instructions = ""
        return payload  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    def _dump_live(
        self,
        dialogue: List[Dict[str, Any]],
        ended: bool,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """每次发出消息后即时写入最终输出文件（原子写）。"""
        if not self.live_path:
            return
        self.live_path.parent.mkdir(parents=True, exist_ok=True)
        clean = _clean_dialogue(dialogue)
        payload = {
            "persona1_id": self.agent_a.persona.persona_id,
            "persona2_id": self.agent_b.persona.persona_id,
            "scenario_id": self.scenario.scenario_id,
            "dialogue": clean,
            "meta": meta or {
                "persona1_id": self.agent_a.persona.persona_id,
                "persona2_id": self.agent_b.persona.persona_id,
                "scenario_id": self.scenario.scenario_id,
                "model": self.agent_a.model,
                "session_tag": self.session_tag,
                "turns": len(clean),
                "ended_naturally": ended,
                "termination_reason": "in_progress",
                "fail_count": self.fail_count,
                "llm_failures": list(self.llm_failures),
                "in_progress": not ended,
                "virtual_duration_ms": self._last_send_at(dialogue),
                "virtual_start_time": self.virtual_start.isoformat(timespec="seconds"),
            },
        }
        tmp = self.live_path.with_suffix(self.live_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.live_path)

    # ------------------------------------------------------------------
    def _default_trace_dir(self) -> Path:
        s = get_settings()
        model = self.agent_a.model or "unknown"
        sid = self.scenario.scenario_id
        p1 = self.agent_a.persona.persona_id
        p2 = self.agent_b.persona.persona_id
        base = (s.path("outputs_dir") / "raw_traces"
                / slug(model) / scenario_path(sid)
                / f"{p1}_{p2}_{self.session_tag}")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _default_live_path(self) -> Path:
        """outputs/dialogues/<model>/A0001/B0002/C0001/chat_<P1>_<P2>_<tag>.json。"""
        s = get_settings()
        model = self.agent_a.model or s.model_for("persona_chat")
        sid = self.scenario.scenario_id
        p1 = self.agent_a.persona.persona_id
        p2 = self.agent_b.persona.persona_id
        base = (s.path("outputs_dir") / "dialogues"
                / slug(model) / scenario_path(sid))
        base.mkdir(parents=True, exist_ok=True)
        return base / f"chat_{p1}_{p2}_{self.session_tag}.json"


# ---------------------------------------------------------------------------
def build_session(
    persona_a_id: str,
    persona_b_id: str,
    scenario_id: str,
    llm: Optional[LLMClient] = None,
    supervisor: Optional[TurnSupervisor] = None,
    max_retries_per_turn: int = 0,
    max_turns: Optional[int] = None,
    opener: Optional[str] = None,
    live_path: Optional[Path] = None,
    on_turn: Optional[TurnCallback] = _default_on_turn,
    resume_dialogue: Optional[List[Dict[str, Any]]] = None,
    async_supervisor: Optional[AsyncSupervisor] = None,
    raw_trace_path: Optional[Path] = None,
    session_tag: Optional[str] = None,
) -> ChatSession:
    """Convenience constructor used by Layer-1 demos & evaluation."""
    llm = llm or LLMClient()
    persona_a = Persona.load(persona_a_id)
    persona_b = Persona.load(persona_b_id)
    scenario = Scenario.load(scenario_id)
    agent_a = ChatAgent.build(persona_a, scenario, persona_b.persona_id, llm)
    agent_b = ChatAgent.build(persona_b, scenario, persona_a.persona_id, llm)
    return ChatSession(
        agent_a=agent_a,
        agent_b=agent_b,
        scenario=scenario,
        supervisor=supervisor,
        max_retries_per_turn=max_retries_per_turn,
        max_turns=max_turns,
        opener=opener,
        live_path=live_path,
        on_turn=on_turn,
        resume_dialogue=resume_dialogue,
        async_supervisor=async_supervisor,
        raw_trace_path=raw_trace_path,
        session_tag=session_tag,
    )


def default_live_path(persona_a_id: str, persona_b_id: str,
                      scenario_id: str,
                      model: Optional[str] = None) -> Optional[Path]:
    """查找某组 (P1, P2, S, model) 对应最新一条对话文件，用于 resume。"""
    s = get_settings()
    model = model or s.model_for("persona_chat")
    base = (s.path("outputs_dir") / "dialogues"
            / slug(model) / scenario_path(scenario_id))
    if not base.exists():
        return None
    candidates = sorted(
        list(base.glob(f"chat_{persona_a_id}_{persona_b_id}_*.json"))
        + list(base.glob(f"{persona_a_id}_{persona_b_id}_*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_resume(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "dialogue" not in data:
        raise ValueError(f"{path} does not look like a dialogue dump")
    return data


def extract_session_tag(path: Path,
                        persona_a_id: Optional[str] = None,
                        persona_b_id: Optional[str] = None) -> Optional[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = data.get("meta") or {}
        tag = meta.get("session_tag")
        if tag:
            return str(tag)
    except Exception:
        pass
    name = Path(path).stem
    if persona_a_id and persona_b_id:
        prefix = f"{persona_a_id}_{persona_b_id}_"
        if name.startswith(prefix):
            return name[len(prefix):]
    parts = name.rsplit("_", 2)
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return None


def _clean_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    """裁剪 turn 为微信格式：只保留 role + response[{content, timestamp}]。"""
    clean_resp = [
        {"content": m.get("content", ""), "timestamp": m.get("timestamp", "")}
        for m in (turn.get("response") or [])
        if (m.get("content") or "").strip()
    ]
    return {"role": turn.get("role"), "response": clean_resp}


def _clean_dialogue(dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """裁剪并**合并相邻同 role**：同一发言者连续多条 turn 的 response 累积到同一个 list 内。

    每条 response 会被追加一个全局递增的 ``turn`` 字段（从 0 开始），
    仅用于组织展示，不参与 LLM 输入。
    """
    merged: List[Dict[str, Any]] = []
    for t in dialogue:
        cleaned = _clean_turn(t)
        if not cleaned["response"]:
            continue
        if merged and merged[-1]["role"] == cleaned["role"]:
            merged[-1]["response"].extend(cleaned["response"])
        else:
            merged.append(cleaned)
    # 为每条 response 添加全局递增的 turn 编号（仅用于展示）
    turn_counter = 0
    for entry in merged:
        for resp in entry["response"]:
            resp["turn"] = turn_counter
            turn_counter += 1
    return merged


def save_dialogue(result: SessionResult,
                  out_dir: Optional[Path] = None,
                  tag: str = "") -> Path:
    """对话已在 run() 期间实时写入 result.output_path；这里并不额外复制一份。

    逻辑：
    - 如果 result.output_path 存在 → 以最终 meta 重写该文件一次，返回该路径。
    - 否则按 (out_dir, tag) 传统逻辑创建新文件。
    这使得 chat-only 与 synthesis 两种入口都会复用同一份即时写盘的文件，避免产生两份重复输出。
    """
    s = get_settings()
    payload = {
        "persona1_id": result.meta.get("persona1_id", "P?"),
        "persona2_id": result.meta.get("persona2_id", "P?"),
        "scenario_id": result.meta.get("scenario_id", "S?"),
        "dialogue": _clean_dialogue(result.dialogue),
        "meta": result.meta,
    }
    if result.output_path is not None:
        out = result.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return out
    p1 = result.meta.get("persona1_id", "P?")
    p2 = result.meta.get("persona2_id", "P?")
    sid = result.meta.get("scenario_id", "S?")
    model = result.meta.get("model") or s.model_for("persona_chat")
    ts = result.meta.get("session_tag") or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir or (s.path("outputs_dir") / "dialogues"
                       / slug(model) / scenario_path(sid))
    base.mkdir(parents=True, exist_ok=True)
    name = f"{p1}_{p2}_{ts}"
    if tag:
        name = f"{tag}_{name}"
    out = base / f"{name}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    result.output_path = out
    return out
