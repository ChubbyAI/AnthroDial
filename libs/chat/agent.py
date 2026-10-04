"""ChatAgent: a single persona-driven speaker (single-turn architecture).

Responsibilities:
- Build its system prompt from persona + scenario + memory + skills.
- Generate a reply given a user snapshot (conversation memory + recent turns).
- Optionally ingest supervisor feedback to retry a turn.
- Write per-turn SFT-friendly trace files (0.json, 1.json, ...).
- Trigger conversation memory folding every N turns.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from libs.chat.compressor import Compressor
from libs.chat.memory import Memory
from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.core.config import REPO_ROOT, get_settings
from libs.llm.client import LLMClient, LLMResult

LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPT_COMPONENTS_DIR = REPO_ROOT / "src" / "chat" / "prompts" / "persona_chat"


def _prompt_components_dir() -> Path:
    """返回 prompt 组件目录。可通过 paths.prompt_components_dir 覆盖默认模板。"""
    try:
        p = get_settings().path("prompt_components_dir")
        if p.exists():
            return p
        LOGGER.warning("paths.prompt_components_dir=%s 不存在，回退默认模板", p)
    except KeyError:
        pass
    return DEFAULT_PROMPT_COMPONENTS_DIR

MAX_PRE_SEND_DELAY_MS = 7_200_000

# ---------------------------------------------------------------------------
# 生成机制（session.mechanism）
# ---------------------------------------------------------------------------
# draft_scheduling：默认的单草稿调度机制（并行决策 + 虚拟时间戳竞争）。
# turn_taking：一对一严格轮流 baseline（你一句我一句）。
DEFAULT_GENERATION_MODE = "draft_scheduling"

_MECHANISM_ALIASES = {
    "draft_scheduling": "draft_scheduling",
    "default": "draft_scheduling",
    "cache": "draft_scheduling",
    "turn_taking": "turn_taking",
    "one_on_one": "turn_taking",
    "alternating": "turn_taking",
}


def normalize_generation_mode(value: Any) -> str:
    """把 ``session.mechanism`` 配置值规范化为标准生成模式名。

    未知值打 warning 并回退默认机制（保证旧配置行为不变）。
    """
    key = str(value or "").strip().lower()
    mode = _MECHANISM_ALIASES.get(key)
    if mode is None:
        LOGGER.warning(
            "unknown session.mechanism=%r; falling back to %s",
            value, DEFAULT_GENERATION_MODE,
        )
        return DEFAULT_GENERATION_MODE
    return mode
# MIN_PRE_SEND_DELAY_MS: agent 层的发送延迟下限，从 session.min_send_interval_ms 配置读取，
# 同时由 session._enforce_delay_schedule() 在调度层强制执行。
# 此常量仅作为 build() 前的全局默认值；实际运行时实例会用 self._min_send_interval_ms。
MIN_PRE_SEND_DELAY_MS = 0

# user 快照里展示的最近轮数（默认 10，可通过 compression.keep_recent_turns 配置）。
RECENT_TURNS_IN_USER_BLOCK = 10

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_TS_FMT_MS = "%Y-%m-%d %H:%M:%S.%f"


def _parse_vt(s: str) -> Optional[datetime]:
    """宽松解析虚拟时间字符串。支持毫秒/秒/分钟精度。失败返 None。"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    for fmt in (_TS_FMT_MS, _TS_FMT, "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _load_skills_block(skills_dir: Optional[Path] = None) -> str:
    """Read all *.md files under the configured skills_dir and concatenate them.

    Args:
        skills_dir: Optional override for the skill library path. When omitted,
            uses ``paths.skills_dir`` from the global settings.
    """
    if skills_dir is None:
        skills_dir = get_settings().path("skills_dir")
    if not skills_dir.exists():
        return "(暂无额外技能)"
    parts: List[str] = []
    for p in sorted(skills_dir.glob("*.md")):
        try:
            text = p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if text:
            parts.append(f"### {p.stem}\n{text}")
    return "\n\n".join(parts) if parts else "(暂无额外技能)"


DEFAULT_CONSECUTIVE_DELAY_SCHEDULE_MS = [3000, 3000, 5000, 20000, 60000, 180000]


def _build_delay_table(schedule: List[int]) -> str:
    """将 consecutive_delay_schedule_ms 转化为 prompt 中的人类可读表格。"""
    if not schedule:
        return "（连发间隔递增，具体由系统控制）"

    def _ms_to_human(ms: int) -> str:
        if ms < 1000:
            return f"{ms} 毫秒"
        elif ms < 60_000:
            return f"{ms // 1000} 秒"
        elif ms < 3_600_000:
            return f"{ms // 60_000} 分钟"
        else:
            return f"{ms // 3_600_000} 小时"

    lines = [
        "| 第几条连发 | 距上一条的最小间隔 |",
        "| --- | --- |",
    ]
    # 只展示前 6 条（再多模型也不太会连发那么多）
    show_count = min(len(schedule), 6)
    for i in range(show_count):
        ordinal = f"第 {i + 2} 条" if i < show_count - 1 else f"第 {i + 2}+ 条"
        lines.append(f"| {ordinal} | ≥ {_ms_to_human(schedule[i])} |")
    return "\n".join(lines)


def build_system_prompt(
    persona: Persona,
    scenario: Scenario,
    partner_id: str,
    memory: Optional[Memory] = None,
    synthesis_hint: str = "",
    keep_recent_turns: int = RECENT_TURNS_IN_USER_BLOCK,
    consecutive_delay_schedule: Optional[List[int]] = None,
    skills_dir: Optional[Path] = None,
) -> str:
    """按 prompt_components_dir 下的模板 + 卡面组装 canonical system prompt。

    synthesis_hint 为空时输出与 benchmark 运行时一致的 prompt；SFT 数据构建
    等离线场景复用本函数以保证训练/推理 prompt 对齐。

    Args:
        skills_dir: 指定使用的技能库路径；默认使用全局配置中的 paths.skills_dir。
    """
    md_files = sorted(_prompt_components_dir().glob("*.md"))
    static_parts = [(f.stem, f.read_text(encoding="utf-8").strip()) for f in md_files]

    skills_block = _load_skills_block(skills_dir)
    persona_block = persona.to_system_prompt()
    scenario_block = scenario.to_system_prompt(
        self_id=persona.persona_id,
        partner_id=partner_id,
    )
    if memory is None:
        memory = Memory(persona.persona_id, persona.nickname, {})
    memory_block = memory.section_for(partner_id)

    dynamic_section = (
        "## 当前生效的拟人写作技能 (Skills)\n\n"
        "下面是从经验中沉淀下来的写作技巧，请把它们自然融入回复，不要照搬：\n\n"
        f"{skills_block}\n\n"
        f"## 你的人设\n{persona_block}\n\n"
        f"## 当前场景\n{scenario_block}\n\n"
        "## 你对对方的记忆（来自以往对话，不是当前这段对话）\n\n"
        '以下是你在**之前的对话/互动中**了解到的关于对方的信息。这是你的"印象"——\n'
        "帮助你了解对方是什么样的人、之前聊过什么话题。\n"
        "**但这不是当前对话的内容**。当前对话的实际历史在下面的 user/assistant 消息中。\n"
        "你应该把记忆当做背景知识，自然地融入对话，而不是去延续或重复记忆里的内容。\n\n"
        f"{memory_block}"
    )
    # 数据合成专用提示（benchmark 等场景不设置此字段，不会出现）
    if synthesis_hint:
        dynamic_section += f"\n\n{synthesis_hint}"

    parts: list[str] = []
    for name, content in static_parts:
        parts.append(content)
        if name == "02_mechanism":
            parts.append(dynamic_section)

    raw = "\n\n".join(parts)
    return raw.replace(
        "{recent_turns_in_user_block}", str(keep_recent_turns)
    ).replace(
        "{consecutive_delay_table}",
        _build_delay_table(consecutive_delay_schedule or DEFAULT_CONSECUTIVE_DELAY_SCHEDULE_MS),
    )


@dataclass
class ChatAgent:
    persona: Persona
    scenario: Scenario
    partner_id: str
    memory: Memory
    llm: LLMClient
    compressor: Optional[Compressor] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_segments: int = 4
    extra_instructions: str = ""  # injected by supervisor on retry
    max_retries: int = 3
    retry_base_sleep: float = 1.0
    soft_hint: str = ""
    synthesis_hint: str = ""  # 仅数据合成时注入的额外 system prompt（对话时长约束等）
    generation_mode: str = DEFAULT_GENERATION_MODE  # session.mechanism 规范化值
    trace_dir: Optional[Path] = None
    _min_send_interval_ms: int = field(default=3000, init=False)
    _turn_counter: int = field(default=0, init=False)
    # ----- 单轮架构状态 -----
    _system_prompt_cache: Optional[str] = field(default=None, init=False)
    _conversation_memory: str = field(default="", init=False)
    _turns_since_last_fold: int = field(default=0, init=False)
    _history_for_fold: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _fold_interval: int = field(default=20, init=False)
    _keep_recent: int = field(default=10, init=False)
    _min_send_interval_ms: int = field(default=3000, init=False)
    _consecutive_delay_schedule: List[int] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        persona: Persona,
        scenario: Scenario,
        partner_id: str,
        llm: LLMClient,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> "ChatAgent":
        s = get_settings()
        memory = Memory.load(persona.persona_id, persona.nickname)
        # Compressor 使用独立的 LLMClient，enable_thinking 从各自 model_config 继承。
        compressor_model_name = s.model_for("compressor")
        comp_mcfg = s.model_config_for(compressor_model_name)
        comp_base_url = comp_mcfg.get("base_url") or s.get("llm", "base_url")
        comp_api_key = comp_mcfg.get("api_key") or s.get("llm", "api_key") or None
        # 解析实际 API model name（config key 可能是别名）
        comp_api_model = comp_mcfg.get("api_model", compressor_model_name)
        comp_thinking = bool(comp_mcfg.get("enable_thinking", False))
        comp_extra_body = comp_mcfg.get("extra_body") or None
        compressor_llm = LLMClient(
            base_url=comp_base_url, api_key=comp_api_key,
            enable_thinking=comp_thinking, extra_body=comp_extra_body,
        )
        compressor = Compressor(compressor_llm, model=comp_api_model)
        agent = cls(
            persona=persona,
            scenario=scenario,
            partner_id=partner_id,
            memory=memory,
            llm=llm,
            compressor=compressor,
            model=model or s.model_for("persona_chat"),
            temperature=(temperature if temperature is not None
                         else s.temperature_for("persona_chat", 0.9)),
            max_segments=int(s.get("session", "reply_max_segments", default=4)),
            max_retries=int(s.get("session", "agent_max_retries", default=3)),
            retry_base_sleep=float(s.get("session", "agent_retry_base_sleep", default=1.0)),
            generation_mode=normalize_generation_mode(
                s.get("session", "mechanism", default=DEFAULT_GENERATION_MODE)),
        )
        agent._fold_interval = int(s.get("compression", "fold_interval", default=20))
        agent._keep_recent = int(s.get("compression", "keep_recent_turns", default=10))
        agent._min_send_interval_ms = int(s.get("session", "min_send_interval_ms", default=3000))
        # 连发延迟调度表（用于动态注入 system prompt）
        raw_schedule = s.get("session", "consecutive_delay_schedule_ms", default=None)
        if raw_schedule and isinstance(raw_schedule, list):
            agent._consecutive_delay_schedule = [int(x) for x in raw_schedule]
        else:
            agent._consecutive_delay_schedule = list(DEFAULT_CONSECUTIVE_DELAY_SCHEDULE_MS)
        return agent

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        """生成 system prompt（首次调用时缓存，后续不再变动）。"""
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache
        self._system_prompt_cache = build_system_prompt(
            persona=self.persona,
            scenario=self.scenario,
            partner_id=self.partner_id,
            memory=self.memory,
            synthesis_hint=self.synthesis_hint,
            keep_recent_turns=self._keep_recent,
            consecutive_delay_schedule=self._consecutive_delay_schedule or None,
        )
        return self._system_prompt_cache

    def reply(
        self,
        history: List[Dict[str, Any]],
        *,
        current_virtual_time: str = "",
        prev_draft: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """生成本轮输出（单轮架构：每次 LLM 调用只有 system + user）。

        单轮架构
        ----------
        每次 reply() 构造独立的 [system, user_snapshot] 提交 LLM：
        - system：稳态 prompt（persona + scenario + skills + memory）
        - user_snapshot：对话记忆 + 最近 N 轮 + 未发出草稿 + 状态标签 + 当前时间

        不维护累积 messages 队列。每轮调用完全独立，trace 直接写入单独文件，
        可作为 SFT 训练样本。
        """
        # ----- 1. 构造 user snapshot -----
        user_block = self._build_user_block(
            history=history,
            current_virtual_time=current_virtual_time,
            prev_draft=prev_draft,
        )

        # ----- 2. 构造单轮 messages -----
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_block},
        ]

        # ----- 3. 调 LLM -----
        data, raw_result = self._call_with_retry(messages)

        # ----- 4. 写入 per-turn trace（SFT 样本） -----
        self._write_turn_trace(messages, raw_result)
        self._turn_counter += 1

        # ----- 5. 记忆折叠检查 -----
        self._turns_since_last_fold += 1
        self._maybe_fold_memory(history)

        # 清理软提示 + 监督提示
        self.extra_instructions = ""
        self.soft_hint = ""
        return self._build_turn_payload(data, current_virtual_time)

    # ------------------------------------------------------------------
    def _build_user_block(
        self,
        *,
        history: List[Dict[str, Any]],
        current_virtual_time: str,
        prev_draft: Optional[Tuple[str, str]],
    ) -> str:
        """构造本轮 user snapshot 块（单轮架构版本）。

        结构：
            【对话记忆】               ← 仅当有压缩记忆时
            <摘要>

            【最近对话】               ← 保留最近 keep_recent 轮
            [segments]

            【你之前准备发但未发出的草稿】 ← 仅当 prev_draft 存在
            内容："..."
            原计划发送时间：...

            [状态标签]
            当前时间：...
        """
        self_id = self.persona.persona_id
        partner_id = self.partner_id

        segments = self._tail_recent_segments(history, self._keep_recent)

        lines: List[str] = []

        # 对话记忆
        if self._conversation_memory:
            lines.append("【对话记忆】")
            lines.append(self._conversation_memory)
            lines.append("")

        # 最近对话
        lines.append(f"【最近 {self._keep_recent} 轮对话】")

        def _render_segment(segment: List[Dict[str, Any]]) -> None:
            if not segment:
                return
            role = segment[0].get("role")
            if role == self_id:
                for t in segment:
                    for m in (t.get("response") or []):
                        txt = (m.get("content") or "").strip()
                        if not txt:
                            continue
                        ts = (m.get("timestamp") or "").strip()
                        if ts:
                            lines.append(f'[{self_id}][{ts}]"{txt}"')
                        else:
                            lines.append(f'[{self_id}]"{txt}"')
            else:
                lines.append(f"[{partner_id}]")
                for t in segment:
                    for m in (t.get("response") or []):
                        txt = (m.get("content") or "").strip()
                        if not txt:
                            continue
                        ts = (m.get("timestamp") or "").strip()
                        if ts:
                            lines.append(f"[{ts}] {txt}")
                        else:
                            lines.append(txt)

        for seg in segments:
            _render_segment(seg)

        # 未发出的草稿（prev_draft）
        if prev_draft is not None:
            draft_content, draft_ts = prev_draft
            lines.append("")
            lines.append("【你之前准备发但未发出的草稿】")
            lines.append(f'内容："{draft_content}"')
            lines.append(f"原计划发送时间：{draft_ts}")

        # 状态标签
        if not history:
            lines.append("[开始对话]")
        elif prev_draft is not None:
            lines.append("[对方的消息先发送过来了，请重新回答内容和时间]")
        else:
            last_role = segments[-1][0].get("role") if segments else None
            if last_role == self_id:
                lines.append("[对方回复中]")

        if self.extra_instructions:
            lines.append(f"[修正] {self.extra_instructions}")
        if self.soft_hint:
            lines.append(f"[状态] {self.soft_hint}")

        vt = (current_virtual_time or "").strip() or "(未知)"
        lines.append(f"当前时间：{vt}")

        # 末尾身份提醒 + 反重复指令
        lines.append("")
        if self.generation_mode == "turn_taking":
            lines.append(
                f"你是 {self_id}。请撰写你要发送的下一条消息（一次只发一条），"
                f"回顾上方你已发送的历史，禁止输出与其中任何一句语义重复的内容"
                f'输出格式：{{"content": "..."}}'
            )
        else:
            lines.append(
                f"你是 {self_id}。请撰写你打算发送的下一条消息，可自由开展话题"
                f"回顾上方你已发送的历史，禁止输出与其中任何一句语义重复的内容"
                f'输出格式：{{"content": "...", "send_timestamp": "..."}}'
            )
        return "\n".join(lines)

    @staticmethod
    def _tail_recent_segments(
        history: List[Dict[str, Any]],
        n: int,
    ) -> List[List[Dict[str, Any]]]:
        """从 history 取最近 ``n`` 个 response 非空的 turn，切为 segments。"""
        if not history or n <= 0:
            return []
        non_empty = [t for t in history if (t.get("response") or [])]
        if not non_empty:
            return []
        recent = non_empty[-n:]
        segments: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        cur_role: Optional[str] = None
        for t in recent:
            r = t.get("role")
            if r != cur_role:
                if cur:
                    segments.append(cur)
                cur = [t]
                cur_role = r
            else:
                cur.append(t)
        if cur:
            segments.append(cur)
        return segments

    # ------------------------------------------------------------------
    def _maybe_fold_memory(self, history: List[Dict[str, Any]]) -> None:
        """达到 fold_interval 时触发 LLM 记忆压缩。"""
        if self._turns_since_last_fold < self._fold_interval:
            return
        if self.compressor is None:
            return

        # 收集待折叠的 turns：所有 history 中超出 recent 窗口的部分
        non_empty = [t for t in history if (t.get("response") or [])]
        overflow = len(non_empty) - self._keep_recent
        if overflow <= 0:
            self._turns_since_last_fold = 0
            return

        turns_to_fold = non_empty[:overflow]
        LOGGER.info(
            "agent %s folding %d turns into conversation memory",
            self.persona.persona_id, len(turns_to_fold),
        )
        try:
            self._conversation_memory = self.compressor.fold_conversation(
                existing_memory=self._conversation_memory,
                turns_to_fold=turns_to_fold,
            )
        except Exception as exc:
            LOGGER.warning("memory fold failed for %s: %s",
                           self.persona.persona_id, exc)
        self._turns_since_last_fold = 0

    # ------------------------------------------------------------------
    def _write_turn_trace(
        self,
        messages: List[Dict[str, Any]],
        raw_result: Optional[LLMResult],
    ) -> None:
        """每轮 LLM I/O 写入独立的 ``<turn_idx>.json`` 文件。

        目录结构：
            <trace_dir>/<persona_id>/<turn_idx>.json

        每个文件是一个独立的 SFT 训练样本：
            messages = [system, user_snapshot]
            assistant = {role, content, reasoning_content}
        """
        if not self.trace_dir:
            return
        llm_out: Dict[str, Any] = {
            "role": "assistant",
            "content": (raw_result.content or "") if raw_result else "",
            "reasoning_content": (
                getattr(raw_result, "reasoning_content", "") or ""
            ) if raw_result else "",
        }
        record: Dict[str, Any] = {
            "persona_id": self.persona.persona_id,
            "partner_id": self.partner_id,
            "model": self.model or "",
            "turn_idx": self._turn_counter,
            "conversation_memory": self._conversation_memory,
            "messages": messages,
            "assistant": llm_out,
            "usage": (raw_result.usage if raw_result and raw_result.usage else None),
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            persona_dir = self.trace_dir / self.persona.persona_id
            persona_dir.mkdir(parents=True, exist_ok=True)
            out_path = persona_dir / f"{self._turn_counter}.json"
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            LOGGER.warning("raw trace write failed for %s turn %d: %s",
                           self.persona.persona_id, self._turn_counter, exc)

    # ------------------------------------------------------------------
    def mark_last_trace_emitted(self, emitted_index: int) -> None:
        """消息真正发出后，给产生该草稿的 trace 文件打 emitted 标记。

        由 session._emit 在发出消息时调用（pending 一定由最近一次调用产生）。
        带标记的 trace 是唯一进入了最终对话的调用，构建 SFT 数据时据此过滤
        被抢发/截断/被调度替换的草稿调用。
        """
        if not self.trace_dir or self._turn_counter <= 0:
            return
        path = (self.trace_dir / self.persona.persona_id
                / f"{self._turn_counter - 1}.json")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            record["emitted"] = True
            record["emitted_index"] = emitted_index
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("failed to mark emitted trace %s: %s", path, exc)

    # ------------------------------------------------------------------
    def _call_with_retry(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Optional[LLMResult]]:
        """调 LLM 并校验契约。最多 `max_retries` 次重试，指数退避。"""
        last_err: Optional[Exception] = None
        last_raw: Optional[LLMResult] = None
        for i in range(max(1, self.max_retries)):
            try:
                data, raw = self.llm.chat_json_with_raw(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                )
                last_raw = raw
                self._validate_contract(data)
                return data, raw
            except Exception as exc:
                last_err = exc
                sleep_s = self.retry_base_sleep * (2 ** i) + random.uniform(0, 0.5)
                LOGGER.warning(
                    "agent %s reply attempt %d/%d failed (%s); retry in %.1fs",
                    self.persona.persona_id, i + 1, self.max_retries, exc, sleep_s,
                )
                if i + 1 < self.max_retries:
                    time.sleep(sleep_s)
        LOGGER.error("agent %s exhausted %d retries; last_err=%s",
                     self.persona.persona_id, self.max_retries, last_err)
        degraded = {
            "content": "",
            "send_timestamp": "",
            "_degraded": True,
        }
        return degraded, last_raw

    @staticmethod
    def _validate_contract(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("LLM output is not a JSON object")
        has_content_field = "content" in data
        has_segments = isinstance(data.get("segments"), list) and data.get("segments")
        has_messages = (isinstance(data.get("messages"), list) and data.get("messages")) or \
                       (isinstance(data.get("messages"), str) and data.get("messages").strip())
        if not (has_content_field or has_segments or has_messages):
            raise ValueError("LLM output missing 'content' (and no fallback segments/messages)")
        # content 不能为空：对话系统不允许空输出，空输出视为失败触发 retry
        content = ChatAgent._normalize_content(data)
        if not content:
            raise ValueError("LLM output has empty content; retry required")

    # ------------------------------------------------------------------
    def _build_turn_payload(self, data: Dict[str, Any],
                            current_virtual_time: str = "") -> Dict[str, Any]:
        """把 LLM 原始输出规范化为轮结构（不含虚拟时钟，由 session 补齐）。"""
        content = self._normalize_content(data)

        send_timestamp_raw = data.get("send_timestamp") or data.get("timestamp") or ""
        if isinstance(send_timestamp_raw, str):
            send_timestamp_raw = send_timestamp_raw.strip()
        else:
            send_timestamp_raw = ""

        now_dt = _parse_vt(current_virtual_time) if current_virtual_time else None
        send_dt = _parse_vt(send_timestamp_raw) if send_timestamp_raw else None

        pre_send_delay_ms: Optional[int] = None
        if send_dt is not None and now_dt is not None:
            delta_ms = int((send_dt - now_dt).total_seconds() * 1000)
            if delta_ms < 0:
                LOGGER.warning("agent %s send_timestamp %s < current %s; clamp to now",
                               self.persona.persona_id, send_timestamp_raw,
                               current_virtual_time)
                delta_ms = 0
            pre_send_delay_ms = delta_ms

        if pre_send_delay_ms is None:
            raw_delay = data.get("pre_send_delay_ms")
            if raw_delay is None:
                raw_delay = data.get("next_reply_delay_ms")
            try:
                pre_send_delay_ms = int(raw_delay if raw_delay is not None else 3000)
            except (TypeError, ValueError):
                pre_send_delay_ms = 3000

        # 强制最小发送间隔：低于 min_send_interval_ms 时 clamp 到该值
        pre_send_delay_ms = max(
            self._min_send_interval_ms,
            min(int(pre_send_delay_ms), MAX_PRE_SEND_DELAY_MS),
        )

        if now_dt is not None:
            send_dt_norm = now_dt + timedelta(milliseconds=pre_send_delay_ms)
            normalized_send_ts = (
                send_dt_norm.strftime("%Y-%m-%d %H:%M:%S.") +
                f"{send_dt_norm.microsecond // 1000:03d}"
            )
        else:
            normalized_send_ts = send_timestamp_raw or ""

        response_compat = [{"content": content, "timestamp": ""}] if content else []
        return {
            "role": self.persona.persona_id,
            "content": content,
            "pre_send_delay_ms": pre_send_delay_ms,
            "send_timestamp": normalized_send_ts,
            "_degraded": bool(data.get("_degraded")),
            "response": response_compat,
        }

    @staticmethod
    def _normalize_content(data: Dict[str, Any]) -> str:
        """取出本轮要发出的单条消息内容。"""
        c = data.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
        segs = data.get("segments")
        if isinstance(segs, list):
            for s in segs:
                if isinstance(s, dict):
                    txt = str(s.get("content") or "").strip()
                    if txt:
                        return txt
                elif isinstance(s, str) and s.strip():
                    return s.strip()
        msgs = data.get("messages")
        if isinstance(msgs, str) and msgs.strip():
            return msgs.strip()
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, str) and m.strip():
                    return m.strip()
                if isinstance(m, dict):
                    txt = str(m.get("content") or "").strip()
                    if txt:
                        return txt
        return ""

    # ------------------------------------------------------------------
    def write_memory(self, dialogue: List[dict], partner_nickname: str,
                     session_date: Optional[str] = None) -> List[str]:
        return self.memory.update_after_session(
            partner_id=self.partner_id,
            partner_nickname=partner_nickname,
            dialogue=dialogue,
            llm=self.llm,
            session_date=session_date,
        )
