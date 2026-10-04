"""Shared benchmark LLM-as-judge scaffolding.

Domain-specific judges (chatbot / game) subclass ``BaseBenchmarkJudge`` and
override the system-prompt builders.  Everything else — L0 checks, per-turn and
holistic scoring, JSON parsing, trace saving — lives here.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.core.config import get_settings, slug
from libs.evaluation.rubric import (
    BenchmarkDimension,
    HardConstraint,
    load_rubric,
    load_hard_constraints,
    per_turn_dimensions,
    holistic_dimensions,
    dimensions_hash,
)
from libs.llm.client import LLMClient

LOGGER = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class L0Violation:
    constraint_id: str
    constraint_name: str
    severity: str  # "fatal" | "major"
    evidence: str = ""
    turn_index: int = -1


@dataclass
class DimResult:
    dim_id: str
    dim_name: str
    weight: float = 10.0
    checks: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""
    score: float = 0.0


@dataclass
class TurnScoreCard:
    turn_index: int
    role: str
    content: str
    dim_results: List[DimResult] = field(default_factory=list)
    avg_score: float = 0.0
    weighted_score: float = 0.0


@dataclass
class HolisticScoreCard:
    role: str
    dim_results: List[DimResult] = field(default_factory=list)
    avg_score: float = 0.0
    weighted_score: float = 0.0


# ── Base judge ────────────────────────────────────────────────────────

class BaseBenchmarkJudge:
    """Domain-agnostic judge implementation."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        s = get_settings()
        judge_name = model or s.model_for("benchmark_judge")
        self.judge_name = judge_name
        mcfg = s.model_config_for(judge_name)
        self.model = mcfg.get("api_model", judge_name)
        if llm is None:
            base_url = mcfg.get("base_url") or None
            api_key = mcfg.get("api_key") or None
            # enable_thinking 从 model_configs 中该模型自身的声明继承
            judge_thinking = bool(mcfg.get("enable_thinking", False))
            judge_extra_body = mcfg.get("extra_body") or None
            self.llm = LLMClient(base_url=base_url, api_key=api_key,
                                  enable_thinking=judge_thinking,
                                  extra_body=judge_extra_body)
        else:
            self.llm = llm
            # 若外部传入 llm，仍以 model_config 中的声明为准
            self.llm.enable_thinking = bool(mcfg.get("enable_thinking", False))
        self.temperature = temperature
        self.all_dims = load_rubric()
        self.pt_dims = per_turn_dimensions(self.all_dims)
        self.ho_dims = holistic_dimensions(self.all_dims)
        # Stable hashes of the rubric groups. Used to invalidate cached scores
        # when the rubric (or the judge subclass prompt) changes.
        judge_salt = self.__class__.__name__
        self.pt_hash = dimensions_hash(self.pt_dims) + "-" + judge_salt
        self.ho_hash = dimensions_hash(self.ho_dims) + "-" + judge_salt
        self.hard_constraints = load_hard_constraints()
        self.trace_root = s.path("outputs_dir") / "eval_traces"
        self.trace_root.mkdir(parents=True, exist_ok=True)

    # ── L0 Hard constraint check ──────────────────────────────────

    def check_l0(
        self,
        dialogue: List[Dict[str, Any]],
        persona: Persona,
        scenario: Scenario,
        evaluated_role: str,
    ) -> List[L0Violation]:
        """Check L0 hard constraints via keyword scan + repetition rules."""
        violations: List[L0Violation] = []

        # 1. Keyword-based scan
        for i, turn in enumerate(dialogue):
            if turn.get("role") != evaluated_role:
                continue
            text = _turn_content(turn).lower()
            for hc in self.hard_constraints:
                if not hc.keywords:
                    continue
                for kw in hc.keywords:
                    if kw.lower() in text:
                        violations.append(L0Violation(
                            constraint_id=hc.id,
                            constraint_name=hc.name,
                            severity=hc.severity,
                            evidence=f"Turn {i}: 关键词命中 '{kw}'",
                            turn_index=i,
                        ))
                        break  # one keyword per constraint per turn is enough

        # 2. Check for hard repetition (L0-05)
        # 真人会原样重发短消息（催促、报数字、确认），只有复读长句才判 fatal
        prev_texts: List[str] = []
        for i, turn in enumerate(dialogue):
            text = _turn_content(turn).strip()
            if turn.get("role") == evaluated_role and text:
                # Check exact copy of any previous message
                for j, prev in enumerate(prev_texts):
                    if text == prev and len(text) > 15:
                        violations.append(L0Violation(
                            constraint_id="L0-05",
                            constraint_name="无硬重复",
                            severity="fatal",
                            evidence=f"Turn {i}: 完全复制了之前的消息",
                            turn_index=i,
                        ))
                        break
                # Check copying opponent's last message
                if i > 0:
                    prev_turn = dialogue[i - 1]
                    if prev_turn.get("role") != evaluated_role:
                        prev_text = _turn_content(prev_turn).strip()
                        if text == prev_text and len(text) > 15:
                            violations.append(L0Violation(
                                constraint_id="L0-05",
                                constraint_name="无硬重复",
                                severity="fatal",
                                evidence=f"Turn {i}: 复制了对方刚说的话",
                                turn_index=i,
                            ))
            prev_texts.append(text)

        return violations

    def has_fatal_violation(self, violations: List[L0Violation]) -> bool:
        return any(v.severity == "fatal" for v in violations)

    # ── Per-turn scoring ──────────────────────────────────────────

    def score_turn(
        self,
        turn_index: int,
        turn: Dict[str, Any],
        dialogue: List[Dict[str, Any]],
        persona: Persona,
        scenario: Scenario,
    ) -> TurnScoreCard:
        """Score a single turn with context window."""
        role = turn.get("role", "")
        content = _turn_content(turn)
        sys_prompt = self._per_turn_system_prompt(persona, scenario)
        user_msg = self._per_turn_user_msg(turn_index, turn, dialogue)

        dim_results = self._judge_dimensions(sys_prompt, user_msg, self.pt_dims)
        avg = _avg_dim_scores(dim_results)
        weighted = _weighted_dim_score(dim_results)

        card = TurnScoreCard(
            turn_index=turn_index,
            role=role,
            content=content,
            dim_results=dim_results,
            avg_score=avg,
            weighted_score=weighted,
        )
        return card

    # ── Holistic scoring ──────────────────────────────────────────

    def score_holistic(
        self,
        dialogue: List[Dict[str, Any]],
        persona: Persona,
        scenario: Scenario,
        evaluated_role: str,
    ) -> HolisticScoreCard:
        sys_prompt = self._holistic_system_prompt(persona, scenario)
        user_msg = self._holistic_user_msg(dialogue, evaluated_role)

        dim_results = self._judge_dimensions(sys_prompt, user_msg, self.ho_dims)
        avg = _avg_dim_scores(dim_results)
        weighted = _weighted_dim_score(dim_results)

        return HolisticScoreCard(
            role=evaluated_role,
            dim_results=dim_results,
            avg_score=avg,
            weighted_score=weighted,
        )

    # ── LLM call ──────────────────────────────────────────────────

    def _call(self, sys_prompt: str, user_msg: str) -> Dict[str, Any]:
        """Call LLM and return parsed JSON dict."""
        return self.llm.chat_json(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            model=self.model,
            temperature=self.temperature,
        )

    _MAX_DIM_ATTEMPTS = 3

    def _judge_dimensions(
        self,
        sys_prompt: str,
        user_msg: str,
        dims: List[BenchmarkDimension],
    ) -> List[DimResult]:
        """Judge a prompt's dimensions, retrying incomplete responses.

        The judge LLM occasionally returns JSON that omits some dimensions;
        _parse_dim_results would silently zero-fill them. Retry until every
        dimension is present or attempts run out.
        """
        dim_results: List[DimResult] = []
        for attempt in range(1, self._MAX_DIM_ATTEMPTS + 1):
            data = self._call(sys_prompt, user_msg)
            dim_results = self._parse_dim_results(data, dims)
            missing = _missing_dim_ids(data, dims)
            if not missing:
                return dim_results
            LOGGER.warning(
                "judge %s response missing dims %s (attempt %d/%d)",
                self.judge_name, missing, attempt, self._MAX_DIM_ATTEMPTS,
            )
        return dim_results

    # ── Prompt builders (neutral defaults; subclasses override) ───

    _CONTEXT_WINDOW = 5

    def _per_turn_system_prompt(self, persona: Persona, scenario: Scenario) -> str:
        dims_block = self._dims_block(self.pt_dims)
        return (
            "你是一名专业的中文对话拟人度评测员，正在评估某个角色的**单轮回复**质量。\n\n"
            "## 被评角色信息\n"
            f"{persona.to_system_prompt()}\n\n"
            f"{scenario.to_system_prompt(self_id=persona.persona_id, partner_id='对方')}\n\n"
            "## 评测维度与 Checkbox\n"
            "对每个维度，逐条判断其下的 checkbox 是否满足（true/false），并给出一句简短理由。\n\n"
            "**判分逻辑**：\n"
            "每个 checkbox 描述的是一种真人中文对话中的优秀特征。判分规则：\n"
            "1. 该特征在当前轮次中**存在** → **true**\n"
            "2. 该特征在当前轮次中**不存在，且场景不适用**（checkbox 描述中标注的「不适用」条件满足时）→ **true**\n"
            "3. 该特征在当前轮次中**不存在，且场景适用** → **false**\n\n"
            "**关键**：严格判断每个 checkbox 描述的具体特征是否存在。不要因为回复整体看起来自然就将所有 checkbox 判 true。\n"
            "注意：包含 / 分隔的多条消息表示真人分条发送，每条只说一件事。\n\n"
            f"{dims_block}\n\n"
            "## 输出格式（只输出 JSON，不要任何其他文字）\n"
            "```json\n"
            '{\n'
            '  "dimensions": {\n'
            '    "<dim_id>": {\n'
            '      "checks": {"<checkbox_id>": true/false, ...},\n'
            '      "reason": "一句话理由"\n'
            '    },\n'
            '    ...\n'
            '  }\n'
            "}\n"
            "```\n"
        )

    def _per_turn_user_msg(
        self,
        turn_index: int,
        turn: Dict[str, Any],
        dialogue: List[Dict[str, Any]],
    ) -> str:
        # 取 [max(0, turn_index - window), turn_index] 的上下文窗口
        start = max(0, turn_index - self._CONTEXT_WINDOW)
        window = dialogue[start:turn_index + 1]

        context_lines = []
        if start > 0:
            context_lines.append(f"... (前 {start} 轮已省略) ...")
        for i, t in enumerate(window, start=start):
            role = t.get("role", "?")
            text = _turn_content(t)
            marker = " <<<< 请评价这一轮" if i == turn_index else ""
            context_lines.append(f"[Turn {i}] {role}: {text}{marker}")
        return (
            "以下是对话上下文（仅保留待评轮次及其前几轮），请针对标记为「请评价这一轮」的那条回复进行 checkbox 评分。\n\n"
            + "\n".join(context_lines)
        )

    def _holistic_system_prompt(self, persona: Persona, scenario: Scenario) -> str:
        dims_block = self._dims_block(self.ho_dims)
        return (
            "你是一名专业的中文对话拟人度评测员，正在评估某个角色在**整段对话**中的整体表现。\n\n"
            "注意：真实人类（或被设计成拟人风格的角色）的表达会自然波动，"
            "不会每轮都严格复现人设卡里的口头禅、长度或话题偏好。"
            "判分时应区分：语气、长度、话题、知识范围的起伏属于正常波动，不应判为不稳定；"
            "但身份、年龄、职业、个人经历等硬事实前后矛盾，或明显自相矛盾、记忆混乱导致角色可信度丧失，仍应判为不稳定。\n\n"
            "## 被评角色信息\n"
            f"{persona.to_system_prompt()}\n\n"
            f"{scenario.to_system_prompt(self_id=persona.persona_id, partner_id='对方')}\n\n"
            "## 评测维度与 Checkbox\n"
            "对每个维度，逐条判断其下的 checkbox 是否满足（true/false），并给出一句简短理由。\n\n"
            "**判分逻辑**：\n"
            "每个 checkbox 描述的是一种真人中文对话中的优秀特征。判分规则：\n"
            "1. 该特征在整段对话中**存在** → **true**\n"
            "2. 该特征在整段对话中**不存在，且场景不适用**（checkbox 描述中标注的「不适用」条件满足时）→ **true**\n"
            "3. 该特征在整段对话中**不存在，且场景适用** → **false**\n\n"
            "**关键**：严格判断每个 checkbox 描述的具体特征是否存在。不要因为对话整体看起来自然就将所有 checkbox 判 true。\n"
            "注意：包含 / 分隔的多条消息表示真人分条发送，每条只说一件事。\n\n"
            f"{dims_block}\n\n"
            "## E1 上下文一致性特别说明（零容忍原则）\n"
            "评估 E1_context_consistency 时，采用'零容忍'原则：只要被评角色在整段对话中出现过一次对应问题，该 checkbox 就必须判 false。\n"
            "每个 checkbox 的 reason 必须引用具体的 Turn 编号作为证据；如果无法指出具体 Turn，必须判 false。\n"
            "1. no_repeat_ask：被评角色是否在对方已经明确回答后，再次询问相同或等价问题（包括换措辞、同一条消息内重复、连续追问）。如有，必须判 false。\n"
            "2. facts_coherent：被评角色自述的身份、职业、状态、经历、时间线是否与对话历史中的 earlier 陈述或对方陈述矛盾。如有，必须判 false。\n"
            "3. speaker_reference_correct：被评角色是否把对方经历说成自己的，或把对方观点错误归因。如有，必须判 false。\n"
            "4. state_tracking：被评角色是否混淆对话状态（如把已确认事项当作未决问题）。如有，必须判 false。\n"
            "5. no_memory_confusion：被评角色是否把之前已确认的信息当作新信息再次质疑，或前后记忆不一致。如有，必须判 false。\n\n"
            "## 输出格式（只输出 JSON，不要任何其他文字）\n"
            "```json\n"
            '{\n'
            '  "dimensions": {\n'
            '    "<dim_id>": {\n'
            '      "checks": {"<checkbox_id>": true/false, ...},\n'
            '      "reason": "一句话理由（需包含具体 Turn 编号与证据）"\n'
            '    },\n'
            '    ...\n'
            '  }\n'
            "}\n"
            "```\n"
        )

    def _holistic_user_msg(
        self,
        dialogue: List[Dict[str, Any]],
        evaluated_role: str,
    ) -> str:
        lines = []
        for i, t in enumerate(dialogue):
            role = t.get("role", "?")
            tag = " [被评角色]" if role == evaluated_role else ""
            msgs = _turn_messages(t)
            if len(msgs) == 1:
                lines.append(f"[Turn {i}] {role}{tag}: {msgs[0]}")
            else:
                lines.append(f"[Turn {i}] {role}{tag}:（{len(msgs)}条消息）")
                for m in msgs:
                    lines.append(f"  → {m}")
        return (
            f"以下是完整对话。请针对角色 {evaluated_role}（标记为 [被评角色]）在整段对话中的表现进行 checkbox 评分。\n"
            "注意：每个 Turn 中用 → 标记的每条是独立发送的消息（真人分条发送），评估消息长度、标点等特征时请以 → 标记的单条消息为单位。\n\n"
            + "\n".join(lines)
        )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _dims_block(dims: List[BenchmarkDimension]) -> str:
        parts = []
        for d in dims:
            cb_lines = "\n".join(
                f"  - `{cb.id}`: {cb.description}" for cb in d.checkboxes
            )
            parts.append(
                f"### {d.id} ({d.name}, 权重={d.weight})\n{d.description}\n{cb_lines}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _parse_dim_results(
        data: Dict[str, Any],
        dims: List[BenchmarkDimension],
    ) -> List[DimResult]:
        raw_dims = data.get("dimensions") or {}
        results = []
        for d in dims:
            raw = raw_dims.get(d.id) or {}
            if not raw:
                prefix = d.id.split("-")[0]
                raw = raw_dims.get(prefix) or {}
            raw_checks = raw.get("checks") or {}
            checks: Dict[str, bool] = {}
            checked = 0
            for cb in d.checkboxes:
                val = bool(raw_checks.get(cb.id, False))
                checks[cb.id] = val
                if val:
                    checked += 1
            score = checked / max(1, d.checkbox_count)
            results.append(DimResult(
                dim_id=d.id,
                dim_name=d.name,
                weight=d.weight,
                checks=checks,
                reason=str(raw.get("reason", "")).strip(),
                score=round(score, 4),
            ))
        return results

    # ── Trace saving ──────────────────────────────────────────────

    def save_trace(
        self,
        case_id: str,
        turn_cards: List[TurnScoreCard],
        holistic_card: HolisticScoreCard,
        dialogue: List[Dict[str, Any]],
        model_name: str | Path,
        l0_violations: Optional[List[L0Violation]] = None,
        status: str = "valid",
    ) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(model_name, Path):
            out_dir = self.trace_root / model_name
        else:
            out_dir = self.trace_root / slug(model_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{case_id}_{ts}.json"
        payload = {
            "case_id": case_id,
            "model": str(model_name),
            "status": status,
            "evaluated_role": holistic_card.role,
            "pt_hash": self.pt_hash,
            "ho_hash": self.ho_hash,
            "l0_violations": [
                {
                    "constraint_id": v.constraint_id,
                    "constraint_name": v.constraint_name,
                    "severity": v.severity,
                    "evidence": v.evidence,
                    "turn_index": v.turn_index,
                }
                for v in (l0_violations or [])
            ],
            "turn_scores": self._turn_cards_to_dict(turn_cards),
            "holistic_scores": self._holistic_card_to_dict(holistic_card),
            "dialogue": dialogue,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

        # Also save per-group traces so per-turn / holistic can be reused
        # independently when only one group changes.
        l0_list = [
            {
                "constraint_id": v.constraint_id,
                "constraint_name": v.constraint_name,
                "severity": v.severity,
                "evidence": v.evidence,
                "turn_index": v.turn_index,
            }
            for v in (l0_violations or [])
        ]
        self._save_group_trace(
            out_dir=out_dir,
            case_id=case_id,
            ts=ts,
            group="per_turn",
            group_hash=self.pt_hash,
            evaluated_role=holistic_card.role,
            status=status,
            model_name=str(model_name),
            l0_violations=l0_list,
            data={"turn_scores": self._turn_cards_to_dict(turn_cards)},
        )
        self._save_group_trace(
            out_dir=out_dir,
            case_id=case_id,
            ts=ts,
            group="holistic",
            group_hash=self.ho_hash,
            evaluated_role=holistic_card.role,
            status=status,
            model_name=str(model_name),
            l0_violations=l0_list,
            data={"holistic_scores": self._holistic_card_to_dict(holistic_card)},
        )
        return path

    @staticmethod
    def _turn_cards_to_dict(turn_cards: List[TurnScoreCard]) -> List[Dict[str, Any]]:
        return [
            {
                "turn_index": tc.turn_index,
                "role": tc.role,
                "content": tc.content,
                "avg_score": tc.avg_score,
                "weighted_score": tc.weighted_score,
                "dimensions": [
                    {
                        "dim_id": dr.dim_id,
                        "weight": dr.weight,
                        "score": dr.score,
                        "checks": dr.checks,
                        "reason": dr.reason,
                    }
                    for dr in tc.dim_results
                ],
            }
            for tc in turn_cards
        ]

    @staticmethod
    def _holistic_card_to_dict(holistic_card: HolisticScoreCard) -> Dict[str, Any]:
        return {
            "avg_score": holistic_card.avg_score,
            "weighted_score": holistic_card.weighted_score,
            "dimensions": [
                {
                    "dim_id": dr.dim_id,
                    "weight": dr.weight,
                    "score": dr.score,
                    "checks": dr.checks,
                    "reason": dr.reason,
                }
                for dr in holistic_card.dim_results
            ],
        }

    def _save_group_trace(
        self,
        out_dir: Path,
        case_id: str,
        ts: str,
        group: str,
        group_hash: str,
        evaluated_role: str,
        status: str,
        model_name: str,
        l0_violations: List[Dict[str, Any]],
        data: Dict[str, Any],
    ) -> Path:
        path = out_dir / f"{case_id}_{ts}_{group}.json"
        payload = {
            "case_id": case_id,
            "group": group,
            "group_hash": group_hash,
            "judge_name": self.judge_name,
            "judge_class": self.__class__.__name__,
            "model": model_name,
            "status": status,
            "evaluated_role": evaluated_role,
            "l0_violations": l0_violations,
            **data,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path


# ── Utility functions ─────────────────────────────────────────────────

def _turn_content(turn: Dict[str, Any]) -> str:
    c = turn.get("content")
    if isinstance(c, str) and c.strip():
        return c.strip()
    parts = []
    for m in (turn.get("response") or []):
        txt = (m.get("content") or "").strip()
        if txt:
            parts.append(txt)
    return " / ".join(parts)


def _turn_messages(turn: Dict[str, Any]) -> List[str]:
    c = turn.get("content")
    if isinstance(c, str) and c.strip():
        return [c.strip()]
    parts = []
    for m in (turn.get("response") or []):
        txt = (m.get("content") or "").strip()
        if txt:
            parts.append(txt)
    return parts or [""]


def _avg_dim_scores(results: List[DimResult]) -> float:
    if not results:
        return 0.0
    return round(sum(r.score for r in results) / len(results), 4)


def _weighted_dim_score(results: List[DimResult]) -> float:
    """Compute weight-normalized score → [0, 1]."""
    if not results:
        return 0.0
    total_weight = sum(r.weight for r in results)
    if total_weight == 0:
        return _avg_dim_scores(results)
    weighted = sum(r.score * r.weight for r in results) / total_weight
    return round(weighted, 4)


def _missing_dim_ids(
    data: Dict[str, Any],
    dims: List[BenchmarkDimension],
) -> List[str]:
    """Return dimension ids absent from a judge response (prefix fallback applies)."""
    raw_dims = data.get("dimensions") or {}
    missing: List[str] = []
    for d in dims:
        if d.id in raw_dims or d.id.split("-")[0] in raw_dims:
            continue
        missing.append(d.id)
    return missing
