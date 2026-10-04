"""LLM-as-judge scorer.

Given a complete dialogue + persona/scenario context, score it across each
dimension defined in ``rubric.py``. Each dialogue → one judge call that
returns a JSON object with per-dimension {score, reason}.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.core.config import get_settings, slug
from libs.evaluation.light_rubric import Dimension, load_dimensions, score_range
from libs.llm.client import LLMClient

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoreCard:
    persona1_id: str
    persona2_id: str
    scenario_id: str
    scores: Dict[str, int] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)
    overall: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona1_id": self.persona1_id,
            "persona2_id": self.persona2_id,
            "scenario_id": self.scenario_id,
            "scores": self.scores,
            "reasons": self.reasons,
            "overall": self.overall,
        }


class Judge:
    def __init__(self, llm: Optional[LLMClient] = None,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None) -> None:
        s = get_settings()
        self.llm = llm or LLMClient()
        self.model = model or s.model_for("judge")
        self.temperature = (s.temperature_for("judge", 0.0)
                            if temperature is None else temperature)
        self.dimensions: List[Dimension] = load_dimensions()
        self.score_min, self.score_max = score_range()
        # eval_traces 根目录，实际路径还会包含 <mode>/<model>/<scenario>。
        self.trace_root = s.path("outputs_dir") / "eval_traces"
        self.trace_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def score(
        self,
        dialogue: List[Dict[str, Any]],
        persona1: Persona,
        persona2: Persona,
        scenario: Scenario,
        tag: str = "",
        mode: str = "online",
        model: Optional[str] = None,
    ) -> ScoreCard:
        sys_prompt = self._system_prompt()
        user_payload = {
            "persona1": persona1.raw,
            "persona2": persona2.raw,
            "scenario": scenario.raw,
            "dialogue": dialogue,
        }
        try:
            data = self.llm.chat_json(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",
                     "content": "请按系统提示给这段对话打分，只输出 JSON。\n"
                                + json.dumps(user_payload, ensure_ascii=False)},
                ],
                model=self.model,
                temperature=self.temperature,
            )
        except Exception as exc:
            LOGGER.exception("judge call failed: %s", exc)
            data = {}

        card = self._parse(data, persona1, persona2, scenario)
        self._save_trace(card, dialogue, data, tag=tag, mode=mode, model=model)
        return card

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        dim_lines = []
        for d in self.dimensions:
            dim_lines.append(f"- `{d.id}` ({d.name})：{d.description}")
        dim_block = "\n".join(dim_lines)
        return (
            "你是一名专业的中文对话拟人度评测员。请阅读完整对话和上下文，"
            f"按照下面 {len(self.dimensions)} 个维度逐项打 {self.score_min}~{self.score_max} 分整数 "
            "（分数越高越好），并附 1 句中文理由。\n\n"
            f"## 评分维度\n{dim_block}\n\n"
            "## 输出格式（只输出 JSON，不要任何其他文字）\n"
            "```json\n"
            '{\n  "scores": { "concise": 4, "knowledge_bounded": 5, ... },\n'
            '  "reasons": { "concise": "...", ... },\n'
            '  "overall_reason": "..."\n}\n'
            "```\n\n"
            "## 评分要点\n"
            "- 综合考虑对话整体表现，不要被单条偶发瑕疵拉低或拉高。\n"
            "- 对每个维度都要给出该维度的独立分数和理由。\n"
            "- 缺失的维度按照默认 3 分处理是不允许的，必须每维都打。\n"
        )

    def _parse(self, data: Dict[str, Any],
               persona1: Persona, persona2: Persona,
               scenario: Scenario) -> ScoreCard:
        scores_raw: Dict[str, Any] = data.get("scores") or {}
        reasons_raw: Dict[str, Any] = data.get("reasons") or {}
        scores: Dict[str, int] = {}
        reasons: Dict[str, str] = {}
        for d in self.dimensions:
            v = scores_raw.get(d.id)
            try:
                v_int = int(round(float(v)))
            except (TypeError, ValueError):
                v_int = self.score_min
            scores[d.id] = max(self.score_min, min(self.score_max, v_int))
            reasons[d.id] = str(reasons_raw.get(d.id, "")).strip()
        overall = round(sum(scores.values()) / max(1, len(scores)), 3)
        return ScoreCard(
            persona1_id=persona1.persona_id,
            persona2_id=persona2.persona_id,
            scenario_id=scenario.scenario_id,
            scores=scores,
            reasons=reasons,
            overall=overall,
            raw=data,
        )

    def _save_trace(self, card: ScoreCard, dialogue: List[Dict[str, Any]],
                    raw: Dict[str, Any], tag: str = "",
                    mode: str = "online", model: Optional[str] = None) -> None:
        """写入 outputs/eval_traces/<mode>/<model>/<scenario>/<P1>_<P2>_<ts>.json。"""
        s = get_settings()
        model = model or s.model_for("judge")
        out_dir = (self.trace_root / slug(mode) / slug(model)
                   / slug(card.scenario_id))
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{tag}_" if tag else ""
        name = f"{prefix}{card.persona1_id}_{card.persona2_id}_{ts}.json"
        path = out_dir / name
        payload = {
            "scorecard": card.to_dict(),
            "raw_judge_output": raw,
            "dialogue": dialogue,
            "meta": {"mode": mode, "model": model},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
