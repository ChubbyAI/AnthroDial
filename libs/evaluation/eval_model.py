"""Online evaluation: spin up dialogues with the model under test, then judge.

For every (persona_a, persona_b, scenario) combination, we:
1. Create a dedicated ``LLMClient`` from the per-model config (base_url +
   api_key_env), so different providers can be benchmarked through one entry.
2. Build a fresh ``ChatSession`` whose two agents are forced to use the model
   under test for every reply.
3. Run the session, save the dialogue under ``outputs/dialogues/`` (tagged
   with the model name), then ask the judge (default project judge model)
   to score it.
4. Aggregate per-model scores into a ``ModelReport`` and dump a
   ``results/eval_summary/<run_ts>/summary.md`` table for human inspection.
"""
from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.chat.agent import ChatAgent
from libs.chat.session import ChatSession, save_dialogue
from libs.core.config import REPO_ROOT, get_settings, slug
from libs.evaluation.light_judge import Judge, ScoreCard
from libs.evaluation.light_rubric import load_dimensions
from libs.llm.client import LLMClient

LOGGER = logging.getLogger(__name__)

_LEGACY_MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"


@dataclass
class ModelReport:
    name: str
    cards: List[ScoreCard] = field(default_factory=list)
    sample_count: int = 0
    avg_overall: float = 0.0
    avg_per_dim: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "avg_overall": self.avg_overall,
            "avg_per_dim": self.avg_per_dim,
            "cards": [c.to_dict() for c in self.cards],
            **({"extra": self.extra} if self.extra else {}),
        }


def load_models_yaml(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = Path(path) if path else _LEGACY_MODELS_YAML
    if not p.exists():
        s = get_settings()
        cfgs = s.get("llm", "model_configs", default={}) or {}
        return [{"name": k, **v} for k, v in cfgs.items() if isinstance(v, dict)]
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get("models") or []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            out.append(item)
    return out


def _list_seed_ids(subdir: str) -> List[str]:
    s = get_settings()
    seeds = s.path("seeds_dir") / subdir
    if not seeds.exists():
        return []
    # 支持子目录结构（BP01/card.yaml）和平铺文件（BS01.yaml）
    out: List[str] = []
    for p in sorted(seeds.iterdir()):
        if p.is_dir() and (p / "card.yaml").exists():
            out.append(p.name)
        elif p.is_file() and p.suffix == ".yaml":
            out.append(p.stem)
    return out


def default_combos(sample: int = 3, seed: int = 7) -> List[Tuple[str, str, str]]:
    personas = _list_seed_ids("personas")
    scenarios = _list_seed_ids("scenarios")
    if len(personas) < 2 or not scenarios:
        return []
    rng = random.Random(seed)
    pairs = [(a, b) for a, b in itertools.combinations(personas, 2)]
    rng.shuffle(pairs)
    rng.shuffle(scenarios)
    combos: List[Tuple[str, str, str]] = []
    for i in range(min(sample, max(len(pairs), 1) * len(scenarios))):
        a, b = pairs[i % len(pairs)]
        s_id = scenarios[i % len(scenarios)]
        combos.append((a, b, s_id))
    return combos[:sample]


def _build_llm_for_model(cfg: Dict[str, Any]) -> LLMClient:
    base_url = cfg.get("base_url") or None
    api_key_env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    return LLMClient(base_url=base_url, api_key_env=api_key_env)


def _sanitize(name: str) -> str:
    return slug(name)


def _run_one_session(model_name: str,
                     llm_under_test: LLMClient,
                     pa_id: str, pb_id: str, s_id: str,
                     max_turns: Optional[int] = None) -> Dict[str, Any]:
    persona_a = Persona.load(pa_id)
    persona_b = Persona.load(pb_id)
    scenario = Scenario.load(s_id)
    agent_a = ChatAgent.build(persona_a, scenario, persona_b.persona_id,
                              llm_under_test, model=model_name)
    agent_b = ChatAgent.build(persona_b, scenario, persona_a.persona_id,
                              llm_under_test, model=model_name)
    session = ChatSession(
        agent_a=agent_a,
        agent_b=agent_b,
        scenario=scenario,
        max_turns=max_turns,
        opener=scenario.raw.get("opener"),
    )
    result = session.run()
    save_dialogue(result, tag=f"eval_{_sanitize(model_name)}")
    return {"persona_a": persona_a, "persona_b": persona_b,
            "scenario": scenario, "result": result}


def evaluate_model(cfg: Dict[str, Any],
                   combos: List[Tuple[str, str, str]],
                   max_turns: Optional[int] = None,
                   judge: Optional[Judge] = None) -> ModelReport:
    name = str(cfg.get("name") or "unknown")
    llm_under_test = _build_llm_for_model(cfg)
    judge = judge or Judge()

    cards: List[ScoreCard] = []
    for (pa, pb, s_id) in combos:
        try:
            ctx = _run_one_session(name, llm_under_test, pa, pb, s_id, max_turns)
        except Exception as exc:
            LOGGER.exception("[%s] session %s/%s/%s failed: %s",
                             name, pa, pb, s_id, exc)
            continue
        try:
            card = judge.score(
                dialogue=ctx["result"].dialogue,
                persona1=ctx["persona_a"],
                persona2=ctx["persona_b"],
                scenario=ctx["scenario"],
                tag=f"online_{_sanitize(name)}",
                mode="online",
                model=name,
            )
        except Exception as exc:
            LOGGER.exception("[%s] judge for %s/%s/%s failed: %s",
                             name, pa, pb, s_id, exc)
            continue
        cards.append(card)

    return _aggregate(name, cards)


def _aggregate(name: str, cards: List[ScoreCard]) -> ModelReport:
    rep = ModelReport(name=name, cards=cards, sample_count=len(cards))
    if not cards:
        return rep
    dims = [d.id for d in load_dimensions()]
    per_dim_sum: Dict[str, float] = {d: 0.0 for d in dims}
    overall_sum = 0.0
    for c in cards:
        overall_sum += c.overall
        for d in dims:
            per_dim_sum[d] += float(c.scores.get(d, 0))
    n = len(cards)
    rep.avg_overall = round(overall_sum / n, 3)
    rep.avg_per_dim = {d: round(per_dim_sum[d] / n, 3) for d in dims}
    return rep


def write_summary(reports: List[ModelReport],
                  out_path: Optional[Path] = None) -> Path:
    """默认写入 results/eval_summary/<run_ts>/summary.md (+ summary.json)。"""
    s = get_settings()
    if out_path is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = s.path("results_dir") / "eval_summary" / run_ts
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "summary.md"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    dims = [d.id for d in load_dimensions()]
    header = ["model", "samples", "overall"] + dims
    sep = ["---"] * len(header)

    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Evaluation Summary ({ts})")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for r in reports:
        row = [r.name, str(r.sample_count), f"{r.avg_overall:.3f}"]
        for d in dims:
            row.append(f"{r.avg_per_dim.get(d, 0):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps([r.to_dict() for r in reports], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
