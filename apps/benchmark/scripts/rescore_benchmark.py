#!/usr/bin/env python
"""Re-score benchmark results from cached eval_traces WITHOUT regenerating dialogues.

Rebuilds every CaseResult from outputs/<...>/eval_traces/, then re-applies:
  - L0 hard-constraint rules (deterministic, current code)
  - the human-style gate
  - current domain score thresholds (chatbot >= 95; game/clam >= 82.5)
and rewrites results/<...>/ summary + leaderboards.

Cases that were L0-fatal under OLD rules but are valid under NEW rules have no
cached turn scores; they are re-judged via the judge LLM (use --no-rejudge to
skip them instead).

Usage::

    python apps/benchmark/scripts/rescore_benchmark.py \
        --config configs/chatbot/qwen3.6-35b-a3b.yaml \
        --model Qwen3.6-35B-A3B
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core.config import get_settings, load_settings, setup_logging  # noqa: E402
from libs.chat.persona import Persona  # noqa: E402
from libs.chat.scenario import Scenario  # noqa: E402
from libs.evaluation import (  # noqa: E402
    ChatbotBenchmarkJudge, GameBenchmarkJudge, GeneralBenchmarkJudge,
)
from libs.evaluation.core.judge import (  # noqa: E402
    BaseBenchmarkJudge, DimResult, TurnScoreCard, HolisticScoreCard,
)
from libs.evaluation.report import (  # noqa: E402
    BenchmarkReport, CaseResult,
    write_summary, write_leaderboard, write_global_leaderboard,
)
from apps.benchmark.src.runner import (  # noqa: E402
    _score_turn_with_retry, _score_holistic_with_retry, slug, BenchmarkRunner,
    reconstruct_untraced_cases,
)

_DOMAIN_TO_JUDGE = {
    "chatbot": ChatbotBenchmarkJudge,
    "game": GameBenchmarkJudge,
    "general": GeneralBenchmarkJudge,
    "cdial": GeneralBenchmarkJudge,
    "clam": GeneralBenchmarkJudge,
}

LOGGER = logging.getLogger(__name__)


def _parse_case_id(case_id: str):
    """P713A22_PFBFF69_A0001_B0001_C0041_as_a -> (pa, pb, sid, tested_role)."""
    parts = case_id.split("_")
    pa, pb = parts[0], parts[1]
    suffix = parts[-1]
    sid = "_".join(parts[2:-2])
    tested_role = "persona_a" if suffix == "a" else "persona_b"
    return pa, pb, sid, tested_role


def _dims_from_trace(dim_list: List[Dict[str, Any]]) -> List[DimResult]:
    return [
        DimResult(
            dim_id=dd.get("dim_id", ""),
            dim_name=dd.get("dim_name", dd.get("dim_id", "")),
            weight=dd.get("weight", 10),
            checks=dd.get("checks", {}),
            reason=dd.get("reason", ""),
            score=dd.get("score", 0.0),
        )
        for dd in (dim_list or [])
    ]


def rescore_trace(
    trace_path: Path,
    judge: BaseBenchmarkJudge,
    pt_weight: float,
    ho_weight: float,
    rejudge: bool,
    domain: str,
) -> Optional[CaseResult]:
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("skip unreadable trace %s: %s", trace_path.name, exc)
        return None

    case_id = data.get("case_id", "")
    dialogue = data.get("dialogue") or []
    if not case_id:
        return None

    pa, pb, sid, tested_role = _parse_case_id(case_id)
    tested_id = pa if tested_role == "persona_a" else pb

    try:
        persona = Persona.load(tested_id)
        scenario = Scenario.load(sid)
    except Exception as exc:
        LOGGER.warning("skip %s: cannot load persona/scenario (%s)", case_id, exc)
        return None

    # Re-run deterministic L0 under CURRENT rules
    l0 = judge.check_l0(dialogue, persona, scenario, evaluated_role=tested_id)
    is_fatal = judge.has_fatal_violation(l0)

    turn_cards = [
        TurnScoreCard(
            turn_index=tc.get("turn_index", 0),
            role=tc.get("role", ""),
            content=tc.get("content", ""),
            dim_results=_dims_from_trace(tc.get("dimensions")),
            avg_score=tc.get("avg_score", 0.0),
            weighted_score=tc.get("weighted_score", 0.0),
        )
        for tc in (data.get("turn_scores") or [])
    ]
    ho = data.get("holistic_scores") or {}
    if not ho.get("dimensions"):
        ho_path = trace_path.with_name(f"{trace_path.stem}_holistic.json")
        if ho_path.exists():
            try:
                ho_data = json.loads(ho_path.read_text(encoding="utf-8"))
                ho = ho_data.get("holistic_scores") or ho
            except Exception as exc:
                LOGGER.warning("skip holistic trace %s: %s", ho_path.name, exc)
    holistic_card = HolisticScoreCard(
        role=tested_id,
        dim_results=_dims_from_trace(ho.get("dimensions")),
        avg_score=ho.get("avg_score", 0.0),
        weighted_score=ho.get("weighted_score", 0.0),
    )

    # 旧规则判 fatal 而新规则放行的 case 没有判分卡 → 需要重新过 judge
    if not is_fatal and not turn_cards and dialogue:
        if not rejudge:
            LOGGER.warning("%s: newly valid but no cached scores; skipped "
                           "(rerun with rejudge enabled)", case_id)
            return None
        LOGGER.info("%s: newly valid under current L0 rules -> re-judging", case_id)
        tasks = [(i, t) for i, t in enumerate(dialogue)
                 if t.get("role") == tested_id]
        for i, turn in tasks:
            try:
                turn_cards.append(_score_turn_with_retry(
                    judge, i, turn, dialogue, persona, scenario))
            except Exception as exc:
                LOGGER.warning("%s: turn %d re-judge failed: %s", case_id, i, exc)
        try:
            holistic_card = _score_holistic_with_retry(
                judge, dialogue, persona, scenario, tested_id)
        except Exception as exc:
            LOGGER.warning("%s: holistic re-judge failed: %s", case_id, exc)
        turn_cards.sort(key=lambda tc: tc.turn_index)

    case = CaseResult(
        case_id=case_id,
        persona_a_id=pa,
        persona_b_id=pb,
        scenario_id=sid,
        tested_role=tested_role,
        category=str(scenario.raw.get("category", "")),
        sub_category=str(scenario.raw.get("sub_category", "")),
        turn_cards=turn_cards,
        holistic_card=holistic_card,
        l0_violations=l0,
        domain=domain,
    )
    case.apply_style_gate(dialogue, tested_id)
    case.compute(pt_weight=pt_weight, ho_weight=ho_weight)
    return case


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", default=None,
                   help="tested model name (eval_traces subdir, e.g. Qwen3.6-35B-A3B; "
                        "defaults to benchmark.tested_model in config)")
    p.add_argument("--ref-model", default=None,
                   help="ref model name (default: from config)")
    p.add_argument("--judge-model", default=None,
                   help="judge model name (default: from config)")
    p.add_argument("--no-rejudge", action="store_true",
                   help="do not re-judge newly-valid cases via LLM")
    args = p.parse_args()

    load_settings(args.config)
    setup_logging()
    s = get_settings()

    model = args.model or s.get("benchmark", "tested_model")
    if not model:
        sys.exit("--model is required when benchmark.tested_model is not set in config")

    ref_model = args.ref_model or s.model_for("ref_partner")
    judge_model = args.judge_model or s.model_for("benchmark_judge")

    trace_dir = (s.path("outputs_dir") / "eval_traces"
                 / f"judge_{slug(judge_model)}"
                 / slug(model))
    if not trace_dir.exists():
        sys.exit(f"no eval traces at {trace_dir}")

    domain = BenchmarkRunner._detect_domain()
    judge_cls = _DOMAIN_TO_JUDGE.get(domain, GameBenchmarkJudge)
    judge = judge_cls()
    pt_weight = float(s.get("benchmark", "per_turn_weight", default=0.5))
    ho_weight = float(s.get("benchmark", "holistic_weight", default=0.5))

    files = sorted(
        f for f in trace_dir.glob("*.json")
        if not f.name.endswith("_holistic.json")
        and not f.name.endswith("_per_turn.json")
    )
    print(f"rescoring {len(files)} cached cases from {trace_dir}")
    seen: Dict[str, CaseResult] = {}
    for f in files:
        case = rescore_trace(f, judge, pt_weight, ho_weight,
                             rejudge=not args.no_rejudge,
                             domain=domain)
        if case:
            seen[case.case_id] = case  # later trace wins (same case re-run)
    cases = list(seen.values())

    # L0-fatal cases never get eval traces; rebuild them from the dialogue
    # files so invalid cases are not silently dropped from the report.
    cases.extend(reconstruct_untraced_cases(
        tested_model=model,
        ref_model=ref_model,
        self_play=bool(s.get("benchmark", "self_play", default=False)),
        judge=judge,
        domain=domain,
        traced_case_ids=set(seen),
        pt_weight=pt_weight,
        ho_weight=ho_weight,
        rejudge=not args.no_rejudge,
    ))

    report = BenchmarkReport(
        model_name=model,
        ref_model=ref_model,
        judge_model=judge_model,
        cases=cases,
    )
    report.compute()

    print(f"\n{'='*60}")
    print(f"  Model:           {report.model_name} (rescored, no regeneration)")
    print(f"  Cases:           {len(cases)}")
    print(f"  Valid rate:      {report.valid_rate:.0%}")
    print(f"  ACC:             {report.acc:.2%}  (all rubric checkboxes pass)")
    print(f"  Style gate rate: {report.style_gate_rate:.2%}")
    print(f"  Final score:     {report.final_score_100:.1f}/100 ({report.grade})")
    print(f"{'='*60}")

    summary_path = write_summary([report])
    lb = write_leaderboard(judge_model=judge_model)
    glb = write_global_leaderboard()
    print(f"\n[OK] summary -> {summary_path}")
    print(f"[OK] leaderboard -> {lb}")
    print(f"[OK] global leaderboard -> {glb}")


if __name__ == "__main__":
    main()
