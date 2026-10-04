#!/usr/bin/env python
"""Re-judge LLM benchmark dialogues with the current judge prompt (new N/A rule).

Reads dialogues from existing eval_traces, re-runs per-turn + holistic judge LLM,
and rewrites the benchmark report. Does NOT regenerate dialogues.

Usage::
    python apps/benchmark/scripts/rejudge_benchmark.py \
        --config configs/game/qwen3.6-35b-a3b.yaml \
        --model Qwen3.6-35B-A3B
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core.config import get_settings, load_settings, setup_logging, slug  # noqa: E402
from libs.chat.persona import Persona  # noqa: E402
from libs.chat.scenario import Scenario  # noqa: E402
from libs.evaluation import (  # noqa: E402
    ChatbotBenchmarkJudge, GameBenchmarkJudge, GeneralBenchmarkJudge,
)
from libs.evaluation.core.judge import BaseBenchmarkJudge  # noqa: E402
from libs.evaluation.report import (  # noqa: E402
    BenchmarkReport, CaseResult,
    write_summary, write_leaderboard, write_global_leaderboard,
)
from apps.benchmark.src.runner import (  # noqa: E402
    _score_turn_with_retry, _score_holistic_with_retry, BenchmarkRunner,
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
    parts = case_id.split("_")
    pa, pb = parts[0], parts[1]
    suffix = parts[-1]
    sid = "_".join(parts[2:-2])
    tested_role = "persona_a" if suffix == "a" else "persona_b"
    return pa, pb, sid, tested_role


def rejudge_one(
    trace_path: Path,
    judge: BaseBenchmarkJudge,
    pt_weight: float,
    ho_weight: float,
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

    l0 = judge.check_l0(dialogue, persona, scenario, evaluated_role=tested_id)

    evaluated_turns = [
        (i, t) for i, t in enumerate(dialogue) if t.get("role") == tested_id
    ]

    turn_cards = []
    for i, turn in evaluated_turns:
        try:
            card = _score_turn_with_retry(
                judge, i, turn, dialogue, persona, scenario)
            turn_cards.append(card)
        except Exception as exc:
            LOGGER.warning("%s: turn %d judge failed: %s", case_id, i, exc)

    holistic_card = None
    try:
        holistic_card = _score_holistic_with_retry(
            judge, dialogue, persona, scenario, tested_id)
    except Exception as exc:
        LOGGER.warning("%s: holistic judge failed: %s", case_id, exc)

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
    p.add_argument("--model", required=True,
                   help="tested model name (eval_traces subdir)")
    p.add_argument("--ref-model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--sample", type=int, default=None,
                   help="only re-judge N cases for quick testing")
    args = p.parse_args()

    load_settings(args.config)
    setup_logging()
    s = get_settings()

    ref_model = args.ref_model or s.model_for("ref_partner")
    judge_model = args.judge_model or s.model_for("benchmark_judge")

    trace_dir = (s.path("outputs_dir") / "eval_traces"
                 / f"judge_{slug(judge_model)}"
                 / slug(args.model))
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
    if args.sample and args.sample < len(files):
        files = files[:args.sample]
    print(f"re-judging {len(files)} cases from {trace_dir}")

    seen: Dict[str, CaseResult] = {}
    if args.concurrency <= 1:
        for i, f in enumerate(files, 1):
            LOGGER.info("[%d/%d] %s", i, len(files), f.stem)
            case = rejudge_one(f, judge, pt_weight, ho_weight, domain)
            if case:
                seen[case.case_id] = case
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(rejudge_one, f, judge, pt_weight, ho_weight, domain): f
                       for f in files}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    case = future.result()
                    if case:
                        seen[case.case_id] = case
                        LOGGER.info("[%d/%d] %s: %.1f", i, len(files),
                                    case.case_id, case.final_score_100)
                except Exception as exc:
                    LOGGER.warning("[%d/%d] failed: %s", i, len(files), exc)

    cases = list(seen.values())

    # L0-fatal cases never get eval traces; rebuild them from the dialogue
    # files so invalid cases are not silently dropped from the report.
    cases.extend(reconstruct_untraced_cases(
        tested_model=args.model,
        ref_model=ref_model,
        self_play=bool(s.get("benchmark", "self_play", default=False)),
        judge=judge,
        domain=domain,
        traced_case_ids=set(seen),
        pt_weight=pt_weight,
        ho_weight=ho_weight,
        rejudge=True,
    ))

    report = BenchmarkReport(
        model_name=args.model,
        ref_model=ref_model,
        judge_model=judge_model,
        cases=cases,
    )
    report.compute()

    print(f"\n{'='*60}")
    print(f"  Model:           {report.model_name} (re-judged)")
    print(f"  Cases:           {len(cases)}")
    print(f"  Valid rate:      {report.valid_rate:.0%}")
    print(f"  ACC:             {report.acc:.2%}")
    print(f"  Final score:     {report.final_score_100:.1f}/100 ({report.grade})")
    print(f"  Per-turn:        {report.per_turn_avg:.4f}")
    print(f"  Holistic:        {report.holistic_avg:.4f}")
    print(f"{'='*60}")
    for dim_id in report.per_dim_scores:
        score = report.per_dim_scores[dim_id]
        dim_acc = report.per_dim_acc.get(dim_id, 0)
        print(f"  {dim_id}: {score:.4f} | ACC={dim_acc:.2%}")

    summary_path = write_summary([report])
    lb = write_leaderboard(judge_model=judge_model)
    glb = write_global_leaderboard()
    print(f"\n[OK] summary -> {summary_path}")
    print(f"[OK] leaderboard -> {lb}")
    print(f"[OK] global leaderboard -> {glb}")


if __name__ == "__main__":
    main()
