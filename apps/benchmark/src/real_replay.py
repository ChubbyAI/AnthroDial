"""Replay-evaluate REAL game dialogues through the benchmark judge pipeline.

Benchmark mode ``benchmark.mode: real_replay`` — instead of generating
dialogues with a model under test, real dialogues from
``inverse_synthesis/results/dialogues.jsonl`` are replayed through the
same L0 → per-turn → holistic judging flow, one case per binding.

Real human dialogues should score near-full marks; any systematically
failing checkbox indicates a rubric/prompt calibration problem, which is
written to ``<outputs_dir>/real_dialogue_eval/report_<ts>.json``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from libs.core.config import REPO_ROOT, get_settings, slug
from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.evaluation import (
    ChatbotBenchmarkJudge, GameBenchmarkJudge, GeneralBenchmarkJudge,
)
from libs.evaluation.core.judge import BaseBenchmarkJudge, HolisticScoreCard
from libs.evaluation.report import BenchmarkReport, CaseResult
from apps.benchmark.src.runner import (
    _score_turn_with_retry,
    _score_holistic_with_retry,
    _load_cached_case,
    BenchmarkRunner,
)

LOGGER = logging.getLogger(__name__)

ABC_SID_RE = re.compile(r"^A\d{4}_B\d{4}_C\d{4}$")


_DOMAIN_TO_JUDGE = {
    "chatbot": ChatbotBenchmarkJudge,
    "game": GameBenchmarkJudge,
    "general": GeneralBenchmarkJudge,
    "cdial": GeneralBenchmarkJudge,
    "clam": GeneralBenchmarkJudge,
}


def _find_completed_real_replay_cases(
    model_name: str, judge_model: str = ""
) -> Dict[str, Path]:
    """Scan eval_traces/judge_<judge>/<model_name>/ for completed traces.

    When no judge_model is provided, fall back to the legacy flat path
    ``eval_traces/<model_name>/`` for backward compatibility.
    """
    s = get_settings()
    rel_dir = slug(model_name)
    if judge_model:
        rel_dir = Path(f"judge_{slug(judge_model)}") / rel_dir
    trace_dir = s.path("outputs_dir") / "eval_traces" / rel_dir
    if not trace_dir.exists():
        return {}
    completed: Dict[str, Path] = {}
    for f in trace_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            cid = data.get("case_id", "")
            if cid and cid not in completed:
                completed[cid] = f
        except Exception:
            continue
    return completed


def _stable_pid(raw_id: str) -> str:
    """P + md5(raw_id)[:8][:6].upper() — mirrors inverse_synthesis pipeline."""
    return "P" + hashlib.md5(str(raw_id).encode("utf-8")).hexdigest()[:8][:6].upper()


def load_real_dialogues(path: Path, require_abc: bool = True) -> List[Dict[str, Any]]:
    """Load records from dialogues.jsonl.

    For game benchmark scenarios keep the ABC filter; for cdial/clam/general
    domains flat scenario ids are allowed.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(rec.get("scenario_id", ""))
            if sid and (not require_abc or ABC_SID_RE.match(sid)):
                records.append(rec)
    return records


def _normalize_dialogue(rec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, str]:
    """Map raw role ids in turns to persona ids; return (dialogue, pid1, pid2)."""
    pid1 = rec["persona1_id"]
    pid2 = rec["persona2_id"]
    dialogue = []
    for turn in rec.get("dialogue", []):
        raw_role = str(turn.get("role", ""))
        pid = raw_role if raw_role in (pid1, pid2) else _stable_pid(raw_role)
        dialogue.append({"role": pid, "response": turn.get("response") or []})
    return dialogue, pid1, pid2


def _parse_ts(ts: str) -> datetime:
    """Parse timestamp like '2026-02-02 21:36:00.000000' or without microseconds."""
    if "." in ts:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def _save_replayed_dialogue(
    rec: Dict[str, Any],
    dialogue: List[Dict[str, Any]],
    pa: str,
    pb: str,
    tested_role: str,
    model_name: str,
    session_tag: str,
) -> Path:
    """Persist a real-replayed dialogue to outputs/game_benchmark/dialogues/ref_human/<model>/<scenario>/.

    This makes real dialogues directly comparable with machine-generated dialogues
    under ``outputs/game_benchmark/dialogues`` for rubric evolution and human-vs-LLM analysis.
    """
    s = get_settings()
    sid = rec["scenario_id"]
    out_dir = s.path("outputs_dir") / "dialogues" / f"ref_human" / slug(model_name) / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "as_a" if tested_role == "persona_a" else "as_b"
    out_path = out_dir / f"chat_{pa}_{pb}_{suffix}_{session_tag}.json"

    # Compute virtual duration from first to last message timestamp
    virtual_duration_ms = 0
    virtual_start_time = ""
    timestamps = [
        m.get("timestamp")
        for turn in dialogue
        for m in (turn.get("response") or [])
        if m.get("timestamp")
    ]
    if timestamps:
        try:
            start_dt = _parse_ts(timestamps[0])
            end_dt = _parse_ts(timestamps[-1])
            virtual_duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
            virtual_start_time = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            virtual_start_time = timestamps[0]

    meta = dict(rec.get("meta") or {})
    meta.update({
        "persona1_id": pa,
        "persona2_id": pb,
        "scenario_id": sid,
        "model": model_name,
        "session_tag": session_tag,
        "turns": len(dialogue),
        "ended_naturally": True,
        "fail_count": 0,
        "interrupts": [],
        "resumed_from_turns": 0,
        "virtual_duration_ms": virtual_duration_ms,
        "virtual_start_time": virtual_start_time,
        "raw_trace_dir": "",
    })

    payload = {
        "persona1_id": pa,
        "persona2_id": pb,
        "scenario_id": sid,
        "dialogue": dialogue,
        "meta": meta,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved real dialogue: %s", out_path)
    return out_path


def _replay_case(
    judge: BaseBenchmarkJudge,
    rec: Dict[str, Any],
    pt_weight: float,
    ho_weight: float,
    model_name: str,
    judge_model: str,
    eval_pid: str,
    case_id: str,
    tested_role: str,
    pa: str,
    pb: str,
    session_tag: str,
    domain: str,
) -> Optional[CaseResult]:
    sid = rec["scenario_id"]
    dialogue, _, _ = _normalize_dialogue(rec)

    try:
        persona = Persona.load(eval_pid)
        scenario = Scenario.load(sid)
    except FileNotFoundError as exc:
        LOGGER.warning("skip %s: %s", sid, exc)
        return None

    l0 = judge.check_l0(dialogue, persona, scenario, evaluated_role=eval_pid)
    is_invalid = judge.has_fatal_violation(l0)

    turn_cards: List = []
    holistic = HolisticScoreCard(role=eval_pid)
    if not is_invalid:
        tasks = [(i, t) for i, t in enumerate(dialogue) if t.get("role") == eval_pid]
        with ThreadPoolExecutor(max_workers=min(4, len(tasks) + 1)) as pool:
            futs = {
                pool.submit(_score_turn_with_retry, judge, i, t, dialogue,
                            persona, scenario): i
                for i, t in tasks
            }
            hfut = pool.submit(_score_holistic_with_retry, judge, dialogue,
                               persona, scenario, eval_pid)
            for fut in as_completed(futs):
                try:
                    turn_cards.append(fut.result())
                except Exception as exc:
                    LOGGER.warning("%s turn %d failed: %s", case_id, futs[fut], exc)
            try:
                holistic = hfut.result()
            except Exception as exc:
                LOGGER.warning("%s holistic failed: %s", case_id, exc)
        turn_cards.sort(key=lambda tc: tc.turn_index)

    # Persist trace under judge-isolated path to match the rest of the
    # benchmark pipeline: eval_traces/judge_<judge>/<model_name>/
    trace_rel = Path(f"judge_{slug(judge_model)}") / slug(model_name)
    judge.save_trace(
        case_id=case_id,
        turn_cards=turn_cards,
        holistic_card=holistic,
        dialogue=dialogue,
        model_name=trace_rel,
        l0_violations=l0,
        status="invalid" if is_invalid else "valid",
    )

    # Persist replayed real dialogue alongside machine-generated dialogues
    # so rubric evolution can compare human vs. LLM outputs directly.
    _save_replayed_dialogue(
        rec=rec,
        dialogue=dialogue,
        pa=pa,
        pb=pb,
        tested_role=tested_role,
        model_name=model_name,
        session_tag=session_tag,
    )

    case = CaseResult(
        case_id=case_id,
        persona_a_id=pa,
        persona_b_id=pb,
        scenario_id=sid,
        tested_role=tested_role,
        category=str(scenario.raw.get("category", "")),
        sub_category=str(scenario.raw.get("sub_category", "")),
        turn_cards=turn_cards,
        holistic_card=holistic,
        l0_violations=l0,
        domain=domain,
    )
    case.apply_style_gate(dialogue, eval_pid)
    case.compute(pt_weight=pt_weight, ho_weight=ho_weight)
    LOGGER.info("%s: score=%.2f grade=%s status=%s acc=%d gate=%d (turns=%d, l0=%d)",
                case_id, case.final_score_100, case.grade, case.status,
                case.acc, case.style_gate_pass, len(turn_cards), len(l0))
    return case


def _analyze(cases: List[CaseResult]) -> Dict[str, Any]:
    """Aggregate scores + failing checkbox rates → calibration feedback."""
    valid = [c for c in cases if c.status == "valid"]
    invalid = [c for c in cases if c.status != "valid"]

    l0_counter: Counter = Counter()
    l0_evidence: Dict[str, List[str]] = defaultdict(list)
    for c in cases:
        for v in c.l0_violations:
            key = f"{v.constraint_id} {v.constraint_name} [{v.severity}]"
            l0_counter[key] += 1
            if len(l0_evidence[key]) < 3:
                l0_evidence[key].append(f"{c.case_id}: {v.evidence}")

    cb_total: Counter = Counter()
    cb_fail: Counter = Counter()
    cb_reasons: Dict[str, List[str]] = defaultdict(list)
    dim_scores: Dict[str, List[float]] = defaultdict(list)

    def _collect(dim_results, prefix: str, case_id: str) -> None:
        for dr in dim_results:
            dim_scores[f"{prefix}:{dr.dim_id}"].append(dr.score)
            for cb_id, ok in dr.checks.items():
                key = f"{dr.dim_id}/{cb_id}"
                cb_total[key] += 1
                if not ok:
                    cb_fail[key] += 1
                    if len(cb_reasons[key]) < 3 and dr.reason:
                        cb_reasons[key].append(f"{case_id}: {dr.reason}")

    for c in valid:
        for tc in c.turn_cards:
            _collect(tc.dim_results, "per_turn", c.case_id)
        if c.holistic_card:
            _collect(c.holistic_card.dim_results, "holistic", c.case_id)

    fail_rates = sorted(
        (
            {
                "checkbox": k,
                "fail_rate": round(cb_fail[k] / cb_total[k], 4),
                "fails": cb_fail[k],
                "total": cb_total[k],
                "sample_reasons": cb_reasons.get(k, []),
            }
            for k in cb_total if cb_fail[k] > 0
        ),
        key=lambda d: -d["fail_rate"],
    )

    scores = [c.final_score_100 for c in valid]
    return {
        "n_cases": len(cases),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "invalid_case_ids": [c.case_id for c in invalid],
        "avg_score_100": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_score_100": min(scores) if scores else 0.0,
        "max_score_100": max(scores) if scores else 0.0,
        "acc": round(sum(c.acc for c in cases) / len(cases), 4) if cases else 0.0,
        "style_gate_rate": round(
            sum(c.style_gate_pass for c in cases) / len(cases), 4) if cases else 0.0,
        "l0_violations": [
            {"constraint": k, "count": n, "samples": l0_evidence[k]}
            for k, n in l0_counter.most_common()
        ],
        "dim_avg_scores": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(dim_scores.items())
        },
        "failing_checkboxes": fail_rates,
        "per_case": [
            {
                "case_id": c.case_id,
                "scenario_id": c.scenario_id,
                "score_100": c.final_score_100,
                "grade": c.grade,
                "status": c.status,
                "acc": c.acc,
                "style_gate_pass": c.style_gate_pass,
                "style_gate_failed_rules": c.style_gate_detail.get("failed_rules", []),
            }
            for c in cases
        ],
    }


def _print_calibration_summary(report: Dict[str, Any], out_path: Path) -> None:
    print("\n" + "=" * 64)
    print("真实对话回放评测报告")
    print("=" * 64)
    print(f"样本数: {report['n_cases']}  有效: {report['n_valid']}  "
          f"无效(L0 fatal): {report['n_invalid']}")
    print(f"平均分: {report['avg_score_100']}  "
          f"区间: [{report['min_score_100']}, {report['max_score_100']}]  "
          f"ACC: {report['acc']}")
    if report["l0_violations"]:
        print("\n[!] L0 违规（真实对话被判违规 = 规则过严）:")
        for item in report["l0_violations"]:
            print(f"  {item['constraint']}  x{item['count']}")
            for ev in item["samples"]:
                print(f"    - {ev}")
    top_fails = [f for f in report["failing_checkboxes"] if f["fail_rate"] >= 0.3]
    if top_fails:
        print("\n[!] 高失败率 checkbox（fail_rate >= 30%，建议校准）:")
        for f in top_fails[:15]:
            print(f"  {f['checkbox']}: {f['fail_rate']:.0%} "
                  f"({f['fails']}/{f['total']})")
            for r in f["sample_reasons"][:2]:
                print(f"    - {r}")
    print(f"\n完整报告: {out_path}")


def run_real_replay(
    bindings: List[Dict[str, Any]],
    *,
    model_name: str = "real_dialogues",
    concurrency: int = 2,
    force: bool = False,
) -> BenchmarkReport:
    """Replay real dialogues aligned with *bindings*; return a BenchmarkReport.

    One case per binding: the exact-match real dialogue for
    (persona_a, persona_b, scenario_id) is evaluated on the binding's
    tested_role side. Bindings without a usable real dialogue are skipped.
    """
    s = get_settings()
    pt_weight = float(s.get("benchmark", "per_turn_weight", default=0.5))
    ho_weight = float(s.get("benchmark", "holistic_weight", default=0.5))
    min_turns = int(s.get("benchmark", "real_min_turns", default=4))

    src = Path(s.get(
        "benchmark", "real_dialogues_path",
        default="apps/inverse_synthesis/results/dialogues.jsonl"))
    if not src.is_absolute():
        src = REPO_ROOT / src

    domain = BenchmarkRunner._detect_domain()
    require_abc = domain in ("", "game")
    records = load_real_dialogues(src, require_abc=require_abc)
    LOGGER.info("loaded %d real dialogues from %s (abc_required=%s)",
                len(records), src, require_abc)

    by_pair_sid: Dict[Tuple[frozenset, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (frozenset([rec["persona1_id"], rec["persona2_id"]]),
               rec["scenario_id"])
        by_pair_sid[key].append(rec)

    judge_model = s.model_for("benchmark_judge")

    # Scan existing eval_traces/judge_<judge>/<model_name>/ to skip
    # already-evaluated cases unless force=True.
    completed_map: Dict[str, Path] = {}
    if not force:
        completed_map = _find_completed_real_replay_cases(model_name, judge_model)
        if completed_map:
            LOGGER.info("found %d completed real_replay traces under eval_traces/judge_%s/%s; will skip LLM calls for those",
                        len(completed_map), slug(judge_model), slug(model_name))

    tasks: List[Tuple[Dict[str, Any], str, str, str, str, str, str]] = []
    cached_cases: List[CaseResult] = []
    single_role = bool(s.get("benchmark", "single_role", default=False))
    usable_bindings = 0
    for bind in bindings:
        pa, pb = bind["persona_a"], bind["persona_b"]
        sid = bind["scenario_id"]
        if single_role:
            roles = [str(bind.get("tested_role", "persona_a"))]
        else:
            roles = ["persona_a", "persona_b"]
        group = [r for r in by_pair_sid.get((frozenset([pa, pb]), sid), [])
                 if len(r.get("dialogue") or []) >= min_turns]
        if not group:
            LOGGER.warning("binding %s_%s_%s: no usable real dialogue, skip",
                           pa, pb, sid)
            continue
        usable_bindings += 1
        rec = max(group, key=lambda r: len(r.get("dialogue") or []))
        for tested_role in roles:
            eval_pid = pa if tested_role == "persona_a" else pb
            suffix = "as_a" if tested_role == "persona_a" else "as_b"
            case_id = f"{pa}_{pb}_{sid}_{suffix}"

            # If already evaluated and not force, reuse cached result and just persist dialogue.
            if case_id in completed_map and not force:
                cached = _load_cached_case(completed_map[case_id])
                if cached:
                    # Persist dialogue if it doesn't exist yet.
                    trace_data = json.loads(completed_map[case_id].read_text(encoding="utf-8"))
                    cached_dialogue = trace_data.get("dialogue", [])
                    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    _save_replayed_dialogue(
                        rec=rec,
                        dialogue=cached_dialogue,
                        pa=pa,
                        pb=pb,
                        tested_role=tested_role,
                        model_name=model_name,
                        session_tag=session_tag,
                    )
                    cached_cases.append(cached)
                    continue

            session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            tasks.append((rec, eval_pid, case_id, tested_role, pa, pb, session_tag))
    LOGGER.info("real_replay: %d/%d bindings have usable real dialogues; %d cases to evaluate from scratch",
                usable_bindings, len(bindings), len(tasks))

    judge_cls = _DOMAIN_TO_JUDGE.get(domain, GameBenchmarkJudge)
    judge = judge_cls()
    cases: List[CaseResult] = list(cached_cases)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {
            pool.submit(_replay_case, judge, rec, pt_weight, ho_weight,
                        model_name, judge_model, eval_pid, case_id, tested_role,
                        pa, pb, session_tag, domain): case_id
            for rec, eval_pid, case_id, tested_role, pa, pb, session_tag in tasks
        }
        for fut in as_completed(futs):
            try:
                case = fut.result()
                if case:
                    cases.append(case)
            except Exception as exc:
                LOGGER.error("case %s failed: %s", futs[fut], exc)
    cases.sort(key=lambda c: c.case_id)

    # 校准报告：真实对话应接近满分，系统性失分 = rubric/prompt 需校准
    if cases:
        analysis = _analyze(cases)
        out_dir = s.path("outputs_dir") / "real_dialogue_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"report_{ts}.json"
        out_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        _print_calibration_summary(analysis, out_path)

    report = BenchmarkReport(
        model_name=model_name,
        ref_model="human",
        judge_model=judge_model,
        cases=cases,
    )
    report.compute()
    return report
