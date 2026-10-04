"""Benchmark report: aggregate per-case results into a model-level summary.

Supports:
- L0 valid/invalid tracking
- Weighted dimension scores
- ACC metric: score threshold on final_score_100 (chatbot >= 95; game/clam >= 82.5).
- Both-sides evaluation: each scenario (binding) is evaluated on persona_a AND
  persona_b; scenario metrics average the two sides first, then overall metrics
  average scenarios, so every scenario weighs equally. Invalid cases count as 0.
- Category-level breakdown
- Quality grade assignment (S/A/B/C/D/E)
- Per-case fine-grained checkbox detail in output
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from libs.evaluation.core.judge import TurnScoreCard, HolisticScoreCard, DimResult, L0Violation
from libs.evaluation.core.style_gate import tested_side_messages
from libs.evaluation.rubric import load_rubric, per_turn_dimensions, holistic_dimensions
from libs.core.config import get_settings, slug

LOGGER = logging.getLogger(__name__)


def _score_to_grade(score_100: float) -> str:
    if score_100 >= 90:
        return "S"
    elif score_100 >= 80:
        return "A"
    elif score_100 >= 70:
        return "B"
    elif score_100 >= 60:
        return "C"
    elif score_100 >= 40:
        return "D"
    else:
        return "E"


def _all_checks_pass(dim_results: List[DimResult]) -> bool:
    for dr in dim_results:
        for v in dr.checks.values():
            if not v:
                return False
    return True


def _dim_results_detail(dim_results: List[DimResult]) -> List[Dict[str, Any]]:
    return [
        {
            "dim_id": dr.dim_id,
            "dim_name": dr.dim_name,
            "weight": dr.weight,
            "score": dr.score,
            "all_pass": all(dr.checks.values()),
            "checks": dr.checks,
            "reason": dr.reason,
        }
        for dr in dim_results
    ]


@dataclass
class CaseResult:
    case_id: str
    persona_a_id: str
    persona_b_id: str
    scenario_id: str
    tested_role: str
    category: str = ""
    sub_category: str = ""
    turn_cards: List[TurnScoreCard] = field(default_factory=list)
    holistic_card: Optional[HolisticScoreCard] = None
    l0_violations: List[L0Violation] = field(default_factory=list)
    status: str = "valid"
    per_turn_avg: float = 0.0
    holistic_avg: float = 0.0
    per_turn_weighted: float = 0.0
    holistic_weighted: float = 0.0
    final_score: float = 0.0
    final_score_100: float = 0.0
    grade: str = ""
    # ACC: score threshold on final_score_100; chatbot requires >= 95, others >= 82.5.
    ACC_SCORE_THRESHOLD: float = 82.5
    ACC_SCORE_THRESHOLD_CHATBOT: float = 95.0
    acc: int = 0
    checkbox_all_pass: int = 0  # 全部 checkbox 通过
    style_gate_pass: int = 1
    style_gate_detail: Dict[str, Any] = field(default_factory=dict)
    dialogue_path: Optional[str] = None
    domain: Optional[str] = None  # "chatbot" | "game"

    def apply_style_gate(self, dialogue: List[Dict[str, Any]], tested_id: str) -> None:
        """Run the deterministic human-style gate on the tested side's messages."""
        domain = self.domain or "game"
        if domain == "chatbot":
            from libs.evaluation.chatbot.style_gate import check_style_gate
        elif domain == "general":
            from libs.evaluation.general.style_gate import check_style_gate
        else:
            from libs.evaluation.game.style_gate import check_style_gate
        msgs = tested_side_messages(dialogue, tested_id)
        ok, detail = check_style_gate(msgs)
        self.style_gate_pass = 1 if ok else 0
        self.style_gate_detail = detail

    def compute(self, pt_weight: float = 0.5, ho_weight: float = 0.5) -> None:
        if any(v.severity == "fatal" for v in self.l0_violations):
            self.status = "invalid"
            self.final_score = 0.0
            self.final_score_100 = 0.0
            self.grade = "Invalid"
            self.acc = 0
            self.checkbox_all_pass = 0
            return

        self.status = "valid"

        if self.turn_cards:
            self.per_turn_avg = round(
                sum(tc.avg_score for tc in self.turn_cards) / len(self.turn_cards), 4
            )
            self.per_turn_weighted = round(
                sum(tc.weighted_score for tc in self.turn_cards) / len(self.turn_cards), 4
            )
        if self.holistic_card:
            self.holistic_avg = self.holistic_card.avg_score
            self.holistic_weighted = self.holistic_card.weighted_score

        self.final_score = round(
            pt_weight * self.per_turn_weighted + ho_weight * self.holistic_weighted, 4
        )
        self.final_score_100 = round(self.final_score * 100, 2)
        self.grade = _score_to_grade(self.final_score_100)

        # checkbox_all_pass: all checkboxes in all turns + holistic pass.
        if not self.turn_cards:
            self.checkbox_all_pass = 0
            self.acc = 0
            return

        all_pass = True
        for tc in self.turn_cards:
            if not _all_checks_pass(tc.dim_results):
                all_pass = False
                break
        if all_pass and self.holistic_card:
            if not _all_checks_pass(self.holistic_card.dim_results):
                all_pass = False
        self.checkbox_all_pass = 1 if all_pass else 0

        # ACC: chatbot applies the 95 score threshold; game/clam use their own.
        threshold = (self.ACC_SCORE_THRESHOLD_CHATBOT if self.domain == "chatbot"
                     else self.ACC_SCORE_THRESHOLD)
        self.acc = 1 if self.final_score_100 >= threshold else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "persona_a_id": self.persona_a_id,
            "persona_b_id": self.persona_b_id,
            "scenario_id": self.scenario_id,
            "tested_role": self.tested_role,
            "category": self.category,
            "sub_category": self.sub_category,
            "status": self.status,
            "grade": self.grade,
            "acc": self.acc,
            "checkbox_all_pass": self.checkbox_all_pass,
            "style_gate_pass": self.style_gate_pass,
            "style_gate_detail": self.style_gate_detail,
            "per_turn_avg": self.per_turn_avg,
            "per_turn_weighted": self.per_turn_weighted,
            "holistic_avg": self.holistic_avg,
            "holistic_weighted": self.holistic_weighted,
            "final_score": self.final_score,
            "final_score_100": self.final_score_100,
            "turn_count": len(self.turn_cards),
            "l0_violation_count": len(self.l0_violations),
            "dialogue_path": self.dialogue_path,
        }

    def to_detail_dict(self) -> Dict[str, Any]:
        """Full detail including per-turn and holistic checkbox breakdown."""
        base = self.to_dict()
        base["l0_violations"] = [
            {
                "constraint_id": v.constraint_id,
                "constraint_name": v.constraint_name,
                "severity": v.severity,
                "evidence": v.evidence,
                "turn_index": v.turn_index,
            }
            for v in self.l0_violations
        ]
        base["turn_scores"] = [
            {
                "turn_index": tc.turn_index,
                "role": tc.role,
                "avg_score": tc.avg_score,
                "weighted_score": tc.weighted_score,
                "all_pass": _all_checks_pass(tc.dim_results),
                "dimensions": _dim_results_detail(tc.dim_results),
            }
            for tc in self.turn_cards
        ]
        if self.holistic_card:
            base["holistic_scores"] = {
                "avg_score": self.holistic_card.avg_score,
                "weighted_score": self.holistic_card.weighted_score,
                "all_pass": _all_checks_pass(self.holistic_card.dim_results),
                "dimensions": _dim_results_detail(self.holistic_card.dim_results),
            }
        return base


@dataclass
class BenchmarkReport:
    model_name: str
    ref_model: str = ""  # reference partner model name
    judge_model: str = ""  # judge model name
    cases: List[CaseResult] = field(default_factory=list)
    # Overall
    valid_rate: float = 0.0
    per_turn_avg: float = 0.0
    holistic_avg: float = 0.0
    final_score: float = 0.0
    final_score_100: float = 0.0
    grade: str = ""
    acc: float = 0.0  # mean ACC across valid cases only
    style_gate_rate: float = 0.0  # fraction of cases passing the human-style gate
    # Per-dimension
    per_dim_scores: Dict[str, float] = field(default_factory=dict)
    per_dim_acc: Dict[str, float] = field(default_factory=dict)
    # Per-category
    category_scores: Dict[str, float] = field(default_factory=dict)
    # Per-scenario (binding) stats: both sides averaged within each scenario.
    # key: "{pa}_{pb}_{scenario_id}"
    scenario_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Failure stats
    l0_failure_count: int = 0
    top_failure_dims: List[str] = field(default_factory=list)

    def compute(self) -> None:
        if not self.cases:
            return

        n = len(self.cases)
        valid_cases = [c for c in self.cases if c.status == "valid"]
        self.l0_failure_count = n - len(valid_cases)

        # ── Per-scenario aggregation (both sides) ─────────────────────
        # Cases of one binding (persona_a, persona_b, scenario_id) come from
        # the same dialogue, one per evaluated side (as_a / as_b). Scenario
        # scores/validity/gate still average the sides, but ACC is strict:
        # a scenario passes only when ALL evaluated sides pass their domain
        # threshold (chatbot threshold = 95; game/clam use their own score
        # threshold). Overall metrics then average scenarios so
        # every scenario weighs equally.
        groups: Dict[Tuple[str, str, str], List[CaseResult]] = defaultdict(list)
        for c in self.cases:
            groups[(c.persona_a_id, c.persona_b_id, c.scenario_id)].append(c)

        scen_acc: List[float] = []
        scen_score: List[float] = []
        scen_pt: List[float] = []
        scen_ho: List[float] = []
        scen_valid: List[float] = []
        scen_gate: List[float] = []
        cat_scores: Dict[str, List[float]] = defaultdict(list)
        self.scenario_scores = {}
        for (pa, pb, sid), group in groups.items():
            m = len(group)
            # Strict AND: scenario ACC = 1 only if every evaluated side passes.
            s_acc = 1.0 if all(c.acc == 1 for c in group) else 0.0
            s_score = sum(c.final_score for c in group) / m
            s_pt = sum(c.per_turn_weighted for c in group) / m
            s_ho = sum(c.holistic_weighted for c in group) / m
            s_valid = sum(1 for c in group if c.status == "valid") / m
            s_gate = sum(c.style_gate_pass for c in group) / m
            scen_acc.append(s_acc)
            scen_score.append(s_score)
            scen_pt.append(s_pt)
            scen_ho.append(s_ho)
            scen_valid.append(s_valid)
            scen_gate.append(s_gate)
            self.scenario_scores[f"{pa}_{pb}_{sid}"] = {
                "scenario_id": sid,
                "persona_a_id": pa,
                "persona_b_id": pb,
                "case_ids": [c.case_id for c in group],
                "sides": [c.tested_role for c in group],
                "acc": round(s_acc, 4),
                "final_score": round(s_score, 4),
                "final_score_100": round(s_score * 100, 2),
                "valid_rate": round(s_valid, 4),
            }
            cat = group[0].category
            if cat:
                cat_scores[cat].append(s_score)

        ns = len(groups)
        self.valid_rate = round(sum(scen_valid) / ns, 4)
        self.per_turn_avg = round(sum(scen_pt) / ns, 4)
        self.holistic_avg = round(sum(scen_ho) / ns, 4)
        self.final_score = round(sum(scen_score) / ns, 4)
        self.final_score_100 = round(self.final_score * 100, 2)
        self.grade = _score_to_grade(self.final_score_100)
        self.acc = round(sum(scen_acc) / ns, 4)
        self.style_gate_rate = round(sum(scen_gate) / ns, 4)

        # Per-dimension aggregation
        dim_sums: Dict[str, List[float]] = {}
        dim_pass_counts: Dict[str, List[bool]] = {}
        for case in valid_cases:
            for tc in case.turn_cards:
                for dr in tc.dim_results:
                    dim_sums.setdefault(dr.dim_id, []).append(dr.score)
                    dim_pass_counts.setdefault(dr.dim_id, []).append(
                        all(dr.checks.values())
                    )
            if case.holistic_card:
                for dr in case.holistic_card.dim_results:
                    dim_sums.setdefault(dr.dim_id, []).append(dr.score)
                    dim_pass_counts.setdefault(dr.dim_id, []).append(
                        all(dr.checks.values())
                    )

        self.per_dim_scores = {
            k: round(sum(v) / len(v), 4)
            for k, v in dim_sums.items()
            if v
        }
        self.per_dim_acc = {
            k: round(sum(v) / len(v), 4)
            for k, v in dim_pass_counts.items()
            if v
        }

        # Top failure dimensions (lowest ACC)
        if self.per_dim_acc:
            sorted_dims = sorted(self.per_dim_acc.items(), key=lambda x: x[1])
            self.top_failure_dims = [d[0] for d in sorted_dims[:3]]

        # Per-category aggregation (scenario-level means)
        self.category_scores = {
            k: round(sum(v) / len(v), 4)
            for k, v in cat_scores.items()
            if v
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "ref_model": self.ref_model,
            "judge_model": self.judge_model,
            "case_count": len(self.cases),
            "scenario_count": len(self.scenario_scores),
            "valid_rate": self.valid_rate,
            "l0_failure_count": self.l0_failure_count,
            "acc": self.acc,
            "style_gate_rate": self.style_gate_rate,
            "final_score": self.final_score,
            "final_score_100": self.final_score_100,
            "grade": self.grade,
            "per_turn_avg": self.per_turn_avg,
            "holistic_avg": self.holistic_avg,
            "per_dim_scores": self.per_dim_scores,
            "per_dim_acc": self.per_dim_acc,
            "category_scores": self.category_scores,
            "scenario_scores": self.scenario_scores,
            "top_failure_dims": self.top_failure_dims,
            "cases": [c.to_dict() for c in self.cases],
        }


def write_summary(
    reports: List[BenchmarkReport],
    out_dir: Optional[Path] = None,
) -> Path:
    """Write per-model report.

    Directory layout: results/benchmark/judge_<judge>/<model_slug>/
    where judge_slug comes from the report field.
    """
    s = get_settings()
    if out_dir is None:
        model_slug = slug(reports[0].model_name) if reports else "unknown"
        judge_model = reports[0].judge_model if reports else ""
        base_dir = s.path("results_dir")
        if judge_model:
            judge_slug = "judge_" + slug(judge_model)
            base_dir = base_dir / judge_slug
        out_dir = base_dir / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = load_rubric()
    pt_ids = [d.id for d in per_turn_dimensions(dims)]
    ho_ids = [d.id for d in holistic_dimensions(dims)]
    all_ids = pt_ids + ho_ids

    # ── Markdown summary ──────────────────────────────────────────
    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Benchmark Summary ({ts})")
    lines.append("")

    # Overview table
    header = ["model", "cases", "valid%", "ACC", "final(100)", "grade",
              "per_turn", "holistic"] + all_ids
    sep = ["---"] * len(header)
    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for r in reports:
        row = [
            r.model_name,
            str(len(r.cases)),
            f"{r.valid_rate:.0%}",
            f"{r.acc:.2%}",
            f"{r.final_score_100:.1f}",
            r.grade,
            f"{r.per_turn_avg:.4f}",
            f"{r.holistic_avg:.4f}",
        ]
        for d_id in all_ids:
            row.append(f"{r.per_dim_scores.get(d_id, 0):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # Per-dimension ACC table
    lines.append("")
    lines.append("## Per-Dimension ACC (all checkboxes pass rate)")
    lines.append("")
    dim_header = ["model"] + all_ids
    lines.append("| " + " | ".join(dim_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(dim_header)) + " |")
    for r in reports:
        row = [r.model_name]
        for d_id in all_ids:
            row.append(f"{r.per_dim_acc.get(d_id, 0):.2%}")
        lines.append("| " + " | ".join(row) + " |")

    # Category breakdown
    lines.append("")
    lines.append("## Category Scores")
    lines.append("")
    for r in reports:
        if r.category_scores:
            lines.append(f"### {r.model_name}")
            lines.append("")
            lines.append("| Category | Score | Score(100) |")
            lines.append("| --- | --- | --- |")
            for cat, score in sorted(r.category_scores.items()):
                lines.append(f"| {cat} | {score:.4f} | {score * 100:.1f} |")
            lines.append("")

    # L0 failures
    lines.append("## L0 Failures")
    lines.append("")
    for r in reports:
        lines.append(f"### {r.model_name}")
        lines.append(f"- L0 failure count: {r.l0_failure_count}/{len(r.cases)}")
        lines.append(f"- Valid rate: {r.valid_rate:.0%}")
        invalid_cases = [c for c in r.cases if c.status == "invalid"]
        if invalid_cases:
            for c in invalid_cases:
                for v in c.l0_violations:
                    lines.append(f"  - [{v.constraint_id}] {v.constraint_name}: {v.evidence}")
        else:
            lines.append("  - (none)")
        lines.append("")

    # Per-case ACC
    lines.append("## Per-Case Results")
    lines.append("")
    for r in reports:
        lines.append(f"### {r.model_name}")
        lines.append("")
        lines.append("| case_id | status | ACC | score(100) | grade | category |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for c in r.cases:
            lines.append(
                f"| {c.case_id} | {c.status} | {c.acc} | "
                f"{c.final_score_100:.1f} | {c.grade} | {c.category} |"
            )
        lines.append("")

    # Per-scenario results: both sides of each dialogue averaged
    lines.append("## Per-Scenario Results (avg of both sides)")
    lines.append("")
    for r in reports:
        lines.append(f"### {r.model_name}")
        lines.append("")
        lines.append("| scenario | pair | sides | ACC | score(100) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for key in sorted(r.scenario_scores, key=lambda k: r.scenario_scores[k]["scenario_id"]):
            st = r.scenario_scores[key]
            lines.append(
                f"| {st['scenario_id']} | {st['persona_a_id']} x {st['persona_b_id']} "
                f"| {len(st['sides'])} | {st['acc']:.2%} | {st['final_score_100']:.1f} |"
            )
        lines.append("")

    # Top weaknesses
    lines.append("## Top Weakness Dimensions (lowest ACC)")
    lines.append("")
    for r in reports:
        lines.append(f"- {r.model_name}: {', '.join(r.top_failure_dims)}")
    lines.append("")

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── JSON summary (overview) ───────────────────────────────────
    json_path = out_dir / "summary.json"
    json_path.write_text(
        json.dumps(
            [r.to_dict() for r in reports],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── JSON detail (per-case with all checkbox breakdowns) ───────
    detail_path = out_dir / "detail.json"
    detail_data = []
    for r in reports:
        detail_data.append({
            "model_name": r.model_name,
            "acc": r.acc,
            "final_score_100": r.final_score_100,
            "per_dim_acc": r.per_dim_acc,
            "cases": [c.to_detail_dict() for c in r.cases],
        })
    detail_path.write_text(
        json.dumps(detail_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("benchmark summary -> %s", md_path)
    LOGGER.info("benchmark detail  -> %s", detail_path)
    return md_path


# Score thresholds shown as ACC@T sensitivity columns in the leaderboards.
ACC_SENSITIVITY_THRESHOLDS: List[float] = [85.0, 90.0, 95.0, 100.0]


def acc_at_thresholds(
    report: Dict[str, Any],
    thresholds: Optional[List[float]] = None,
) -> Dict[float, Optional[float]]:
    """Recompute scenario ACC at alternative score thresholds from per-case data.

    A case passes at T when status == "valid" and final_score_100 >= T; a
    scenario (binding) passes only when ALL evaluated sides pass. Returns
    {T: acc}; T maps to None when the report carries no per-case detail.
    """
    if thresholds is None:
        thresholds = ACC_SENSITIVITY_THRESHOLDS
    cases = report.get("cases") or []
    if not cases:
        return {t: None for t in thresholds}
    groups: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for c in cases:
        groups[(c.get("persona_a_id"), c.get("persona_b_id"),
                c.get("scenario_id"))].append(c)
    out: Dict[float, Optional[float]] = {}
    for t in thresholds:
        passed = sum(
            1 for group in groups.values()
            if all(c.get("status") == "valid"
                   and float(c.get("final_score_100") or 0.0) >= t
                   for c in group)
        )
        out[t] = round(passed / len(groups), 4) if groups else None
    return out


def fmt_acc_cell(value: Optional[float]) -> str:
    return f"{value:.2%}" if value is not None else "-"


def write_leaderboard(
    results_dir: Optional[Path] = None,
    ref_model: str = "",
    judge_model: str = "",
) -> Path:
    """Scan model subdirs under results_dir and generate a unified leaderboard.md.

    When *judge_model* is given, results are stored inside
    ``results_dir/judge_<slug>/`` and the leaderboard title shows it.
    Each model subdirectory should contain a summary.json produced by write_summary.
    """
    s = get_settings()
    if results_dir is None:
        results_dir = s.path("results_dir")

    # Build scan directory based on judge_model
    scan_dir = results_dir
    if judge_model:
        judge_slug = "judge_" + slug(judge_model)
        scan_dir = scan_dir / judge_slug

    # Collect all model reports
    reports: List[Dict[str, Any]] = []
    if scan_dir.exists():
        for sub in sorted(scan_dir.iterdir()):
            summary_file = sub / "summary.json"
            if sub.is_dir() and summary_file.exists():
                try:
                    data = json.loads(summary_file.read_text(encoding="utf-8"))
                    if isinstance(data, list) and data:
                        reports.append(data[0])
                    elif isinstance(data, dict):
                        reports.append(data)
                except Exception:
                    continue

    if not reports:
        LOGGER.warning("no model results found under %s", scan_dir)
        lb_path = scan_dir / "leaderboard.md"
        return lb_path

    # Load dimension IDs for table headers
    dims = load_rubric()
    pt_ids = [d.id for d in per_turn_dimensions(dims)]
    ho_ids = [d.id for d in holistic_dimensions(dims)]
    all_ids = pt_ids + ho_ids

    # Sort by final_score_100 descending, then by ACC descending
    reports.sort(key=lambda r: (r.get("final_score_100", 0), r.get("acc", 0)), reverse=True)

    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if judge_model:
        lines.append(f"# Benchmark Leaderboard (Judge: {judge_model})")
    else:
        lines.append(f"# Benchmark Leaderboard")
    lines.append(f"")
    lines.append(f"_Last updated: {ts}_")
    lines.append("")
    lines.append("_Metrics aggregate both sides (persona_a + persona_b) of each scenario, "
                 "then average scenarios. Scenario ACC counts only when ALL evaluated sides "
                 "pass their domain threshold._")
    lines.append("")
    lines.append("_ACC@T columns are recomputed from per-case final_score_100: a case "
                 "passes at T when L0-valid and final_score_100 >= T; a scenario passes "
                 "only when all its evaluated sides pass._")
    lines.append("")

    # ── Overall ranking table ─────────────────────────────────────
    lines.append("## Overall Ranking")
    lines.append("")
    sens_rows = [acc_at_thresholds(r) for r in reports]
    header = ["Rank", "Model", "Scenarios", "Cases", "Valid%", "ACC",
              *[f"ACC@{t:g}" for t in ACC_SENSITIVITY_THRESHOLDS],
              "Score(100)", "Grade", "Per-Turn", "Holistic"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for rank, (r, sens) in enumerate(zip(reports, sens_rows), 1):
        row = [
            str(rank),
            r.get("model_name", "?"),
            str(r.get("scenario_count", len(r.get("scenario_scores", {})) or "-")),
            str(r.get("case_count", 0)),
            f"{r.get('valid_rate', 0):.0%}",
            f"{r.get('acc', 0):.2%}",
            *(fmt_acc_cell(sens[t]) for t in ACC_SENSITIVITY_THRESHOLDS),
            f"{r.get('final_score_100', 0):.1f}",
            r.get("grade", "?"),
            f"{r.get('per_turn_avg', 0):.4f}",
            f"{r.get('holistic_avg', 0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # ── Per-dimension score table ─────────────────────────────────
    lines.append("")
    lines.append("## Per-Dimension Scores")
    lines.append("")
    dim_header = ["Model"] + all_ids
    lines.append("| " + " | ".join(dim_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(dim_header)) + " |")
    for r in reports:
        dim_scores = r.get("per_dim_scores", {})
        row = [r.get("model_name", "?")]
        for d_id in all_ids:
            row.append(f"{dim_scores.get(d_id, 0):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # ── Per-dimension ACC table ───────────────────────────────────
    lines.append("")
    lines.append("## Per-Dimension ACC")
    lines.append("")
    lines.append("| " + " | ".join(dim_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(dim_header)) + " |")
    for r in reports:
        dim_acc = r.get("per_dim_acc", {})
        row = [r.get("model_name", "?")]
        for d_id in all_ids:
            row.append(f"{dim_acc.get(d_id, 0):.2%}")
        lines.append("| " + " | ".join(row) + " |")

    # ── Per-scenario ACC matrix (both sides averaged) ─────────────
    scen_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in reports:
        scen = r.get("scenario_scores") or {}
        if scen:
            scen_by_model[r.get("model_name", "?")] = scen
    if scen_by_model:
        all_scen_keys = sorted({k for scen in scen_by_model.values() for k in scen})
        lines.append("")
        lines.append("## Per-Scenario ACC (avg of both sides)")
        lines.append("")
        scen_header = ["scenario"] + list(scen_by_model.keys())
        lines.append("| " + " | ".join(scen_header) + " |")
        lines.append("| " + " | ".join(["---"] * len(scen_header)) + " |")
        for key in all_scen_keys:
            sid = key
            for scen in scen_by_model.values():
                if key in scen:
                    sid = scen[key].get("scenario_id", key)
                    break
            row = [sid]
            for model, scen in scen_by_model.items():
                st = scen.get(key)
                row.append(f"{st['acc']:.2%}" if st else "-")
            lines.append("| " + " | ".join(row) + " |")

    # ── Category breakdown ────────────────────────────────────────
    all_cats = set()
    for r in reports:
        all_cats.update(r.get("category_scores", {}).keys())
    if all_cats:
        sorted_cats = sorted(all_cats)
        lines.append("")
        lines.append("## Category Scores")
        lines.append("")
        cat_header = ["Model"] + sorted_cats
        lines.append("| " + " | ".join(cat_header) + " |")
        lines.append("| " + " | ".join(["---"] * len(cat_header)) + " |")
        for r in reports:
            cat_scores = r.get("category_scores", {})
            row = [r.get("model_name", "?")]
            for cat in sorted_cats:
                s_val = cat_scores.get(cat)
                row.append(f"{s_val * 100:.1f}" if s_val is not None else "-")
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    scan_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = scan_dir / "leaderboard.md"
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("leaderboard -> %s", leaderboard_path)
    return leaderboard_path


def write_global_leaderboard(
    results_dir: Optional[Path] = None,
) -> Path:
    """Scan ALL judge_*/<model>/summary.json and write a unified leaderboard.

    Output: results/benchmark/leaderboard.md (flat table with ref/judge columns).
    """
    s = get_settings()
    if results_dir is None:
        results_dir = s.path("results_dir")

    def _is_backup_judge_dir(name: str) -> bool:
        return name.endswith(("-bp", ".bak", "_backup", "_old"))

    # Collect all model reports across all judge combinations
    reports: List[Dict[str, Any]] = []
    if results_dir.exists():
        for judge_dir in sorted(results_dir.iterdir()):
            if (not judge_dir.is_dir()
                    or not judge_dir.name.startswith("judge_")
                    or _is_backup_judge_dir(judge_dir.name)):
                continue
            for model_dir in sorted(judge_dir.iterdir()):
                summary_file = model_dir / "summary.json"
                if not model_dir.is_dir() or not summary_file.exists():
                    continue
                try:
                    data = json.loads(summary_file.read_text(encoding="utf-8"))
                    if isinstance(data, list) and data:
                        reports.append(data[0])
                    elif isinstance(data, dict):
                        reports.append(data)
                except Exception:
                    continue

    if not reports:
        LOGGER.warning("no model results found under %s", results_dir)
        lb_path = results_dir / "leaderboard.md"
        return lb_path

    # Load dimension IDs for table headers
    dims = load_rubric()
    pt_ids = [d.id for d in per_turn_dimensions(dims)]
    ho_ids = [d.id for d in holistic_dimensions(dims)]
    all_ids = pt_ids + ho_ids

    # Sort by final_score_100 descending, then by ACC descending
    reports.sort(key=lambda r: (r.get("final_score_100", 0), r.get("acc", 0)), reverse=True)

    lines: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("# Benchmark Leaderboard (Global)")
    lines.append("")
    lines.append(f"_Last updated: {ts}_")
    lines.append("")
    lines.append(f"_Total entries: {len(reports)} (across all judge combinations)_")
    lines.append("")
    lines.append("_Metrics aggregate both sides (persona_a + persona_b) of each scenario, "
                 "then average scenarios. Scenario ACC counts only when ALL evaluated sides "
                 "pass their domain threshold._")
    lines.append("")
    lines.append("_ACC@T columns are recomputed from per-case final_score_100: a case "
                 "passes at T when L0-valid and final_score_100 >= T; a scenario passes "
                 "only when all its evaluated sides pass._")
    lines.append("")

    # ── Overall ranking table ─────────────────────────────────────
    lines.append("## Overall Ranking")
    lines.append("")
    sens_rows = [acc_at_thresholds(r) for r in reports]
    header = ["Rank", "Model", "Ref", "Judge", "Scenarios", "Cases", "Valid%",
              "ACC", *[f"ACC@{t:g}" for t in ACC_SENSITIVITY_THRESHOLDS],
              "Score(100)", "Grade", "Per-Turn", "Holistic"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for rank, (r, sens) in enumerate(zip(reports, sens_rows), 1):
        row = [
            str(rank),
            r.get("model_name", "?"),
            r.get("ref_model", "-"),
            r.get("judge_model", "-"),
            str(r.get("scenario_count", len(r.get("scenario_scores", {})) or "-")),
            str(r.get("case_count", 0)),
            f"{r.get('valid_rate', 0):.0%}",
            f"{r.get('acc', 0):.2%}",
            *(fmt_acc_cell(sens[t]) for t in ACC_SENSITIVITY_THRESHOLDS),
            f"{r.get('final_score_100', 0):.1f}",
            r.get("grade", "?"),
            f"{r.get('per_turn_avg', 0):.4f}",
            f"{r.get('holistic_avg', 0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # ── Per-dimension score table ─────────────────────────────────
    lines.append("")
    lines.append("## Per-Dimension Scores")
    lines.append("")
    dim_header = ["Model", "Ref", "Judge"] + all_ids
    lines.append("| " + " | ".join(dim_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(dim_header)) + " |")
    for r in reports:
        dim_scores = r.get("per_dim_scores", {})
        row = [r.get("model_name", "?"), r.get("ref_model", "-"), r.get("judge_model", "-")]
        for d_id in all_ids:
            row.append(f"{dim_scores.get(d_id, 0):.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # ── Per-dimension ACC table ───────────────────────────────────
    lines.append("")
    lines.append("## Per-Dimension ACC")
    lines.append("")
    lines.append("| " + " | ".join(dim_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(dim_header)) + " |")
    for r in reports:
        dim_acc = r.get("per_dim_acc", {})
        row = [r.get("model_name", "?"), r.get("ref_model", "-"), r.get("judge_model", "-")]
        for d_id in all_ids:
            row.append(f"{dim_acc.get(d_id, 0):.2%}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    results_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = results_dir / "leaderboard.md"
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("global leaderboard -> %s", leaderboard_path)
    return leaderboard_path
