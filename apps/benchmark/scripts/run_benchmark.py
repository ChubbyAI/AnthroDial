#!/usr/bin/env python
"""Benchmark evaluation entry point.

Usage::

    python apps/benchmark/scripts/run_benchmark.py --config configs/chatbot/qwen3.5-397b-a17b.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.core.config import get_settings, load_settings, setup_logging, slug  # noqa: E402
from apps.benchmark.src.runner import (  # noqa: E402
    BenchmarkRunner,
    DOMAIN_TO_JUDGE,
    _load_cached_case,
    load_benchmark_bindings,
    reconstruct_untraced_cases,
)
from libs.evaluation import GameBenchmarkJudge  # noqa: E402
from libs.evaluation.report import (  # noqa: E402
    BenchmarkReport,
    write_summary,
    write_leaderboard,
    write_global_leaderboard,
)
from libs.llm.client import LLMClient  # noqa: E402


def _build_llm(model_name: str, enable_thinking: Optional[bool] = None) -> tuple:
    """Build LLMClient from settings' model_configs.

    Args:
        model_name: model name key in model_configs.
        enable_thinking: explicitly control thinking switch.
            If None, reads from model_configs[model_name].enable_thinking (default False).

    Returns (client, api_model).
    """
    s = get_settings()
    mcfg = s.model_config_for(model_name)
    base_url = mcfg.get("base_url") or None
    api_key = mcfg.get("api_key") or None
    api_key_env = mcfg.get("api_key_env") or None
    api_model = mcfg.get("api_model", model_name)
    # enable_thinking 优先级：调用方显式指定 > model_configs 中声明 > 默认 False
    if enable_thinking is None:
        enable_thinking = bool(mcfg.get("enable_thinking", False))
    # per-model extra_body（如 DeepSeek 的 thinking.type / reasoning_effort）
    model_extra_body = mcfg.get("extra_body") or None
    return LLMClient(
        base_url=base_url, api_key_env=api_key_env, api_key=api_key,
        enable_thinking=enable_thinking, extra_body=model_extra_body,
    ), api_model


def _rebuild_from_cached_traces(
    model_name: str,
    judge_model: str,
) -> BenchmarkReport:
    """Rebuild a report solely from existing evaluation traces."""
    s = get_settings()
    trace_dir = (
        s.path("outputs_dir")
        / "eval_traces"
        / f"judge_{slug(judge_model)}"
        / slug(model_name)
    )
    if not trace_dir.exists():
        raise SystemExit(f"No cached evaluation traces found at {trace_dir}.")

    trace_files = sorted(
        path for path in trace_dir.glob("*.json")
        if not path.name.endswith("_holistic.json")
        and not path.name.endswith("_per_turn.json")
    )
    if not trace_files:
        raise SystemExit(f"No complete cached evaluation traces found at {trace_dir}.")

    pt_weight = float(s.get("benchmark", "per_turn_weight", default=0.5))
    ho_weight = float(s.get("benchmark", "holistic_weight", default=0.5))
    domain = BenchmarkRunner._detect_domain()
    cases = {}
    for trace_path in trace_files:
        case = _load_cached_case(trace_path, domain=domain)
        if case is None:
            logging.getLogger(__name__).warning("Skipping unreadable trace: %s", trace_path)
            continue
        case.compute(pt_weight=pt_weight, ho_weight=ho_weight)
        cases[case.case_id] = case

    if not cases:
        raise SystemExit(f"No usable cached cases found at {trace_dir}.")

    # L0-fatal cases never get eval traces; rebuild them from dialogue files so
    # invalid cases are not silently dropped from the rebuilt report.
    mode = str(s.get("benchmark", "mode", default="generate"))
    judge_cls = DOMAIN_TO_JUDGE.get(domain, GameBenchmarkJudge)
    for case in reconstruct_untraced_cases(
        tested_model=model_name,
        ref_model="human" if mode == "real_replay" else s.model_for("ref_partner"),
        self_play=bool(s.get("benchmark", "self_play", default=False)),
        judge=judge_cls(),
        domain=domain,
        traced_case_ids=set(cases),
        pt_weight=pt_weight,
        ho_weight=ho_weight,
        rejudge=False,
    ):
        cases[case.case_id] = case

    report = BenchmarkReport(model_name=model_name, cases=list(cases.values()))
    report.compute()
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark evaluation of a chat model.")
    p.add_argument("--config", type=Path, required=True,
                   help="config file (e.g. configs/chatbot/qwen3.5-397b-a17b.yaml)")
    p.add_argument(
        "--rebuild-from-traces",
        action="store_true",
        help="rebuild reports from existing eval traces without model calls or dialogue writes",
    )
    p.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        help="evaluate only this scenario ID; may be supplied more than once",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="regenerate selected cases instead of reusing dialogue and score caches",
    )
    p.add_argument(
        "--force-group",
        choices=["per_turn", "holistic", "all"],
        default=None,
        help="invalidate one score group while still reusing cached dialogues "
             "(with 'all', both groups are re-judged without dialogue regeneration)",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="do not write summary or leaderboard artifacts after evaluation",
    )
    args = p.parse_args()

    load_settings(args.config)
    setup_logging(entry="benchmark")
    s = get_settings()

    tested_model = s.get("benchmark", "tested_model")
    if not tested_model:
        raise SystemExit("benchmark.tested_model not set in config.")

    sample = s.get("benchmark", "sample", default=None)
    concurrency = int(s.get("benchmark", "concurrency", default=1))
    quiet = bool(s.get("benchmark", "quiet", default=False))
    enable_memory = bool(s.get("benchmark", "enable_memory", default=False))
    force = args.force or bool(s.get("benchmark", "force", default=False))
    force_group = args.force_group or s.get("benchmark", "force_group", default=None)
    if force_group is not None:
        force_group = str(force_group).strip().lower() or None
    self_play = bool(s.get("benchmark", "self_play", default=False))
    single_role = bool(s.get("benchmark", "single_role", default=False))
    sample_strategy = str(s.get("benchmark", "sample_strategy", default="first_n"))
    max_turns = s.get("session", "max_turns", default=None)

    bindings = load_benchmark_bindings()
    if not bindings:
        raise SystemExit(
            "No benchmark bindings found. "
            f"Provide the input file configured by benchmark.bindings_path: "
            f"{s.get('benchmark', 'bindings_path')}. See docs/INPUTS.md."
        )

    if sample_strategy == "one_per_c_class":
        seen_c_classes = set()
        filtered = []
        for b in bindings:
            sid = str(b.get("scenario_id", ""))
            c_class = sid.split("_")[-1] if "_" in sid else sid
            if c_class and c_class not in seen_c_classes:
                seen_c_classes.add(c_class)
                filtered.append(b)
        bindings = filtered

    if args.scenario_ids:
        selected_scenarios = set(args.scenario_ids)
        bindings = [
            binding for binding in bindings
            if str(binding.get("scenario_id", "")) in selected_scenarios
        ]
        if not bindings:
            raise SystemExit(
                f"No bindings found for scenario IDs: {', '.join(sorted(selected_scenarios))}."
            )

    if sample and sample < len(bindings):
        bindings = bindings[:sample]

    judge_model_name = s.model_for("benchmark_judge")
    mode = str(s.get("benchmark", "mode", default="generate"))

    if args.rebuild_from_traces:
        report = _rebuild_from_cached_traces(tested_model, judge_model_name)
        report.ref_model = "human" if mode == "real_replay" else s.model_for("ref_partner")
        report.judge_model = judge_model_name
        print(f"[rebuild] loaded {len(report.cases)} cached cases; no model calls or dialogue writes")
        print(f"  Scenarios: {len(report.scenario_scores)}")
        print(f"  Cases:     {len(report.cases)}")
        print(f"  ACC:       {report.acc:.2%}")
        summary_path = write_summary([report])
        leaderboard_path = write_leaderboard(judge_model=judge_model_name)
        print(f"[OK] summary -> {summary_path}")
        print(f"[OK] leaderboard -> {leaderboard_path}")
        return

    if mode == "real_replay":
        # 真实对话回放评测：不生成对话，直接把 bindings 对应的真实对话
        # 送入同一套 L0 → per-turn → holistic 评分流程
        from apps.benchmark.src.real_replay import run_real_replay  # noqa: E402

        print(f"[mode]   real_replay (no model under test, ref=human)")
        print(f"[judge]  {judge_model_name}")
        print(f"\n=== Benchmark: {tested_model} ({len(bindings)} cases, real replay) ===\n")
        report = run_real_replay(
            bindings, model_name=tested_model, concurrency=concurrency, force=force)
        ref_model_name = "human"
    else:
        tested_llm, tested_api_model = _build_llm(tested_model)

        ref_model_name = s.model_for("ref_partner")
        ref_llm, ref_api_model = _build_llm(ref_model_name)

        if self_play:
            ref_model_name = tested_model
            ref_llm, ref_api_model = tested_llm, tested_api_model
            print(f"[tested] {tested_model} -> base_url={tested_llm.base_url} thinking={tested_llm.enable_thinking} (api_model={tested_api_model})")
            print(f"[ref]    {ref_model_name} (self-play, same as tested)")
        else:
            print(f"[tested] {tested_model} -> base_url={tested_llm.base_url} thinking={tested_llm.enable_thinking} (api_model={tested_api_model})")
            print(f"[ref]    {ref_model_name} -> base_url={ref_llm.base_url} thinking={ref_llm.enable_thinking} (api_model={ref_api_model})")
        print(f"[judge]  {judge_model_name}")
        if single_role:
            print("[mode]   single_role: only evaluate binding.tested_role, no persona swap")
        else:
            print("[mode]   both_roles: evaluate persona_a AND persona_b on the same dialogue")

        runner = BenchmarkRunner(
            tested_model=tested_model,
            ref_model=ref_model_name,
            judge_model=judge_model_name,
            tested_llm=tested_llm,
            ref_llm=ref_llm,
            tested_api_model=tested_api_model,
            ref_api_model=ref_api_model,
            max_turns=max_turns,
            quiet=quiet,
            enable_memory=enable_memory,
            concurrency=concurrency,
            force=force,
            force_group=force_group,
            self_play=self_play,
            single_role=single_role,
        )

        print(f"\n=== Benchmark: {tested_model} ({len(bindings)} bindings) ===\n")
        report = runner.run(bindings)
    report.ref_model = ref_model_name
    report.judge_model = judge_model_name

    print(f"\n{'='*60}")
    print(f"  Model:        {report.model_name}")
    print(f"  Scenarios:    {len(report.scenario_scores)}")
    print(f"  Cases:        {len(report.cases)}")
    print(f"  Valid rate:   {report.valid_rate:.0%}")
    print(f"  L0 failures:  {report.l0_failure_count}")
    print(f"  ACC:          {report.acc:.2%}")
    print(f"  Final score:  {report.final_score_100:.1f}/100 ({report.grade})")
    print(f"  Per-turn:     {report.per_turn_avg:.4f}")
    print(f"  Holistic:     {report.holistic_avg:.4f}")
    print(f"{'='*60}")

    print(f"\n  Dimension scores (score / ACC):")
    for dim_id in report.per_dim_scores:
        score = report.per_dim_scores[dim_id]
        dim_acc = report.per_dim_acc.get(dim_id, 0)
        print(f"    {dim_id}: {score:.4f} ({score*100:.1f}/100) | ACC={dim_acc:.2%}")

    if report.category_scores:
        print(f"\n  Category scores:")
        for cat, score in sorted(report.category_scores.items()):
            print(f"    {cat}: {score:.4f} ({score*100:.1f}/100)")

    if report.top_failure_dims:
        print(f"\n  Top weakness dims (lowest ACC): {', '.join(report.top_failure_dims)}")

    print(f"\n  Per-case results:")
    for c in report.cases:
        status_mark = "X" if c.status == "invalid" else "O"
        acc_mark = "✓" if c.acc == 1 else "·"
        print(f"    [{status_mark}] [{acc_mark}] {c.case_id}: "
              f"{c.final_score_100:.1f} ({c.grade}) "
              f"[{c.category}/{c.sub_category}]")

    if args.no_report:
        print("[OK] evaluation completed without writing report artifacts")
        return

    summary_path = write_summary([report])
    print(f"\n[OK] summary -> {summary_path}")

    leaderboard_path = write_leaderboard(judge_model=judge_model_name)
    print(f"[OK] leaderboard -> {leaderboard_path}")

    global_lb_path = write_global_leaderboard()
    print(f"[OK] global leaderboard -> {global_lb_path}")


if __name__ == "__main__":
    main()
