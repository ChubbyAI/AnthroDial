"""Benchmark runner: generate dialogues with the model under test, then judge.

For each benchmark binding (persona_a, persona_b, scenario):
1. Create two ChatAgents (self-play: both use the model under test; otherwise
   one tested + one reference model).
2. Run a ChatSession and save the dialogue.
3. Evaluate BOTH roles on the dialogue (persona_a and persona_b perspectives):
   for each role, check L0 hard constraints, then score every turn of that
   role (per-turn) plus a holistic evaluation.
4. Aggregate into a BenchmarkReport with category-level breakdown.

Supports:
- Concurrent execution via ThreadPoolExecutor (--concurrency N)
- Skip already-completed cases (--resume)
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from libs.chat.agent import ChatAgent, DEFAULT_GENERATION_MODE, normalize_generation_mode
from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.chat.session import ChatSession, SessionResult, save_dialogue, _default_on_turn
from libs.core.config import REPO_ROOT, get_settings, slug
from libs.evaluation import (
    ChatbotBenchmarkJudge, GameBenchmarkJudge, GeneralBenchmarkJudge,
    detect_domain,
)
from libs.evaluation.core.judge import BaseBenchmarkJudge
from libs.evaluation.report import BenchmarkReport, CaseResult
from libs.llm.client import LLMClient


DOMAIN_TO_JUDGE = {
    "chatbot": ChatbotBenchmarkJudge,
    "game": GameBenchmarkJudge,
    "general": GeneralBenchmarkJudge,
    "cdial": GeneralBenchmarkJudge,
    "clam": GeneralBenchmarkJudge,
}

LOGGER = logging.getLogger(__name__)

# 单个评分任务的重试配置（LLMClient 内部已有 3 次重试，此层处理完全失败后的再次尝试）
_SCORE_TASK_MAX_RETRIES = 2
_SCORE_TASK_BACKOFF_BASE = 3.0

# 对话生成空对话（0 turns）时的外层重试配置
_DIALOGUE_EMPTY_MAX_RETRIES = 2
_DIALOGUE_EMPTY_BACKOFF_BASE = 3.0


def load_benchmark_bindings(path: Path | str | None = None) -> List[Dict[str, Any]]:
    if path is None:
        s = get_settings()
        rel = s.get("benchmark", "bindings_path",
                     default="data/chatbot_benchmark/bindings.yaml")
        path = Path(rel)
        if not path.is_absolute():
            path = REPO_ROOT / path
    else:
        path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, list):
        return data
    return []


def _run_dir(tested_model: str, ref_model: str, self_play: bool = False) -> Path:
    """Return relative sub-path for dialogues/raw_traces.

    With a reference partner: ``ref_<ref>/<tested>``.
    Self-play: ``<tested>``.
    """
    if self_play:
        return Path(slug(tested_model))
    return Path(f"ref_{slug(ref_model)}") / slug(tested_model)


def _eval_dir(
    tested_model: str, ref_model: str, judge_model: str, self_play: bool = False
) -> Path:
    """Return relative sub-path for eval_traces/results.

    With a reference partner: ``ref_<ref>/judge_<judge>/<tested>``.
    Self-play: ``judge_<judge>/<tested>``.
    """
    if self_play:
        return Path(f"judge_{slug(judge_model)}") / slug(tested_model)
    return (Path(f"ref_{slug(ref_model)}")
            / f"judge_{slug(judge_model)}"
            / slug(tested_model))


def _is_empty_dialogue_trace(data: Dict[str, Any]) -> bool:
    """Return True if the trace represents a failed run with 0-turn dialogue."""
    if data.get("status") != "invalid":
        return False
    return any(
        v.get("constraint_id") == "L0-SYS"
        and "轮次为 0" in (v.get("evidence") or "")
        for v in (data.get("l0_violations") or [])
    )


def _find_completed_cases(
    tested_model: str,
    ref_model: str,
    judge_model: str,
    pt_hash: str,
    ho_hash: str,
    self_play: bool = False,
) -> Dict[str, Path]:
    """Scan eval_traces for completed case_ids (full-trace cache).

    Returns {case_id: trace_file_path} so cached results can be loaded.
    Empty-dialogue invalid traces are treated as not completed so they will be re-run.

    Traces written after this change carry ``pt_hash`` and ``ho_hash``; only
    traces whose hashes match the current judge are considered valid. Legacy
    traces without these fields are accepted for backward compatibility.
    """
    s = get_settings()
    trace_dir = (s.path("outputs_dir") / "eval_traces"
                 / _eval_dir(tested_model, ref_model, judge_model, self_play))
    if not trace_dir.exists():
        return {}
    completed: Dict[str, Path] = {}
    # Newest first: timestamps are embedded in filenames, so reverse sort means
    # a case re-run after a failed generation takes precedence over stale traces.
    for f in sorted(trace_dir.glob("*.json"), reverse=True):
        # Ignore per-group split traces.
        if "_per_turn.json" in f.name or "_holistic.json" in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            cid = data.get("case_id", "")
            if not cid or cid in completed:
                continue
            # Skip cached invalid traces caused by empty dialogue (generation failure).
            # These should be re-attempted rather than treated as finished.
            if _is_empty_dialogue_trace(data):
                LOGGER.info("ignoring empty-dialogue cached trace: %s", f.name)
                continue
            # If the trace carries group hashes, validate them against current rubric.
            stored_pt = data.get("pt_hash")
            stored_ho = data.get("ho_hash")
            if stored_pt is not None and stored_pt != pt_hash:
                continue
            if stored_ho is not None and stored_ho != ho_hash:
                continue
            completed[cid] = f
        except Exception:
            continue
    return completed


def _find_group_caches(
    tested_model: str,
    ref_model: str,
    judge_model: str,
    pt_hash: str,
    ho_hash: str,
    self_play: bool = False,
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Scan eval_traces for per-group (per_turn / holistic) caches.

    Returns {case_id: {"per_turn": Path|None, "holistic": Path|None}}.
    Only files whose ``group_hash`` matches the current judge's hash are
    considered valid; this ensures rubric changes invalidate stale caches.
    """
    s = get_settings()
    trace_dir = (s.path("outputs_dir") / "eval_traces"
                 / _eval_dir(tested_model, ref_model, judge_model, self_play))
    result: Dict[str, Dict[str, Optional[Path]]] = {}
    if not trace_dir.exists():
        return result

    def _latest(existing: Optional[Path], candidate: Path) -> Path:
        if existing is None:
            return candidate
        # Timestamp is embedded in filename as YYYYMMDD_HHMMSS; lexicographic sort works.
        return candidate if candidate.name > existing.name else existing

    for f in trace_dir.glob("*_per_turn.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            cid = data.get("case_id", "")
            # Skip empty-dialogue invalid traces so they are regenerated.
            if (data.get("status") == "invalid"
                    and not (data.get("turn_scores") or [])):
                continue
            if not cid or data.get("group_hash") != pt_hash:
                continue
            result.setdefault(cid, {})["per_turn"] = _latest(
                result.get(cid, {}).get("per_turn"), f)
        except Exception:
            continue

    for f in trace_dir.glob("*_holistic.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            cid = data.get("case_id", "")
            hs = data.get("holistic_scores") or {}
            if (data.get("status") == "invalid"
                    and not (hs.get("dimensions") or [])):
                continue
            if not cid or data.get("group_hash") != ho_hash:
                continue
            result.setdefault(cid, {})["holistic"] = _latest(
                result.get(cid, {}).get("holistic"), f)
        except Exception:
            continue

    return result


def _raw_dialogue(dialogue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize dialogue to the raw ChatSession turn format.

    Cached dialogue files are written by ``save_dialogue()``, which merges
    consecutive same-role messages into a single turn with multiple response
    items and no top-level ``content`` field. For scoring consistency we
    expand those back into one turn per emitted message, matching the format
    produced by ``ChatSession.run()``.
    """
    raw: List[Dict[str, Any]] = []
    for turn in dialogue:
        role = turn.get("role", "")
        content = turn.get("content")
        responses = turn.get("response") or []

        # Already raw: has a non-empty top-level content field. Keep as-is.
        if isinstance(content, str) and content.strip():
            raw.append(turn)
            continue

        # Cleaned/merged: expand each response into its own raw turn.
        for m in responses:
            txt = (m.get("content") or "").strip()
            if not txt:
                continue
            resp = {k: v for k, v in m.items() if k != "turn"}
            raw.append({
                "role": role,
                "response": [resp],
                "content": txt,
            })
    return raw


def _find_cached_dialogue(
    tested_model: str,
    ref_model: str,
    case_id: str,
    scenario_id: str,
    self_play: bool = False,
    generation_mode: str = DEFAULT_GENERATION_MODE,
) -> Optional[Path]:
    """Find the latest completed cached dialogue JSON for a case_id, or None.

    Empty dialogue files (0 turns) and files marked ``in_progress: true`` are
    ignored so that the case can be regenerated. Dialogues whose
    ``meta.generation_mode`` (missing = draft_scheduling) does not match the
    requested mechanism are also ignored, so turn_taking and draft_scheduling
    caches never mix.
    """
    s = get_settings()
    run_sub = _run_dir(tested_model, ref_model, self_play)
    dialogue_dir = s.path("outputs_dir") / "dialogues" / run_sub / slug(scenario_id)
    if not dialogue_dir.exists():
        return None
    pa_pb_suffix = case_id.rsplit(f"_{scenario_id}", 1)
    if len(pa_pb_suffix) != 2:
        return None
    pa_pb = pa_pb_suffix[0]
    as_part = pa_pb_suffix[1].lstrip("_")
    pattern = f"chat_{pa_pb}_{as_part}_*.json"
    matches = sorted(dialogue_dir.glob(pattern))
    # Pick the latest non-empty, completed dialogue file.
    for candidate in reversed(matches):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            dialogue = data.get("dialogue") or []
            if not isinstance(dialogue, list) or not dialogue:
                LOGGER.info("ignoring empty cached dialogue: %s", candidate.name)
                continue
            meta = data.get("meta") or {}
            if meta.get("in_progress"):
                LOGGER.info("ignoring in-progress cached dialogue: %s", candidate.name)
                continue
            if str(meta.get("generation_mode") or DEFAULT_GENERATION_MODE) != generation_mode:
                LOGGER.info(
                    "ignoring cached dialogue with different generation mode: %s",
                    candidate.name,
                )
                continue
            return candidate
        except Exception:
            continue
    return None


def _dim_result_from_dict(dd: Dict[str, Any]) -> "DimResult":
    from libs.evaluation.judge import DimResult
    return DimResult(
        dim_id=dd.get("dim_id", ""),
        dim_name=dd.get("dim_name", dd.get("dim_id", "")),
        weight=dd.get("weight", 10),
        checks=dd.get("checks", {}),
        reason=dd.get("reason", ""),
        score=dd.get("score", 0.0),
    )


def _turn_cards_from_data(data: Dict[str, Any]) -> List["TurnScoreCard"]:
    from libs.evaluation.judge import TurnScoreCard
    turn_cards = []
    for tc_data in (data.get("turn_scores") or []):
        dims = [_dim_result_from_dict(dd) for dd in (tc_data.get("dimensions") or [])]
        turn_cards.append(TurnScoreCard(
            turn_index=tc_data.get("turn_index", 0),
            role=tc_data.get("role", ""),
            content=tc_data.get("content", ""),
            dim_results=dims,
            avg_score=tc_data.get("avg_score", 0.0),
            weighted_score=tc_data.get("weighted_score", 0.0),
        ))
    return turn_cards


def _holistic_card_from_data(
    data: Dict[str, Any], evaluated_role: str
) -> "HolisticScoreCard":
    from libs.evaluation.judge import HolisticScoreCard
    holistic_data = data.get("holistic_scores") or {}
    ho_dims = [_dim_result_from_dict(dd) for dd in (holistic_data.get("dimensions") or [])]
    return HolisticScoreCard(
        role=evaluated_role,
        dim_results=ho_dims,
        avg_score=holistic_data.get("avg_score", 0.0),
        weighted_score=holistic_data.get("weighted_score", 0.0),
    )


def _l0_violations_from_data(data: Dict[str, Any]) -> List["L0Violation"]:
    from libs.evaluation.judge import L0Violation
    return [
        L0Violation(
            constraint_id=v.get("constraint_id", ""),
            constraint_name=v.get("constraint_name", ""),
            severity=v.get("severity", "major"),
            evidence=v.get("evidence", ""),
            turn_index=v.get("turn_index", -1),
        )
        for v in (data.get("l0_violations") or [])
    ]


def _case_id_to_role(case_id: str) -> str:
    parts = case_id.rsplit("_", 1)
    suffix = parts[-1] if len(parts) == 2 else ""
    return "persona_a" if suffix == "a" else "persona_b"


def _case_id_to_ids(case_id: str) -> tuple[str, str, str]:
    # Parse case_id like BP01_BP02_BS01_as_a (legacy flat sid) or
    # BP01_BP02_A0001_B0001_C0001_as_a (three-part sid).
    parts = case_id.split("_")
    pa_id = parts[0] if len(parts) > 0 else ""
    pb_id = parts[1] if len(parts) > 1 else ""
    if len(parts) > 4:
        scenario_id = "_".join(parts[2:-2])
    else:
        scenario_id = parts[2] if len(parts) > 2 else ""
    return pa_id, pb_id, scenario_id


def _load_cached_case(
    trace_path: Optional[Path] = None,
    per_turn_path: Optional[Path] = None,
    holistic_path: Optional[Path] = None,
    domain: Optional[str] = None,
) -> Optional[CaseResult]:
    """Load a CaseResult from cached eval trace(s) (no LLM calls).

    Accepts either a legacy full trace, or per-turn / holistic group traces
    that are merged together. When group traces are used, both must exist
    unless the caller intends to fill the missing part later.
    """
    data: Optional[Dict[str, Any]] = None
    if trace_path is not None:
        try:
            data = json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    evaluated_role = ""
    status = "valid"
    l0_violations: List[Any] = []
    if data is not None:
        evaluated_role = data.get("evaluated_role", "")
        status = data.get("status", "valid")
        l0_violations = _l0_violations_from_data(data)
        turn_cards = _turn_cards_from_data(data)
        holistic_card = _holistic_card_from_data(data, evaluated_role)
    else:
        turn_cards = []
        holistic_card = None

    if per_turn_path is not None:
        try:
            pt_data = json.loads(per_turn_path.read_text(encoding="utf-8"))
            evaluated_role = evaluated_role or pt_data.get("evaluated_role", "")
            status = pt_data.get("status", status)
            l0_violations = l0_violations or _l0_violations_from_data(pt_data)
            turn_cards = _turn_cards_from_data(pt_data)
        except Exception:
            return None

    if holistic_path is not None:
        try:
            ho_data = json.loads(holistic_path.read_text(encoding="utf-8"))
            evaluated_role = evaluated_role or ho_data.get("evaluated_role", "")
            status = ho_data.get("status", status)
            l0_violations = l0_violations or _l0_violations_from_data(ho_data)
            holistic_card = _holistic_card_from_data(ho_data, evaluated_role)
        except Exception:
            return None

    if not turn_cards and holistic_card is None:
        return None

    case_id = ""
    if data is not None:
        case_id = data.get("case_id", "")
    if per_turn_path is not None and not case_id:
        case_id = json.loads(per_turn_path.read_text(encoding="utf-8")).get("case_id", "")
    if holistic_path is not None and not case_id:
        case_id = json.loads(holistic_path.read_text(encoding="utf-8")).get("case_id", "")
    if not case_id:
        return None

    tested_role = _case_id_to_role(case_id)
    pa_id, pb_id, scenario_id = _case_id_to_ids(case_id)

    from libs.evaluation.judge import HolisticScoreCard as HCard
    if holistic_card is None:
        holistic_card = HCard(role=evaluated_role)

    case = CaseResult(
        case_id=case_id,
        persona_a_id=pa_id,
        persona_b_id=pb_id,
        scenario_id=scenario_id,
        tested_role=tested_role,
        turn_cards=turn_cards,
        holistic_card=holistic_card,
        l0_violations=l0_violations,
        domain=domain,
        status=status,
    )
    # Traces do not persist category or style-gate results; reload them from
    # the scenario card / embedded dialogue so cached cases match fresh ones.
    if scenario_id:
        try:
            scenario = Scenario.load(scenario_id)
            case.category = str(scenario.raw.get("category", ""))
            case.sub_category = str(scenario.raw.get("sub_category", ""))
        except Exception:
            pass
    dialogue = None
    if data is not None:
        dialogue = data.get("dialogue")
    if dialogue:
        tested_id = pa_id if tested_role == "persona_a" else pb_id
        try:
            case.apply_style_gate(dialogue, tested_id)
        except Exception:
            pass
    case.compute()
    return case


def reconstruct_untraced_cases(
    tested_model: str,
    ref_model: str,
    self_play: bool,
    judge: BaseBenchmarkJudge,
    domain: str,
    traced_case_ids: Set[str],
    pt_weight: float = 0.5,
    ho_weight: float = 0.5,
    rejudge: bool = False,
) -> List[CaseResult]:
    """Rebuild CaseResults for cases that have no eval trace.

    L0-fatal cases are never written to eval_traces, so trace-based re-scoring
    that only scans eval_traces silently drops every invalid case and inflates
    valid_rate / per-turn averages. Those cases are reconstructed from the
    dialogue files on disk instead: L0 is re-checked deterministically, and
    only cases that are no longer fatal need a (costly) judge LLM call.
    """
    from libs.evaluation.judge import HolisticScoreCard

    s = get_settings()
    single_role = bool(s.get("benchmark", "single_role", default=False))
    generation_mode = normalize_generation_mode(
        s.get("session", "mechanism", default=DEFAULT_GENERATION_MODE))
    cases: List[CaseResult] = []
    no_dialogue = 0
    awaiting_rejudge = 0

    for binding in load_benchmark_bindings():
        pa_id = str(binding.get("persona_a", ""))
        pb_id = str(binding.get("persona_b", ""))
        scenario_id = str(binding.get("scenario_id", ""))
        if not (pa_id and pb_id and scenario_id):
            continue
        if single_role:
            tested_roles = [str(binding.get("tested_role", "persona_a"))]
        else:
            tested_roles = ["persona_a", "persona_b"]
        for tested_role in tested_roles:
            suffix = "as_a" if tested_role == "persona_a" else "as_b"
            case_id = f"{pa_id}_{pb_id}_{scenario_id}_{suffix}"
            if case_id in traced_case_ids:
                continue

            dialogue_path = _find_cached_dialogue(
                tested_model, ref_model, case_id, scenario_id, self_play,
                generation_mode=generation_mode)
            if dialogue_path is None and self_play:
                # Self-play shares one dialogue between both roles, canonically
                # saved with the as_a suffix; look it up for as_b cases too.
                alt = "as_a" if suffix == "as_b" else "as_b"
                alt_case_id = f"{pa_id}_{pb_id}_{scenario_id}_{alt}"
                dialogue_path = _find_cached_dialogue(
                    tested_model, ref_model, alt_case_id, scenario_id, self_play,
                    generation_mode=generation_mode)
            if dialogue_path is None:
                LOGGER.warning("%s: no trace and no dialogue on disk; skipped",
                               case_id)
                no_dialogue += 1
                continue

            try:
                data = json.loads(dialogue_path.read_text(encoding="utf-8"))
            except Exception as exc:
                LOGGER.warning("%s: unreadable dialogue %s: %s",
                               case_id, dialogue_path.name, exc)
                no_dialogue += 1
                continue
            dialogue = _raw_dialogue(data.get("dialogue") or [])

            tested_id = pa_id if tested_role == "persona_a" else pb_id
            persona = Persona.load(tested_id)
            scenario = Scenario.load(scenario_id)
            l0_violations = judge.check_l0(
                dialogue, persona, scenario, evaluated_role=tested_id)

            turn_cards: List["TurnScoreCard"] = []
            holistic_card = HolisticScoreCard(role=tested_id)
            if not judge.has_fatal_violation(l0_violations):
                # Was traced before (e.g. L0 rules relaxed), or the earlier
                # scoring run was incomplete — only a fresh judge call can
                # produce its scores.
                if not rejudge:
                    LOGGER.warning(
                        "%s: no trace but L0 is not fatal; skipped "
                        "(rerun with rejudge enabled)", case_id)
                    awaiting_rejudge += 1
                    continue
                LOGGER.info("%s: no trace but L0 is not fatal -> re-judging",
                            case_id)
                for i, turn in enumerate(dialogue):
                    if turn.get("role") != tested_id:
                        continue
                    try:
                        turn_cards.append(_score_turn_with_retry(
                            judge, i, turn, dialogue, persona, scenario))
                    except Exception as exc:
                        LOGGER.warning("%s: turn %d re-judge failed: %s",
                                       case_id, i, exc)
                try:
                    holistic_card = _score_holistic_with_retry(
                        judge, dialogue, persona, scenario, tested_id)
                except Exception as exc:
                    LOGGER.warning("%s: holistic re-judge failed: %s",
                                   case_id, exc)

            case = CaseResult(
                case_id=case_id,
                persona_a_id=pa_id,
                persona_b_id=pb_id,
                scenario_id=scenario_id,
                tested_role=tested_role,
                category=str(scenario.raw.get("category", "")),
                sub_category=str(scenario.raw.get("sub_category", "")),
                turn_cards=turn_cards,
                holistic_card=holistic_card,
                l0_violations=l0_violations,
                dialogue_path=str(dialogue_path),
                domain=domain,
            )
            case.apply_style_gate(dialogue, tested_id)
            case.compute(pt_weight=pt_weight, ho_weight=ho_weight)
            cases.append(case)

    LOGGER.info(
        "reconstructed %d untraced cases (no dialogue on disk: %d, "
        "awaiting rejudge: %d)", len(cases), no_dialogue, awaiting_rejudge)
    return cases


class BenchmarkRunner:

    def __init__(
        self,
        tested_model: str,
        ref_model: Optional[str] = None,
        judge_model: Optional[str] = None,
        tested_llm: Optional[LLMClient] = None,
        ref_llm: Optional[LLMClient] = None,
        tested_api_model: Optional[str] = None,
        ref_api_model: Optional[str] = None,
        judge: Optional[BenchmarkJudge] = None,
        max_turns: Optional[int] = None,
        quiet: bool = False,
        enable_memory: bool = False,
        concurrency: int = 1,
        force: bool = False,
        force_group: Optional[str] = None,
        self_play: bool = False,
        single_role: bool = False,
    ) -> None:
        s = get_settings()
        self.tested_model = tested_model
        self.ref_model = ref_model or s.model_for("ref_partner")
        self.judge_model = judge_model or s.model_for("benchmark_judge")
        self.tested_api_model = tested_api_model or s.api_model_for(tested_model)
        self.ref_api_model = ref_api_model or s.api_model_for(self.ref_model)
        self.tested_llm = tested_llm or LLMClient()
        self.ref_llm = ref_llm or LLMClient()
        self.domain = self._detect_domain()
        if judge is not None:
            self.judge = judge
        else:
            judge_cls = DOMAIN_TO_JUDGE.get(self.domain, GameBenchmarkJudge)
            self.judge = judge_cls()
        self.max_turns = max_turns or int(s.get("session", "max_turns", default=30))
        self.quiet = quiet
        self.enable_memory = enable_memory
        self.concurrency = max(1, concurrency)
        self.force = force
        self.force_group = (force_group or "").strip().lower()
        self.self_play = self_play
        self.single_role = single_role
        self.generation_mode = normalize_generation_mode(
            s.get("session", "mechanism", default=DEFAULT_GENERATION_MODE))
        self.pt_weight = float(s.get("benchmark", "per_turn_weight", default=0.5))
        self.ho_weight = float(s.get("benchmark", "holistic_weight", default=0.5))

    @staticmethod
    def _detect_domain() -> str:
        """Return domain key based on benchmark config."""
        return detect_domain()

    def run(
        self,
        bindings: List[Dict[str, Any]],
    ) -> BenchmarkReport:
        report = BenchmarkReport(model_name=self.tested_model)

        # Build task list: one task per binding. Each binding evaluates BOTH
        # roles (persona_a and persona_b) so per-scenario stats average both
        # sides; with single_role=True only binding.tested_role is evaluated.
        tasks: List[Dict[str, Any]] = []
        for binding in bindings:
            pa_id = str(binding.get("persona_a", ""))
            pb_id = str(binding.get("persona_b", ""))
            scenario_id = str(binding.get("scenario_id", ""))
            if not (pa_id and pb_id and scenario_id):
                continue
            if self.single_role:
                tested_roles = [str(binding.get("tested_role", "persona_a"))]
            else:
                tested_roles = ["persona_a", "persona_b"]
            roles = []
            for tested_role in tested_roles:
                suffix = "as_a" if tested_role == "persona_a" else "as_b"
                roles.append({
                    "tested_role": tested_role,
                    "case_id": f"{pa_id}_{pb_id}_{scenario_id}_{suffix}",
                })
            tasks.append({
                "pa_id": pa_id,
                "pb_id": pb_id,
                "scenario_id": scenario_id,
                "roles": roles,
            })

        # Load cached results and skip completed cases unless --force.
        # Two-level cache:
        # 1. Full-trace cache (whole case already done).
        # 2. Per-group cache (per_turn / holistic) so a rubric change in one
        #    group only requires re-running that group.
        # force_group can invalidate one or both groups:
        #   "per_turn" | "holistic" | "all".
        force_all = self.force or self.force_group == "all"
        force_pt = force_all or self.force_group == "per_turn"
        force_ho = force_all or self.force_group == "holistic"

        completed_map: Dict[str, Path] = {}
        group_cache_map: Dict[str, Dict[str, Optional[Path]]] = {}
        if not force_all:
            completed_map = _find_completed_cases(
                self.tested_model, self.ref_model, self.judge_model,
                pt_hash=self.judge.pt_hash,
                ho_hash=self.judge.ho_hash,
                self_play=self.self_play,
            )
            group_cache_map = _find_group_caches(
                self.tested_model, self.ref_model, self.judge_model,
                pt_hash=self.judge.pt_hash,
                ho_hash=self.judge.ho_hash,
                self_play=self.self_play,
            )

            before = sum(len(t["roles"]) for t in tasks)
            new_tasks: List[Dict[str, Any]] = []
            full_cached = 0
            group_cached = 0

            def _resolve_cached(
                case_id: str,
            ) -> tuple:
                """Resolve one case_id against the two-level cache.

                Returns (cached_case, pt_path, ho_path, source):
                - cached_case: CaseResult when the case is fully cached, else None.
                - pt_path / ho_path: reusable per-group trace paths for re-running.
                - source: "full" | "group" when cached_case is not None, else None.
                """
                if case_id in completed_map:
                    trace_path = completed_map[case_id]
                    try:
                        trace_data = json.loads(
                            trace_path.read_text(encoding="utf-8"))
                    except Exception:
                        trace_data = {}
                    stored_pt = trace_data.get("pt_hash")
                    stored_ho = trace_data.get("ho_hash")

                    # Determine whether each group in the full trace is usable.
                    # Hash-bearing traces are valid only when the hash matches.
                    # Legacy traces without hashes are trusted for a group when
                    # that group is not being forced to re-run (the caller has
                    # confirmed the rubric for that group has not changed).
                    has_turn_scores = bool(trace_data.get("turn_scores"))
                    hs = trace_data.get("holistic_scores") or {}
                    has_holistic = bool(hs.get("dimensions"))

                    pt_usable = (
                        (stored_pt is not None
                         and stored_pt == self.judge.pt_hash)
                        or (stored_pt is None and has_turn_scores
                            and not force_pt)
                    ) and not force_pt
                    ho_usable = (
                        (stored_ho is not None
                         and stored_ho == self.judge.ho_hash)
                        or (stored_ho is None and has_holistic
                            and not force_ho)
                    ) and not force_ho

                    if pt_usable and ho_usable:
                        # Full trace matches current rubric for both groups.
                        cached = _load_cached_case(
                            trace_path=trace_path, domain=self.domain)
                        if cached:
                            return cached, None, None, "full"
                    elif pt_usable or ho_usable:
                        # Partial match: reuse usable groups from the full
                        # trace, then try to fill the missing groups from
                        # per-group caches.
                        full_pt_path = trace_path if pt_usable else None
                        full_ho_path = trace_path if ho_usable else None
                        gc = group_cache_map.get(case_id, {})
                        pt_path = full_pt_path or gc.get("per_turn")
                        ho_path = full_ho_path or gc.get("holistic")

                        if pt_path and ho_path:
                            cached = _load_cached_case(
                                per_turn_path=pt_path,
                                holistic_path=ho_path,
                                domain=self.domain,
                            )
                            if cached:
                                return cached, None, None, "group"

                        # Partial cache: carry forward existing group paths.
                        return None, pt_path, ho_path, None

                gc = group_cache_map.get(case_id, {})
                pt_path = gc.get("per_turn")
                ho_path = gc.get("holistic")
                if pt_path and ho_path:
                    cached = _load_cached_case(
                        per_turn_path=pt_path,
                        holistic_path=ho_path,
                        domain=self.domain,
                    )
                    if cached:
                        return cached, None, None, "group"
                return None, pt_path, ho_path, None

            for t in tasks:
                remaining_roles: List[Dict[str, Any]] = []
                for r in t["roles"]:
                    cid = r["case_id"]
                    cached_case, pt_path, ho_path, source = _resolve_cached(cid)
                    if cached_case is not None:
                        report.cases.append(cached_case)
                        if source == "full":
                            full_cached += 1
                        else:
                            group_cached += 1
                    else:
                        remaining_roles.append({
                            **r,
                            "cached_per_turn_path": pt_path,
                            "cached_holistic_path": ho_path,
                        })
                if remaining_roles:
                    new_tasks.append({
                        "pa_id": t["pa_id"],
                        "pb_id": t["pb_id"],
                        "scenario_id": t["scenario_id"],
                        "roles": remaining_roles,
                    })

            tasks = new_tasks
            skipped = full_cached + group_cached
            if skipped > 0:
                LOGGER.info(
                    "loaded %d/%d cached cases from eval_traces "
                    "(full=%d, group=%d)",
                    skipped, before, full_cached, group_cached)
                print(
                    f"[cached] loaded {skipped}/{before} cases from eval_traces "
                    f"(full={full_cached}, group={group_cached})"
                )

        total = len(tasks)
        if total == 0:
            LOGGER.info("no new tasks to run")
            print("[OK] all cases completed (loaded from cache)")
            report.compute()
            return report

        if self.concurrency <= 1:
            # Sequential execution
            for step, task in enumerate(tasks, 1):
                LOGGER.info(
                    "[binding %d/%d] %s_%s_%s (%d roles)",
                    step, total, task["pa_id"], task["pb_id"],
                    task["scenario_id"], len(task["roles"]),
                )
                try:
                    cases = self._run_binding(**task)
                except Exception as exc:
                    LOGGER.exception(
                        "binding %s_%s_%s failed: %s",
                        task["pa_id"], task["pb_id"], task["scenario_id"], exc)
                    continue
                report.cases.extend(cases)
        else:
            # Concurrent execution
            print(f"[concurrency] running {total} bindings with {self.concurrency} workers")
            completed_count = 0
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                future_to_task = {
                    pool.submit(self._run_binding, **task): task
                    for task in tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    completed_count += 1
                    try:
                        cases = future.result()
                        report.cases.extend(cases)
                        for case in cases:
                            LOGGER.info("[done %d/%d] %s: %s %.1f",
                                        completed_count, total,
                                        case.case_id, case.grade,
                                        case.final_score_100)
                    except Exception as exc:
                        LOGGER.exception("[fail %d/%d] binding %s_%s_%s: %s",
                                         completed_count, total,
                                         task["pa_id"], task["pb_id"],
                                         task["scenario_id"], exc)

        report.compute()
        return report

    def _generate_one_dialogue(
        self,
        persona_a: Persona,
        persona_b: Persona,
        scenario: Scenario,
        tested_role: str,
        pa_id: str,
        pb_id: str,
        scenario_id: str,
    ) -> SessionResult:
        """Generate a single dialogue for one case."""
        if self.self_play:
            agent_a = ChatAgent.build(
                persona_a, scenario, persona_b.persona_id,
                self.tested_llm, model=self.tested_api_model,
            )
            agent_b = ChatAgent.build(
                persona_b, scenario, persona_a.persona_id,
                self.tested_llm, model=self.tested_api_model,
            )
        elif tested_role == "persona_a":
            agent_a = ChatAgent.build(
                persona_a, scenario, persona_b.persona_id,
                self.tested_llm, model=self.tested_api_model,
            )
            agent_b = ChatAgent.build(
                persona_b, scenario, persona_a.persona_id,
                self.ref_llm, model=self.ref_api_model,
            )
        else:
            agent_a = ChatAgent.build(
                persona_a, scenario, persona_b.persona_id,
                self.ref_llm, model=self.ref_api_model,
            )
            agent_b = ChatAgent.build(
                persona_b, scenario, persona_a.persona_id,
                self.tested_llm, model=self.tested_api_model,
            )

        s = get_settings()
        session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "as_a" if tested_role == "persona_a" else "as_b"
        run_sub = _run_dir(self.tested_model, self.ref_model, self.self_play)

        live_dir = (s.path("outputs_dir") / "dialogues"
                    / run_sub / slug(scenario_id))
        live_dir.mkdir(parents=True, exist_ok=True)
        live_path = live_dir / f"chat_{pa_id}_{pb_id}_{suffix}_{session_tag}.json"

        raw_trace_dir = (s.path("outputs_dir") / "raw_traces"
                         / run_sub / slug(scenario_id)
                         / f"{pa_id}_{pb_id}_{suffix}_{session_tag}")
        raw_trace_dir.mkdir(parents=True, exist_ok=True)

        if self.generation_mode == "turn_taking":
            from libs.chat.turn_session import TurnTakingSession
            session_cls = TurnTakingSession
        else:
            session_cls = ChatSession
        session = session_cls(
            agent_a=agent_a,
            agent_b=agent_b,
            scenario=scenario,
            max_turns=scenario.max_turns or self.max_turns,
            opener=scenario.raw.get("opener"),
            on_turn=(None if self.quiet else _default_on_turn),
            live_path=live_path,
            raw_trace_path=raw_trace_dir,
            session_tag=session_tag,
            enable_memory=self.enable_memory,
        )
        result = session.run()
        save_dialogue(result, tag=f"bench_ref_{slug(self.ref_model)}_{slug(self.tested_model)}")
        return result

    def _generate_one_dialogue_with_retry(
        self,
        persona_a: Persona,
        persona_b: Persona,
        scenario: Scenario,
        tested_role: str,
        pa_id: str,
        pb_id: str,
        scenario_id: str,
    ) -> SessionResult:
        """Generate dialogue with retries when the result is empty (0 turns).

        This handles transient upstream failures (502/429) that cause the
        ChatSession to produce no dialogue at all.
        """
        last_result: Optional[SessionResult] = None
        for attempt in range(1, _DIALOGUE_EMPTY_MAX_RETRIES + 2):
            result = self._generate_one_dialogue(
                persona_a, persona_b, scenario, tested_role,
                pa_id, pb_id, scenario_id,
            )
            last_result = result
            if result.dialogue:
                return result

            # Empty dialogue: clean up the empty file and retry after backoff.
            LOGGER.warning(
                "case %s_%s_%s: empty dialogue on generation attempt %d/%d",
                pa_id, pb_id, scenario_id, attempt, _DIALOGUE_EMPTY_MAX_RETRIES + 1,
            )
            if result.output_path and result.output_path.exists():
                try:
                    result.output_path.unlink()
                    LOGGER.info("removed empty dialogue file: %s", result.output_path.name)
                except Exception as exc:
                    LOGGER.warning("failed to remove empty dialogue file %s: %s",
                                   result.output_path, exc)

            if attempt <= _DIALOGUE_EMPTY_MAX_RETRIES:
                sleep_s = _DIALOGUE_EMPTY_BACKOFF_BASE ** attempt
                LOGGER.info("sleeping %.1fs before dialogue generation retry", sleep_s)
                time.sleep(sleep_s)

        LOGGER.error(
            "case %s_%s_%s: dialogue generation exhausted %d attempts (0 turns)",
            pa_id, pb_id, scenario_id, _DIALOGUE_EMPTY_MAX_RETRIES + 1,
        )
        return last_result or SessionResult(dialogue=[])

    def _run_binding(
        self,
        pa_id: str,
        pb_id: str,
        scenario_id: str,
        roles: List[Dict[str, Any]],
    ) -> List[CaseResult]:
        """Run one binding: acquire dialogue(s), then evaluate every role.

        Self-play: the tested model plays BOTH sides, so one dialogue is
        generated (or loaded from cache) and evaluated from both
        perspectives. Reference-partner mode: each role gets its own
        dialogue with the tested model on that side.
        """
        persona_a = Persona.load(pa_id)
        persona_b = Persona.load(pb_id)
        scenario = Scenario.load(scenario_id)

        def _load_cached_dialogue(case_id: str) -> Optional[SessionResult]:
            cached_dlg = _find_cached_dialogue(
                self.tested_model, self.ref_model, case_id, scenario_id,
                self.self_play, generation_mode=self.generation_mode)
            if not cached_dlg:
                return None
            LOGGER.info("[dialogue-cache] %s: loading %s", case_id, cached_dlg.name)
            dlg_data = json.loads(cached_dlg.read_text(encoding="utf-8"))
            return SessionResult(
                dialogue=_raw_dialogue(dlg_data.get("dialogue", [])),
                meta=dlg_data.get("meta", {}),
                output_path=cached_dlg,
            )

        if self.self_play:
            # One shared dialogue for both roles. Try both case suffixes so a
            # dialogue generated under either side is reused.
            result: Optional[SessionResult] = None
            if not self.force:
                for suffix in ("as_a", "as_b"):
                    alt_case_id = f"{pa_id}_{pb_id}_{scenario_id}_{suffix}"
                    result = _load_cached_dialogue(alt_case_id)
                    if result:
                        break
            if result is None:
                # Canonical generation uses the as_a suffix so the filename
                # is deterministic for later cache lookups.
                result = self._generate_one_dialogue_with_retry(
                    persona_a, persona_b, scenario, "persona_a",
                    pa_id, pb_id, scenario_id,
                )
            return [
                self._evaluate_one_role(
                    persona_a=persona_a,
                    persona_b=persona_b,
                    scenario=scenario,
                    pa_id=pa_id,
                    pb_id=pb_id,
                    scenario_id=scenario_id,
                    tested_role=r["tested_role"],
                    case_id=r["case_id"],
                    result=result,
                    cached_per_turn_path=r.get("cached_per_turn_path"),
                    cached_holistic_path=r.get("cached_holistic_path"),
                )
                for r in roles
            ]

        # Reference-partner mode: each role needs its own dialogue with the
        # tested model playing that side.
        cases: List[CaseResult] = []
        for r in roles:
            case_id = r["case_id"]
            result = None
            if not self.force:
                result = _load_cached_dialogue(case_id)
            if result is None:
                result = self._generate_one_dialogue_with_retry(
                    persona_a, persona_b, scenario, r["tested_role"],
                    pa_id, pb_id, scenario_id,
                )
            cases.append(self._evaluate_one_role(
                persona_a=persona_a,
                persona_b=persona_b,
                scenario=scenario,
                pa_id=pa_id,
                pb_id=pb_id,
                scenario_id=scenario_id,
                tested_role=r["tested_role"],
                case_id=case_id,
                result=result,
                cached_per_turn_path=r.get("cached_per_turn_path"),
                cached_holistic_path=r.get("cached_holistic_path"),
            ))
        return cases

    def _evaluate_one_role(
        self,
        persona_a: Persona,
        persona_b: Persona,
        scenario: Scenario,
        pa_id: str,
        pb_id: str,
        scenario_id: str,
        tested_role: str,
        case_id: str,
        result: SessionResult,
        cached_per_turn_path: Optional[Path] = None,
        cached_holistic_path: Optional[Path] = None,
    ) -> CaseResult:
        tested_persona_id = pa_id if tested_role == "persona_a" else pb_id
        tested_persona = persona_a if tested_role == "persona_a" else persona_b

        # ── 空对话保护：若对话生成 0 轮，直接标记为 invalid ──────
        if not result.dialogue:
            LOGGER.warning("  case %s: EMPTY dialogue (0 turns generated)", case_id)
            from libs.evaluation.judge import L0Violation, HolisticScoreCard
            l0_violations = [L0Violation(
                constraint_id="L0-SYS",
                constraint_name="对话生成失败",
                severity="fatal",
                evidence="对话轮次为 0，模型未能产出任何内容",
                turn_index=-1,
            )]
            holistic_card = HolisticScoreCard(role=tested_persona.persona_id)
            # NOTE: 故意不调用 self.judge.save_trace() —— 空对话没有评测价值,
            # 避免往 eval_traces/ 写入 dialogue=[] / turn_scores=[] 的无用文件。
            dialogue_path = str(result.output_path) if result.output_path else ""
            case = CaseResult(
                case_id=case_id,
                persona_a_id=pa_id,
                persona_b_id=pb_id,
                scenario_id=scenario_id,
                tested_role=tested_role,
                category=str(scenario.raw.get("category", "")),
                sub_category=str(scenario.raw.get("sub_category", "")),
                turn_cards=[],
                holistic_card=holistic_card,
                l0_violations=l0_violations,
                dialogue_path=dialogue_path,
                domain=self.domain,
            )
            case.compute(pt_weight=self.pt_weight, ho_weight=self.ho_weight)
            LOGGER.info(
                "  case %s: status=%s (empty dialogue) dialogue=%s",
                case_id, case.status, dialogue_path,
            )
            return case

        # ── L0 Hard constraint check ──────────────────────────────
        l0_violations = self.judge.check_l0(
            dialogue=result.dialogue,
            persona=tested_persona,
            scenario=scenario,
            evaluated_role=tested_persona_id,
        )
        is_invalid = self.judge.has_fatal_violation(l0_violations)

        if is_invalid:
            LOGGER.warning("  case %s: L0 FAILED (%d violations)",
                           case_id, len(l0_violations))

        # ── Per-turn + holistic scoring (skip if L0 fatal) ────────
        from libs.evaluation.judge import HolisticScoreCard
        turn_cards: List["TurnScoreCard"] = []
        holistic_card = HolisticScoreCard(role=tested_persona_id)
        pt_cached = False
        ho_cached = False

        force_all = self.force or self.force_group == "all"
        force_pt = force_all or self.force_group == "per_turn"
        force_ho = force_all or self.force_group == "holistic"

        if cached_per_turn_path and not is_invalid and not force_pt:
            try:
                pt_data = json.loads(cached_per_turn_path.read_text(encoding="utf-8"))
                turn_cards = _turn_cards_from_data(pt_data)
                turn_cards.sort(key=lambda tc: tc.turn_index)
                pt_cached = True
                LOGGER.info("  case %s: per-turn scores loaded from cache", case_id)
            except Exception as exc:
                LOGGER.warning(
                    "  case %s: failed to load cached per-turn scores: %s",
                    case_id, exc,
                )

        if cached_holistic_path and not is_invalid and not force_ho:
            try:
                ho_data = json.loads(cached_holistic_path.read_text(encoding="utf-8"))
                holistic_card = _holistic_card_from_data(ho_data, tested_persona_id)
                ho_cached = True
                LOGGER.info("  case %s: holistic scores loaded from cache", case_id)
            except Exception as exc:
                LOGGER.warning(
                    "  case %s: failed to load cached holistic scores: %s",
                    case_id, exc,
                )

        if not is_invalid and (not turn_cards or not holistic_card.dim_results):
            # Collect turn indices for this persona
            turn_tasks = [
                (i, turn)
                for i, turn in enumerate(result.dialogue)
                if turn.get("role") == tested_persona_id
            ]
            # 根据对话长度自适应并发数，避免超长对话压垮后端
            total_msgs = sum(len(t.get("response") or []) for t in result.dialogue)
            if total_msgs > 200:
                cap = 2
            elif total_msgs > 80:
                cap = 3
            else:
                cap = 4
            need_pt = not turn_cards
            need_ho = not holistic_card.dim_results
            max_workers = min(cap, (len(turn_tasks) if need_pt else 0) + (1 if need_ho else 0))
            if max_workers < 1:
                max_workers = 1
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                turn_futures = {}
                if need_pt:
                    turn_futures = {
                        pool.submit(
                            _score_turn_with_retry,
                            self.judge, i, turn, result.dialogue,
                            tested_persona, scenario,
                        ): i
                        for i, turn in turn_tasks
                    }
                holistic_future = None
                if need_ho:
                    holistic_future = pool.submit(
                        _score_holistic_with_retry,
                        self.judge, result.dialogue,
                        tested_persona, scenario, tested_persona_id,
                    )
                for future in as_completed(turn_futures):
                    idx = turn_futures[future]
                    try:
                        tc = future.result()
                        turn_cards.append(tc)
                    except Exception as exc:
                        LOGGER.warning(
                            "  case %s: turn %d scoring failed: %s",
                            case_id, idx, exc,
                        )
                if holistic_future is not None:
                    try:
                        holistic_card = holistic_future.result()
                    except Exception as exc:
                        LOGGER.warning(
                            "  case %s: holistic scoring failed: %s",
                            case_id, exc,
                        )
            if turn_cards:
                turn_cards.sort(key=lambda tc: tc.turn_index)

        # ── Save eval trace (only when evaluation completed) ─────────
        # 中断 / 不完整的评测(L0 fatal 跳过打分,或打分阶段部分失败导致
        # turn_cards / holistic.dim_results 缺失)不写 trace,避免产出无价值文件。
        status = "invalid" if is_invalid else "valid"
        holistic_complete = bool(
            holistic_card and getattr(holistic_card, "dim_results", None)
        )
        eval_complete = (
            not is_invalid
            and bool(turn_cards)
            and holistic_complete
        )
        if eval_complete:
            eval_sub = _eval_dir(
                self.tested_model, self.ref_model, self.judge_model, self.self_play)
            self.judge.save_trace(
                case_id=case_id,
                turn_cards=turn_cards,
                holistic_card=holistic_card,
                dialogue=result.dialogue,
                model_name=eval_sub,
                l0_violations=l0_violations,
                status=status,
            )
        else:
            LOGGER.warning(
                "  case %s: INCOMPLETE evaluation (is_invalid=%s, turns=%d, "
                "holistic_dims=%d) — skip eval trace",
                case_id, is_invalid, len(turn_cards),
                len(getattr(holistic_card, "dim_results", None) or []),
            )

        # ── Build CaseResult ──────────────────────────────────────
        dialogue_path = str(result.output_path) if result.output_path else ""

        case = CaseResult(
            case_id=case_id,
            persona_a_id=pa_id,
            persona_b_id=pb_id,
            scenario_id=scenario_id,
            tested_role=tested_role,
            category=str(scenario.raw.get("category", "")),
            sub_category=str(scenario.raw.get("sub_category", "")),
            turn_cards=turn_cards,
            holistic_card=holistic_card,
            l0_violations=l0_violations,
            dialogue_path=dialogue_path,
            domain=self.domain,
        )
        case.apply_style_gate(result.dialogue, tested_persona_id)
        case.compute(pt_weight=self.pt_weight, ho_weight=self.ho_weight)

        cache_tag = ""
        if pt_cached or ho_cached:
            parts = []
            if pt_cached:
                parts.append("pt=cache")
            if ho_cached:
                parts.append("ho=cache")
            cache_tag = " (" + ",".join(parts) + ")"

        LOGGER.info(
            "  case %s: status=%s per_turn=%.4f holistic=%.4f final=%.4f grade=%s dialogue=%s%s",
            case_id, case.status, case.per_turn_weighted,
            case.holistic_weighted, case.final_score, case.grade,
            dialogue_path, cache_tag,
        )
        return case


# ── Module-level retry helpers (used by ThreadPoolExecutor) ───────────

def _score_turn_with_retry(
    judge: BaseBenchmarkJudge,
    turn_index: int,
    turn: Dict[str, Any],
    dialogue: List[Dict[str, Any]],
    persona: "Persona",
    scenario: "Scenario",
) -> "TurnScoreCard":
    """Score a single turn with retry logic."""
    last_err = None
    for attempt in range(1, _SCORE_TASK_MAX_RETRIES + 1):
        try:
            return judge.score_turn(
                turn_index=turn_index,
                turn=turn,
                dialogue=dialogue,
                persona=persona,
                scenario=scenario,
            )
        except Exception as exc:
            last_err = exc
            if attempt < _SCORE_TASK_MAX_RETRIES:
                sleep_s = _SCORE_TASK_BACKOFF_BASE ** attempt
                LOGGER.warning(
                    "score_turn(turn=%d) attempt %d/%d failed: %s; retrying in %.1fs",
                    turn_index, attempt, _SCORE_TASK_MAX_RETRIES, exc, sleep_s,
                )
                time.sleep(sleep_s)
    raise RuntimeError(
        f"score_turn(turn={turn_index}) exhausted {_SCORE_TASK_MAX_RETRIES} retries"
    ) from last_err


def _score_holistic_with_retry(
    judge: BaseBenchmarkJudge,
    dialogue: List[Dict[str, Any]],
    persona: "Persona",
    scenario: "Scenario",
    evaluated_role: str,
) -> "HolisticScoreCard":
    """Score holistic with retry logic."""
    last_err = None
    for attempt in range(1, _SCORE_TASK_MAX_RETRIES + 1):
        try:
            return judge.score_holistic(
                dialogue=dialogue,
                persona=persona,
                scenario=scenario,
                evaluated_role=evaluated_role,
            )
        except Exception as exc:
            last_err = exc
            if attempt < _SCORE_TASK_MAX_RETRIES:
                sleep_s = _SCORE_TASK_BACKOFF_BASE ** attempt
                LOGGER.warning(
                    "score_holistic(role=%s) attempt %d/%d failed: %s; retrying in %.1fs",
                    evaluated_role, attempt, _SCORE_TASK_MAX_RETRIES, exc, sleep_s,
                )
                time.sleep(sleep_s)
    raise RuntimeError(
        f"score_holistic(role={evaluated_role}) exhausted {_SCORE_TASK_MAX_RETRIES} retries"
    ) from last_err
