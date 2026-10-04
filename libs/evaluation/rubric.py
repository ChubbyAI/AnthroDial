"""Benchmark rubric: L0 hard constraints + D/E checkbox scoring dimensions.

Loads from ``rubric.yaml``:
- ``hard_constraints``: L0 one-vote-veto rules (keyword + LLM check).
- ``dimensions``: D1-D5 (per_turn) + E1-E5 (holistic).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from libs.core.config import REPO_ROOT, get_settings


@dataclass
class Checkbox:
    id: str
    description: str


@dataclass
class HardConstraint:
    """L0 硬约束定义."""
    id: str
    name: str
    description: str
    severity: str  # "fatal" | "major"
    keywords: List[str] = field(default_factory=list)


@dataclass
class BenchmarkDimension:
    id: str
    name: str
    description: str
    scope: str  # "per_turn" | "holistic"
    weight: float = 10.0
    checkboxes: List[Checkbox] = field(default_factory=list)

    @property
    def checkbox_count(self) -> int:
        return len(self.checkboxes)


def _rubric_path() -> Path:
    s = get_settings()
    rel = s.get("benchmark", "rubric_path",
                 default="data/chatbot_benchmark/rubric.yaml")
    path = Path(rel)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


_VALID_DOMAINS = ("chatbot", "game", "cdial", "clam", "general")


def detect_domain() -> str:
    """Detect the evaluation domain from benchmark settings.

    Priority: explicit ``benchmark.domain`` config key, then the
    ``data/<domain>_benchmark`` directory marker in ``benchmark.rubric_path``.
    A plain substring match on the whole path must be avoided: the repo root
    directory is itself named "chatbot", so every absolute rubric path under
    it would otherwise resolve to the chatbot domain.
    """
    s = get_settings()
    domain = str(s.get("benchmark", "domain", default="") or "").strip().lower()
    if domain in _VALID_DOMAINS:
        return domain
    rubric_path = str(s.get("benchmark", "rubric_path", default=""))
    p = rubric_path.lower()
    for key in _VALID_DOMAINS:
        if f"{key}_benchmark" in p:
            return key
    # Fallback: match on individual path components (never a substring of
    # the whole path, for the repo-root reason above).
    parts = {q.lower() for q in Path(rubric_path).parts}
    for key in _VALID_DOMAINS:
        if key in parts:
            return key
    return "game"


def _load_raw(path: Path | str | None = None) -> Dict[str, Any]:
    if path is None:
        path = _rubric_path()
    else:
        path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── Hard constraints ──────────────────────────────────────────────────

def load_hard_constraints(path: Path | str | None = None) -> List[HardConstraint]:
    raw = _load_raw(path)
    out: List[HardConstraint] = []
    for hc in (raw.get("hard_constraints") or []):
        if not isinstance(hc, dict):
            continue
        out.append(HardConstraint(
            id=str(hc.get("id", "")),
            name=str(hc.get("name", "")),
            description=str(hc.get("description", "")),
            severity=str(hc.get("severity", "major")),
            keywords=[str(k) for k in (hc.get("keywords") or [])],
        ))
    return out


# ── Dimensions ────────────────────────────────────────────────────────

def load_rubric(path: Path | str | None = None) -> List[BenchmarkDimension]:
    raw = _load_raw(path)
    dims_raw: List[Dict[str, Any]] = raw.get("dimensions") or []
    out: List[BenchmarkDimension] = []
    for d in dims_raw:
        if not isinstance(d, dict):
            continue
        cbs = []
        for cb in (d.get("checkboxes") or []):
            if isinstance(cb, dict) and cb.get("id"):
                cbs.append(Checkbox(
                    id=str(cb["id"]),
                    description=str(cb.get("description", "")),
                ))
        out.append(BenchmarkDimension(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            scope=str(d.get("scope", "per_turn")),
            weight=float(d.get("weight", 10)),
            checkboxes=cbs,
        ))
    return out


def per_turn_dimensions(dims: List[BenchmarkDimension] | None = None) -> List[BenchmarkDimension]:
    dims = dims if dims is not None else load_rubric()
    return [d for d in dims if d.scope == "per_turn"]


def holistic_dimensions(dims: List[BenchmarkDimension] | None = None) -> List[BenchmarkDimension]:
    dims = dims if dims is not None else load_rubric()
    return [d for d in dims if d.scope == "holistic"]


def dimensions_hash(dims: List[BenchmarkDimension]) -> str:
    """Return a stable hash of the dimension definitions (rubric version key).

    The hash captures dim id, name, description, weight, scope and all checkbox
    descriptions. It is used to invalidate dimension-group caches when the rubric
    changes.
    """
    parts = []
    for d in sorted(dims, key=lambda x: x.id):
        parts.append(f"dim:{d.id}:{d.name}:{d.description}:{d.weight}:{d.scope}")
        for cb in sorted(d.checkboxes, key=lambda x: x.id):
            parts.append(f"cb:{d.id}:{cb.id}:{cb.description}")
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()
