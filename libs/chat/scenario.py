"""Scenario card I/O. Mirrors persona.py but for §4.3 schema.

场景卡 ID 采用三层路径编号：``A####_B####_C####``，例如
``A0001_B0001_C0001`` 表示大类 A0001 下子类 B0001 的叶子场景 C0001。
路径解析依赖 ``data/catalog.yaml`` 中的 slug 映射。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from libs.core.config import get_settings


# Fields that may carry per-persona dict values (双方状态各自描述)。
DUAL_FIELDS = {"place", "trigger_event", "emotion_tone"}

FIELD_LABELS: List[tuple] = [
    ("name", "场景名称"),
    ("category", "场景大类"),
    ("sub_category", "场景子类"),
    ("relationship", "对话关系"),
    ("time", "时间背景"),
    ("place", "空间背景"),
    ("trigger_event", "触发事件"),
    ("emotion_tone", "情绪基调"),
]

# 全路径 scenario_id 正则。
SID_REGEX = re.compile(r"^A(\d{4})_B(\d{4})_C(\d{4})$")
CATALOG_FILENAME = "syn_catalog.yaml"


@lru_cache(maxsize=1)
def _load_catalog() -> Dict[str, Any]:
    """读取 ``data/syn_catalog.yaml`` 并缓存。

    返回原始 dict（键为 A编号，值含 name/slug/children）。
    文件不存在时返回空 dict。
    """
    s = get_settings()
    if s.get("paths", "catalog_path"):
        path = s.path("catalog_path")
    else:
        try:
            path = s.path("seeds_dir").parent / CATALOG_FILENAME
        except KeyError:
            path = s.path("data_dir").parent / CATALOG_FILENAME
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _parse_sid(sid: str) -> Tuple[str, str, str]:
    """拆分 ``A0001_B0001_C0001`` 为 (a_code, b_code, c_code)。

    打不中正则时抛 ``ValueError``。
    """
    m = SID_REGEX.match(sid)
    if not m:
        raise ValueError(
            f"scenario_id {sid!r} 不符合新格式 A####_B####_C####。"
            " 请使用如 'A0002_B0001_C0001' 这样的全路径 ID。"
        )
    return f"A{m.group(1)}", f"B{m.group(2)}", f"C{m.group(3)}"


def _resolve_scenario_path(sid: str) -> Path:
    """根据 sid 并结合 catalog 拼出 yaml 路径。

    返回：``<scenarios_dir>/A0001_<slugA>/B0001_<slugB>/A0001_B0001_C0001.yaml``。
    如果 catalog 没有该 (A,B) 节点，抛 ``KeyError``。
    """
    a, b, _c = _parse_sid(sid)
    catalog = _load_catalog()
    a_node = catalog.get(a)
    if not a_node:
        raise KeyError(f"大类 {a} 在 catalog.yaml 中不存在")
    children = a_node.get("children") or {}
    b_node = children.get(b)
    if not b_node:
        raise KeyError(f"子类 {a}/{b} 在 catalog.yaml 中不存在")
    a_slug = str(a_node.get("slug") or "").strip() or a.lower()
    b_slug = str(b_node.get("slug") or "").strip() or b.lower()
    base = get_settings().path("scenarios_dir")
    return base / f"{a}_{a_slug}" / f"{b}_{b_slug}" / f"{sid}.yaml"


@dataclass
class Scenario:
    scenario_id: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Scenario":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        sid = str(data.get("scenario_id") or path.stem)
        return cls(scenario_id=sid, raw=data)

    @classmethod
    def load(cls, scenario_id: str) -> "Scenario":
        """按 sid 加载场景卡。

        查找顺序：
        1. catalog 路径解析（A####_B####_C#### 格式）
        2. scenarios_dir 下 flat 文件（如 BS01.yaml）
        3. scenarios_dir 下递归扫描
        4. seeds 目录递归扫描
        """
        s = get_settings()
        candidates: List[Path] = []
        # 1. 主路径：全路径 ID → catalog 拼接。
        try:
            candidates.append(_resolve_scenario_path(scenario_id))
        except (ValueError, KeyError):
            pass
        # 2. Flat：scenarios_dir/<sid>.yaml（benchmark 等简单结构）。
        scenarios_root = s.path("scenarios_dir")
        candidates.append(scenarios_root / f"{scenario_id}.yaml")
        # 3. scenarios_dir 递归扫描。
        if scenarios_root.exists():
            for c in scenarios_root.rglob(f"{scenario_id}.yaml"):
                if c not in candidates:
                    candidates.append(c)
        # 4. seeds 目录递归扫描。
        try:
            seeds_root = s.path("seeds_dir") / "scenarios"
            if seeds_root.exists():
                for c in seeds_root.rglob(f"{scenario_id}.yaml"):
                    candidates.append(c)
        except KeyError:
            pass
        for c in candidates:
            if c.exists():
                return cls.from_yaml(c)
        raise FileNotFoundError(
            f"Scenario {scenario_id} not found. tried: {[str(p) for p in candidates]}"
        )

    def save(self, target_dir: Optional[Path] = None) -> Path:
        """按 catalog 结构保存 yaml。

        传入 ``target_dir`` 时直接写入该目录（仅供测试使用）；否则根据
        ``self.scenario_id`` 拆出 A/B 路径自动 mkdir 父目录。
        """
        if target_dir is not None:
            base = Path(target_dir)
            base.mkdir(parents=True, exist_ok=True)
            out = base / f"{self.scenario_id}.yaml"
        else:
            out = _resolve_scenario_path(self.scenario_id)
            out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.raw, fh, allow_unicode=True, sort_keys=False)
        return out

    def to_system_prompt(self, self_id: str, partner_id: str) -> str:
        lines: List[str] = [
            f"## 当前场景起点 (场景ID: {self.scenario_id})",
            f"你 ({self_id}) 正在和对方 ({partner_id}) 通过类似微信的方式聊天。",
            "",
            "以下信息定义你们**开始聊天时的状态**（在哪、刚发生了什么、心情如何），",
            "而不是限定你们只能聊这个话题。触发事件只是找对方聊天的**契机**，",
            "对话可以自由发展到任何符合你们关系和人设的话题。",
            "",
            "唯一的约束：你的状态要符合客观物理世界的规律和常识。",
        ]
        for key, label in FIELD_LABELS:
            if key not in self.raw:
                continue
            val = self.raw[key]
            if val in (None, "", []):
                continue
            if key in DUAL_FIELDS and isinstance(val, dict):
                self_v = val.get(self_id)
                partner_v = val.get(partner_id)
                lines.append(f"- {label}:")
                if self_v:
                    lines.append(f"    - 你的状态: {self_v}")
                if partner_v:
                    lines.append(f"    - 对方状态: {partner_v}")
                # 兼容多人场景：其他 key 也原样带上
                for k, v in val.items():
                    if k in (self_id, partner_id) or not v:
                        continue
                    lines.append(f"    - {k}: {v}")
            else:
                lines.append(f"- {label}: {val}")
        return "\n".join(lines)

    @property
    def name(self) -> str:
        return str(self.raw.get("name") or self.scenario_id)

    @property
    def first_speaker(self) -> Optional[str]:
        """场景指定的开场方 persona_id。为空时调度方可退退到 agent_a 先开口。"""
        v = self.raw.get("first_speaker")
        if v in (None, ""):
            return None
        return str(v).strip() or None

    @property
    def virtual_start_time(self) -> datetime:
        """虚拟时钟的起始塔钟。

        优先读取 ``raw['start_time']``，支持多种格式：
        - ``2026-05-26 22:00`` / ``2026-05-26 22:00:00`` / ISO 8601
        - 仅时分 ``22:00`` → 取今天该时分。

        后退到 ``raw['time']`` 的轻量解析（如「工作日晚上 10 点」→ 22:00）。
        完全解析失败时返回当前壁钟（这是退退退，会在日志中折在不可复现）。
        """
        # 1. start_time 优先
        st = self.raw.get("start_time")
        if st:
            text = str(st).strip()
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    continue
            # 仅时分
            try:
                t = datetime.strptime(text, "%H:%M").time()
                today = datetime.now().date()
                return datetime.combine(today, t)
            except ValueError:
                pass
        # 2. 从 time 字段轻量推断（“晚上 10 点”→ 22:00）
        time_text = str(self.raw.get("time") or "").strip()
        if time_text:
            today = datetime.now().date()
            base = datetime.combine(today, datetime.min.time())
            # “N 点 / N：MM”
            m = re.search(r"(\d{1,2})\s*[:点](\d{0,2})", time_text)
            hour = minute = None
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2)) if m.group(2) else 0
            # 上下午课正
            if hour is not None:
                if ("晚上" in time_text or "夜间" in time_text or "pm" in time_text.lower()) and hour < 12:
                    hour += 12
                if "凌晨" in time_text and hour >= 12:
                    hour -= 12
                hour = max(0, min(hour, 23))
                minute = max(0, min(minute, 59))
                return base + timedelta(hours=hour, minutes=minute)
        # 3. 完全退退
        return datetime.now()

    @property
    def max_turns(self) -> Optional[int]:
        """从场景卡中读取 max_turns 字段。
    
        用于控制对话的最大轮次上限。返回 None 表示场景卡未指定，
        由系统全局默认值决定。
        """
        raw_val = self.raw.get("max_turns")
        if raw_val is None:
            return None
        try:
            return int(raw_val)
        except (TypeError, ValueError):
            return None
    
    @property
    def duration_ms(self) -> int:
        """[Deprecated] 解析场景卡 duration 字段为毫秒数。
    
        新版场景卡使用 max_turns 代替 duration。
        保留此属性仅为向后兼容，新场景卡不应使用 duration 字段。
        """
        raw_val = self.raw.get("duration", "")
        if not raw_val:
            return 900_000  # default 15min
        text = str(raw_val).strip()
        # 中文"N分钟"
        m = re.search(r"(\d+(?:\.\d+)?)\s*分钟?", text)
        if m:
            return int(float(m.group(1)) * 60_000)
        # 中文"N小时"
        m = re.search(r"(\d+(?:\.\d+)?)\s*小时", text)
        if m:
            return int(float(m.group(1)) * 3_600_000)
        # 英文 "Nmin" / "Nm"
        m = re.search(r"(\d+(?:\.\d+)?)\s*m(?:in)?\b", text, re.IGNORECASE)
        if m:
            return int(float(m.group(1)) * 60_000)
        # 英文 "Nh" / "Nhour"
        m = re.search(r"(\d+(?:\.\d+)?)\s*h(?:our)?\b", text, re.IGNORECASE)
        if m:
            return int(float(m.group(1)) * 3_600_000)
        # 纯数字（假定分钟）
        m = re.match(r"(\d+(?:\.\d+)?)", text)
        if m:
            return int(float(m.group(1)) * 60_000)
        return 900_000
