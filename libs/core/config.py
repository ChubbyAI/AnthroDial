"""Global configuration loader.

Reads a YAML config file (e.g. ``configs/benchmark/xxx.yaml``) via
``load_settings(path)``, exposes a ``Settings`` singleton plus path helpers.
API keys are NEVER stored in YAML; they are pulled from environment variables.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    raw: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None

    # ----- generic accessors -----
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # ----- path helpers (always absolute) -----
    def path(self, key: str) -> Path:
        rel = self.get("paths", key)
        if rel is None:
            raise KeyError(f"paths.{key} not configured")
        p = Path(rel)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p

    def ensure_dirs(self) -> None:
        for key in ("data_dir", "personas_dir", "scenarios_dir",
                    "outputs_dir", "results_dir"):
            try:
                self.path(key).mkdir(parents=True, exist_ok=True)
            except KeyError:
                pass
        # outputs subdirs
        for sub in ("dialogues", "interrupts", "skill_history", "eval_traces",
                    "raw_traces", "logs"):
            (self.path("outputs_dir") / sub).mkdir(parents=True, exist_ok=True)

    # ----- llm helpers -----
    def model_for(self, role: str) -> str:
        return self.get("llm", "models", role,
                        default=self.get("llm", "default_model", default="gpt-4o"))

    def temperature_for(self, role: str, default: float = 0.7) -> float:
        return float(self.get("llm", "temperature", role, default=default))

    def model_config_for(self, name: str) -> Dict[str, Any]:
        """Return per-model config overrides from llm.model_configs, or empty dict."""
        configs = self.get("llm", "model_configs", default={}) or {}
        if not isinstance(configs, dict):
            return {}
        return configs.get(name, {})

    def api_model_for(self, name: str) -> str:
        """Resolve display/alias name to actual API model name."""
        cfg = self.model_config_for(name)
        return cfg.get("api_model", name)


_settings: Optional[Settings] = None


def load_settings(path: Path) -> Settings:
    global _settings
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    _settings = Settings(raw=raw, config_path=cfg_path)
    _settings.ensure_dirs()
    return _settings


def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError(
            "Settings not initialized. Call load_settings(path) first, "
            "e.g. via --config on the command line."
        )
    return _settings


def get_api_key(env_var: str = "OPENAI_API_KEY") -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. "
            "Export it before running, e.g. `export OPENAI_API_KEY=sk-...`."
        )
    return key


def slug(name: str, fallback: str = "unknown") -> str:
    """把任意名字转为合法的目录名（保留字母、数字、-、_、.）。"""
    if not name:
        return fallback
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
    return safe or fallback


def scenario_path(scenario_id: str) -> Path:
    """Split scenario_id (e.g. 'A0001_B0002_C0001') into nested Path: A0001/B0002/C0001.

    Falls back to a flat slug if format does not match.
    """
    parts = scenario_id.split("_") if scenario_id else []
    if len(parts) >= 3:
        return Path(parts[0]) / parts[1] / parts[2]
    return Path(slug(scenario_id))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(
    entry: str = "run",
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> Optional[Path]:
    """统一初始化根 logger：控制台 + 可选文件。

    - 控制台以 ``level`` 输出（默认 INFO）。
    - 文件写入 ``outputs/logs/<entry>_<ts>.log``，并包含代码位置，方便事后追查。
    - 把 ``openai._base_client`` / ``httpx`` 的 INFO 也汇入文件（控制台默认静音为 WARNING）。
      以免底层 SDK 的重试提示淹没业务日志。
    """
    root = logging.getLogger()
    # 重复调用时清理之前的 handler，避免重复输出。
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    fmt_console = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_file = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s [%(filename)s:%(lineno)d] | %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt_console)
    root.addHandler(console)

    log_path: Optional[Path] = None
    if log_to_file:
        try:
            settings = get_settings()
            log_dir = settings.path("outputs_dir") / "logs"
        except Exception:
            log_dir = REPO_ROOT / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{slug(entry)}_{ts}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # 文件记一切
        fh.setFormatter(fmt_file)
        root.addHandler(fh)

    # 底层 SDK 日志：文件 INFO、控制台 WARNING（避免被 "Retrying request..." 刷屏）。
    for noisy in ("openai", "openai._base_client", "httpx", "httpcore"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.INFO)
        lg.propagate = True
        # 重点：为控制台加个高阈值 filter，让 INFO 只走文件、不走控制台。
    class _ConsoleHigh(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.name.startswith(("openai", "httpx", "httpcore")):
                return record.levelno >= logging.WARNING
            return True
    console.addFilter(_ConsoleHigh())

    if log_path:
        logging.getLogger(__name__).info("logging to %s", log_path)
    return log_path
