"""Lightweight wrapper around the OpenAI chat completions API.

- API key is pulled from the environment (never stored in code/yaml).
- Supports per-call ``base_url`` / ``api_key_env`` overrides for evaluating
  models from different providers.
- Adds simple retry logic and JSON-mode parsing helpers.
- 重试只走本文件里的循环；SDK 内置重试被明确关闭（``max_retries=0``），
  以避免跨层重试乘积并屏蔽错误详情。
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from libs.core.config import get_api_key, get_settings

# 匹配任意位置的 <think>...</think> 块（含换行）。
# 部分后端（如 vLLM 开启 reasoning-parser）会直接返回 message.reasoning_content；
# 但其它部署会把思维链放在 content 里，需要手动抽取。
# 注意：Qwen3.5 的 think 模式输出 *没有* 开头 <think>，只用 </think> 闭合标记
# 思维链与正式回答的分界，因此另用 _CLOSE_TAG 处理这种特殊形态。
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)
_CLOSE_TAG = "</think>"

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMResult:
    content: str
    reasoning_content: str = ""   # DeepSeek-R1 / o1 等模型的思维链，普通模型为空字符串
    raw: Any = None  # full response object for debugging
    usage: Optional[Dict[str, int]] = None  # {prompt_tokens, completion_tokens, total_tokens}


class LLMClient:
    """Thin wrapper that hides repeated boilerplate around openai.chat."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        s = get_settings()
        self.base_url = base_url or s.get("llm", "base_url")
        self.api_key_env = api_key_env or s.get("llm", "api_key_env", default="OPENAI_API_KEY")
        self.timeout = timeout or s.get("llm", "request", "timeout", default=60)
        self.max_retries = max_retries or s.get("llm", "request", "max_retries", default=10)
        # API key 获取优先级：构造参数 api_key > yaml 中 llm.api_key 字面量 > 环境变量。
        # 本地 vLLM/sglang 部署不验证 key，可在 yaml 中直接写 api_key: "EMPTY"。
        literal_key = api_key if api_key is not None else s.get("llm", "api_key", default=None)
        if literal_key:
            resolved_key = str(literal_key)
        else:
            resolved_key = get_api_key(self.api_key_env)
        # thinking 开关：构造函数参数显式指定，None 时默认 False。
        # 每个模型的 thinking 应在 model_configs 中各自声明，通过构造参数传入。
        if enable_thinking is None:
            enable_thinking = False
        self.enable_thinking: bool = bool(enable_thinking)
        # 默认 extra_body（yaml 中常填 top_k 等 OpenAI 不直接暴露的字段）。
        cfg_extra = s.get("llm", "extra_body", default={}) or {}
        if not isinstance(cfg_extra, dict):
            cfg_extra = {}
        if extra_body:
            cfg_extra = {**cfg_extra, **extra_body}
        self.default_extra_body: Dict[str, Any] = cfg_extra
        # 明确关闭 SDK 内置重试；重试仅由本类控制，保证每次错误都能被我们 log 下来。
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            max_retries=0,
        )

    # ------------------------------------------------------------------
    def _build_extra_body(
        self,
        enable_thinking: Optional[bool],
        call_extra_body: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], bool]:
        """合并默认 extra_body / 此次调用覆盖 / thinking 开关。

        返回 (extra_body, effective_enable_thinking)。
        """
        eff_thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        merged: Dict[str, Any] = copy.deepcopy(self.default_extra_body)
        if call_extra_body:
            for k, v in call_extra_body.items():
                merged[k] = v
        # 根据 thinking 开关注入 chat_template_kwargs：
        # 这是 Qwen3 系列控制思考的标准姿势（走 chat_template）。
        ctk = merged.get("chat_template_kwargs")
        ctk = dict(ctk) if isinstance(ctk, dict) else {}
        ctk["enable_thinking"] = eff_thinking
        merged["chat_template_kwargs"] = ctk
        return merged, eff_thinking

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        enable_thinking: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> LLMResult:
        last_err: Optional[Exception] = None
        merged_extra_body, eff_thinking = self._build_extra_body(enable_thinking, extra_body)
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": self.timeout,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if response_format is not None:
                    kwargs["response_format"] = response_format
                # extra_body 总是带上（至少含 chat_template_kwargs.enable_thinking）。
                kwargs["extra_body"] = merged_extra_body
                kwargs.update(extra)
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                raw_content = msg.content or ""
                # 1) 后端如果已分离 reasoning（如 vLLM reasoning-parser），直接取。
                #    不同 parser 可能用不同字段名：
                #    - deepseek_r1: msg.reasoning_content
                #    - qwen3: msg.model_extra['reasoning']
                # 2) 否则从 content 中抽取 <think>...</think> 作为 reasoning，并从 content 中剥离。
                reasoning_content = getattr(msg, "reasoning_content", "") or ""
                if not reasoning_content:
                    # vLLM --reasoning-parser qwen3 把思维链放在 model_extra.reasoning
                    _extra = getattr(msg, "model_extra", None)
                    if isinstance(_extra, dict):
                        reasoning_content = _extra.get("reasoning", "") or ""
                content, extracted_reasoning = _split_think_block(raw_content)
                if not reasoning_content:
                    reasoning_content = extracted_reasoning
                # 关闭 thinking 时强制清空，避免污染下游（许多后端在 enable_thinking=False 时
                # 仍会返回一个空的 <think></think>）。
                if not eff_thinking:
                    reasoning_content = ""
                if attempt > 1:
                    LOGGER.info("LLM call OK on attempt %d/%d (model=%s)",
                                attempt, self.max_retries, model)
                # 提取 token usage 统计（不需要 logprobs）
                token_usage: Optional[Dict[str, int]] = None
                if hasattr(resp, "usage") and resp.usage is not None:
                    token_usage = {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
                    }
                return LLMResult(
                    content=content,
                    reasoning_content=reasoning_content,
                    raw=resp,
                    usage=token_usage,
                )
            except Exception as exc:  # network / 4xx / 5xx / parse
                last_err = exc
                detail = _format_exc_detail(exc)
                preview = _preview_messages(messages)
                LOGGER.warning(
                    "LLM call failed (attempt %s/%s, model=%s) %s\n  prompt_preview=%s",
                    attempt, self.max_retries, model, detail, preview,
                )
                if attempt < self.max_retries:
                    sleep_s = min(2 ** attempt, 10)
                    LOGGER.info("sleeping %.1fs before next attempt", sleep_s)
                    time.sleep(sleep_s)
        # 重试耗尽：取一个含上下文的 RuntimeError 抛出去，上层会再重试 / 降级。
        LOGGER.error("LLM call exhausted %d attempts; last_err_type=%s msg=%s",
                     self.max_retries, type(last_err).__name__, last_err)
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts: "
            f"{type(last_err).__name__}: {last_err}"
        ) from last_err

    # ------------------------------------------------------------------
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Call chat() and parse the result as JSON.

        Tries ``response_format={type: json_object}`` first; if the model does
        not support it, falls back to extracting a JSON block from raw text.
        """
        try:
            res = self.chat(
                messages,
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                **extra,
            )
            return json.loads(res.content)
        except Exception as exc:
            LOGGER.warning(
                "chat_json with response_format=json_object failed (%s: %s); "
                "falling back to raw extraction.",
                type(exc).__name__, exc,
            )
            res = self.chat(messages, model=model, temperature=temperature, **extra)
            return _safe_extract_json(res.content)

    def chat_json_with_raw(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        **extra: Any,
    ) -> Tuple[Dict[str, Any], LLMResult]:
        """chat_json 的带原始输出版本，同时返回解析后字典和原始 LLMResult。

        兑为 SFT 数据制作：
        - raw.content 即模型原始输出文本
        - raw.reasoning_content 即思维链（普通模型为空字符串）
        """
        try:
            raw = self.chat(
                messages,
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                **extra,
            )
            return json.loads(raw.content), raw
        except Exception as exc:
            LOGGER.warning(
                "chat_json_with_raw json_object mode failed (%s: %s); "
                "falling back to raw extraction.",
                type(exc).__name__, exc,
            )
            raw = self.chat(messages, model=model, temperature=temperature, **extra)
            return _safe_extract_json(raw.content), raw


def _split_think_block(text: str) -> Tuple[str, str]:
    """从模型原始输出中抽出思维链，并返回 (clean_content, reasoning)。

    支持三种形态：
    1. 标准成对 <think>...</think>（DeepSeek-R1 / 通用形态）：抽出标签内内容，
       多块拼接，clean_content 是去掉标签后的纯文本。
    2. **Qwen3.5 形态**：开头无 <think>，只有 </think> 闭合：
       </think> 之前的所有内容视作 reasoning，</think> 之后视作 clean_content。
    3. 只有开 <think> 但无闭合（流式截断或后端占位）：标签之后全部当 reasoning。
    4. 完全不含 think 标签：原样返回。
    """
    if not text:
        return "", ""
    lower = text.lower()
    open_idx = lower.find("<think>")
    close_idx = lower.find(_CLOSE_TAG)

    # 1) 标准成对（至少一对完整 <think>...</think>）。
    if open_idx != -1 and close_idx != -1 and close_idx > open_idx:
        matches = list(_THINK_RE.finditer(text))
        if matches:
            reasoning_parts = [m.group(1).strip() for m in matches]
            cleaned = _THINK_RE.sub("", text).strip()
            return cleaned, "\n\n".join(p for p in reasoning_parts if p)

    # 2) Qwen3.5 形态：只出现 </think>，没有 <think>。
    #    </think> 前是思维链，之后是正式回答。
    if open_idx == -1 and close_idx != -1:
        reasoning = text[:close_idx].strip()
        cleaned = text[close_idx + len(_CLOSE_TAG):].strip()
        return cleaned, reasoning

    # 3) 只有 <think>，没有 </think>：标签之后都是 thinking。
    if open_idx != -1 and close_idx == -1:
        head = text[:open_idx].strip()
        tail = text[open_idx + len("<think>"):].strip()
        return head, tail

    # 4) 不含任何 think 标签。
    return text, ""


def _format_exc_detail(exc: BaseException) -> str:
    """拼出一段包含状态码 / body 预览的故障描述，方便事后追查。"""
    parts = [f"{type(exc).__name__}: {exc}"]
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status={status}")
    resp = getattr(exc, "response", None)
    if resp is not None:
        text = None
        for attr in ("text", "content"):
            try:
                v = getattr(resp, attr, None)
                if v:
                    text = v if isinstance(v, str) else v.decode("utf-8", "replace")
                    break
            except Exception:
                continue
        if text:
            text = text.strip().replace("\n", " ")
            parts.append(f"body={text[:300]!r}")
    req_id = getattr(exc, "request_id", None)
    if req_id:
        parts.append(f"request_id={req_id}")
    return " | ".join(parts)


def _preview_messages(messages: List[Dict[str, Any]], max_chars: int = 120) -> str:
    """取 system 首句 + 最后一条 user，各截断到 max_chars，仅供调试。"""
    if not messages:
        return "<empty>"
    sys_msg = next((m for m in messages if m.get("role") == "system"), None)
    last = messages[-1]

    def _trim(text: str) -> str:
        text = (text or "").replace("\n", " ").strip()
        return text[:max_chars] + ("…" if len(text) > max_chars else "")

    parts = []
    if sys_msg:
        parts.append(f"sys={_trim(sys_msg.get('content', ''))}")
    parts.append(f"last[{last.get('role')}]={_trim(last.get('content', ''))}")
    return " || ".join(parts)


def _safe_extract_json(text: str) -> Dict[str, Any]:
    """Best-effort: locate the first {...} block and json.loads it."""
    if not text:
        return {}
    text = text.strip()
    # Strip common code fences.
    if text.startswith("```"):
        text = text.strip("`")
        # remove possible language tag
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Cannot parse JSON from LLM output: {text!r}")
    return json.loads(text[start:end + 1])
