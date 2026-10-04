"""TurnTakingSession（一对一严格轮流 baseline）与机制开关的单元测试。

不依赖网络：agent 用 stub，settings 用临时目录下的极简 yaml。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from libs.chat.agent import (
    ChatAgent,
    DEFAULT_GENERATION_MODE,
    build_system_prompt,
    normalize_generation_mode,
)
from libs.chat.memory import Memory
from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.chat.turn_session import TurnTakingSession
from libs.core.config import load_settings


REPO_ROOT = ROOT


def _write_config(tmp: Path, prompt_components_dir: str) -> Path:
    cfg = {
        "paths": {
            "data_dir": str(tmp / "data"),
            "personas_dir": str(tmp / "personas"),
            "scenarios_dir": str(tmp / "scenarios"),
            "outputs_dir": str(tmp / "outputs"),
            "results_dir": str(tmp / "results"),
            "skills_dir": str(tmp / "skills"),
            "prompt_components_dir": prompt_components_dir,
        },
        "session": {
            "max_turns": 100,
            "min_send_interval_ms": 3000,
        },
    }
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


class StubAgent:
    """最小 agent：按脚本逐条返回内容，记录被调用次数。"""

    def __init__(self, persona_id: str, replies: list) -> None:
        self.persona = SimpleNamespace(persona_id=persona_id, nickname=persona_id)
        self.replies = list(replies)
        self.calls = 0
        self.trace_dir = None
        self.model = "stub-model"
        self.extra_instructions = ""

    def reply(self, history, *, current_virtual_time="", prev_draft=None):
        self.calls += 1
        item = self.replies.pop(0) if self.replies else "嗯嗯"
        if isinstance(item, str):
            item = {"content": item}
        return dict(item)


def _scenario(first_speaker: str) -> Scenario:
    return Scenario(
        scenario_id="A0001_B0001_C0001",
        raw={
            "name": "测试场景",
            "start_time": "2026-06-03 18:30",
            "first_speaker": first_speaker,
        },
    )


class TurnTakingSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="turn-taking-test-"))
        load_settings(_write_config(self.tmp, "src/chat/prompts/chatbot"))
        self.live = self.tmp / "live" / "chat.json"

    def _session(self, agent_a, agent_b, scenario, **kw):
        return TurnTakingSession(
            agent_a=agent_a,
            agent_b=agent_b,
            scenario=scenario,
            max_turns=kw.pop("max_turns", 8),
            opener=kw.pop("opener", None),
            on_turn=None,
            live_path=self.live,
            raw_trace_path=self.tmp / "traces",
            session_tag="test",
            enable_memory=False,
            resume_dialogue=kw.pop("resume_dialogue", None),
        )

    def test_strict_alternation_and_call_count(self) -> None:
        a = StubAgent("PA", ["a1", "a2", "a3", "a4"])
        b = StubAgent("PB", ["b1", "b2", "b3", "b4"])
        result = self._session(a, b, _scenario("PA"), max_turns=8).run()

        roles = [t["role"] for t in result.dialogue]
        self.assertEqual(roles, ["PA", "PB"] * 4)
        # 严格轮流：每次发出的消息恰好对应一次 LLM 调用，无并行预决策。
        self.assertEqual(a.calls, 4)
        self.assertEqual(b.calls, 4)
        self.assertEqual(result.meta["turns"], 8)
        self.assertEqual(result.meta["termination_reason"], "max_turns")
        self.assertEqual(result.meta["generation_mode"], "turn_taking")
        # 虚拟时间单调递增且满足最小发送间隔。
        times = [t["t_send_ms"] for t in result.dialogue]
        self.assertEqual(times, sorted(times))
        self.assertTrue(all(y - x >= 3000 for x, y in zip(times, times[1:])))

    def test_meta_keeps_parent_schema(self) -> None:
        a = StubAgent("PA", ["a1"])
        b = StubAgent("PB", ["b1"])
        result = self._session(a, b, _scenario("PA"), max_turns=2).run()
        for key in (
            "persona1_id", "persona2_id", "scenario_id", "model", "session_tag",
            "turns", "ended_naturally", "termination_reason", "fail_count",
            "llm_failures", "interrupts", "resumed_from_turns",
            "virtual_duration_ms", "virtual_start_time", "raw_trace_dir",
        ):
            self.assertIn(key, result.meta)
        self.assertEqual(result.meta["persona1_id"], "PA")

    def test_first_speaker_from_scenario(self) -> None:
        a = StubAgent("PA", ["a1"])
        b = StubAgent("PB", ["b1"])
        result = self._session(a, b, _scenario("PB"), max_turns=2).run()
        self.assertEqual([t["role"] for t in result.dialogue], ["PB", "PA"])

    def test_opener_wrapped_then_partner_replies(self) -> None:
        a = StubAgent("PA", ["a1"])
        b = StubAgent("PB", ["b1"])
        result = self._session(a, b, _scenario("PA"), max_turns=2,
                               opener="在吗").run()
        self.assertEqual(result.dialogue[0]["role"], "PA")
        self.assertEqual(result.dialogue[0]["content"], "在吗")
        self.assertEqual(result.dialogue[1]["role"], "PB")
        # opener 不调 LLM；PA 只在轮到自己时被调用一次。
        self.assertEqual(a.calls, 0)
        self.assertEqual(b.calls, 1)

    def test_resume_picks_other_side(self) -> None:
        a = StubAgent("PA", ["a1"])
        b = StubAgent("PB", ["b1"])
        resumed = [{
            "role": "PA", "content": "之前的话",
            "response": [{"content": "之前的话", "timestamp": "x"}],
            "t_send_ms": 3000,
        }]
        result = self._session(a, b, _scenario("PA"), max_turns=2,
                               resume_dialogue=resumed).run()
        self.assertEqual(result.meta["resumed_from_turns"], 1)
        self.assertEqual(result.dialogue[1]["role"], "PB")

    def test_empty_content_is_llm_failure(self) -> None:
        a = StubAgent("PA", ["a1", ""])
        b = StubAgent("PB", ["b1"])
        result = self._session(a, b, _scenario("PA"), max_turns=8).run()
        self.assertEqual(result.meta["termination_reason"], "llm_failure")
        self.assertEqual(result.meta["llm_failures"], ["PA"])
        self.assertEqual([t["role"] for t in result.dialogue], ["PA", "PB"])

    def test_repeat_drop_reasks_same_speaker(self) -> None:
        # PA 第二轮先复读自己上一条（被防御丢弃），重问后给出新内容。
        a = StubAgent("PA", ["a1", "a1", "a2"])
        b = StubAgent("PB", ["b1", "b2"])
        result = self._session(a, b, _scenario("PA"), max_turns=4).run()
        contents = [(t["role"], t["content"]) for t in result.dialogue]
        self.assertEqual(
            contents, [("PA", "a1"), ("PB", "b1"), ("PA", "a2"), ("PB", "b2")])
        self.assertEqual(a.calls, 3)  # 1 次 + 复读丢弃后重问 1 次 + 正常 1 次
        self.assertEqual(result.meta["termination_reason"], "max_turns")

    def test_repeat_drop_exhaustion_terminates(self) -> None:
        a = StubAgent("PA", ["a1"] + ["a1"] * 10)  # 永远复读
        b = StubAgent("PB", ["b1"])
        result = self._session(a, b, _scenario("PA"), max_turns=8).run()
        self.assertEqual([t["role"] for t in result.dialogue], ["PA", "PB"])
        self.assertTrue(result.meta["ended_naturally"])
        self.assertEqual(result.meta["termination_reason"], "natural")
        # 1 次正常尝试 + _MAX_REASK 次重问。
        self.assertEqual(a.calls, 1 + TurnTakingSession._MAX_REASK)


class GenerationModeSwitchTest(unittest.TestCase):
    def test_normalize_aliases(self) -> None:
        self.assertEqual(normalize_generation_mode("turn_taking"), "turn_taking")
        self.assertEqual(normalize_generation_mode("one_on_one"), "turn_taking")
        self.assertEqual(normalize_generation_mode("ALTERNATING"), "turn_taking")
        self.assertEqual(normalize_generation_mode("draft_scheduling"),
                         "draft_scheduling")
        self.assertEqual(normalize_generation_mode("default"), "draft_scheduling")
        self.assertEqual(normalize_generation_mode(""), DEFAULT_GENERATION_MODE)

    def _agent(self, mode: str) -> ChatAgent:
        return ChatAgent(
            persona=Persona("PA", "PA", {"persona_id": "PA", "nickname": "PA"}),
            scenario=_scenario("PA"),
            partner_id="PB",
            memory=Memory("PA", "PA", {}),
            llm=None,
            generation_mode=mode,
        )

    def test_default_contract_line_unchanged(self) -> None:
        agent = self._agent("draft_scheduling")
        block = agent._build_user_block(
            history=[], current_virtual_time="2026-06-03 18:30:00",
            prev_draft=None)
        expected = ('你是 PA。请撰写你打算发送的下一条消息，可自由开展话题'
                    '回顾上方你已发送的历史，禁止输出与其中任何一句语义重复的内容'
                    '输出格式：{"content": "...", "send_timestamp": "..."}')
        self.assertIn(expected, block)

    def test_turn_taking_contract_line_is_content_only(self) -> None:
        agent = self._agent("turn_taking")
        block = agent._build_user_block(
            history=[], current_virtual_time="2026-06-03 18:30:00",
            prev_draft=None)
        self.assertIn('输出格式：{"content": "..."}', block)
        self.assertNotIn("send_timestamp", block)


class BaselinePromptTest(unittest.TestCase):
    def _render(self, components_dir: str) -> str:
        tmp = Path(tempfile.mkdtemp(prefix="prompt-test-"))
        load_settings(_write_config(tmp, components_dir))
        persona = Persona("PA", "小明", {"persona_id": "PA", "nickname": "小明"})
        return build_system_prompt(
            persona=persona,
            scenario=_scenario("PA"),
            partner_id="PB",
            memory=Memory("PA", "小明", {}),
        )

    def test_baseline_prompt_is_minimal(self) -> None:
        prompt = self._render("src/chat/prompts/chatbot_baseline")
        self.assertIn("你一句，我一句", prompt)
        # 动态块注入在 02_mechanism 之后。
        self.assertGreater(prompt.index("当前生效的拟人写作技能"),
                           prompt.index("# 系统机制"))
        for word in ("send_timestamp", "草稿", "连发", "抢发"):
            self.assertNotIn(word, prompt, f"baseline prompt 不应包含 {word}")
        self.assertNotIn("{consecutive_delay_table}", prompt)
        self.assertNotIn("{recent_turns_in_user_block}", prompt)

    def test_default_prompt_dir_still_renders(self) -> None:
        prompt = self._render("src/chat/prompts/chatbot")
        self.assertIn("send_timestamp", prompt)
        # 占位符已被替换为实际轮数。
        self.assertIn("最近 10 轮对话", prompt)


class CachedDialogueModeFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cache-test-"))
        load_settings(_write_config(self.tmp, "src/chat/prompts/chatbot"))
        from apps.benchmark.src.runner import _find_cached_dialogue
        self.find = _find_cached_dialogue
        self.dlg_dir = (self.tmp / "outputs" / "dialogues" / "TM"
                        / "A0001_B0001_C0001")
        self.dlg_dir.mkdir(parents=True)

    def _write(self, name: str, mode=None) -> Path:
        meta: dict = {"session_tag": name}
        if mode is not None:
            meta["generation_mode"] = mode
        payload = {
            "persona1_id": "P1", "persona2_id": "P2",
            "scenario_id": "A0001_B0001_C0001",
            "dialogue": [{"role": "P1", "response": [{"content": "hi"}]}],
            "meta": meta,
        }
        path = self.dlg_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_mode_filter_both_ways(self) -> None:
        draft = self._write("chat_P1_P2_as_a_1.json")            # 无 mode 字段 = 默认
        turn = self._write("chat_P1_P2_as_a_2.json", "turn_taking")

        case = "P1_P2_A0001_B0001_C0001_as_a"
        self.assertEqual(
            self.find("TM", "ref", case, "A0001_B0001_C0001", True), draft)
        self.assertEqual(
            self.find("TM", "ref", case, "A0001_B0001_C0001", True,
                      generation_mode="turn_taking"), turn)

    def test_mode_mismatch_returns_none(self) -> None:
        turn = self._write("chat_P1_P2_as_a_1.json", "turn_taking")
        case = "P1_P2_A0001_B0001_C0001_as_a"
        self.assertIsNone(
            self.find("TM", "ref", case, "A0001_B0001_C0001", True))
        turn.unlink()
        self._write("chat_P1_P2_as_a_2.json")  # 无 mode 字段 = 默认机制
        self.assertIsNone(
            self.find("TM", "ref", case, "A0001_B0001_C0001", True,
                      generation_mode="turn_taking"))


if __name__ == "__main__":
    unittest.main()
