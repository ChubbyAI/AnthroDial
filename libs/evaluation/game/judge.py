"""Game-domain benchmark judge.

Preserves the original game-chat prompt framing and punctuation/fragmentation
examples used for ``data/game_benchmark/rubric.yaml``.
"""
from __future__ import annotations

from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.evaluation.core.judge import BaseBenchmarkJudge


class GameBenchmarkJudge(BaseBenchmarkJudge):
    """Judge for ``data/game_benchmark/rubric.yaml``."""

    def _per_turn_system_prompt(self, persona: Persona, scenario: Scenario) -> str:
        dims_block = self._dims_block(self.pt_dims)
        return (
            "你是一名专业的中文对话拟人度评测员，正在评估某个角色的**单轮回复**质量。\n\n"
            "## 被评角色信息\n"
            f"{persona.to_system_prompt()}\n\n"
            f"{scenario.to_system_prompt(self_id=persona.persona_id, partner_id='对方')}\n\n"
            "## 评测维度与 Checkbox\n"
            "对每个维度，逐条判断其下的 checkbox 是否满足（true/false），并给出一句简短理由。\n\n"
            "**判分逻辑**：\n"
            "每个 checkbox 描述的是一种真人游戏私聊中的优秀特征。判分规则：\n"
            "1. 该特征在当前轮次中**存在** → **true**\n"
            "2. 该特征在当前轮次中**不存在，且场景不适用**（checkbox 描述中标注的「不适用」条件满足时）→ **true**\n"
            "3. 该特征在当前轮次中**不存在，且场景适用** → **false**\n\n"
            "**关键**：严格判断每个 checkbox 描述的具体特征是否存在。不要因为回复整体看起来自然就将所有 checkbox 判 true。\n"
            "注意：包含 / 分隔的多条消息表示真人分条发送，每条只说一件事。\n"
            "如果单条消息内部用逗号、顿号、空格等连接多个独立短语或分句（例如'排位吧，手感热，想冲分'、'在么 上号'、'来了，快点'），"
            "属于把多条消息压缩成一条的 AI 式打包，对应 checkbox 应判为 false。\n"
            "真实玩家快速打字时极少在单条消息内使用逗号、顿号、分号、冒号、破折号、括号等句中连接/分隔标点；"
            "只要单条消息内部出现这些标点，即视为 AI 式书面化表达，应判为 false。"
            "句末的句号、问号、感叹号、省略号用于表达语气，可判为 true。\n\n"
            f"{dims_block}\n\n"
            "## 输出格式（只输出 JSON，不要任何其他文字）\n"
            "```json\n"
            '{\n'
            '  "dimensions": {\n'
            '    "<dim_id>": {\n'
            '      "checks": {"<checkbox_id>": true/false, ...},\n'
            '      "reason": "一句话理由"\n'
            '    },\n'
            '    ...\n'
            '  }\n'
            "}\n"
            "```\n"
        )

    def _holistic_system_prompt(self, persona: Persona, scenario: Scenario) -> str:
        dims_block = self._dims_block(self.ho_dims)
        return (
            "你是一名专业的中文对话拟人度评测员，正在评估某个角色在**整段对话**中的整体表现。\n\n"
            "## 被评角色信息\n"
            f"{persona.to_system_prompt()}\n\n"
            f"{scenario.to_system_prompt(self_id=persona.persona_id, partner_id='对方')}\n\n"
            "## 评测维度与 Checkbox\n"
            "对每个维度，逐条判断其下的 checkbox 是否满足（true/false），并给出一句简短理由。\n\n"
            "**判分逻辑**：\n"
            "每个 checkbox 描述的是一种真人游戏私聊中的优秀特征。判分规则：\n"
            "1. 该特征在整段对话中**存在** → **true**\n"
            "2. 该特征在整段对话中**不存在，且场景不适用**（checkbox 描述中标注的「不适用」条件满足时）→ **true**\n"
            "3. 该特征在整段对话中**不存在，且场景适用** → **false**\n\n"
            "**关键**：严格判断每个 checkbox 描述的具体特征是否存在。不要因为对话整体看起来自然就将所有 checkbox 判 true。\n"
            "注意：包含 / 分隔的多条消息表示真人分条发送，每条只说一件事。\n"
            "如果单条消息内部用逗号、顿号、空格等连接多个独立短语或分句（例如'排位吧，手感热，想冲分'、'在么 上号'、'来了，快点'），"
            "属于把多条消息压缩成一条的 AI 式打包，对应 checkbox 应判为 false。\n"
            "真实玩家快速打字时极少在单条消息内使用逗号、顿号、分号、冒号、破折号、括号等句中连接/分隔标点；"
            "只要单条消息内部出现这些标点，即视为 AI 式书面化表达，应判为 false。"
            "句末的句号、问号、感叹号、省略号用于表达语气，可判为 true。\n\n"
            f"{dims_block}\n\n"
            "## 输出格式（只输出 JSON，不要任何其他文字）\n"
            "```json\n"
            '{\n'
            '  "dimensions": {\n'
            '    "<dim_id>": {\n'
            '      "checks": {"<checkbox_id>": true/false, ...},\n'
            '      "reason": "一句话理由"\n'
            '    },\n'
            '    ...\n'
            '  }\n'
            "}\n"
            "```\n"
        )
