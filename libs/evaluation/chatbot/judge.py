"""Chatbot-domain benchmark judge.

Tailored for general Chinese IM-style conversation (e.g. WeChat).  The prompts
stay neutral and avoid game-specific punctuation/fragmentation examples.
"""
from __future__ import annotations

from libs.chat.persona import Persona
from libs.chat.scenario import Scenario
from libs.evaluation.core.judge import BaseBenchmarkJudge


class ChatbotBenchmarkJudge(BaseBenchmarkJudge):
    """Judge for ``data/chatbot_benchmark/rubric.yaml``."""

    def _per_turn_system_prompt(self, persona: Persona, scenario: Scenario) -> str:
        dims_block = self._dims_block(self.pt_dims)
        return (
            "你是一名专业的中文对话拟人度评测员，正在评估某个角色在中文私聊（如微信）中的**单轮回复**质量。\n\n"
            "## 被评角色信息\n"
            f"{persona.to_system_prompt()}\n\n"
            f"{scenario.to_system_prompt(self_id=persona.persona_id, partner_id='对方')}\n\n"
            "## 评测维度与 Checkbox\n"
            "对每个维度，逐条判断其下的 checkbox 是否满足（true/false），并给出一句简短理由。\n\n"
            "**判分逻辑**：\n"
            "每个 checkbox 描述的是一种真人中文私聊中的优秀特征。判分规则：\n"
            "1. 该特征在当前轮次中**存在** → **true**\n"
            "2. 该特征在当前轮次中**不存在，且场景不适用**（checkbox 描述中标注的「不适用」条件满足时）→ **true**\n"
            "3. 该特征在当前轮次中**不存在，且场景适用** → **false**\n\n"
            "**关键**：严格判断每个 checkbox 描述的具体特征是否存在。不要因为回复整体看起来自然就将所有 checkbox 判 true。\n"
            "注意：包含 / 分隔的多条消息表示真人分条发送，每条只说一件事。\n\n"
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
            "你是一名专业的中文对话拟人度评测员，正在评估某个角色在中文私聊（如微信）**整段对话**中的整体表现。\n\n"
            "## 被评角色信息\n"
            f"{persona.to_system_prompt()}\n\n"
            f"{scenario.to_system_prompt(self_id=persona.persona_id, partner_id='对方')}\n\n"
            "## 评测维度与 Checkbox\n"
            "对每个维度，逐条判断其下的 checkbox 是否满足（true/false），并给出一句简短理由。\n\n"
            "**判分逻辑**：\n"
            "每个 checkbox 描述的是一种真人中文私聊中的优秀特征。判分规则：\n"
            "1. 该特征在整段对话中**存在** → **true**\n"
            "2. 该特征在整段对话中**不存在，且场景不适用**（checkbox 描述中标注的「不适用」条件满足时）→ **true**\n"
            "3. 该特征在整段对话中**不存在，且场景适用** → **false**\n\n"
            "**关键**：严格判断每个 checkbox 描述的具体特征是否存在。不要因为对话整体看起来自然就将所有 checkbox 判 true。\n"
            "注意：包含 / 分隔的多条消息表示真人分条发送，每条只说一件事。\n\n"
            f"{dims_block}\n\n"
            "## E1 上下文一致性——强制判定流程（必须遵守）\n"
            "评估 E1_context_consistency 时，对 5 个 checkbox 采用'零容忍'原则：只要被评角色在整段对话中出现过一次对应问题，该 checkbox 就必须判 false；reason 必须引用具体 Turn 编号。\n\n"
            "判定流程（按顺序执行）：\n"
            "1. 先通读被评角色的所有发言，针对每个 checkbox 列出所有候选违规 Turn。\n"
            "2. 只要找到任一候选违规，该 checkbox 直接判 false；不得以'整体自然'、'情绪表达'、'强调'、'补充说明'等理由将其解释为 true。\n"
            "3. 只有完全找不到候选违规时，才判 true。\n\n"
            "具体 hard rules（满足任一即判 false）：\n"
            "- no_repeat_ask：\n"
            "  - 同一 Turn 内发送两条内容基本相同的消息；\n"
            "  - 向伴侣提出一个已经由伴侣回答过的问题；\n"
            "  - 在伴侣未回答或仅部分回答时，连续多轮（间隔不超过两轮）重复提出同一核心问题；\n"
            "  - 对已确认过的事实再次以问句形式确认或追问。\n"
            "- facts_coherent：\n"
            "  - 被评角色自己前后陈述的身份、职业、状态、时间线等事实矛盾；\n"
            "  - 伴侣错误陈述被评角色的身份/经历/状态，被评角色未在同一或下一轮及时纠正，反而顺着错误语境回应。\n"
            "- speaker_reference_correct：\n"
            "  - 把被评角色自己说过的话/经历说成是伴侣的；\n"
            "  - 把伴侣说过的话/经历说成是被评角色自己的。\n"
            "- state_tracking：\n"
            "  - 伴侣已明确的状态变化，被评角色在后续仍按旧状态提问或回应。\n"
            "- no_memory_confusion：\n"
            "  - 把本对话中较早发生过的确认信息当作新信息再次质疑或追问；\n"
            "  - 接受伴侣错误声称被评角色曾说过/做过某事（实际没有），并顺着该错误信息回应。\n\n"
            "判 false 示例（来自真实 bad case）：\n"
            "- no_repeat_ask：Turn 20 刚描述完'就是那种慢悠悠的...'，Turn 23 几乎原样重复同样内容；或同一 Turn 内连续发送两条'是不是已经成功把小神兽“放倒”啦～'。\n"
            "- facts_coherent：被评角色之前说自己是学生，伴侣错误称其'上班/做销售'，被评角色未立即纠正，反而顺着该错误语境回应。\n"
            "- speaker_reference_correct：Turn 12 被评角色自己提出'钛铝复合工艺'，Turn 14 又说'你说到钛铝复合工艺'，把己方观点归给对方。\n"
            "- state_tracking：Turn 13 对方已说加班到八九点，Turn 20 又问'加班到几点'。\n"
            "- no_memory_confusion：Turn 6 已确认每月投 1000，伴侣在 Turn 15 又问'打算每个月投多少'，被评角色未指出已确认；或伴侣虚构'你之前说想换组'，被评角色未否认并顺着回应。\n\n"
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
