"""轻量意图分类模块 (阶段1 预分类)。

基于关键词+句式规则实现粗粒度意图路由，不依赖 LLM 推理，
降低 Agent 推理负担，保证系统稳定性和响应时间。

意图类型 → 执行路径:
- GREETING          → 直接回复模板 (不调用 LLM/Tool)
- COMMAND           → 直接回复确认
- FACTUAL           → 优先 RAG (retrieve_vehicle_knowledge)
- COMPLEX_EXPLANATION → 混合 (RAG + generate_with_context)
- OPEN_GENERATION   → 直接 LLM (generate_answer)
- UNKNOWN           → 全量工具 (让 Agent 自己决定)
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class IntentType(Enum):
    GREETING = auto()
    COMMAND = auto()
    FACTUAL = auto()
    COMPLEX_EXPLANATION = auto()
    OPEN_GENERATION = auto()
    UNKNOWN = auto()


# 每个意图对应的推荐工具集
INTENT_TOOL_MAP: Dict[IntentType, List[str]] = {
    IntentType.GREETING: [],                        # 直接回复
    IntentType.COMMAND: [],                          # 直接回复
    IntentType.FACTUAL: ["retrieve_vehicle_knowledge", "generate_with_context"],
    IntentType.COMPLEX_EXPLANATION: ["retrieve_vehicle_knowledge", "generate_with_context"],
    IntentType.OPEN_GENERATION: ["generate_answer"],
    IntentType.UNKNOWN: ["retrieve_vehicle_knowledge", "generate_answer", "generate_with_context"],
}


# 每个意图的默认回复模板
INTENT_FALLBACK_REPLIES: Dict[IntentType, str] = {
    IntentType.GREETING: "你好！有什么关于车辆使用的问题可以帮你解答吗？",
    IntentType.COMMAND: "好的，已收到你的指令。",
    IntentType.UNKNOWN: "我没太理解你的意思，可以换个方式再说一遍吗？",
}


@dataclass
class IntentResult:
    """意图分类结果。"""
    intent: IntentType
    recommended_tools: List[str]   # 推荐的 Tool 名称列表
    fallback_reply: Optional[str]  # 快速回复文本 (GREETING/COMMAND 直接使用)
    confidence: float = 1.0        # 简单规则分类置信度固定为 1.0


class IntentClassifier:
    """轻量级意图分类器：关键词 + 句式规则。

    两层匹配策略:
    1. 精确关键词匹配 (问候/指令)
    2. 疑问句式匹配 (事实型/复杂解释型)
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        self.greeting_patterns: List[str] = cfg.get(
            "greeting_patterns",
            ["你好", "嗨", "hello", "hi", "早上好", "晚上好", "再见", "拜拜", "谢谢"]
        )
        self.command_patterns: List[str] = cfg.get(
            "command_patterns",
            ["打开", "关闭", "启动", "停止", "设置", "调节", "播放"]
        )
        self.factual_patterns: List[str] = cfg.get(
            "factual_patterns",
            ["怎么", "如何", "什么是", "是什么", "多少", "多久", "哪里", "哪个",
             "怎样", "怎么样", "什么", "有没有", "能不能", "可以"]
        )
        self.complex_patterns: List[str] = cfg.get(
            "complex_patterns",
            ["为什么", "原因", "原理", "解释", "区别", "比较", "详细", "分析"]
        )

    def classify(self, text: str) -> IntentResult:
        """分类用户输入文本。"""
        text_stripped = text.strip()
        if not text_stripped:
            return IntentResult(
                intent=IntentType.UNKNOWN,
                recommended_tools=INTENT_TOOL_MAP[IntentType.UNKNOWN],
                fallback_reply=None,
            )

        # 1. 问候检测 (精确匹配，最高优先级)
        for pat in self.greeting_patterns:
            if pat in text_stripped:
                return IntentResult(
                    intent=IntentType.GREETING,
                    recommended_tools=[],
                    fallback_reply=INTENT_FALLBACK_REPLIES[IntentType.GREETING],
                )

        # 2. 指令检测
        for pat in self.command_patterns:
            if pat in text_stripped:
                return IntentResult(
                    intent=IntentType.COMMAND,
                    recommended_tools=[],
                    fallback_reply=INTENT_FALLBACK_REPLIES[IntentType.COMMAND],
                )

        # 3. 复杂解释型 (先检测，因"为什么"等优先级高于事实型)
        for pat in self.complex_patterns:
            if pat in text_stripped:
                return IntentResult(
                    intent=IntentType.COMPLEX_EXPLANATION,
                    recommended_tools=INTENT_TOOL_MAP[IntentType.COMPLEX_EXPLANATION],
                    fallback_reply=None,
                )

        # 4. 事实型 (疑问句)
        for pat in self.factual_patterns:
            if pat in text_stripped:
                return IntentResult(
                    intent=IntentType.FACTUAL,
                    recommended_tools=INTENT_TOOL_MAP[IntentType.FACTUAL],
                    fallback_reply=None,
                )

        # 5. 开放生成型 (无关键词匹配但有实质内容)
        if len(text_stripped) > 2:
            return IntentResult(
                intent=IntentType.OPEN_GENERATION,
                recommended_tools=INTENT_TOOL_MAP[IntentType.OPEN_GENERATION],
                fallback_reply=None,
            )

        # 6. 兜底
        return IntentResult(
            intent=IntentType.UNKNOWN,
            recommended_tools=INTENT_TOOL_MAP[IntentType.UNKNOWN],
            fallback_reply=INTENT_FALLBACK_REPLIES[IntentType.UNKNOWN],
        )
