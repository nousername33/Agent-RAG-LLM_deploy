"""请求编排层。

整合四大模块:
1. 输入归一化 - 整理 ASR 文本、对话历史、运行态信息
2. 阶段1 轻量意图分类 - 粗粒度路由，决定工具约束
3. 阶段2 手动 ReAct Agent 循环 - 智能工具调用决策
4. Memory & State 管理 - 短期上下文 + 结构化对话状态

数据流:
  ASR text → InputNormalizer → Stage1 IntentClassifier
    → [GREETING/COMMAND: 直接回复]
    → [其他: Stage2 ReAct Loop → Tool calls → 最终回复]
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from communication.llm_client import LLMClient
from intent.intent_classifier import (
    IntentClassifier,
    IntentResult,
    IntentType,
)
from intent.query_rewriter import QueryRewriter
from memory.conversation_memory import ConversationMemory
from tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# ReAct Prompt 模板 (中文)
REACT_SYSTEM_PROMPT = """你是一个车载语音助手，帮助用户解答车辆使用相关的问题。

你可以使用以下工具来完成任务:

{tools}

严格遵循以下格式进行推理:

Question: 用户当前的问题是什么
Thought: 我需要分析用户的问题，决定是否需要使用工具
Action: 要使用的工具名称，必须是可用工具列表中的一个
Action Input: 工具的输入参数，使用 JSON 格式
Observation: 工具返回的结果
... (上述 Thought/Action/Action Input/Observation 可以重复)
Thought: 我现在已经获得了足够的信息
Final Answer: 对用户问题的最终回答，简洁自然，适合语音播报

重要规则:
- 车辆操作、故障处理、保养等问题，先检索再回答
- 普通聊天、问候不需要检索，直接回答
- Action Input 必须是合法 JSON
- 不确定时说明"根据现有资料暂时无法回答"，不要编造
"""


class InputNormalizer:
    """输入归一化模块。

    将 ASR 文本、最近对话历史、当前对话主题整理为统一输入。
    """

    def __init__(self, query_rewriter: QueryRewriter):
        self._rewriter = query_rewriter

    def normalize(
        self,
        asr_text: str,
        memory: ConversationMemory,
    ) -> Tuple[str, List[str], Optional[str]]:
        """归一化输入。

        Returns:
            (normalized_query, history_strs, last_entity)
        """
        last_entity = memory.get_last_entity()
        rewritten = self._rewriter.rewrite(asr_text, last_entity)
        history = memory.get_recent_history()
        history_strs = [
            f"用户: {h['user']}\n助手: {h['agent']}" for h in history
        ]
        return rewritten, history_strs, last_entity


class RequestOrchestrator:
    """请求编排器: 两阶段决策 + 手动 ReAct Agent 调度。

    不使用 langchain.agents (避免 Python 3.13 兼容性问题)，
    而是手写轻量 ReAct 循环，更适合端侧场景。
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_client: LLMClient,
        memory: ConversationMemory,
        intent_classifier: IntentClassifier,
        query_rewriter: QueryRewriter,
        config: Optional[Dict] = None,
    ):
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.memory = memory
        self.intent_classifier = intent_classifier
        self.normalizer = InputNormalizer(query_rewriter)
        self.max_iterations = (config or {}).get("max_agent_iterations", 5)

    def process(self, asr_text: str) -> str:
        """处理单轮请求的完整流程。

        Args:
            asr_text: ASR 识别文本

        Returns:
            Agent 生成的回复文本
        """
        # 1. 输入归一化 (含 Query 重写)
        normalized_query, history_strs, _ = self.normalizer.normalize(
            asr_text, self.memory
        )

        # 2. 阶段1: 轻量意图预分类
        intent_result = self.intent_classifier.classify(normalized_query)
        logger.info(
            "Stage1 intent: %s, tools: %s",
            intent_result.intent.name,
            intent_result.recommended_tools,
        )

        # 3. 快速路径: 问候/指令直接回复
        if intent_result.fallback_reply is not None:
            self.memory.add_turn(
                asr_text, intent_result.fallback_reply,
                intent_result.intent.name,
            )
            return intent_result.fallback_reply

        # 4. 阶段2: 手动 ReAct Agent 循环
        response = self._run_react_loop(normalized_query, history_strs, intent_result)

        if not response:
            response = "抱歉，我暂时无法处理这个问题，请换个方式试试。"

        # 5. 保存对话记忆
        self.memory.add_turn(asr_text, response, intent_result.intent.name)
        return response

    def _run_react_loop(
        self,
        query: str,
        history: List[str],
        intent_result: IntentResult,
    ) -> Optional[str]:
        """阶段2: 手动 ReAct 循环。

        手写轻量 ReAct 推理循环，不依赖 langchain.agents 包。
        """
        tools = self.tool_registry.get_tools_for_intent(intent_result)

        # 构建工具描述
        tool_descriptions = []
        tool_map: Dict[str, Any] = {}
        for t in tools:
            desc = f"- {t.name}: {t.description}"
            tool_descriptions.append(desc)
            tool_map[t.name] = t

        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools="\n".join(tool_descriptions)
        )

        # 对话历史
        history_text = "\n".join(history) if history else "（无历史）"

        full_prompt = f"""{system_prompt}

=== 对话历史 ===
{history_text}

Question: {query}
"""
        # ReAct 循环
        scratchpad = ""
        for iteration in range(self.max_iterations):
            prompt = full_prompt + scratchpad + "\n"
            response = self.llm_client.generate(prompt)

            if response is None:
                logger.warning("LLM returned None at iteration %d", iteration)
                break

            # 提取 Action 和 Final Answer
            action_match = re.search(
                r"Action:\s*(\S+)", response, re.IGNORECASE
            )
            action_input_match = re.search(
                r"Action Input:\s*(\{.*?\})", response, re.DOTALL
            )
            final_answer_match = re.search(
                r"Final Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE
            )

            # 如果有 Final Answer，结束循环
            if final_answer_match:
                return final_answer_match.group(1).strip()

            # 如果有 Action，执行工具调用
            if action_match and action_input_match:
                tool_name = action_match.group(1).strip()
                try:
                    tool_input = json.loads(action_input_match.group(1))
                except json.JSONDecodeError:
                    tool_input = {"query": action_input_match.group(1)}

                logger.info("ReAct iteration %d: calling tool '%s'",
                           iteration + 1, tool_name)

                observation = self._execute_tool(tool_map, tool_name, tool_input)
                scratchpad += (
                    f"{response}\nObservation: {observation}\n"
                )
                continue

            # 没匹配到 Action，把整个响应当 Final Answer
            return self._clean_final_answer(response)

        return None

    def _execute_tool(
        self,
        tool_map: Dict[str, Any],
        tool_name: str,
        tool_input: dict,
    ) -> str:
        """执行工具调用并返回观察结果。"""
        tool = tool_map.get(tool_name)
        if tool is None:
            return f"错误: 未知工具 '{tool_name}'"

        try:
            # LangChain BaseTool.invoke() 接受 dict 或 str
            result = tool.invoke(tool_input)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", tool_name, e)
            return f"工具调用失败: {e}"

    @staticmethod
    def _clean_final_answer(response: str) -> str:
        """清理响应中的标记。"""
        # 尝试提取 Final Answer
        m = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # 移除非内容行
        lines = response.strip().split("\n")
        cleaned = []
        for line in lines:
            if line.startswith(("Thought:", "Action:", "Action Input:", "Observation:")):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()
