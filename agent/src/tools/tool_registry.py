"""Tool 注册中心。

基于当前阶段1意图分类结果，构建约束后的 LangChain Tool 列表。

LangChain 负责:
1. 统一消息组织 - system/user/history/tool observation 用规范 message 格式
2. 封装 Tool - RAG/LLM 能力以标准 Tool 接口暴露
3. 短期记忆和执行链 - 多轮问答不退化
"""

import logging
from typing import Dict, List

from langchain_core.tools import tool

from intent.intent_classifier import IntentType, IntentResult
from tools.rag_search import RAGSearchTool
from tools.llm_generate import LLMGenerateTool
from tools.llm_rag_generate import LLMRAGGenerateTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """管理所有可用 Tool 并根据意图上下文提供约束后的 Tool 列表。"""

    def __init__(
        self,
        rag_tool: RAGSearchTool,
        llm_gen_tool: LLMGenerateTool,
        llm_rag_gen_tool: LLMRAGGenerateTool,
    ):
        self.rag_tool = rag_tool
        self.llm_gen_tool = llm_gen_tool
        self.llm_rag_gen_tool = llm_rag_gen_tool

        self._all_tools: Dict[str, callable] = {}
        self._build_all_tools()

    def _build_all_tools(self):
        """构建所有可用 Tool 的 LangChain 封装。"""

        rag = self.rag_tool
        llm_gen = self.llm_gen_tool
        llm_rag_gen = self.llm_rag_gen_tool

        @tool
        def retrieve_vehicle_knowledge(
            query: str,
            top_k: int = 5,
            threshold: float = 0.5,
        ) -> str:
            """检索车辆使用手册获取相关知识。当用户询问车辆功能操作、故障处理、保养指南等需要查阅手册的问题时使用此工具。

            Args:
                query: 要检索的问题或关键词
                top_k: 返回的文档数量，默认5
                threshold: 相似度阈值(0-1)，低于此值的文档会被过滤，默认0.5
            """
            results = rag.search(query, top_k=top_k, threshold=threshold)
            if not results:
                return "（未找到相关文档）"
            return rag.format_for_prompt(results)

        @tool
        def generate_answer(query: str) -> str:
            """直接生成回答，适用于不需要查阅车辆手册的泛化聊天或常识问题。

            Args:
                query: 用户的问题
            """
            return llm_gen.generate(query)

        @tool
        def generate_with_context(
            query: str,
            retrieved_docs_json: str,
        ) -> str:
            """基于检索到的车辆手册知识生成准确的回答。先用retrieve_vehicle_knowledge检索，再将结果作为retrieved_docs_json传入。

            Args:
                query: 用户的问题
                retrieved_docs_json: 从retrieve_vehicle_knowledge获取的JSON格式检索结果
            """
            import json
            docs = json.loads(retrieved_docs_json)
            return llm_rag_gen.generate_with_context(query, docs)

        self._all_tools = {
            "retrieve_vehicle_knowledge": retrieve_vehicle_knowledge,
            "generate_answer": generate_answer,
            "generate_with_context": generate_with_context,
        }

    def get_all_tools(self) -> List:
        """获取所有可用 Tool。"""
        return list(self._all_tools.values())

    def get_tools_for_intent(self, intent_result: IntentResult) -> List:
        """根据阶段1意图分类结果获取约束后的 Tool 列表。

        核心设计: 阶段1预分类剪枝，减少 Agent 的推理负担。
        """
        tool_names = intent_result.recommended_tools
        if not tool_names:
            return self.get_all_tools()

        tools = []
        for name in tool_names:
            if name in self._all_tools:
                tools.append(self._all_tools[name])

        logger.debug(
            "Tools for intent %s: %s",
            intent_result.intent.name,
            [t.name for t in tools],
        )
        return tools
