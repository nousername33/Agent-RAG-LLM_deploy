"""Tool: generate_with_context

融合类 Tool - RAG 检索 + LLM 生成的融合链路。
输入 query + retrieved_docs + history，输出基于知识的回答。

将检索结果组织为 context prompt，要求模型优先依据提供的知识回答，
上下文不足时明确说明而非补充未经支持的信息。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMRAGGenerateTool:
    """封装 RAG+LLM 融合生成为 Agent 可调用工具。

    Agent 先通过 retrieve_vehicle_knowledge 拿到结构化知识片段，
    再调用本 Tool 将检索结果作为 context 交给 LLM 生成。
    """

    def __init__(self, llm_client: "LLMClient", default_max_tokens: int = 512):
        self._client = llm_client
        self.default_max_tokens = default_max_tokens

    def generate_with_context(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        history: Optional[List[str]] = None,
    ) -> str:
        """基于检索文档生成回答。

        Args:
            query: 当前用户问题
            retrieved_docs: RAG 检索返回的结构化文档列表
                [{id, content, score, section, subsection}, ...]
            history: 最近几轮对话上下文 (可选)

        Returns:
            基于文档知识的回答
        """
        context_text = self._format_retrieved_docs(retrieved_docs)
        prompt = self._build_rag_prompt(query, context_text, history)
        response = self._client.generate(prompt)
        return self._clean_response(response) if response else "抱歉，我暂时无法回答这个问题。"

    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        """将结构化检索结果格式化为 prompt 上下文。"""
        if not docs:
            return "（未检索到相关文档）"

        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[文档{i}]")
            if doc.get("section"):
                section_info = f"章节: {doc['section']}"
                if doc.get("subsection"):
                    section_info += f" > {doc['subsection']}"
                parts.append(section_info)
            parts.append(f"内容: {doc['content']}")
            parts.append("")

        return "\n".join(parts)

    def _build_rag_prompt(
        self,
        query: str,
        context: str,
        history: Optional[List[str]] = None,
    ) -> str:
        """构建 RAG+生成 融合 prompt。

        核心设计: 将检索结果作为可靠知识源，要求模型基于证据回答。
        """
        parts = []

        # 系统指令
        parts.append("你是一个车载语音助手，请根据以下车辆使用手册中的知识回答用户的问题。")
        parts.append("回答规则:")
        parts.append("1. 优先依据提供的文档内容回答")
        parts.append("2. 如果文档中有明确答案，直接引用并解释")
        parts.append("3. 如果文档信息不足以回答问题，明确说明'根据现有资料，这个问题暂时无法回答'")
        parts.append("4. 回答要简洁、准确，适合语音播报")
        parts.append("")

        # 检索上下文
        parts.append("=== 车辆使用手册相关知识 ===")
        parts.append(context)
        parts.append("=== 知识引用结束 ===")
        parts.append("")

        # 对话历史
        if history:
            parts.append("=== 对话历史 ===")
            for i, turn in enumerate(history, 1):
                parts.append(f"用户: {turn}")
            parts.append("")

        # 当前问题
        parts.append(f"用户问题: {query}")
        parts.append("请回答：")

        return "\n".join(parts)

    @staticmethod
    def _clean_response(response: str) -> str:
        """清理 LLM 响应。"""
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"<\|Assistant\|>", "", response)
        return response.strip()
