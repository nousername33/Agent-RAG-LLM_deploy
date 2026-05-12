"""Tool: generate_answer

生成类 Tool - 对应纯 LLM 推理能力。
输入 query + history + 生成参数，输出自然语言文本。
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class LLMGenerateTool:
    """封装 LLM 生成为 Agent 可调用工具。

    通过 ZMQ 与 RKNN LLM 通信，支持伪流式输出。
    """

    def __init__(self, llm_client: "LLMClient", default_max_tokens: int = 256):
        self._client = llm_client
        self.default_max_tokens = default_max_tokens

    def generate(self, query: str, history: Optional[List[str]] = None) -> str:
        """直接生成回答 (不走 RAG)。

        Args:
            query: 当前用户问题
            history: 最近几轮对话上下文 (可选)

        Returns:
            生成的回答文本
        """
        prompt = self._build_prompt(query, history)
        response = self._client.generate(prompt)
        return self._clean_response(response) if response else "抱歉，我暂时无法回答这个问题。"

    def generate_streaming(self, query: str, history: Optional[List[str]] = None):
        """流式生成回答 (伪流式，逐 chunk yield)。"""
        prompt = self._build_prompt(query, history)
        for chunk in self._client.generate_streaming(prompt):
            yield chunk

    def _build_prompt(self, query: str, history: Optional[List[str]] = None) -> str:
        """构建生成 prompt。

        RKNN LLM 使用 DeepSeek chat template:
        <｜User｜>...<｜Assistant｜><think>\n</think>
        """
        parts = []
        if history:
            parts.append("以下是最近的对话历史：")
            for i, turn in enumerate(history, 1):
                parts.append(f"用户: {turn}")
            parts.append("")

        parts.append(f"请回答用户的问题：{query}")
        parts.append("如果你不确定答案，请直接说明，不要编造信息。")

        return "\n".join(parts)

    @staticmethod
    def _clean_response(response: str) -> str:
        """清理 LLM 响应中的 think 标签和空白。"""
        # 移除 <think>...</think> 块 (DeepSeek-R1 特性)
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"<\|Assistant\|>", "", response)
        response = response.strip()
        return response
