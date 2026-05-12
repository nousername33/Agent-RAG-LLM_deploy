"""LangChain LLM 封装 RKNN LLM。

通过 ZMQ 与 C++ LLM 进程通信，将 LangChain 的 prompt 发送给 RKNN LLM。

使用 BaseLLM (而非 BaseChatModel) 以避免 langchain_protocol
在 Python 3.13 上的兼容性问题。
"""

import re
import logging
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult

logger = logging.getLogger(__name__)


class ZMQLLM(BaseLLM):
    """LangChain 兼容的 LLM，通过 ZMQ 调用 RKNN LLM。

    将 prompt 字符串发送给 C++ LLM 进程，解析返回结果。
    """

    llm_client: Any = None  # LLMClient 实例
    model_name: str = "deepseek-r1-distill-qwen-1.5b"
    temperature: float = 0.7
    max_tokens: int = 256

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "zmq-rknn-llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """发送 prompt 到 RKNN LLM，返回生成的文本。"""
        logger.debug("ZMQLLM prompt (len=%d): %.200s...", len(prompt), prompt)

        raw_response = self.llm_client.generate(prompt)
        if raw_response is None:
            return "抱歉，模型暂时无法响应。"

        cleaned = self._clean_response(raw_response)
        logger.debug("ZMQLLM response (len=%d): %.200s...", len(cleaned), cleaned)
        return cleaned

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[Generation]:
        """流式生成。"""
        for chunk in self.llm_client.generate_streaming(prompt):
            yield Generation(text=chunk)

    @staticmethod
    def _clean_response(response: str) -> str:
        """清理 LLM 输出: 移除 think 块、特殊 token 等。"""
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"<think>.*", "", response, flags=re.DOTALL)
        for token in ["<｜User｜>", "<｜Assistant｜>", "<|endoftext|>"]:
            response = response.replace(token, "")
        return response.strip()
