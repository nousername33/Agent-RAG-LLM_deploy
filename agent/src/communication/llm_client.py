"""LLM 通信客户端。

与 RKNN LLM 进程通信：REQ→:8899 发送 prompt，PULL :5559 接收生成结果。
"""

import logging
import time
from typing import Optional

from communication.zmq_bridge import ReqSocket, PullSocket

logger = logging.getLogger(__name__)


class LLMClient:
    """封装与 RKNN LLM 的 ZMQ 通信协议。

    协议:
    1. REQ → tcp://127.0.0.1:8899 发送 prompt
    2. 等待 ACK ("llm sucess reply !!!")
    3. PULL ← tcp://*:5559 接收生成结果
    """

    def __init__(
        self,
        req_address: str = "tcp://127.0.0.1:8899",
        result_port: int = 5559,
        req_timeout_ms: int = 30000,
        result_timeout_ms: int = 60000,
    ):
        self.req_address = req_address
        self.result_port = result_port
        self._req = ReqSocket(req_address, timeout_ms=req_timeout_ms)
        self._result = PullSocket(
            f"tcp://*:{result_port}", timeout_ms=100
        )

    def generate(self, prompt: str, timeout_ms: int = 60000) -> Optional[str]:
        """发送 prompt 并等待完整生成结果。"""
        logger.debug("Sending prompt to LLM (len=%d): %.100s...", len(prompt), prompt)

        # 1. REQ 发送 prompt
        ack = self._req.request(prompt)
        if ack is None:
            logger.error("LLM REQ timeout on %s", self.req_address)
            return None
        logger.debug("LLM ACK: %s", ack)

        # 2. PULL 接收结果，非阻塞轮询收集完整输出
        full_response = []
        start = time.time()
        while True:
            chunk = self._result.recv()
            if chunk:
                full_response.append(chunk)
                continue
            if full_response and time.time() - start > 2.0:
                break
            if time.time() - start > (timeout_ms / 1000):
                break

        if not full_response:
            logger.warning("No response from LLM within timeout")
            return None

        result = "".join(full_response)
        logger.debug("LLM response (len=%d): %.200s...", len(result), result)
        return result

    def generate_streaming(self, prompt: str, timeout_ms: int = 60000):
        """流式生成: 逐 chunk yield 结果。

        与上述 generate() 使用完全相同的协议，
        但在收到 ACK 后逐 chunk yield 而非一次性返回完整结果。
        """

        logger.debug("Sending prompt to LLM (len=%d)", len(prompt))

        # 1. REQ 发送 prompt
        ack = self._req.request(prompt)
        if ack is None:
            logger.error("LLM REQ timeout on %s", self.req_address)
            return
        logger.debug("LLM ACK: %s", ack)

        # 2. PULL 接收结果，非阻塞轮询逐 chunk yield
        start = time.time()
        no_data_since = time.time()
        while True:
            chunk = self._result.recv()
            if chunk:
                yield chunk
                no_data_since = time.time()
                continue
            if time.time() - no_data_since > 2.0:
                break
            if time.time() - start > (timeout_ms / 1000):
                break

    def close(self):
        self._req.close()
        self._result.close()
