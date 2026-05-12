"""ZMQ 底层封装。

复用现有 ZMQ 端口协议，使用 pyzmq 与 C++ 模块通信。
"""

import logging
from typing import Optional

import zmq

logger = logging.getLogger(__name__)


class ZMQContext:
    """全局单例 ZMQ Context，所有 socket 共享。"""

    _instance: Optional["ZMQContext"] = None

    def __new__(cls) -> "ZMQContext":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._context = zmq.Context()
        return cls._instance

    @property
    def context(self) -> zmq.Context:
        return self._context

    def destroy(self):
        """销毁 context (进程退出时调用)。"""
        self._context.destroy()
        ZMQContext._instance = None


class PullSocket:
    """ZMQ_PULL socket，用于接收数据 (如 ASR → Agent :5555)。"""

    def __init__(self, address: str, timeout_ms: int = 100):
        self.address = address
        self.timeout_ms = timeout_ms
        ctx = ZMQContext()
        self._socket = ctx.context.socket(zmq.PULL)
        self._socket.bind(address)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        logger.info("PullSocket bound to %s (timeout=%dms)", address, timeout_ms)

    def recv(self) -> Optional[str]:
        """接收一条消息。超时返回 None。"""
        try:
            msg = self._socket.recv(zmq.NOBLOCK)
            return msg.decode("utf-8")
        except zmq.Again:
            return None

    def recv_blocking(self) -> str:
        """阻塞接收一条消息。"""
        msg = self._socket.recv()
        return msg.decode("utf-8")

    def close(self):
        self._socket.close()


class ReqSocket:
    """ZMQ_REQ socket，用于请求-应答模式 (如 Agent → TTS :7777, Agent → LLM :8899)。"""

    def __init__(self, address: str, timeout_ms: int = 30000):
        self.address = address
        self.timeout_ms = timeout_ms
        ctx = ZMQContext()
        self._socket = ctx.context.socket(zmq.REQ)
        self._socket.connect(address)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        logger.info("ReqSocket connected to %s (timeout=%dms)", address, timeout_ms)

    def request(self, data: str) -> Optional[str]:
        """发送请求并等待应答。超时返回 None。"""
        try:
            self._socket.send_string(data)
            return self._socket.recv_string()
        except zmq.Again:
            logger.warning("ReqSocket timeout waiting for reply from %s", self.address)
            return None

    def send_only(self, data: str):
        """只发送不等待应答 (fire-and-forget 模式)。"""
        self._socket.send_string(data, zmq.NOBLOCK)

    def close(self):
        self._socket.close()
