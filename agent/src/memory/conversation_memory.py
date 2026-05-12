"""三层多轮对话记忆管理。

Layer 1 - 短期上下文窗口: 保留最近 k 轮对话解决指代消解
Layer 2 - Query 重写: 代词检测 + 实体替换 (在 intent/query_rewriter.py)
Layer 3 - 结构化状态管理: 维护当前主题和最近实体，避免 token 膨胀
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage


@dataclass
class DialogueState:
    """结构化对话状态。

    轻量状态追踪，不直接送入 LLM prompt，仅用于 Query 重写和上下文管理。
    """
    current_topic: str = ""
    recent_entities: List[str] = field(default_factory=list)
    turn_count: int = 0
    last_intent: Optional[str] = None

    def update(self, user_text: str, agent_reply: str, intent: str,
               entities: Optional[List[str]] = None):
        self.turn_count += 1
        self.last_intent = intent
        if entities:
            self.recent_entities = entities[-5:]  # 保留最近 5 个实体

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_topic": self.current_topic,
            "recent_entities": self.recent_entities,
            "turn_count": self.turn_count,
            "last_intent": self.last_intent,
        }


class ConversationMemory:
    """三层记忆管理器。

    封装 LangChain ConversationBufferWindowMemory + 自定义 DialogueState。
    """

    def __init__(self, max_turns: int = 5, timeout_seconds: int = 30):
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self._buffer: deque = deque()  # 保存 (timestamp, user, agent) 元组
        self.state = DialogueState()

    def add_turn(self, user_text: str, agent_reply: str, intent: str,
                 entities: Optional[List[str]] = None):
        """添加一轮对话。"""
        self._clean_timeout()
        self._buffer.append((time.time(), user_text, agent_reply))
        if len(self._buffer) > self.max_turns:
            self._buffer.popleft()
        self.state.update(user_text, agent_reply, intent, entities)

    def get_recent_history(self) -> List[Dict[str, str]]:
        """获取最近 k 轮对话历史 (用于 LLM prompt 组装)。"""
        self._clean_timeout()
        return [
            {"user": u, "agent": a}
            for _, u, a in self._buffer
        ]

    def get_langchain_messages(self) -> List[Any]:
        """获取 LangChain 消息格式的历史记录。"""
        self._clean_timeout()
        messages = []
        for _, user, agent in self._buffer:
            messages.append(HumanMessage(content=user))
            messages.append(AIMessage(content=agent))
        return messages

    def get_last_entity(self) -> Optional[str]:
        """获取最近提到的实体 (用于指代消解)。"""
        entities = self.state.recent_entities
        return entities[-1] if entities else None

    def _clean_timeout(self):
        """清理超时对话。"""
        now = time.time()
        while self._buffer and (now - self._buffer[0][0] > self.timeout_seconds):
            self._buffer.popleft()

    def clear(self):
        """清空所有记忆。"""
        self._buffer.clear()
        self.state = DialogueState()
