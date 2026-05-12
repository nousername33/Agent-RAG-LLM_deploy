"""测试三层记忆管理。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory.conversation_memory import ConversationMemory, DialogueState


def test_memory_add_and_retrieve():
    mem = ConversationMemory(max_turns=3)
    mem.add_turn("怎么打开后视镜加热", "后视镜加热按钮在车门控制面板上",
                 "FACTUAL", ["后视镜加热"])

    history = mem.get_recent_history()
    assert len(history) == 1
    assert history[0]["user"] == "怎么打开后视镜加热"
    assert history[0]["agent"] == "后视镜加热按钮在车门控制面板上"


def test_memory_window_limit():
    mem = ConversationMemory(max_turns=3)
    for i in range(5):
        mem.add_turn(f"问题{i}", f"回答{i}", "FACTUAL")

    history = mem.get_recent_history()
    assert len(history) == 3
    assert history[0]["user"] == "问题2"


def test_memory_entity_tracking():
    mem = ConversationMemory(max_turns=5)
    mem.add_turn("发动机灯亮了怎么办", "请立即停车检查", "FACTUAL",
                 entities=["发动机警告灯"])
    mem.add_turn("它亮了要检查什么", "检查机油和冷却液", "FACTUAL",
                 entities=["发动机警告灯"])

    assert mem.get_last_entity() == "发动机警告灯"


def test_dialogue_state():
    state = DialogueState()
    state.update("测试", "回复", "FACTUAL", ["实体A", "实体B"])
    assert state.turn_count == 1
    assert state.last_intent == "FACTUAL"
    assert len(state.recent_entities) == 2

    d = state.to_dict()
    assert d["current_topic"] == ""
    assert d["turn_count"] == 1


def test_memory_timeout():
    mem = ConversationMemory(max_turns=5, timeout_seconds=0)
    mem.add_turn("问题", "回答", "FACTUAL")
    history = mem.get_recent_history()
    assert len(history) == 0  # 立即超时


if __name__ == "__main__":
    test_memory_add_and_retrieve()
    test_memory_window_limit()
    test_memory_entity_tracking()
    test_dialogue_state()
    test_memory_timeout()
    print("All memory tests passed!")
