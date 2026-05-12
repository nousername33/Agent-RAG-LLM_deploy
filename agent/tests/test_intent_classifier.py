"""测试意图分类器。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from intent.intent_classifier import IntentClassifier, IntentType


def test_greeting():
    clf = IntentClassifier()
    result = clf.classify("你好")
    assert result.intent == IntentType.GREETING
    assert result.fallback_reply is not None

    result = clf.classify("早上好")
    assert result.intent == IntentType.GREETING

    result = clf.classify("再见")
    assert result.intent == IntentType.GREETING


def test_command():
    clf = IntentClassifier()
    result = clf.classify("打开空调")
    assert result.intent == IntentType.COMMAND

    result = clf.classify("关闭导航")
    assert result.intent == IntentType.COMMAND


def test_factual():
    clf = IntentClassifier()
    result = clf.classify("发动机故障灯亮了怎么办")
    assert result.intent == IntentType.FACTUAL
    assert "retrieve_vehicle_knowledge" in result.recommended_tools

    result = clf.classify("如何更换轮胎")
    assert result.intent == IntentType.FACTUAL


def test_complex():
    clf = IntentClassifier()
    result = clf.classify("为什么发动机会过热")
    assert result.intent == IntentType.COMPLEX_EXPLANATION

    result = clf.classify("解释一下ABS系统工作原理")
    assert result.intent == IntentType.COMPLEX_EXPLANATION


def test_open_generation():
    clf = IntentClassifier()
    result = clf.classify("今天天气不错啊")
    assert result.intent == IntentType.OPEN_GENERATION
    assert "generate_answer" in result.recommended_tools


def test_unknown():
    clf = IntentClassifier()
    result = clf.classify("嗯")
    assert result.intent == IntentType.UNKNOWN


def test_empty_input():
    clf = IntentClassifier()
    result = clf.classify("")
    assert result.intent == IntentType.UNKNOWN


if __name__ == "__main__":
    test_greeting()
    test_command()
    test_factual()
    test_complex()
    test_open_generation()
    test_unknown()
    test_empty_input()
    print("All intent classifier tests passed!")
