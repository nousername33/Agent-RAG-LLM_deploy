"""测试 Tool 模块 (不依赖 ZMQ/RAG 基础设施)。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.llm_rag_generate import LLMRAGGenerateTool


class MockLLMClient:
    """模拟 LLM 客户端。"""
    def generate(self, prompt: str, timeout_ms: int = 60000):
        return "根据用户手册，发动机故障灯亮起时应立即停车检查。"

    def generate_streaming(self, prompt: str, timeout_ms: int = 60000):
        yield "根据"

    def close(self):
        pass


def test_llm_rag_generate_format():
    client = MockLLMClient()
    tool = LLMRAGGenerateTool(client)

    docs = [
        {
            "id": 0,
            "content": "发动机警告灯亮起表示发动机可能存在故障，应立即检查。",
            "score": 0.95,
            "section": "仪表盘警告灯说明",
            "subsection": "发动机警告灯",
        },
    ]

    result = tool.generate_with_context("发动机灯亮了怎么办", docs)
    assert "发动机" in result or "故障" in result or "检查" in result


def test_rag_tool_format():
    # 测试检索结果格式化逻辑
    from tools.rag_search import RAGSearchTool

    # 不加载真实模型，仅测试格式化方法
    results = [
        {
            "id": 0,
            "content": "测试内容",
            "score": 0.9,
            "section": "测试章节",
            "subsection": "测试子章节",
        }
    ]

    # 通过实例化(但不调用 search)来测试 format_for_prompt
    # format_for_prompt 是纯字符串处理，不依赖模型
    tool = RAGSearchTool(model_path="/fake/path")
    formatted = tool.format_for_prompt(results)
    assert "测试内容" in formatted
    assert "测试章节" in formatted
    assert "0.90" in formatted


def test_clean_response():
    from tools.llm_generate import LLMGenerateTool
    from tools.llm_rag_generate import LLMRAGGenerateTool

    gen_clean = LLMGenerateTool._clean_response
    rag_clean = LLMRAGGenerateTool._clean_response

    assert gen_clean("<think>推理过程</think>答案内容") == "答案内容"
    assert rag_clean("<think>...</think>最终回答") == "最终回答"
    assert gen_clean("<|Assistant|>回复") == "回复"


if __name__ == "__main__":
    test_llm_rag_generate_format()
    test_rag_tool_format()
    test_clean_response()
    print("All tools tests passed!")
