#!/usr/bin/env python3
"""Edge Agent 入口。

基于 LangChain 的 Tool-based Agent，统一调度 RAG 检索和 LLM 推理。
通过 ZMQ 与 ASR、LLM、TTS 等 C++ 模块通信。

启动方式:
    python -m agent.src.main          # 从项目根目录
    python agent/src/main.py          # 直接运行
"""

import logging
import os
import sys
import time
from pathlib import Path

# 确保 src/ 在 sys.path 中 (支持 python agent/src/main.py 直接运行)
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import yaml


def load_config(config_path: str = None) -> dict:
    """加载 YAML 配置文件。"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict):
    """配置日志。"""
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )


def resolve_path(base_dir: str, rel_path: str) -> str:
    """将相对路径解析为绝对路径 (相对于 agent/ 目录)。"""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.normpath(os.path.join(base_dir, rel_path))


def main():
    # ---- 1. 加载配置 ----
    agent_dir = Path(__file__).parent.parent
    config_path = agent_dir / "config.yaml"
    cfg = load_config(str(config_path))
    setup_logging(cfg)

    logger = logging.getLogger("agent")
    logger.info("============================================")
    logger.info("Edge Agent (LangChain Tool-based) 启动中...")
    logger.info("============================================")

    agent_cfg = cfg.get("agent", {})
    zmq_cfg = cfg.get("zmq", {})
    rag_cfg = cfg.get("rag", {})
    llm_cfg = cfg.get("llm", {})
    intent_cfg = cfg.get("intent", {})

    # ---- 2. 初始化通信层 ----
    from communication.zmq_bridge import PullSocket, ReqSocket
    from communication.llm_client import LLMClient

    # ASR 接收端
    asr_port = zmq_cfg.get("asr_recv_port", 5555)
    asr_socket = PullSocket(
        f"tcp://*:{asr_port}",
        timeout_ms=zmq_cfg.get("recv_timeout_ms", 100),
    )
    logger.info("ASR receiver on tcp://*:%d", asr_port)

    # TTS 发送端
    tts_address = zmq_cfg.get("tts_req_address", "tcp://127.0.0.1:7777")
    tts_socket = ReqSocket(tts_address)

    # LLM 通信客户端
    llm_client = LLMClient(
        req_address=zmq_cfg.get("llm_req_address", "tcp://127.0.0.1:8899"),
        result_port=zmq_cfg.get("llm_result_port", 5559),
        req_timeout_ms=zmq_cfg.get("llm_timeout_ms", 30000),
        result_timeout_ms=zmq_cfg.get("llm_timeout_ms", 60000),
    )

    # ---- 3. 初始化 RAG 工具 ----
    from tools.rag_search import RAGSearchTool

    model_path = resolve_path(str(agent_dir), rag_cfg.get("model_path", "../automotive_edge_rag/models"))
    vector_db_path = resolve_path(str(agent_dir), rag_cfg.get("vector_db_path", "../automotive_edge_rag/python/vector_db"))

    rag_search = RAGSearchTool(
        model_path=model_path,
        vector_db_path=vector_db_path,
        default_top_k=rag_cfg.get("default_top_k", 5),
        default_threshold=rag_cfg.get("default_threshold", 0.5),
    )

    # ---- 4. 初始化 LLM 工具 ----
    from tools.llm_generate import LLMGenerateTool
    from tools.llm_rag_generate import LLMRAGGenerateTool

    llm_gen = LLMGenerateTool(llm_client)
    llm_rag_gen = LLMRAGGenerateTool(llm_client)

    # ---- 5. 构建 Tool Registry ----
    from tools.tool_registry import ToolRegistry
    tool_registry = ToolRegistry(rag_search, llm_gen, llm_rag_gen)
    logger.info("Tools registered: %s", [t.name for t in tool_registry.get_all_tools()])

    # ---- 6. 初始化 Memory ----
    from memory.conversation_memory import ConversationMemory
    memory = ConversationMemory(
        max_turns=agent_cfg.get("max_context_turns", 5),
        timeout_seconds=agent_cfg.get("context_timeout_seconds", 30),
    )

    # ---- 7. 初始化意图分类和 Query 重写 ----
    from intent.intent_classifier import IntentClassifier
    from intent.query_rewriter import QueryRewriter

    intent_classifier = IntentClassifier(config=intent_cfg)
    query_rewriter = QueryRewriter()

    # ---- 8. 初始化请求编排器 ----
    from core.request_orchestrator import RequestOrchestrator
    orchestrator = RequestOrchestrator(
        tool_registry=tool_registry,
        llm_client=llm_client,
        memory=memory,
        intent_classifier=intent_classifier,
        query_rewriter=query_rewriter,
        config={
            "max_agent_iterations": agent_cfg.get("max_agent_iterations", 5),
            "tts_req_address": tts_address,
        },
    )

    logger.info("Edge Agent 初始化完成，等待 ASR 输入...")

    # ---- 9. 主循环 ----
    try:
        while True:
            asr_text = asr_socket.recv()
            if not asr_text:
                time.sleep(0.01)
                continue

            logger.info("[ASR→Agent] %s", asr_text)

            # 全流程处理: 归一化 → 意图分类 → Agent决策 → 回复生成
            response = orchestrator.process(asr_text)
            logger.info("[Agent→TTS] %s", response)

            # 发送到 TTS
            if response:
                echo = tts_socket.request(response)
                logger.debug("TTS echo: %s", echo)

    except KeyboardInterrupt:
        logger.info("Agent shutting down...")
    except Exception as e:
        logger.error("Agent fatal error: %s", e, exc_info=True)
        raise
    finally:
        asr_socket.close()
        tts_socket.close()
        llm_client.close()
        logger.info("Agent stopped.")


if __name__ == "__main__":
    main()
