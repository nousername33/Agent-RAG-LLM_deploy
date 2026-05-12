"""Tool: retrieve_vehicle_knowledge

检索类 Tool - 封装底层 RAG 检索模块，向 Agent 暴露标准检索接口。
输入 query + top_k + threshold，输出结构化文档结果列表。
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGSearchTool:
    """封装 VehicleVectorSearch 为 Agent 可调用工具。

    直接 import Python RAG 模块 (同进程内调用，无 ZMQ 开销)。
    """

    def __init__(
        self,
        model_path: str,
        vector_db_path: str = "vector_db",
        default_top_k: int = 5,
        default_threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.vector_db_path = vector_db_path
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold
        self._searcher: Optional[Any] = None

    def _lazy_init(self):
        """延迟加载 RAG 模型 (首次调用时初始化)。"""
        if self._searcher is not None:
            return
        import sys
        import os
        rag_path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "automotive_edge_rag", "python")
        rag_path = os.path.normpath(rag_path)
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)

        from vehicle_vector_search import VehicleVectorSearch

        self._searcher = VehicleVectorSearch(
            model_path=self.model_path,
            vector_db_path=self.vector_db_path,
        )
        if self._searcher.model is None:
            self._searcher.load_model(self.model_path)
        logger.info("RAG model loaded from %s", self.model_path)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """执行向量检索。

        Returns:
            [{id, content: str, score: float, section: str, subsection: str}, ...]
        """
        self._lazy_init()
        results = self._searcher.search(query, top_k=top_k, threshold=threshold)
        return [
            {
                "id": r["id"],
                "content": r["text"],
                "score": r["similarity"],
                "section": r.get("section", ""),
                "subsection": r.get("subsection", ""),
            }
            for r in results
        ]

    def format_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """将检索结果格式化为 LLM 可理解的上下文文本。"""
        if not results:
            return "（未找到相关文档）"

        parts = []
        for i, r in enumerate(results, 1):
            header = f"[文档{i}] 章节: {r['section']}"
            if r["subsection"]:
                header += f" > {r['subsection']}"
            header += f" (相关度: {r['score']:.2f})"
            parts.append(f"{header}\n{r['content']}")

        return "\n\n---\n\n".join(parts)
