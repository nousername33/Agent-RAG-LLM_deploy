"""Query 重写模块 (三层记忆 - Layer 2)。

当当前输入依赖上下文时 (如代词指代、省略表达)，
利用 DialogueState 将缩写 query 改写成完整问题。

示例:
- "它怎么开" + entity="后视镜加热" → "后视镜加热怎么开"
- "那个是什么" + entity="发动机警告灯" → "发动机警告灯是什么"
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QueryRewriter:
    """基于规则的 Query 重写器。

    检测中文指代词 (它/这/那/这个/那个) 并用最近实体进行替换。
    """

    # 指代词模式: 匹配中文常见代词 + 英文 it/that
    PRONOUN_PATTERNS = [
        (re.compile(r"^(它|他|她)(怎么|如何|是什么|能|可以|要|会|在|是)"), "replace"),
        (re.compile(r"^(这个|那个)(是|怎么|如何|能|可以|要|会)"), "replace"),
        (re.compile(r"^(这|那)(是|怎么|如何)"), "replace"),
        (re.compile(r"^它的?"), "replace"),
        (re.compile(r"^(上面|前面|刚才)说的"), "strip_prefix"),
    ]

    def rewrite(self, query: str, last_entity: Optional[str] = None) -> str:
        """尝试用最近实体改写代词指代。

        Args:
            query: 当前用户输入
            last_entity: 对话状态中最近的实体

        Returns:
            改写后的 query (无法改写时返回原文)
        """
        if not last_entity or not query:
            return query
        if last_entity in query:
            return query

        for pattern, action in self.PRONOUN_PATTERNS:
            match = pattern.search(query)
            if match:
                if action == "replace":
                    suffix = query[match.end():]
                    rewritten = f"{last_entity}{suffix}" if suffix else last_entity
                    logger.info("Query rewritten: '%s' → '%s'", query, rewritten)
                    return rewritten
                elif action == "strip_prefix":
                    rewritten = f"{last_entity}，{query[match.end():]}"
                    logger.info("Query rewritten: '%s' → '%s'", query, rewritten)
                    return rewritten

        return query
