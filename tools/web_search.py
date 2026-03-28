"""联网搜索工具门面：统一参数、日志，供编排层或 Agent 调用。"""

from __future__ import annotations

import logging
import time
from typing import Any, List

from retrievers.web import WebRetriever
from schemas.models import Document

logger = logging.getLogger(__name__)


def web_search(
    query: str,
    *,
    max_results: int = 5,
    retriever: WebRetriever | None = None,
) -> tuple[List[Document], dict[str, Any]]:
    r = retriever or WebRetriever()
    t0 = time.perf_counter()
    try:
        docs = r.search(query, max_results=max_results)
        err = None
    except Exception as e:
        logger.exception("web_search failed: %s", query[:120])
        docs = []
        err = str(e)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    audit: dict[str, Any] = {
        "tool": "web_search",
        "query": query,
        "count": len(docs),
        "elapsed_ms": round(elapsed_ms, 2),
        "error": err,
    }
    return docs, audit
