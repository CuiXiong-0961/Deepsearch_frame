"""联网搜索工具门面：统一参数、日志；可选拉取网页正文与图中 OCR 文本。"""

from __future__ import annotations

import logging
import time
from typing import Any, List

from retrievers.web import WebRetriever
from schemas.models import Document
from tools.page_reader import PageEnrichConfig, enrich_documents

logger = logging.getLogger(__name__)


def web_search(
    query: str,
    *,
    max_results: int = 5,
    retriever: WebRetriever | None = None,
    fetch_full_page: bool = True,
    page_cfg: PageEnrichConfig | None = None,
) -> tuple[List[Document], dict[str, Any]]:
    """
    先走搜索引擎拿 URL 与摘要；若 ``fetch_full_page=True``，再对每条 URL 拉 HTML，
    抽取正文并尝试图片 OCR（见 ``page_reader`` 模块说明）。
    """
    r = retriever or WebRetriever()
    t0 = time.perf_counter()
    try:
        docs = r.search(query, max_results=max_results)
        err = None
    except Exception as e:
        logger.exception("web_search failed: %s", query[:120])
        docs = []
        err = str(e)

    page_stats: dict[str, Any] = {"enabled": fetch_full_page, "skipped": not fetch_full_page}
    if docs and fetch_full_page:
        docs, page_stats = enrich_documents(docs, cfg=page_cfg, enabled=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    audit: dict[str, Any] = {
        "tool": "web_search",
        "query": query,
        "count": len(docs),
        "elapsed_ms": round(elapsed_ms, 2),
        "error": err,
        "page_fetch": page_stats,
    }
    return docs, audit
