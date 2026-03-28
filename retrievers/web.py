"""联网检索：优先 Tavily（若配置 API Key），否则 DuckDuckGo 文本搜索。"""

from __future__ import annotations

import hashlib
import logging
from typing import List

import httpx

from retrievers.base import Retriever
from schemas.models import Document
from utils.env_utils import TAVILY_API_KEY

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


def _doc_id(url: str, snippet: str) -> str:
    h = hashlib.sha256(f"{url}|{snippet[:200]}".encode()).hexdigest()[:16]
    return f"web_{h}"


class WebRetriever(Retriever):
    def __init__(self, timeout_s: float = 30.0):
        self._timeout = timeout_s
        self._client = httpx.Client(timeout=timeout_s)

    def search(self, query: str, max_results: int = 5) -> List[Document]:
        if TAVILY_API_KEY:
            return self._search_tavily(query, max_results)
        return self._search_ddg(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> List[Document]:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        r = self._client.post(_TAVILY_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        docs: List[Document] = []
        for item in results[:max_results]:
            url = item.get("url") or ""
            content = (item.get("content") or item.get("snippet") or "").strip()
            title = (item.get("title") or "").strip()
            if not content and not title:
                continue
            docs.append(
                Document(
                    id=_doc_id(url, content or title),
                    content=f"{title}\n{content}".strip(),
                    title=title,
                    source="web_tavily",
                    url=url,
                    metadata={"raw": item},
                )
            )
        logger.info("Tavily returned %s docs for query=%r", len(docs), query[:80])
        return docs

    def _search_ddg(self, query: str, max_results: int) -> List[Document]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise RuntimeError(
                "未配置 TAVILY_API_KEY 时需安装: pip install duckduckgo-search"
            ) from e
        docs: List[Document] = []
        ddgs = DDGS()
        for i, row in enumerate(ddgs.text(query, max_results=max_results)):
            title = (row.get("title") or "").strip()
            body = (row.get("body") or "").strip()
            href = (row.get("href") or "").strip()
            text = f"{title}\n{body}".strip()
            if not text:
                continue
            docs.append(
                Document(
                    id=_doc_id(href, text),
                    content=text,
                    title=title,
                    source="web_ddg",
                    url=href,
                    metadata={"index": i},
                )
            )
        logger.info("DDG returned %s docs for query=%r", len(docs), query[:80])
        return docs
