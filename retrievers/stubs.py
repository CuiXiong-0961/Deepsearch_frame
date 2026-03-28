"""向量库 / 混合检索占位：接口与 README 一致，默认未实现。"""

from __future__ import annotations

from typing import List

from retrievers.base import Retriever
from schemas.models import Document


class VectorRetriever(Retriever):
    def search(self, query: str, max_results: int = 5) -> List[Document]:
        raise NotImplementedError(
            "VectorRetriever 需接入向量库与嵌入；请实现 search 或使用 WebRetriever。"
        )


class HybridRetriever(Retriever):
    def __init__(self, *parts: Retriever):
        self._parts = parts

    def search(self, query: str, max_results: int = 5) -> List[Document]:
        raise NotImplementedError(
            "HybridRetriever 融合策略未默认实现；请组合 WebRetriever 等自定义合并逻辑。"
        )
