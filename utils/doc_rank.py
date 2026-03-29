"""按查询与文档正文的相关性排序，用于结果条数裁剪。"""

from __future__ import annotations

import logging
from typing import List

from schemas.models import Document

logger = logging.getLogger(__name__)


def rank_documents_by_query(
    query: str,
    docs: List[Document],
    *,
    top_k: int = 10,
) -> List[Document]:
    """
    当 ``len(docs) > top_k`` 时，用查询与 ``content`` 的 TF-IDF（字符 n-gram）余弦相似度取前 ``top_k``。
    条数不超过 ``top_k`` 时原样返回。若 sklearn 不可用，退化为保留列表前 ``top_k`` 条。
    """
    if len(docs) <= top_k:
        return docs
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning("sklearn 未安装，超过 %s 条时仅保留前 %s 条（未排序）", top_k, top_k)
        return docs[:top_k]

    q = (query or "").strip()
    texts = [d.content or "" for d in docs]
    corpus = [q] + texts
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=8192)
    tfidf = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    order = np.argsort(sim)[::-1][:top_k]
    return [docs[int(i)] for i in order]
