"""
Query-aware 正文压缩：单条网页正文超过阈值时，用本地小模型抽取与查询最相关的短段落（目标约 500 字）。
失败或未安装 transformers/torch 时回退为硬截断到 threshold 字。
"""

from __future__ import annotations

import logging
from typing import Any

from schemas.models import Document
from utils.doc_rank import rank_documents_by_query
from utils.env_utils import COMPRESS_MODEL_ID, ENABLE_QUERY_COMPRESS

logger = logging.getLogger(__name__)

# 与业务约定：超过该长度才走压缩
DEFAULT_THRESHOLD_CHARS = 2000
# 送入小模型的文档片段上限（避免撑爆上下文）
DOC_SLICE_FOR_MODEL = 12000
# 输出目标长度（字）
DEFAULT_MAX_OUT_CHARS = 500

_pipeline: Any = None


def _get_generator():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    model_id = COMPRESS_MODEL_ID
    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError("需要 pip install transformers torch 才能使用本地压缩模型") from e

    logger.info("加载 query-aware 压缩模型: %s（首次较慢）", model_id)
    _pipeline = pipeline(
        "text-generation",
        model=model_id,
        trust_remote_code=True,
        device_map="auto",
    )
    return _pipeline


def _compress_with_pipeline(query: str, doc_text: str, max_new_tokens: int = 512) -> str:
    generator = _get_generator()
    body = doc_text.strip()
    if len(body) > DOC_SLICE_FOR_MODEL:
        body = body[:DOC_SLICE_FOR_MODEL] + "\n…[输入已截断用于压缩]"
    prompt = (
        f'[INST] 给定查询："{query}"，从以下文档中提取出能够回答该查询所需的最短段落。'
        f"不要添加额外信息。文档：{body} [/INST]"
    )
    try:
        out = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
    except TypeError:
        out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    if not out:
        return ""
    chunk = out[0].get("generated_text") or ""
    chunk = chunk.strip()
    if chunk.startswith(prompt):
        chunk = chunk[len(prompt) :].strip()
    return chunk


def compress_document_content(
    query: str,
    text: str,
    *,
    threshold: int = DEFAULT_THRESHOLD_CHARS,
    max_out_chars: int = DEFAULT_MAX_OUT_CHARS,
) -> tuple[str, dict[str, Any]]:
    """
    若 ``len(text) <= threshold`` 原样返回；否则尝试 pipeline 压缩，失败则 ``text[:threshold]``。
    返回 (正文, 元信息)。
    """
    meta: dict[str, Any] = {
        "compressed": False,
        "original_len": len(text),
        "method": "none",
    }
    if len(text) <= threshold:
        return text, meta

    if not ENABLE_QUERY_COMPRESS:
        meta["method"] = "truncate_disabled_compress"
        meta["compressed"] = True
        return text[:threshold], meta

    try:
        compressed = _compress_with_pipeline(query, text)
        if not compressed:
            raise ValueError("empty generation")
        if len(compressed) > max_out_chars:
            compressed = compressed[:max_out_chars] + "…"
        meta["compressed"] = True
        meta["method"] = "local_transformers"
        meta["result_len"] = len(compressed)
        return compressed, meta
    except Exception as e:
        logger.warning("query-aware 压缩失败，回退截断: %s", e)
        meta["compressed"] = True
        meta["method"] = f"truncate_fallback:{e!s}"
        return text[:threshold], meta


def apply_compression_to_document(
    query: str,
    doc: Document,
    *,
    threshold: int = DEFAULT_THRESHOLD_CHARS,
    max_out_chars: int = DEFAULT_MAX_OUT_CHARS,
) -> Document:
    """返回新 ``Document``：更新 ``content`` 与 ``metadata['query_compress']``。"""
    new_text, meta = compress_document_content(
        query,
        doc.content,
        threshold=threshold,
        max_out_chars=max_out_chars,
    )
    md = dict(doc.metadata)
    md["query_compress"] = meta
    return Document(
        id=doc.id,
        content=new_text,
        title=doc.title,
        source=doc.source,
        url=doc.url,
        metadata=md,
    )


def prepare_documents_for_analysis(
    query: str,
    docs: list[Document],
    *,
    top_k: int = 10,
    threshold: int = DEFAULT_THRESHOLD_CHARS,
    max_out_chars: int = DEFAULT_MAX_OUT_CHARS,
) -> list[Document]:
    """
    编排层调用：先按查询取 Top-K，再对过长正文做 query-aware 压缩。
    """
    ranked = rank_documents_by_query(query, docs, top_k=top_k)
    return [
        apply_compression_to_document(query, d, threshold=threshold, max_out_chars=max_out_chars)
        for d in ranked
    ]
