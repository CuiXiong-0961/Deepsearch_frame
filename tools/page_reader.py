"""
网页正文拉取与「图中文字」补充 — 主要思路
========================================
1) 搜索 API（如 Tavily）返回的多为标题 + 短摘要，单独用作研究证据信息度不足。对每条结果携带的 URL
   再发起一次 HTTP GET，拿到完整 HTML，用 trafilatura 做与「阅读模式」类似的处理：去掉导航、侧栏、
   页脚等模板噪声，抽取主体文章文本，作为摘要之上的主要证据正文。
2) 正文中常含截图、幻灯片、扫描件等以图片形式承载的大段文字。对页面内有限个 <img> 拉取图片字节，
   过滤过小图标与超大文件；对剩余图片用 OCR（pytesseract + 本机 Tesseract 引擎）识别纯文本。
   仅当识别结果长度超过阈值时才拼入结果，用于过滤装饰图、表情包等非文字图；不识别图片语义、
   不处理视频/音频，也不做通用多模态理解。
3) 工程上：全局超时、单页图片数量与总耗时上限；任一步失败则保留搜索引擎原始 snippet，不阻断主流程。
   OCR 为可选增强：未安装 Tesseract 或语言包时自动跳过 OCR，仅使用正文抽取。
   Windows 可安装 Tesseract 安装包并将安装路径加入 PATH；中文识别需额外安装 chi_sim 语言包。
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from schemas.models import Document

logger = logging.getLogger(__name__)

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

_MAX_MAIN_TEXT = 100_000
_MAX_IMAGE_BYTES = 2 * 1024 * 1024
_MAX_IMAGES_PER_PAGE = 12
_MIN_IMAGE_SIDE_PX = 80
_MIN_OCR_CHARS = 40


@dataclass
class PageEnrichConfig:
    timeout_s: float = 25.0
    max_main_text: int = _MAX_MAIN_TEXT
    max_image_bytes: int = _MAX_IMAGE_BYTES
    max_images: int = _MAX_IMAGES_PER_PAGE
    min_side_px: int = _MIN_IMAGE_SIDE_PX
    min_ocr_chars: int = _MIN_OCR_CHARS


def _looks_like_html(content_type: str, body_sample: bytes) -> bool:
    ct = (content_type or "").lower()
    if "pdf" in ct or "image/" in ct:
        return False
    if "html" in ct or "text/plain" in ct:
        return True
    head = body_sample[:200].lstrip().lower()
    return head.startswith(b"<html") or head.startswith(b"<!doctype")


def _ocr_image_png(image_bytes: bytes, cfg: PageEnrichConfig) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.debug("pytesseract/PIL 未安装，跳过 OCR")
        return ""

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        if w < cfg.min_side_px or h < cfg.min_side_px:
            return ""
    except Exception as e:
        logger.debug("打开图片失败: %s", e)
        return ""

    text = ""
    for lang in ("chi_sim+eng", "eng"):
        try:
            text = pytesseract.image_to_string(img, lang=lang)
            if text and len(text.strip()) >= 10:
                break
        except Exception:
            continue
    text = (text or "").strip()
    if len(text) < cfg.min_ocr_chars:
        return ""
    return text


def _collect_img_urls(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if not src or src.startswith("data:"):
            continue
        absolute = urljoin(base_url, src)
        p = urlparse(absolute)
        if p.scheme not in ("http", "https"):
            continue
        urls.append(absolute)
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_page_enriched(
    url: str,
    *,
    fallback_snippet: str = "",
    title: str = "",
    cfg: PageEnrichConfig | None = None,
    client: httpx.Client | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    拉取单页：正文（trafilatura）+ 可选图片 OCR 文本。
    返回 (合并后的可读文本, 元信息 dict)。
    """
    cfg = cfg or PageEnrichConfig()
    meta: dict[str, Any] = {"url": url, "fetched": False, "error": None, "ocr_blocks": 0}

    if not url.startswith(("http://", "https://")):
        meta["error"] = "invalid_url"
        return _merge_texts(title, fallback_snippet, "", ""), meta

    own_client = client is None
    c = client or httpx.Client(
        timeout=cfg.timeout_s,
        headers={"User-Agent": _DEFAULT_UA},
        follow_redirects=True,
    )
    try:
        r = c.get(url)
        r.raise_for_status()
        body = r.content
        ct = r.headers.get("content-type", "")
        if not _looks_like_html(ct, body):
            meta["error"] = f"skip_content_type:{ct}"
            return _merge_texts(title, fallback_snippet, "", ""), meta

        html = body.decode(r.encoding or "utf-8", errors="replace")

        main = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
        )
        main = (main or "").strip()
        if len(main) > cfg.max_main_text:
            main = main[: cfg.max_main_text] + "\n…[正文截断]"

        ocr_parts: list[str] = []
        img_urls = _collect_img_urls(html, url)[: cfg.max_images]
        for iu in img_urls:
            try:
                ir = c.get(iu, timeout=min(20.0, cfg.timeout_s))
                ir.raise_for_status()
                if len(ir.content) > cfg.max_image_bytes:
                    continue
                if not (ir.headers.get("content-type") or "").lower().startswith("image"):
                    # 部分 CDN 未返回正确 type，仍尝试 PIL
                    pass
                chunk = _ocr_image_png(ir.content, cfg)
                if chunk:
                    ocr_parts.append(chunk)
                    meta["ocr_blocks"] = meta.get("ocr_blocks", 0) + 1
            except Exception as e:
                logger.debug("OCR 跳过图片 %s: %s", iu[:80], e)

        ocr_merged = "\n\n".join(ocr_parts) if ocr_parts else ""
        meta["fetched"] = True
        meta["main_len"] = len(main)
        meta["ocr_len"] = len(ocr_merged)
        return _merge_texts(title, fallback_snippet, main, ocr_merged), meta
    except Exception as e:
        logger.info("fetch_page_enriched failed %s: %s", url[:100], e)
        meta["error"] = str(e)
        return _merge_texts(title, fallback_snippet, "", ""), meta
    finally:
        if own_client:
            c.close()


def _merge_texts(title: str, snippet: str, main: str, ocr: str) -> str:
    parts: list[str] = []
    if title:
        parts.append(title.strip())
    if main:
        parts.append(main)
    elif snippet:
        parts.append(f"[搜索摘要]\n{snippet.strip()}")
    if ocr:
        parts.append(f"[图片内文字 OCR]\n{ocr.strip()}")
    return "\n\n".join(p for p in parts if p).strip()


def enrich_documents(
    docs: list[Document],
    *,
    cfg: PageEnrichConfig | None = None,
    enabled: bool = True,
) -> tuple[list[Document], dict[str, Any]]:
    """
    对检索得到的 Document 列表逐条 enrich：写回 ``content``，并在 ``metadata['page_enrichment']`` 记元信息。
    """
    cfg = cfg or PageEnrichConfig()
    stats = {"attempted": 0, "fetched_ok": 0, "errors": []}

    if not enabled:
        return docs, {**stats, "enabled": False}

    client = httpx.Client(timeout=cfg.timeout_s, headers={"User-Agent": _DEFAULT_UA}, follow_redirects=True)
    try:
        out: list[Document] = []
        for d in docs:
            url = (d.url or "").strip()
            if not url:
                out.append(d)
                continue
            stats["attempted"] += 1
            text, meta = fetch_page_enriched(
                url,
                fallback_snippet=d.content,
                title=d.title,
                cfg=cfg,
                client=client,
            )
            md = dict(d.metadata)
            md["page_enrichment"] = meta
            if meta.get("fetched"):
                stats["fetched_ok"] += 1
            elif meta.get("error"):
                stats["errors"].append({"url": url[:120], "error": meta["error"]})
            out.append(
                Document(
                    id=d.id,
                    content=text if text.strip() else d.content,
                    title=d.title,
                    source=d.source,
                    url=d.url,
                    metadata=md,
                )
            )
        return out, {**stats, "enabled": True}
    finally:
        client.close()
