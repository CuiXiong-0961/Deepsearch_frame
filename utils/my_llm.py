"""
OpenAI 兼容 LLM 工厂：按厂商 + 模型枚举懒加载，带超时与重试等工业级默认行为。
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Final, Union

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from utils.env_utils import BAILIAN_API_KEY, BAILIAN_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 对外：厂商与模型（枚举）
# ---------------------------------------------------------------------------


class LLMVendor(str, Enum):
    """LLM 厂商（与 .env 中的 base_url / api_key 对应）。"""

    QWEN = "qwen"
    DEEPSEEK = "deepseek"


class QwenModel(str, Enum):
    """通义千问兼容模式常用模型 id（DashScope OpenAI 兼容）。"""

    QWEN_FLASH = "qwen-flash"
    QWEN_TURBO = "qwen-turbo"
    QWEN_PLUS = "qwen-plus"
    QWEN_MAX = "qwen-max"


class DeepSeekModel(str, Enum):
    """DeepSeek OpenAI 兼容 API 模型 id。"""

    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"
    DEEPSEEK_CODER = "deepseek-coder"


ModelEnum = Union[QwenModel, DeepSeekModel]

# ---------------------------------------------------------------------------
# 默认工业级 HTTP / 调用参数
# ---------------------------------------------------------------------------
# 以下四项对应 httpx.Timeout 的四个阶段，会同时传给独立 httpx.Client 与 ChatOpenAI.timeout，
# 避免「只配一层」导致实际仍用库默认值。
#
# _DEFAULT_CONNECT_S：与远端建立 TCP/TLS 握手的上限；网络差或 DNS 慢时在此阶段超时。
# _DEFAULT_READ_S：连接已建立后，等待服务端返回数据（含流式首包/后续块）的上限；长推理场景宜偏大。
# _DEFAULT_WRITE_S：把本请求 body 写完的上限；大上下文上传时主要受它约束。
# _DEFAULT_POOL_S：从 httpx 连接池取一条可用连接的最大等待；池满或复用排队过久时触发。
#
# _DEFAULT_MAX_RETRIES：传给 ChatOpenAI.max_retries，由底层 OpenAI 兼容客户端对可重试错误
#（如 429、5xx、连接重置等，具体以所用 langchain-openai / openai 版本为准）做有限次重试。
#
# _DEFAULT_HTTP_LIMITS：限制本进程内该 httpx.Client 的总并发连接数与长连接复用条数，
# 避免高并发时文件句柄/内存暴涨；与「单次请求超时」无关，管的是连接池规模。

_DEFAULT_CONNECT_S: Final[float] = 15.0
_DEFAULT_READ_S: Final[float] = 180.0
_DEFAULT_WRITE_S: Final[float] = 120.0
_DEFAULT_POOL_S: Final[float] = 5.0
_DEFAULT_MAX_RETRIES: Final[int] = 3
_DEFAULT_HTTP_LIMITS = httpx.Limits(max_connections=32, max_keepalive_connections=16)


def _vendor_endpoint(vendor: LLMVendor) -> tuple[str, str | None]:
    if vendor is LLMVendor.QWEN:
        return BAILIAN_BASE_URL or "", BAILIAN_API_KEY
    if vendor is LLMVendor.DEEPSEEK:
        return DEEPSEEK_BASE_URL or "", DEEPSEEK_API_KEY
    raise ValueError(f"unsupported vendor: {vendor}")


def _validate_model(vendor: LLMVendor, model: ModelEnum) -> None:
    if vendor is LLMVendor.QWEN and not isinstance(model, QwenModel):
        raise TypeError(f"vendor {vendor!r} requires QwenModel, got {type(model).__name__}")
    if vendor is LLMVendor.DEEPSEEK and not isinstance(model, DeepSeekModel):
        raise TypeError(f"vendor {vendor!r} requires DeepSeekModel, got {type(model).__name__}")


def _build_http_client(
    *,
    connect_s: float,
    read_s: float,
    write_s: float,
    pool_s: float,
) -> httpx.Client:
    timeout = httpx.Timeout(
        connect=connect_s,
        read=read_s,
        write=write_s,
        pool=pool_s,
    )
    return httpx.Client(timeout=timeout, limits=_DEFAULT_HTTP_LIMITS)


@lru_cache(maxsize=64)
def _create_chat_client(
    vendor_value: str,
    model_value: str,
    temperature: float,
    max_retries: int,
    connect_s: float,
    read_s: float,
    write_s: float,
    pool_s: float,
) -> ChatOpenAI:
    vendor = LLMVendor(vendor_value)
    base_url, api_key = _vendor_endpoint(vendor)
    if not base_url:
        raise RuntimeError(f"base_url for {vendor.value} is not set in environment")
    if not api_key:
        raise RuntimeError(f"api_key for {vendor.value} is not set in environment")

    http_client = _build_http_client(
        connect_s=connect_s, read_s=read_s, write_s=write_s, pool_s=pool_s
    )

    llm = ChatOpenAI(
        model=model_value,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        # 与 httpx.Timeout.read 对齐，避免仅套一层无意义的大超时
        timeout=httpx.Timeout(
            connect=connect_s,
            read=read_s,
            write=write_s,
            pool=pool_s,
        ),
        http_client=http_client,
    )
    logger.debug(
        "initialized ChatOpenAI vendor=%s model=%s max_retries=%s connect=%ss read=%ss",
        vendor.value,
        model_value,
        max_retries,
        connect_s,
        read_s,
    )
    return llm


def get_llm(
    vendor: LLMVendor,
    model: ModelEnum,
    *,
    temperature: float = 0.6,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    connect_timeout_s: float = _DEFAULT_CONNECT_S,
    read_timeout_s: float = _DEFAULT_READ_S,
    write_timeout_s: float = _DEFAULT_WRITE_S,
    pool_timeout_s: float = _DEFAULT_POOL_S,
) -> ChatOpenAI:
    """
    按厂商与模型懒加载并缓存 ``ChatOpenAI`` 实例。

    - 使用 ``lru_cache`` 按 (vendor, model, temperature, 超时与重试参数) 维度缓存。
    - HTTP：独立 ``httpx.Client``，连接/读/写/pool 超时与 OpenAI 客户端一致。
    - 重试：LangChain ``ChatOpenAI.max_retries``（网络与可重试状态码，具体行为以所用版本为准）。

    Parameters
    ----------
    vendor:
        ``LLMVendor.QWEN``（百炼兼容）或 ``LLMVendor.DEEPSEEK``。
    model:
        与 vendor 对应的 ``QwenModel`` 或 ``DeepSeekModel`` 枚举值。
    temperature, max_retries, *_timeout_s:
        生成参数与 HTTP 粒度超时；一般无需改动默认值。
    """
    _validate_model(vendor, model)
    return _create_chat_client(
        vendor.value,
        model.value,
        temperature,
        max_retries,
        connect_timeout_s,
        read_timeout_s,
        write_timeout_s,
        pool_timeout_s,
    )


def clear_llm_cache() -> None:
    """测试或切换环境变量后清空缓存，使下次 ``get_llm`` 重新建连。"""
    _create_chat_client.cache_clear()


def test_ping() -> None:
    llm = get_llm(LLMVendor.QWEN, QwenModel.QWEN_FLASH)
    resp = llm.invoke([HumanMessage(content="用一句话介绍南京。")])
    print(resp.content)


if __name__ == "__main__":
    # 勿对 root 使用 DEBUG：否则 openai / httpx / httpcore 等第三方会全部打出 DEBUG，日志爆炸。
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    for _noisy in ("openai", "httpcore", "httpx", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    test_ping()
