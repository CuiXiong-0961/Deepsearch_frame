"""
Redis 存储层（中间结果外置）。

目的：
- 将检索得到的文档、audit、reflection 等“大对象”放入 Redis，避免 LangGraph state 过大。
- LangGraph state 中只保存 redis key（或 key 列表），降低序列化与内存压力。
- 记录每轮 run 的所有 key，便于在“下一问开始”时批量清理。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class RedisConfig:
    """Redis 连接配置（优先 REDIS_URL，否则使用 host/port/db）。"""

    url: Optional[str]
    host: str
    port: int
    db: int
    prefix: str = "deepsearch"


def load_redis_config() -> RedisConfig:
    """
    从环境变量读取 Redis 配置。

    - REDIS_URL：例如 redis://localhost:6379/0
    - REDIS_HOST/REDIS_PORT/REDIS_DB：当未提供 REDIS_URL 时使用
    - REDIS_PREFIX：key 前缀（默认 deepsearch）
    """

    url = os.getenv("REDIS_URL")
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    prefix = os.getenv("REDIS_PREFIX", "deepsearch").strip() or "deepsearch"
    return RedisConfig(url=url, host=host, port=port, db=db, prefix=prefix)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


class RedisStore:
    """
    简单的 JSON Redis 存储封装。

    注意：这里只做“存/取/批量删 + 元数据记录”，不做复杂的 schema 约束；
    由上层 runner 负责保证 key 命名与写入内容的一致性。
    """

    def __init__(self, cfg: RedisConfig):
        # 方法注释：初始化 redis client（惰性依赖，便于没有 redis 时更明确报错）
        import redis  # type: ignore

        self.cfg = cfg
        if cfg.url:
            self._r = redis.Redis.from_url(cfg.url, decode_responses=True)
        else:
            self._r = redis.Redis(
                host=cfg.host,
                port=cfg.port,
                db=cfg.db,
                decode_responses=True,
            )

    def key(self, run_id: str, *parts: str) -> str:
        """按约定生成 key：{prefix}:{run_id}:{parts...}"""

        clean = [p.strip(":") for p in parts if p]
        return ":".join([self.cfg.prefix, run_id, *clean])

    def meta_key(self, run_id: str) -> str:
        """存储本轮所有 key 的集合（用于批量清理）。"""

        return self.key(run_id, "meta", "keys")

    def put_json(self, key: str, value: Any) -> None:
        """写入 JSON 字符串。"""

        self._r.set(key, _json_dumps(value))

    def get_json(self, key: str) -> Any:
        """读取 JSON 字符串并反序列化；不存在返回 None。"""

        raw = self._r.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return raw

    def track_key(self, run_id: str, key: str) -> None:
        """把 key 记录到本轮 meta set，便于清理。"""

        self._r.sadd(self.meta_key(run_id), key)

    def track_keys(self, run_id: str, keys: Iterable[str]) -> None:
        """批量记录 key 到本轮 meta set。"""

        mk = self.meta_key(run_id)
        ks = [k for k in keys if k]
        if ks:
            self._r.sadd(mk, *ks)

    def list_tracked_keys(self, run_id: str) -> List[str]:
        """列出本轮记录过的所有 key。"""

        return sorted(self._r.smembers(self.meta_key(run_id)))

    def delete_keys(self, keys: Iterable[str]) -> int:
        """删除给定 keys，返回删除数量（尽力而为）。"""

        ks = [k for k in keys if k]
        if not ks:
            return 0
        return int(self._r.delete(*ks))

    def cleanup_run(self, run_id: str) -> int:
        """
        清理某个 run：删除 meta 记录的所有 key（含 meta 自身）。

        设计意图：在“下一问开始”时调用，删除上一问的缓存，避免长期堆积。
        """

        mk = self.meta_key(run_id)
        keys = list(self._r.smembers(mk))
        # 也删掉 meta set 自身
        keys.append(mk)
        return self.delete_keys(keys)

