"""
子任务展示与执行顺序工具。

Planner 常生成 ``s1``、``s2`` … 这类 id，并可能给不同子任务不同 ``priority``。
若仅按 ``(priority, id 字符串)`` 排序，会出现 **s6（P0）排在 s3（P1）前** 等与用户阅读顺序不一致的情况。

本模块约定：当 id 匹配 ``s`` + 数字（如 `s1`、`s12`）时，**优先按该数字升序** 排列，
与「研究步骤编号」一致；无法解析的 id 再按优先级与字符串兜底。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from schemas.models import SubTask

# 匹配 s1、s01、S12 等（统一取数字部分）
_RE_S_NUM = re.compile(r"^s(\d+)$", re.IGNORECASE)


def subtask_suffix_number(subtask_id: str) -> Optional[int]:
    """
    从子任务 id 中解析 ``s`` 后的数字；无法解析则返回 None。

    示例：``s1`` -> 1，``s12`` -> 12，``foo`` -> None。
    """

    m = _RE_S_NUM.match((subtask_id or "").strip())
    if not m:
        return None
    return int(m.group(1))


def _priority_rank(p: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2}.get((p or "").strip().upper(), 1)


def sort_key_subtask(st: SubTask) -> tuple:
    """
    用于 ``sorted(plan.subtasks, key=sort_key_subtask)``。

    - 可解析为 s+数字：按数字升序（保证 s1→s2→…→s6 的阅读顺序）。
    - 否则：按 priority 再按 id 字符串。
    """

    n = subtask_suffix_number(st.id)
    if n is not None:
        return (0, n)
    return (1, _priority_rank(st.priority), st.id)


def sort_key_subtask_dict(d: Dict[str, Any]) -> tuple:
    """Gradio / JSON 子任务 dict 的排序键，与 ``sort_key_subtask`` 语义一致。"""

    sid = str(d.get("id") or "")
    n = subtask_suffix_number(sid)
    if n is not None:
        return (0, n)
    return (1, _priority_rank(str(d.get("priority", "P1"))), sid)


def sort_subtasks_ordered(subtasks: List[SubTask]) -> List[SubTask]:
    """返回按「s 序号优先」排好序的子任务列表。"""

    return sorted(subtasks, key=sort_key_subtask)


def sort_subtask_dicts_ordered(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对子任务 dict 列表做相同排序。"""

    return sorted(rows, key=sort_key_subtask_dict)
