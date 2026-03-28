"""全流程共享数据结构（与 process.md / 各模块 README 对齐）。"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    id: str
    content: str
    priority: str = "P1"  # P0 / P1 / P2
    status: str = "pending"  # pending / running / done


class Plan(BaseModel):
    task: str
    subtasks: List[SubTask] = Field(default_factory=list)
    version: int = 1
    notes: str = ""


class Document(BaseModel):
    """统一检索结果单元。"""

    id: str
    content: str
    title: str = ""
    source: str = "web"
    url: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReflectionResult(BaseModel):
    sufficient: bool = False
    need_more: bool = False
    new_queries: List[str] = Field(default_factory=list)
    # 保留的文档在本轮 docs 列表中的下标（0-based）
    filtered_docs: List[int] = Field(default_factory=list)
    adequacy_score: int = 3  # 1-5
    consistency_score: int = 3
    relevance_score: int = 3
    rationale: str = ""
