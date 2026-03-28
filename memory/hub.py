from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemas.models import Document, Plan, ReflectionResult


class MemoryHub:
    """统一中间状态：计划、按子任务的文档与小结、反思记录。"""

    def __init__(self) -> None:
        self.task: str = ""
        self.plan: Optional[Plan] = None
        self.current_subtask_id: str = ""
        self.docs: Dict[str, List[Document]] = {}
        self.summaries: Dict[str, str] = {}
        self.reflections: List[dict[str, Any]] = []
        self.meta: Dict[str, Any] = {}

    def set_plan(self, plan: Plan) -> None:
        self.plan = plan
        self.task = plan.task

    def set_docs(self, subtask_id: str, docs: List[Document]) -> None:
        self.docs[subtask_id] = docs

    def append_docs(self, subtask_id: str, docs: List[Document]) -> None:
        cur = self.docs.get(subtask_id, [])
        seen = {d.id for d in cur}
        for d in docs:
            if d.id not in seen:
                seen.add(d.id)
                cur.append(d)
        self.docs[subtask_id] = cur

    def set_summary(self, subtask_id: str, text: str) -> None:
        self.summaries[subtask_id] = text

    def record_reflection(self, subtask_id: str, result: ReflectionResult) -> None:
        self.reflections.append(
            {
                "subtask_id": subtask_id,
                "sufficient": result.sufficient,
                "need_more": result.need_more,
                "rationale": result.rationale,
            }
        )
