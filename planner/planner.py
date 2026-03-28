from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from planner.prompts import PLANNER_PROMPT, PLAN_UPDATE_PROMPT
from schemas.models import Plan, SubTask
from utils.json_extract import extract_json_object

logger = logging.getLogger(__name__)


class Planner:
    """任务 → 子任务计划；支持根据上下文更新。"""

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    def create_plan(self, task: str) -> Plan:
        raw = self._llm.invoke([HumanMessage(content=PLANNER_PROMPT.format(task=task))])
        text = raw.content if hasattr(raw, "content") else str(raw)
        data = extract_json_object(text)
        return self._plan_from_json(task, data)

    def update_plan(self, plan: Plan, context: dict[str, Any]) -> Plan:
        feedback = context.get("feedback") or str(context)
        plan_json = plan.model_dump_json(indent=2)
        prompt = PLAN_UPDATE_PROMPT.format(task=plan.task, plan_json=plan_json, feedback=feedback)
        raw = self._llm.invoke([HumanMessage(content=prompt)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            data = extract_json_object(text)
        except Exception:
            logger.warning("update_plan JSON parse failed, keeping previous plan")
            return plan
        subtasks_data = data.get("subtasks")
        if not subtasks_data:
            return plan
        subtasks = [SubTask(**s) for s in subtasks_data]
        notes = data.get("notes", plan.notes)
        return Plan(
            task=plan.task,
            subtasks=subtasks,
            version=plan.version + 1,
            notes=notes,
        )

    @staticmethod
    def _plan_from_json(task: str, data: dict[str, Any]) -> Plan:
        items = data.get("subtasks") or []
        subtasks: list[SubTask] = []
        for i, s in enumerate(items):
            sid = str(s.get("id") or f"s{i+1}")
            subtasks.append(
                SubTask(
                    id=sid,
                    content=str(s.get("content", "")).strip(),
                    priority=str(s.get("priority", "P1")),
                    status=str(s.get("status", "pending")),
                )
            )
        return Plan(task=task, subtasks=subtasks, notes=str(data.get("notes", "")))
