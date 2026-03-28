from __future__ import annotations

import logging
from typing import List

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from reflection.prompts import REFLECTION_PROMPT
from schemas.models import Document, ReflectionResult, SubTask
from utils.json_extract import extract_json_object

logger = logging.getLogger(__name__)


def _format_docs(docs: List[Document]) -> str:
    lines: list[str] = []
    for i, d in enumerate(docs):
        snippet = d.content[:1200] + ("…" if len(d.content) > 1200 else "")
        lines.append(f"[{i}] 标题:{d.title or '无'} URL:{d.url or '无'}\n{snippet}\n")
    return "\n".join(lines) if lines else "(无检索结果)"


class Reflector:
    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    def evaluate(
        self,
        subtask: SubTask,
        docs: List[Document],
        *,
        anchor_task: str = "",
    ) -> ReflectionResult:
        prompt = REFLECTION_PROMPT.format(
            anchor_task=anchor_task or subtask.content,
            subtask=subtask.content,
            search_results=_format_docs(docs),
        )
        raw = self._llm.invoke([HumanMessage(content=prompt)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            data = extract_json_object(text)
        except Exception:
            logger.warning("reflection JSON parse failed, using heuristic fallback")
            return self._fallback(docs)
        return self._result_from_json(data, docs)

    @staticmethod
    def _result_from_json(data: dict, docs: List[Document]) -> ReflectionResult:
        filt = data.get("filtered_docs")
        if not isinstance(filt, list):
            filt = list(range(len(docs)))
        else:
            parsed: list[int] = []
            for x in filt:
                if isinstance(x, int):
                    parsed.append(x)
                elif isinstance(x, str) and x.strip().isdigit():
                    parsed.append(int(x.strip()))
            filt = [i for i in parsed if 0 <= i < len(docs)]
            if not filt and docs:
                filt = list(range(len(docs)))

        return ReflectionResult(
            sufficient=bool(data.get("sufficient", False)),
            need_more=bool(data.get("need_more", True)),
            new_queries=[str(q) for q in (data.get("new_queries") or []) if str(q).strip()],
            filtered_docs=filt,
            adequacy_score=int(data.get("adequacy_score", 3)),
            consistency_score=int(data.get("consistency_score", 3)),
            relevance_score=int(data.get("relevance_score", 3)),
            rationale=str(data.get("rationale", "")),
        )

    @staticmethod
    def _fallback(docs: List[Document]) -> ReflectionResult:
        if not docs:
            return ReflectionResult(
                sufficient=False,
                need_more=True,
                new_queries=[],
                filtered_docs=[],
                rationale="无检索结果或解析失败",
            )
        return ReflectionResult(
            sufficient=True,
            need_more=False,
            new_queries=[],
            filtered_docs=list(range(len(docs))),
            rationale="解析失败，默认保留全部结果",
        )
