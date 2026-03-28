"""
Deep Research 主循环：Planner → 子任务 [检索 → 反思 → 可选补搜] → 子任务摘要 → Synthesizer 成文。
"""

from __future__ import annotations

import logging
from typing import List

from langchain_openai import ChatOpenAI

from logger.recorder import SessionStepLogger
from memory.hub import MemoryHub
from planner.planner import Planner
from reflection.reflector import Reflector
from retrievers.web import WebRetriever
from schemas.models import Document, Plan, SubTask
from synthesizer.synthesizer import Synthesizer
from tools.web_search import web_search

logger = logging.getLogger(__name__)

# 预算锚定（reflection README）
MAX_GLOBAL_SEARCHES = 30
MAX_SEARCHES_PER_SUBTASK = 8
MAX_EXTRA_RETRIEVAL_ROUNDS = 2  # 除首轮外最多补搜 2 次


def _priority_rank(p: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2}.get(p.strip().upper(), 1)


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen: set[str] = set()
    out: List[Document] = []
    for d in docs:
        if d.id not in seen:
            seen.add(d.id)
            out.append(d)
    return out


def _pick_docs(refl_filtered: List[int], pool: List[Document]) -> List[Document]:
    if not pool:
        return []
    if not refl_filtered:
        return pool
    picked = [pool[i] for i in refl_filtered if 0 <= i < len(pool)]
    return picked if picked else pool


def _sort_subtasks(plan: Plan) -> List[SubTask]:
    return sorted(plan.subtasks, key=lambda s: (_priority_rank(s.priority), s.id))


def _docs_snapshot(docs: List[Document], preview: int = 500) -> list[dict]:
    return [
        {
            "id": d.id,
            "title": d.title,
            "url": d.url,
            "source": d.source,
            "content_preview": d.content[:preview] + ("…" if len(d.content) > preview else ""),
        }
        for d in docs
    ]


def run_deep_research(
    task: str,
    *,
    llm: ChatOpenAI | None = None,
    web_retriever: WebRetriever | None = None,
    step_logger: SessionStepLogger | None = None,
    file_log: bool = True,
) -> str:
    """
    执行完整深度检索研究，返回最终报告 Markdown 正文。

    file_log 为 True 时，在 ``logger/records/`` 下创建按时间命名的 txt，
    记录各步骤输入输出；也可传入已构造的 ``step_logger``。
    """
    from utils.my_llm import QwenModel, LLMVendor, get_llm

    own_logger = step_logger is None and file_log
    step_logger = step_logger or (SessionStepLogger() if file_log else None)

    llm = llm or get_llm(LLMVendor.QWEN, QwenModel.QWEN_FLASH)
    web_retriever = web_retriever or WebRetriever()

    memory = MemoryHub()
    memory.task = task

    planner = Planner(llm)
    reflector = Reflector(llm)
    synthesizer = Synthesizer(llm)

    try:
        if step_logger:
            step_logger.log_step("00_task_input", {"task": task}, {"note": "进入主流程"})

        plan = planner.create_plan(task)
        memory.set_plan(plan)
        logger.info("plan: %s subtasks", len(plan.subtasks))
        if step_logger:
            step_logger.log_step(
                "01_planner.create_plan",
                {"task": task},
                plan.model_dump(),
            )

        global_searches = 0
        ordered = _sort_subtasks(plan)

        for st in ordered:
            if global_searches >= MAX_GLOBAL_SEARCHES:
                logger.warning("达到全局检索上限，跳过剩余子任务")
                break

            memory.current_subtask_id = st.id
            query = st.content.strip()
            all_docs: List[Document] = []
            extra_done = 0
            searches_this = 0
            last_refl = None
            round_idx = 0

            while searches_this < MAX_SEARCHES_PER_SUBTASK and global_searches < MAX_GLOBAL_SEARCHES:
                global_searches += 1
                searches_this += 1
                round_idx += 1
                new_docs, audit = web_search(query, max_results=5, retriever=web_retriever)
                logger.info("search audit: %s", audit)
                all_docs = _dedupe_docs(all_docs + new_docs)
                if step_logger:
                    step_logger.log_step(
                        f"02_search[{st.id}]#{round_idx}",
                        {"subtask_id": st.id, "subtask": st.content, "query": query},
                        {"audit": audit, "new_docs": _docs_snapshot(new_docs), "merged_doc_count": len(all_docs)},
                    )

                last_refl = reflector.evaluate(st, all_docs, anchor_task=task)
                memory.record_reflection(st.id, last_refl)
                if step_logger:
                    step_logger.log_step(
                        f"03_reflection[{st.id}]#{round_idx}",
                        {
                            "anchor_task": task,
                            "subtask": st.model_dump(),
                            "docs_for_eval": _docs_snapshot(all_docs),
                        },
                        last_refl.model_dump(),
                    )

                if last_refl.sufficient and not last_refl.need_more:
                    break
                if extra_done >= MAX_EXTRA_RETRIEVAL_ROUNDS:
                    break
                if last_refl.need_more and last_refl.new_queries:
                    query = last_refl.new_queries[0]
                    extra_done += 1
                else:
                    break

            final_pool = all_docs
            if last_refl is not None:
                final_pool = _pick_docs(last_refl.filtered_docs, all_docs)

            summary = synthesizer.summarize_subtask(st.content, final_pool)
            memory.set_docs(st.id, final_pool)
            memory.set_summary(st.id, summary)
            logger.info("subtask %s summary len=%s", st.id, len(summary))
            if step_logger:
                step_logger.log_step(
                    f"04_summarize_subtask[{st.id}]",
                    {
                        "subtask": st.content,
                        "docs_used": _docs_snapshot(final_pool),
                    },
                    {"summary": summary},
                )

        summaries = [memory.summaries[s.id] for s in ordered if s.id in memory.summaries]
        if not summaries:
            out = "未能生成有效摘要：请检查网络检索与 API 配置。"
            if step_logger:
                step_logger.log_step("99_abort", {"reason": "no_summaries"}, out)
            if own_logger and step_logger is not None:
                logger.info("步骤记录文件: %s", step_logger.path)
            return out

        outline = synthesizer.generate_outline(task, summaries)
        if step_logger:
            step_logger.log_step(
                "05_synthesizer.generate_outline",
                {"task": task, "summaries": summaries},
                {"outline": outline},
            )

        report = synthesizer.generate_report(outline, summaries)
        memory.meta["outline"] = outline
        memory.meta["global_searches"] = global_searches
        if step_logger:
            step_logger.log_step(
                "06_synthesizer.generate_report",
                {"outline": outline, "summaries": summaries},
                {"report": report},
            )
        if own_logger and step_logger is not None:
            logger.info("步骤记录文件: %s", step_logger.path)
        return report
    finally:
        if own_logger and step_logger is not None:
            step_logger.close()


def run_deep_research_demo(*, file_log: bool = True) -> str:
    """默认演示问题（偏科普，便于联网检索）。"""
    q = "2024–2025 年主流大语言模型在中文场景下有哪些代表性进展？请分技术路线简述。"
    return run_deep_research(q, file_log=file_log)
