"""
LangGraph 版本 Deep Research Runner（支持并行子任务 + Redis 外置中间结果 + 子任务日志合并）。

本模块提供与 `orchestrator/runner.py` 类似的对外入口，但实现方式不同：
- 使用 LangGraph 表达 “plan → 按波次并行执行子任务 →（每波后可选）更新计划 → 汇总合成”。
- 每波最多并行 `parallel` 个子任务；每波结束后调用 `Planner.update_plan`，根据已完成摘要调整后续子任务。
- 大对象（docs/audit/reflection 等）写入 Redis；图 state 仅保存轻量字段与 key。
- 每个子任务独立写日志；run 结束后按最终 plan 顺序合并写入 `logger/records/`。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph  # type: ignore

from logger.subtask_logs import (
    SubtaskStepLogger,
    default_subtask_records_dir,
    merge_subtask_logs,
)
from planner.planner import Planner
from reflection.reflector import Reflector
from retrievers.web import WebRetriever
from schemas.models import Document, Plan, SubTask
from synthesizer.synthesizer import Synthesizer
from tools.page_reader import PageEnrichConfig
from tools.web_search import web_search
from utils.query_compress import prepare_documents_for_analysis
from utils.redis_store import RedisStore, load_redis_config
from utils.subtask_order import sort_subtasks_ordered

logger = logging.getLogger(__name__)

# 与串行 runner 对齐的预算
MAX_GLOBAL_SEARCHES = 30
MAX_SEARCHES_PER_SUBTASK = 8
MAX_EXTRA_RETRIEVAL_ROUNDS = 2
# 防止 plan 无限循环更新
MAX_WAVES = 50


def _sort_subtasks(plan: Plan) -> List[SubTask]:
    """按 s1/s2/… 数字序优先（与串行 runner 一致），再处理非标准 id。"""

    return sort_subtasks_ordered(plan.subtasks)


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    """按 Document.id 去重，保留首次出现顺序。"""

    seen: set[str] = set()
    out: List[Document] = []
    for d in docs:
        if d.id in seen:
            continue
        seen.add(d.id)
        out.append(d)
    return out


def _pick_docs(refl_filtered: List[int], pool: List[Document]) -> List[Document]:
    """按 reflection 过滤下标挑选 docs；空则回退为全量。"""

    if not pool:
        return []
    if not refl_filtered:
        return pool
    picked = [pool[i] for i in refl_filtered if 0 <= i < len(pool)]
    return picked if picked else pool


def _docs_snapshot(docs: List[Document], preview: int = 500) -> list[dict]:
    """把 Document 列表转换为可写入日志/redis 的轻量快照（避免全文）。"""

    return [
        {
            "id": d.id,
            "title": d.title,
            "url": d.url,
            "source": d.source,
            "content_preview": d.content[:preview] + ("…" if len(d.content) > preview else ""),
            "metadata": d.metadata,
        }
        for d in docs
    ]


@dataclass
class SubtaskResult:
    """子任务执行结果（图 state 里只保留轻量字段 + redis key）。"""

    subtask_id: str
    subtask: str
    summary: str
    redis_keys: Dict[str, str]
    log_path: str


def _build_wave_feedback(
    task: str,
    batch: List[SubTask],
    wave_results: List[SubtaskResult],
    remaining_after_batch: List[SubTask],
) -> str:
    """
    构造传给 Planner.update_plan 的反馈文本：本波小结 + 尚未执行子任务摘要，供模型决定是否调整后续子任务。
    """

    lines = [
        f"用户总任务：{task}",
        "",
        f"本波已完成 {len(batch)} 个子任务，摘要如下：",
    ]
    for st, r in zip(batch, wave_results):
        lines.append(f"- [{st.id}] {st.content[:200]}")
        lines.append(f"  小结摘录：{r.summary[:600]}…")
    lines.append("")
    lines.append("当前计划中尚未执行的子任务（供你参考是否合并/拆分/增删）：")
    for st in remaining_after_batch[:20]:
        lines.append(f"- [{st.id}] {st.content[:200]}")
    if len(remaining_after_batch) > 20:
        lines.append(f"... 共 {len(remaining_after_batch)} 条，此处省略部分")
    lines.append("")
    lines.append(
        "请根据已掌握信息，在必要时调整后续子任务（可增删改、改 priority），"
        "保持任务 id 稳定或清晰；不要重复已完成的子任务。"
    )
    return "\n".join(lines)


def _subtask_fields_signature(st: SubTask) -> tuple:
    """用于对比同一 id 子任务是否被 planner 改写（content / priority / status）。"""

    return (st.content, st.priority, st.status)


def _log_parallel_wave(
    *,
    run_id: str,
    wave_idx: int,
    batch: List[SubTask],
    parallel_limit: int,
) -> None:
    """在控制台打印本波并行执行的子任务 id 列表。"""

    ids = [st.id for st in batch]
    logger.info(
        "[parallel] run_id=%s wave=%s 本波并行子任务 (%s 个，parallel 上限=%s): %s",
        run_id,
        wave_idx,
        len(batch),
        parallel_limit,
        " + ".join(ids),
    )


def _log_plan_diff_after_update(
    *,
    before: Plan,
    after: Plan,
    run_id: str,
    wave_idx: int,
) -> None:
    """
    在控制台打印 update_plan 前后差异：新增/删除/同一 id 下字段变更。

    notes 变更单独打一行，便于区分「仅说明更新」与「子任务结构变化」。
    """

    old_map = {s.id: s for s in before.subtasks}
    new_map = {s.id: s for s in after.subtasks}
    o_ids, n_ids = set(old_map), set(new_map)
    added = sorted(n_ids - o_ids)
    removed = sorted(o_ids - n_ids)
    modified = sorted(
        sid
        for sid in (o_ids & n_ids)
        if _subtask_fields_signature(old_map[sid]) != _subtask_fields_signature(new_map[sid])
    )

    logger.info(
        "[plan_update] run_id=%s wave=%s plan 版本 %s -> %s",
        run_id,
        wave_idx,
        before.version,
        after.version,
    )
    if before.notes != after.notes:
        logger.info(
            "[plan_update] notes 已更新（摘录）: %s",
            (after.notes or "")[:200] + ("…" if len(after.notes or "") > 200 else ""),
        )

    if added:
        logger.info("[plan_update] 新增子任务 id: %s", ", ".join(added))
    if removed:
        logger.info("[plan_update] 移除子任务 id: %s", ", ".join(removed))
    if modified:
        logger.info("[plan_update] 已修改子任务（content/priority/status）id: %s", ", ".join(modified))
        for sid in modified:
            o, n = old_map[sid], new_map[sid]
            if o.content != n.content:
                logger.info(
                    "[plan_update]   [%s] content: %r -> %r",
                    sid,
                    (o.content[:120] + "…") if len(o.content) > 120 else o.content,
                    (n.content[:120] + "…") if len(n.content) > 120 else n.content,
                )
            if o.priority != n.priority or o.status != n.status:
                logger.info(
                    "[plan_update]   [%s] priority/status: %s/%s -> %s/%s",
                    sid,
                    o.priority,
                    o.status,
                    n.priority,
                    n.status,
                )

    if not added and not removed and not modified and before.notes == after.notes:
        logger.info(
            "[plan_update] 子任务 id 与 content/priority/status 相对上一版无变化（可能仅内部版本号递增）",
        )


class GraphState(TypedDict, total=False):
    """LangGraph 全局 state（只存轻量信息与 plan 快照）。"""

    task: str
    run_id: str
    parallel: int
    plan_dump: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    ordered_subtask_ids: List[str]
    results: List[Dict[str, Any]]
    merged_log_path: str
    outline: str
    report: str


async def _run_one_subtask(
    *,
    task: str,
    run_id: str,
    st: SubTask,
    llm: ChatOpenAI,
    web_retriever: WebRetriever,
    redis_store: RedisStore,
    fetch_full_page: bool,
    page_cfg: PageEnrichConfig | None,
    semaphore: asyncio.Semaphore,
    event_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> SubtaskResult:
    """
    执行单个子任务链路（可并行）。

    说明：
    - 每个子任务独立 logger 文件；
    - 大对象写入 Redis，返回时仅携带 redis key。
    """

    async with semaphore:
        subtask_dir = default_subtask_records_dir() / run_id
        filename = f"{run_id}_{st.id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        step_logger = SubtaskStepLogger(records_dir=subtask_dir, filename=filename)

        reflector = Reflector(llm)
        synthesizer = Synthesizer(llm)

        query = st.content.strip()
        all_docs: List[Document] = []
        extra_done = 0
        searches_this = 0
        global_used = 0
        last_refl = None
        round_idx = 0

        docs_key = redis_store.key(run_id, "subtask", st.id, "docs")
        audit_key = redis_store.key(run_id, "subtask", st.id, "audit")
        refl_key = redis_store.key(run_id, "subtask", st.id, "reflection")
        log_key = redis_store.key(run_id, "subtask", st.id, "log_path")

        try:
            if event_cb:
                event_cb(
                    {
                        "type": "subtask_start",
                        "run_id": run_id,
                        "subtask_id": st.id,
                        "subtask": st.content,
                    }
                )
            while (
                searches_this < MAX_SEARCHES_PER_SUBTASK
                and global_used < MAX_GLOBAL_SEARCHES
            ):
                global_used += 1
                searches_this += 1
                round_idx += 1

                new_docs, audit = web_search(
                    query,
                    max_results=5,
                    retriever=web_retriever,
                    fetch_full_page=fetch_full_page,
                    page_cfg=page_cfg,
                )
                all_docs = _dedupe_docs(all_docs + new_docs)
                all_docs = prepare_documents_for_analysis(query, all_docs, top_k=10)

                step_logger.log_step(
                    f"02_search[{st.id}]#{round_idx}",
                    {"subtask_id": st.id, "subtask": st.content, "query": query},
                    {"audit": audit, "new_docs": _docs_snapshot(new_docs), "merged_doc_count": len(all_docs)},
                )

                last_refl = reflector.evaluate(st, all_docs, anchor_task=task)
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
            step_logger.log_step(
                f"04_summarize_subtask[{st.id}]",
                {"subtask": st.content, "docs_used": _docs_snapshot(final_pool)},
                {"summary": summary},
            )

            redis_store.put_json(docs_key, _docs_snapshot(final_pool, preview=2000))
            redis_store.put_json(audit_key, {"searches": searches_this})
            redis_store.put_json(refl_key, last_refl.model_dump() if last_refl else {})
            redis_store.put_json(log_key, str(step_logger.path))
            redis_store.track_keys(run_id, [docs_key, audit_key, refl_key, log_key])

            if event_cb:
                event_cb(
                    {
                        "type": "subtask_done",
                        "run_id": run_id,
                        "subtask_id": st.id,
                        "subtask": st.content,
                        "log_path": str(step_logger.path),
                    }
                )
            return SubtaskResult(
                subtask_id=st.id,
                subtask=st.content,
                summary=summary,
                redis_keys={
                    "docs": docs_key,
                    "audit": audit_key,
                    "reflection": refl_key,
                    "log_path": log_key,
                },
                log_path=str(step_logger.path),
            )
        finally:
            step_logger.close()


async def run_deep_research_graph(
    task: str,
    *,
    llm: ChatOpenAI | None = None,
    web_retriever: WebRetriever | None = None,
    file_log: bool = True,
    fetch_full_page: bool = True,
    page_cfg: PageEnrichConfig | None = None,
    parallel: int = 3,
    run_id: Optional[str] = None,
    event_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> str:
    """
    LangGraph 版深度研究入口。

    - parallel：每波并行子任务数上限（默认 3）。
    - 每波结束后调用 Planner.update_plan，根据反馈调整后续子任务（可增删改）。
    """

    from utils.my_llm import QwenModel, LLMVendor, get_llm

    if not task.strip():
        return "任务为空。"
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    llm = llm or get_llm(LLMVendor.QWEN, QwenModel.QWEN_FLASH)
    web_retriever = web_retriever or WebRetriever()
    redis_store = RedisStore(load_redis_config())

    planner = Planner(llm)
    synthesizer = Synthesizer(llm)

    sem = asyncio.Semaphore(max(1, int(parallel)))

    async def plan_node(state: GraphState) -> GraphState:
        """生成初始计划（串行节点）。"""

        plan = planner.create_plan(state["task"])
        ordered = _sort_subtasks(plan)
        if event_cb:
            event_cb(
                {
                    "type": "plan_ready",
                    "run_id": state["run_id"],
                    "task": state["task"],
                    "subtasks": [s.model_dump() for s in ordered],
                }
            )
        return {
            "plan_dump": plan.model_dump(),
            "subtasks": [s.model_dump() for s in ordered],
            "ordered_subtask_ids": [s.id for s in ordered],
        }

    async def parallel_waves_node(state: GraphState) -> GraphState:
        """
        按波次并行执行子任务；每波结束后调用 update_plan 评估是否调整后续子任务。

        该节点替代「一次性 fan-out 全部子任务」：避免无法在中间插入规划更新。
        """

        plan = Plan.model_validate(state["plan_dump"])
        task_s = state["task"]
        run_id_s = state["run_id"]
        par = max(1, int(state.get("parallel") or 3))

        completed_ids: set[str] = set()
        all_results: List[SubtaskResult] = []
        wave_idx = 0

        while wave_idx < MAX_WAVES:
            wave_idx += 1
            pending = [st for st in _sort_subtasks(plan) if st.id not in completed_ids]
            if not pending:
                break

            batch = pending[:par]
            remaining = pending[par:]

            _log_parallel_wave(
                run_id=run_id_s,
                wave_idx=wave_idx,
                batch=batch,
                parallel_limit=par,
            )

            plan_before_update = plan
            wave_results = await asyncio.gather(
                *[
                    _run_one_subtask(
                        task=task_s,
                        run_id=run_id_s,
                        st=st,
                        llm=llm,
                        web_retriever=web_retriever,
                        redis_store=redis_store,
                        fetch_full_page=fetch_full_page,
                        page_cfg=page_cfg,
                        semaphore=sem,
                        event_cb=event_cb,
                    )
                    for st in batch
                ]
            )
            for r in wave_results:
                completed_ids.add(r.subtask_id)
                all_results.append(r)

            feedback = _build_wave_feedback(task_s, batch, wave_results, remaining)
            plan = planner.update_plan(plan, {"feedback": feedback})
            _log_plan_diff_after_update(
                before=plan_before_update,
                after=plan,
                run_id=run_id_s,
                wave_idx=wave_idx,
            )

            if event_cb:
                event_cb(
                    {
                        "type": "plan_updated",
                        "run_id": run_id_s,
                        "wave_index": wave_idx,
                        "subtasks": [s.model_dump() for s in _sort_subtasks(plan)],
                        "notes": plan.notes,
                    }
                )

        if wave_idx >= MAX_WAVES and any(
            st.id not in completed_ids for st in _sort_subtasks(plan)
        ):
            logger.warning("parallel_waves_node: 达到 MAX_WAVES=%s，未执行子任务将被跳过", MAX_WAVES)

        final_ordered = _sort_subtasks(plan)
        ordered_ids = [s.id for s in final_ordered if s.id in completed_ids]

        return {
            "plan_dump": plan.model_dump(),
            "subtasks": [s.model_dump() for s in final_ordered],
            "ordered_subtask_ids": ordered_ids,
            "results": [r.__dict__ for r in all_results],
        }

    async def synthesize_node(state: GraphState) -> GraphState:
        """汇总子任务摘要、生成 outline/report，并合并子任务日志。"""

        ordered_ids = state.get("ordered_subtask_ids") or []
        results = state.get("results") or []
        by_id = {str(r.get("subtask_id")): r for r in results if isinstance(r, dict)}
        summaries = [str(by_id[sid].get("summary", "")) for sid in ordered_ids if sid in by_id]
        if not summaries:
            return {"report": "未能生成有效摘要：请检查网络检索与 API 配置。"}

        outline = synthesizer.generate_outline(state["task"], summaries)
        report = synthesizer.generate_report(outline, summaries)

        merged_path = Path(__file__).resolve().parents[1] / "logger" / "records" / (
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_merged_{state['run_id']}.txt"
        )
        if file_log:
            merge_subtask_logs(
                ordered_subtask_ids=ordered_ids,
                subtask_log_paths={sid: Path(by_id[sid]["log_path"]) for sid in ordered_ids if sid in by_id},
                merged_out_path=merged_path,
                title=f"task={state['task']}",
            )
            mk = redis_store.key(state["run_id"], "meta", "merged_log_path")
            redis_store.put_json(mk, str(merged_path))
            redis_store.track_key(state["run_id"], mk)

        return {"outline": outline, "report": report, "merged_log_path": str(merged_path)}

    g = StateGraph(GraphState)
    g.add_node("plan", plan_node)
    g.add_node("parallel_waves", parallel_waves_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "parallel_waves")
    g.add_edge("parallel_waves", "synthesize")
    g.add_edge("synthesize", END)

    graph = g.compile()

    final_state = await graph.ainvoke(
        {
            "task": task.strip(),
            "run_id": run_id,
            "parallel": int(parallel),
            "results": [],
        }
    )
    return str(final_state.get("report") or "")
