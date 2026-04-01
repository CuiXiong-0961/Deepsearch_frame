from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from orchestrator.langgraph_runner import run_deep_research_graph
from utils.redis_store import RedisStore, load_redis_config
from utils.subtask_order import sort_subtask_dicts_ordered


@dataclass
class RunState:
    log_path: Optional[str] = None
    steps_done: int = 0
    steps_min_total: Optional[int] = None
    current_step: str = ""
    running: bool = False
    run_id: Optional[str] = None
    prev_run_id: Optional[str] = None
    subtasks: List[Dict[str, Any]] = None  # type: ignore[assignment]
    done_ids: List[str] = None  # type: ignore[assignment]
    # 当前正在执行的子任务 id（并行时多个，用于展示「s1 + s2」）
    running_ids: List[str] = None  # type: ignore[assignment]


def _records_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "logger" / "records"


def _feedback_dir() -> Path:
    p = Path(__file__).resolve().parents[1] / "logger" / "feedback"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _try_delete_run_logs(run_id: str) -> None:
    """
    删除本轮 run 的日志文件（尽力而为）。

    - 合并日志：logger/records/*_merged_{run_id}.txt
    - 子任务日志：logger/subtasks/{run_id}/*.txt
    """

    base = Path(__file__).resolve().parents[1]
    merged_dir = base / "logger" / "records"
    subtask_dir = base / "logger" / "subtasks" / run_id

    try:
        for p in merged_dir.glob(f"*_merged_{run_id}.txt"):
            if p.is_file():
                p.unlink()
    except Exception:
        pass

    try:
        if subtask_dir.exists():
            for p in subtask_dir.glob("*.txt"):
                if p.is_file():
                    p.unlink()
            # 尝试删除空目录
            try:
                subtask_dir.rmdir()
            except Exception:
                pass
    except Exception:
        pass


def _format_running_line(rs: RunState) -> str:
    """并行执行时展示「当前任务：s1 + s2」；无运行中则显示状态文案。"""

    ids = rs.running_ids or []
    if ids:
        return "当前任务：" + " + ".join(sorted(ids))
    return rs.current_step or "（等待首个步骤日志…）"


def _format_status(rs: RunState) -> str:
    if not rs.running and rs.steps_done == 0:
        return "空闲。"
    total = (
        f"预计最少 {rs.steps_min_total} 步"
        if rs.steps_min_total is not None
        else "总步数：规划后可估计"
    )
    cur = _format_running_line(rs)
    prefix = "运行中" if rs.running else "已完成"
    return f"{prefix}｜{total}｜已执行 {rs.steps_done} 步｜{cur}"


def _sort_subtasks_for_ui(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    与 ``utils.subtask_order`` / runner 一致：**s1、s2、… 按数字升序**，
    避免 P0 的 s6 显示在 P1 的 s3 前面。
    """

    return sort_subtask_dicts_ordered(list(subtasks))


def _render_task_list(rs: RunState) -> str:
    """
    渲染子任务列表（按与 runner 一致的顺序逐行），已完成的追加（已完成）。
    """

    subtasks = rs.subtasks or []
    done = set(rs.done_ids or [])
    if not subtasks:
        return ""
    lines: List[str] = ["### 子任务列表（顺序：s1、s2… 按编号；与执行/合并日志一致）"]
    for st in subtasks:
        sid = str(st.get("id") or "")
        content = str(st.get("content") or "").strip()
        suffix = "（已完成）" if sid in done else ""
        lines.append(f"- {sid}. {content}{suffix}")
    return "\n".join(lines)


def _run_deep_research_in_thread(task: str, q: "queue.Queue[dict]") -> None:
    try:
        def _cb(evt: Dict[str, Any]) -> None:
            q.put({"type": "event", "event": evt})

        import asyncio

        report = asyncio.run(
            run_deep_research_graph(
                task.strip(),
                file_log=True,
                fetch_full_page=False,
                parallel=3,
                event_cb=_cb,
            )
        )
        q.put({"type": "final", "report": report})
    except Exception as e:
        q.put({"type": "error", "error": str(e)})


def submit_task(
    task: str,
    chat: List[Tuple[str, str]],
    rs: RunState,
) -> Any:
    task = (task or "").strip()
    if not task:
        yield (
            chat,
            rs,
            "请输入问题/任务。",
            gr.update(value=None, interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(interactive=True),
        )
        return

    # 运行期间锁定输入框，结束后再解锁（见各 yield 中的 task_in 更新）
    task_in_locked = gr.update(interactive=False)

    # reset rating/feedback ui for new run，同时隐藏原因与日志保留信息
    rating_ui = gr.update(value=None, interactive=False)
    reasons_ui = gr.update(visible=False, value=None, interactive=True)
    reasons_label_ui = gr.update(visible=False)
    feedback_ui = gr.update(value="", visible=False)
    task_list_ui = gr.update(value="", visible=False)
    current_task_ui = gr.update(value="", visible=False)

    # 下一问触发清理：删除上一轮 run_id 在 Redis 中记录的所有 key
    if rs and rs.run_id:
        try:
            RedisStore(load_redis_config()).cleanup_run(rs.run_id)
        except Exception:
            pass

    chat = (chat or []) + [(task, "")]
    rs = RunState(
        log_path=None,
        steps_done=0,
        steps_min_total=None,
        current_step="",
        running=True,
        prev_run_id=rs.run_id if rs else None,
        run_id=None,
        subtasks=[],
        done_ids=[],
        running_ids=[],
    )
    yield (
        chat,
        rs,
        _format_status(rs),
        rating_ui,
        reasons_label_ui,
        reasons_ui,
        feedback_ui,
        task_list_ui,
        current_task_ui,
        task_in_locked,
    )

    q: "queue.Queue[dict]" = queue.Queue()
    t = threading.Thread(target=_run_deep_research_in_thread, args=(task, q), daemon=True)
    t.start()

    report: Optional[str] = None
    error: Optional[str] = None

    while True:
        try:
            msg = q.get(timeout=0.25)
        except queue.Empty:
            if not t.is_alive():
                break
            # keep yielding status so UI shows "still running"
            yield (
                chat,
                rs,
                _format_status(rs),
                rating_ui,
                reasons_label_ui,
                reasons_ui,
                feedback_ui,
                task_list_ui,
                current_task_ui,
                task_in_locked,
            )
            continue

        mtype = msg.get("type")
        if mtype == "event":
            evt = msg.get("event") or {}
            et = evt.get("type")
            if et == "plan_ready":
                rs.run_id = evt.get("run_id")
                rs.subtasks = _sort_subtasks_for_ui(list(evt.get("subtasks") or []))
                # 估计总步数：plan + 波次执行（近似每子任务 3 步）+ 每波可能 update_plan + outline+report
                n = len(rs.subtasks or [])
                rs.steps_min_total = 2 + (4 * n) + 2 if n else None
                task_list_ui = gr.update(value=_render_task_list(rs), visible=True)
                current_task_ui = gr.update(
                    value="当前任务：准备开始执行子任务…", visible=True
                )
            elif et == "plan_updated":
                # 某波结束后计划可能被调整：刷新列表，已完成的仍带（已完成）
                rs.subtasks = _sort_subtasks_for_ui(list(evt.get("subtasks") or []))
                task_list_ui = gr.update(value=_render_task_list(rs), visible=True)
            elif et == "subtask_start":
                rs.current_step = f"subtask_start[{evt.get('subtask_id')}]"
                rid = str(evt.get("subtask_id") or "")
                if rid and rid not in (rs.running_ids or []):
                    rs.running_ids = list(rs.running_ids or []) + [rid]
                current_task_ui = gr.update(
                    value=_format_running_line(rs), visible=True
                )
            elif et == "subtask_done":
                sid = str(evt.get("subtask_id") or "")
                if sid and sid not in (rs.done_ids or []):
                    rs.done_ids.append(sid)
                if sid and rs.running_ids and sid in rs.running_ids:
                    rs.running_ids = [x for x in rs.running_ids if x != sid]
                rs.steps_done += 1
                task_list_ui = gr.update(value=_render_task_list(rs), visible=True)
                current_task_ui = gr.update(
                    value=_format_running_line(rs), visible=True
                )
            yield (
                chat,
                rs,
                _format_status(rs),
                rating_ui,
                reasons_label_ui,
                reasons_ui,
                feedback_ui,
                task_list_ui,
                current_task_ui,
                task_in_locked,
            )
            continue
        if mtype == "final":
            report = str(msg.get("report") or "")
            break
        if mtype == "error":
            error = str(msg.get("error") or "unknown error")
            break

        yield (
            chat,
            rs,
            _format_status(rs),
            rating_ui,
            reasons_label_ui,
            reasons_ui,
            feedback_ui,
            task_list_ui,
            current_task_ui,
            task_in_locked,
        )

    rs.running = False
    rs.running_ids = []
    rs.current_step = "都已完成" if not error else (rs.current_step or "error")

    assistant_text = report if report else f"运行失败：{error}"
    chat[-1] = (chat[-1][0], assistant_text)

    rating_ui = gr.update(value=None, interactive=True)
    # 回答结束：清空输入并解锁；当前任务显示「都已完成」
    task_in_unlocked = gr.update(value="", interactive=True)
    current_task_ui = gr.update(
        value="当前任务：都已完成" if not error else f"当前任务：结束（错误）",
        visible=True,
    )
    yield (
        chat,
        rs,
        _format_status(rs),
        rating_ui,
        reasons_label_ui,
        reasons_ui,
        feedback_ui,
        task_list_ui,
        current_task_ui,
        task_in_unlocked,
    )


REASONS = ["搜索补全不足", "总结异常/跑题", "引用证据不足", "信息过时或不可靠", "格式不符合要求", "其他"]


def on_rating_change(
    rating: Optional[str],
    bad_reason: Optional[str],
    rs: RunState,
    chat: List[Tuple[str, str]],
) -> Any:
    rating = (rating or "").strip()
    log_path = Path(rs.log_path) if rs.log_path else None

    if rating == "好":
        # 删除本次记录（按 run_id 清理合并日志 + 子任务日志）
        if rs and rs.run_id:
            _try_delete_run_logs(rs.run_id)
        # 同时清理本轮 Redis
        if rs and rs.run_id:
            try:
                RedisStore(load_redis_config()).cleanup_run(rs.run_id)
            except Exception:
                pass
        return (
            rs,
            gr.update(value="好", interactive=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="已标记为好：已删除本次步骤日志（并清理 Redis 缓存）。", visible=True),
        )

    if rating == "坏":
        # 展示原因下拉框；若已选择原因则落盘反馈（保留日志）
        status = "已标记为坏：请选择原因（会保留本次步骤日志）。"
        rating_ui = gr.update(value="坏", interactive=True)
        reasons_label_ui = gr.update(visible=True)
        reasons_ui = gr.update(visible=True, interactive=True)
        feedback_ui = gr.update(value=status, visible=True)
        if bad_reason:
            fb = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "rating": "bad",
                "reason": bad_reason,
                "log_path": str(log_path) if log_path else "",
                "last_user": chat[-1][0] if chat else "",
            }
            out = _feedback_dir() / f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out.write_text(json.dumps(fb, ensure_ascii=False, indent=2), encoding="utf-8")
            status = f"已保存反馈：{out.name}（日志已保留）。"
            # 一旦选择原因：锁死评分与原因选择，避免二次改动导致日志/反馈不一致
            rating_ui = gr.update(value="坏", interactive=False)
            reasons_ui = gr.update(visible=True, interactive=False, value=bad_reason)
            feedback_ui = gr.update(value=status, visible=True)
        return rs, rating_ui, reasons_label_ui, reasons_ui, feedback_ui

    return (
        rs,
        gr.update(value=None, interactive=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="请选择评分：好 / 坏。", visible=True),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Deepsearch 评估与反馈") as demo:
        gr.Markdown("### Deepsearch 对话 + 进度提示 + 人工评分\n- 运行过程中会提示执行到第几步；完成后可选择 **好/坏**。")

        rs = gr.State(RunState())
        chat = gr.State([])  # List[Tuple[user, assistant]]

        chatbot = gr.Chatbot(label="对话框", height=520)
        status = gr.Markdown(value="空闲。")
        task_list = gr.Markdown(value="", visible=False)
        current_task = gr.Markdown(value="", visible=False)

        with gr.Row():
            task_in = gr.Textbox(label="输入", placeholder="输入你的研究问题/任务，然后回车或点击发送", scale=6)
            send = gr.Button("发送", variant="primary", scale=1)

        with gr.Row():
            rating = gr.Radio(choices=["好", "坏"], label="评分（输出质量）", value=None, interactive=False)

        bad_reason_label = gr.Markdown("### 选择原因（坏）", visible=False)
        bad_reason = gr.Dropdown(choices=REASONS, label="原因", value=None, visible=False)
        feedback_status = gr.Markdown(value="", visible=False)

        def _sync_chat(chat_pairs: List[Tuple[str, str]]):
            return chat_pairs

        # submit
        send.click(
            submit_task,
            inputs=[task_in, chat, rs],
            outputs=[
                chatbot,
                rs,
                status,
                rating,
                bad_reason_label,
                bad_reason,
                feedback_status,
                task_list,
                current_task,
                task_in,
            ],
        )

        task_in.submit(
            submit_task,
            inputs=[task_in, chat, rs],
            outputs=[
                chatbot,
                rs,
                status,
                rating,
                bad_reason_label,
                bad_reason,
                feedback_status,
                task_list,
                current_task,
                task_in,
            ],
        )

        # keep chatbot synced with state (mainly for first paint)
        demo.load(_sync_chat, inputs=[chat], outputs=[chatbot])

        # rating change
        rating.change(
            on_rating_change,
            inputs=[rating, bad_reason, rs, chat],
            outputs=[rs, rating, bad_reason_label, bad_reason, feedback_status],
        )
        bad_reason.change(
            on_rating_change,
            inputs=[rating, bad_reason, rs, chat],
            outputs=[rs, rating, bad_reason_label, bad_reason, feedback_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()

