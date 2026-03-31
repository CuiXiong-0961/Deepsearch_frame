from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from logger.recorder import SessionStepLogger
from orchestrator.runner import run_deep_research


@dataclass
class RunState:
    log_path: Optional[str] = None
    steps_done: int = 0
    steps_min_total: Optional[int] = None
    current_step: str = ""
    running: bool = False


class ProgressLogger(SessionStepLogger):
    def __init__(self, records_dir: Path, q: "queue.Queue[dict]") -> None:
        super().__init__(records_dir=records_dir)
        self._q = q
        self._steps_done = 0
        self._steps_min_total: Optional[int] = None

    @property
    def steps_done(self) -> int:
        return self._steps_done

    @property
    def steps_min_total(self) -> Optional[int]:
        return self._steps_min_total

    def log_step(self, step_name: str, inputs: Any, outputs: Any) -> None:
        super().log_step(step_name, inputs, outputs)
        self._steps_done += 1

        if step_name == "01_planner.create_plan" and isinstance(outputs, dict):
            subtasks = outputs.get("subtasks")
            if isinstance(subtasks, list):
                n = len(subtasks)
                # 最少步数估计：task_input + create_plan + (search+reflection+summarize)*n + outline + report
                self._steps_min_total = 2 + (3 * n) + 2

        self._q.put(
            {
                "type": "progress",
                "step_name": step_name,
                "steps_done": self._steps_done,
                "steps_min_total": self._steps_min_total,
                "log_path": str(self.path),
            }
        )


def _records_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "logger" / "records"


def _feedback_dir() -> Path:
    p = Path(__file__).resolve().parents[1] / "logger" / "feedback"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _format_status(rs: RunState) -> str:
    if not rs.running and rs.steps_done == 0:
        return "空闲。"
    total = (
        f"预计最少 {rs.steps_min_total} 步"
        if rs.steps_min_total is not None
        else "总步数：规划后可估计"
    )
    cur = rs.current_step or "（等待首个步骤日志…）"
    prefix = "运行中" if rs.running else "已完成"
    return f"{prefix}｜{total}｜已执行 {rs.steps_done} 步｜当前：{cur}"


def _run_deep_research_in_thread(task: str, q: "queue.Queue[dict]") -> None:
    logger = ProgressLogger(records_dir=_records_dir(), q=q)
    try:
        q.put({"type": "start", "log_path": str(logger.path)})
        report = run_deep_research(
            task.strip(),
            step_logger=logger,
            file_log=False,
            fetch_full_page=False,
        )
        q.put({"type": "final", "report": report, "log_path": str(logger.path)})
    except Exception as e:
        q.put({"type": "error", "error": str(e), "log_path": str(logger.path)})
    finally:
        logger.close()


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
        )
        return

    # reset rating/feedback ui for new run
    rating_ui = gr.update(value=None, interactive=False)
    reasons_ui = gr.update(visible=False, value=None, interactive=True)
    reasons_label_ui = gr.update(visible=False)
    feedback_ui = gr.update(value="", visible=False)

    chat = (chat or []) + [(task, "")]
    rs = RunState(log_path=None, steps_done=0, steps_min_total=None, current_step="", running=True)
    yield chat, rs, _format_status(rs), rating_ui, reasons_label_ui, reasons_ui, feedback_ui

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
            yield chat, rs, _format_status(rs), rating_ui, reasons_label_ui, reasons_ui, feedback_ui
            continue

        mtype = msg.get("type")
        if mtype == "start":
            rs.log_path = msg.get("log_path")
        elif mtype == "progress":
            rs.log_path = msg.get("log_path") or rs.log_path
            rs.steps_done = int(msg.get("steps_done") or rs.steps_done)
            rs.steps_min_total = msg.get("steps_min_total", rs.steps_min_total)
            rs.current_step = str(msg.get("step_name") or rs.current_step)
        elif mtype == "final":
            rs.log_path = msg.get("log_path") or rs.log_path
            report = str(msg.get("report") or "")
            break
        elif mtype == "error":
            rs.log_path = msg.get("log_path") or rs.log_path
            error = str(msg.get("error") or "unknown error")
            break

        yield chat, rs, _format_status(rs), rating_ui, reasons_label_ui, reasons_ui, feedback_ui

    rs.running = False
    rs.current_step = rs.current_step or ("error" if error else "done")

    assistant_text = report if report else f"运行失败：{error}"
    chat[-1] = (chat[-1][0], assistant_text)

    rating_ui = gr.update(value=None, interactive=True)
    yield chat, rs, _format_status(rs), rating_ui, reasons_label_ui, reasons_ui, feedback_ui


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
        # 删除本次记录
        if log_path and log_path.exists() and log_path.is_file():
            try:
                log_path.unlink()
            except Exception:
                pass
        return (
            rs,
            gr.update(value="好", interactive=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="已标记为好：已删除本次步骤日志。", visible=True),
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
            outputs=[chatbot, rs, status, rating, bad_reason_label, bad_reason, feedback_status],
        ).then(lambda: gr.update(value=""), None, task_in)

        task_in.submit(
            submit_task,
            inputs=[task_in, chat, rs],
            outputs=[chatbot, rs, status, rating, bad_reason_label, bad_reason, feedback_status],
        ).then(lambda: gr.update(value=""), None, task_in)

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

