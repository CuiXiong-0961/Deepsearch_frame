"""
子任务级日志与合并工具。

目的：
- 并行执行时避免多个线程/任务写同一个日志文件导致乱序或竞争。
- 为每个 subtask 生成独立日志文件（基于现有 SessionStepLogger 格式）。
- 在 run 结束后按 plan 中的 subtask 顺序合并为“总日志”，落到 logger/records/。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from logger.recorder import SessionStepLogger


class SubtaskStepLogger(SessionStepLogger):
    """
    每个子任务一个 logger。

    与 SessionStepLogger 的差异：
    - 支持指定 filename，便于带上 run_id/subtask_id，且避免同秒并发冲突。
    """

    def __init__(self, *, records_dir: Path, filename: str):
        # 方法注释：复用父类初始化，但覆盖文件名规则
        self._dir = Path(records_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.path = self._dir / self.filename
        self._fp = open(self.path, "w", encoding="utf-8")
        self._write_line(f"会话开始: {datetime.now().isoformat(timespec='seconds')}")
        self._write_line(f"记录文件: {self.path}")
        self._write_line("")


def default_subtask_records_dir() -> Path:
    """子任务日志目录：logger/subtasks/"""

    return Path(__file__).resolve().parent / "subtasks"


def merge_subtask_logs(
    *,
    ordered_subtask_ids: List[str],
    subtask_log_paths: dict[str, Path],
    merged_out_path: Path,
    title: Optional[str] = None,
) -> Path:
    """
    按 ordered_subtask_ids 顺序，把每个 subtask 的日志拼接成一个总日志文件。

    - 不要求子任务按完成顺序拼接；必须按 plan 顺序，便于阅读。
    - 缺失的 subtask 日志会被跳过，但会在总日志中写明。
    """

    merged_out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "=" * 88
    with open(merged_out_path, "w", encoding="utf-8") as out:
        out.write(f"合并日志开始: {datetime.now().isoformat(timespec='seconds')}\n")
        if title:
            out.write(f"标题: {title}\n")
        out.write(f"子任务数: {len(ordered_subtask_ids)}\n\n")

        for sid in ordered_subtask_ids:
            out.write(sep + "\n")
            out.write(f"[SUBTASK {sid}]\n")
            p = subtask_log_paths.get(sid)
            if p is None or not p.exists():
                out.write(f"(missing log file) subtask_id={sid}\n\n")
                continue
            out.write(f"log_file: {p}\n")
            out.write("-" * 40 + "\n")
            out.write(p.read_text(encoding="utf-8"))
            out.write("\n\n")

        out.write(sep + "\n")
        out.write(f"合并日志结束: {datetime.now().isoformat(timespec='seconds')}\n")
    return merged_out_path

