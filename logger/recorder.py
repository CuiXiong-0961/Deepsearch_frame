"""
会话级步骤记录：每一步的输入 / 输出写入纯文本文件。
文件名：YYYY-MM-DD_HH-MM-SS.txt（Windows 文件名不可含冒号，故时间用连字符分隔）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel


def _default_records_dir() -> Path:
    return Path(__file__).resolve().parent / "records"


def _to_printable(obj: Any, max_str: int = 80000) -> str:
    if obj is None:
        return ""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if isinstance(obj, (dict, list)):
        try:
            s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            s = repr(obj)
    else:
        s = str(obj)
    if len(s) > max_str:
        return s[:max_str] + f"\n... [截断，原长 {len(s)} 字符]"
    return s


class SessionStepLogger:
    """
    单次运行一个 txt 文件；按步骤追加「步骤名 → INPUT → OUTPUT」块。
    """

    def __init__(self, records_dir: Path | None = None) -> None:
        self._dir = Path(records_dir) if records_dir is not None else _default_records_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        # 与「年-月-日 时:分」对应：2026-03-28_14-30-45.txt（秒用于同分钟内区分多次运行）
        self.filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        self.path = self._dir / self.filename
        self._fp: TextIO = open(self.path, "w", encoding="utf-8")
        self._write_line(f"会话开始: {datetime.now().isoformat(timespec='seconds')}")
        self._write_line(f"记录文件: {self.path}")
        self._write_line("")

    def _write_line(self, line: str) -> None:
        self._fp.write(line + "\n")
        self._fp.flush()

    def log_step(self, step_name: str, inputs: Any, outputs: Any) -> None:
        sep = "=" * 72
        self._write_line(sep)
        self._write_line(f"步骤: {step_name}")
        self._write_line("--- INPUT ---")
        self._write_line(_to_printable(inputs))
        self._write_line("--- OUTPUT ---")
        self._write_line(_to_printable(outputs))
        self._write_line("")

    def log_plain(self, title: str, body: str) -> None:
        self._write_line("-" * 48)
        self._write_line(title)
        self._write_line(body)
        self._write_line("")

    def close(self) -> None:
        if not self._fp.closed:
            self._write_line(f"会话结束: {datetime.now().isoformat(timespec='seconds')}")
            self._fp.close()

    def __enter__(self) -> SessionStepLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
