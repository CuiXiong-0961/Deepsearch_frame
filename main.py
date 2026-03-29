"""
Deepsearch_frame 入口：从项目根目录运行
    python main.py
    python main.py --task "你的研究问题"
"""

from __future__ import annotations

import argparse
import logging
import sys

from orchestrator.runner import run_deep_research, run_deep_research_demo


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    for name in ("openai", "httpcore", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    p = argparse.ArgumentParser(description="Deep Research 框架（Planner→检索→反思→合成）")
    p.add_argument("--task", type=str, default="AI Infra对于大模型应用、算法做了哪些方面的支撑？需要学哪些方面的知识，怎么去入门？", help="研究任务；省略则运行内置演示问题")
    p.add_argument(
        "--no-file-log",
        action="store_true",
        help="不写 logger/records 下的步骤 txt（默认写入）",
    )
    p.add_argument(
        "--no-fetch-page",
        action="store_true",
        help="搜索后不再拉取网页正文/OCR，仅用引擎摘要（更快）",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    args = p.parse_args()
    _configure_logging(args.verbose)

    try:
        fl = not args.no_file_log
        fetch_page = not args.no_fetch_page
        if args.task.strip():
            text = run_deep_research(
                args.task.strip(),
                file_log=fl,
                fetch_full_page=fetch_page,
            )
        else:
            text = run_deep_research_demo(file_log=fl, fetch_full_page=fetch_page)
        print()
        print("=" * 60)
        print(text)
        print("=" * 60)
    except Exception:
        logging.exception("run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
