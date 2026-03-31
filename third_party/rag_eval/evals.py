from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 让 evals.py 既能在项目根目录运行，也能在 third_party/rag_eval 下直接运行
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
sys.path.insert(0, str(_REPO_ROOT))
# 使用仓库内 vendored ragas（无需额外 pip install ragas）
_RAGAS_SRC = _REPO_ROOT / "third_party" / "ragas" / "src"
if _RAGAS_SRC.exists():
    sys.path.insert(0, str(_RAGAS_SRC))

from datasets import Dataset as HFDataset  # type: ignore

from logger.recorder import SessionStepLogger
from orchestrator.runner import run_deep_research

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness


@dataclass
class CapturedSample:
    subtask_id: str
    question: str
    contexts: List[str]
    answer: str
    meta: Dict[str, Any]


class RagasCaptureLogger(SessionStepLogger):
    """
    在不改你主流程代码的前提下，把「每步记录」转成 Ragas 可评估样本：
    - 仍写 `evals/records/*.txt`（便于回溯）
    - 从 `04_summarize_subtask[xxx]` 步骤提取：question / contexts / answer
    """

    def __init__(self, records_dir: Path) -> None:
        super().__init__(records_dir=records_dir)
        self.samples: List[CapturedSample] = []

    def log_step(self, step_name: str, inputs: Any, outputs: Any) -> None:
        super().log_step(step_name, inputs, outputs)

        if not isinstance(step_name, str) or not step_name.startswith("04_summarize_subtask["):
            return
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            return

        subtask_id = step_name.split("[", 1)[1].split("]", 1)[0].strip()
        question = str(inputs.get("subtask", "")).strip() or subtask_id
        answer = str(outputs.get("summary", "")).strip()
        docs_used = inputs.get("docs_used") or []

        contexts: List[str] = []
        context_ids: List[str] = []
        sources: List[str] = []
        urls: List[str] = []

        if isinstance(docs_used, list):
            for d in docs_used:
                if not isinstance(d, dict):
                    continue
                title = str(d.get("title") or "").strip()
                url = str(d.get("url") or "").strip()
                source = str(d.get("source") or "").strip()
                cid = str(d.get("id") or "").strip()
                text = str(d.get("content_preview") or "").strip()
                if not text:
                    continue

                ctx = text if not title else f"{title}\n{text}"
                if url:
                    ctx = f"{ctx}\nURL: {url}"
                if source:
                    ctx = f"{ctx}\nSOURCE: {source}"

                contexts.append(ctx)
                if cid:
                    context_ids.append(cid)
                if source:
                    sources.append(source)
                if url:
                    urls.append(url)

        self.samples.append(
            CapturedSample(
                subtask_id=subtask_id,
                question=question,
                contexts=contexts,
                answer=answer,
                meta={
                    "captured_from_step": step_name,
                    "context_ids": context_ids,
                    "sources": sorted(set(sources)),
                    "urls": urls[:20],
                },
            )
        )


def _ensure_dirs() -> Dict[str, Path]:
    evals_dir = _THIS_DIR / "evals"
    datasets_dir = evals_dir / "datasets"
    records_dir = evals_dir / "records"
    results_dir = evals_dir / "results"
    for p in (evals_dir, datasets_dir, records_dir, results_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "evals": evals_dir,
        "datasets": datasets_dir,
        "records": records_dir,
        "results": results_dir,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _pick_latest_record_file(records_dir: Path) -> Optional[Path]:
    if not records_dir.exists():
        return None
    files = sorted(records_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _resolve_log_path(maybe_path: Optional[str]) -> Optional[Path]:
    """
    解析用户传入的 EVAL_LOG_PATH：
    - None/空串：返回 None
    - 指向目录：返回该目录下最新的 *.txt
    - 指向文件：返回该文件
    - 不存在：返回 None
    """
    if maybe_path is None:
        return None
    s = str(maybe_path).strip().strip('"').strip("'")
    if not s:
        return None
    p = Path(s)
    if not p.exists():
        return None
    if p.is_dir():
        return _pick_latest_record_file(p)
    return p


def _safe_json_loads(text: str) -> Any:
    s = text.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


def parse_session_record_txt(path: Path) -> List[Dict[str, Any]]:
    """
    解析 `logger/records/*.txt` 的步骤日志。

    输出格式：[{ "step": str, "inputs": Any, "outputs": Any }, ...]
    """
    steps: List[Dict[str, Any]] = []
    cur_step: Optional[str] = None
    mode: Optional[str] = None  # "in" | "out" | None
    buf_in: List[str] = []
    buf_out: List[str] = []

    def flush():
        nonlocal cur_step, mode, buf_in, buf_out
        if cur_step is None:
            return
        steps.append(
            {
                "step": cur_step,
                "inputs": _safe_json_loads("".join(buf_in)),
                "outputs": _safe_json_loads("".join(buf_out)),
            }
        )
        cur_step = None
        mode = None
        buf_in = []
        buf_out = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # 新步骤开始
            if line.startswith("步骤: "):
                flush()
                cur_step = line[len("步骤: ") :].strip()
                continue
            if line.startswith("--- INPUT ---"):
                mode = "in"
                continue
            if line.startswith("--- OUTPUT ---"):
                mode = "out"
                continue
            # 步骤分隔线；如果当前已经有 step 且模式在 out，遇到分隔线通常意味着块结束
            if line.startswith("========================================================================"):
                if cur_step is not None and (buf_in or buf_out):
                    flush()
                continue

            if cur_step is None or mode is None:
                continue
            if mode == "in":
                buf_in.append(line)
            elif mode == "out":
                buf_out.append(line)

    flush()
    return steps


def samples_from_steps(steps: List[Dict[str, Any]]) -> List[CapturedSample]:
    out: List[CapturedSample] = []
    for st in steps:
        step_name = st.get("step")
        if not isinstance(step_name, str) or not step_name.startswith("04_summarize_subtask["):
            continue
        inputs = st.get("inputs")
        outputs = st.get("outputs")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue

        subtask_id = step_name.split("[", 1)[1].split("]", 1)[0].strip()
        question = str(inputs.get("subtask", "")).strip() or subtask_id
        answer = str(outputs.get("summary", "")).strip()
        docs_used = inputs.get("docs_used") or []

        contexts: List[str] = []
        context_ids: List[str] = []
        sources: List[str] = []
        urls: List[str] = []

        if isinstance(docs_used, list):
            for d in docs_used:
                if not isinstance(d, dict):
                    continue
                title = str(d.get("title") or "").strip()
                url = str(d.get("url") or "").strip()
                source = str(d.get("source") or "").strip()
                cid = str(d.get("id") or "").strip()
                text = str(d.get("content_preview") or "").strip()
                if not text:
                    continue

                ctx = text if not title else f"{title}\n{text}"
                if url:
                    ctx = f"{ctx}\nURL: {url}"
                if source:
                    ctx = f"{ctx}\nSOURCE: {source}"

                contexts.append(ctx)
                if cid:
                    context_ids.append(cid)
                if source:
                    sources.append(source)
                if url:
                    urls.append(url)

        out.append(
            CapturedSample(
                subtask_id=subtask_id,
                question=question,
                contexts=contexts,
                answer=answer,
                meta={
                    "captured_from_step": step_name,
                    "context_ids": context_ids,
                    "sources": sorted(set(sources)),
                    "urls": urls[:20],
                    "from_log": True,
                },
            )
        )
    return out


def _get_judge_llm():
    """
    Ragas 指标需要一个「评审 LLM」。
    优先复用你项目的 OpenAI 兼容配置（百炼/DeepSeek），这样不必额外配置 OPENAI_API_KEY。
    """
    try:
        from utils.my_llm import LLMVendor, QwenModel, get_llm

        return get_llm(LLMVendor.QWEN, QwenModel.QWEN_FLASH, temperature=0.0)
    except Exception:
        return None


def build_dataset_from_run(task: str) -> tuple[HFDataset, Path, Path, Path]:
    dirs = _ensure_dirs()
    cap_logger = RagasCaptureLogger(records_dir=dirs["records"])

    fetch_full_page = os.getenv("EVAL_FETCH_FULL_PAGE", "false").lower() in ("1", "true", "yes")

    _ = run_deep_research(
        task.strip(),
        step_logger=cap_logger,
        file_log=False,
        fetch_full_page=fetch_full_page,
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: List[Dict[str, Any]] = []
    for s in cap_logger.samples:
        if not s.answer or not s.contexts:
            continue
        rows.append(
            {
                "question": s.question,
                "answer": s.answer,
                "contexts": s.contexts,
                "subtask_id": s.subtask_id,
                "meta": json.dumps(s.meta, ensure_ascii=False),
            }
        )

    dataset = HFDataset.from_list(rows)

    csv_path = dirs["datasets"] / f"deepsearch_subtasks_{now}.csv"
    jsonl_path = dirs["datasets"] / f"deepsearch_subtasks_{now}.jsonl"
    score_path = dirs["results"] / f"ragas_scores_{now}.csv"

    # CSV：把 list 字段序列化，方便直接打开查看
    _write_csv(
        csv_path,
        [{**r, "contexts": json.dumps(r["contexts"], ensure_ascii=False)} for r in rows],
    )
    # JSONL：保留原生 list（Ragas/HF Dataset 直接可用）
    _write_jsonl(jsonl_path, rows)

    cap_logger.close()
    return dataset, csv_path, jsonl_path, score_path


def build_dataset_from_log(log_path: Path) -> tuple[HFDataset, Path, Path, Path]:
    dirs = _ensure_dirs()
    steps = parse_session_record_txt(log_path)
    samples = samples_from_steps(steps)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: List[Dict[str, Any]] = []
    for s in samples:
        if not s.answer or not s.contexts:
            continue
        rows.append(
            {
                "question": s.question,
                "answer": s.answer,
                "contexts": s.contexts,
                "subtask_id": s.subtask_id,
                "meta": json.dumps({**s.meta, "log_path": str(log_path)}, ensure_ascii=False),
            }
        )

    dataset = HFDataset.from_list(rows)

    csv_path = dirs["datasets"] / f"deepsearch_from_log_{now}.csv"
    jsonl_path = dirs["datasets"] / f"deepsearch_from_log_{now}.jsonl"
    score_path = dirs["results"] / f"ragas_scores_from_log_{now}.csv"

    _write_csv(
        csv_path,
        [{**r, "contexts": json.dumps(r["contexts"], ensure_ascii=False)} for r in rows],
    )
    _write_jsonl(jsonl_path, rows)
    return dataset, csv_path, jsonl_path, score_path


def main() -> None:
    # 优先：从你已有的 `logger/records/*.txt` 解析样本并评估
    log_path = _resolve_log_path(os.getenv("EVAL_LOG_PATH"))
    if log_path is None:
        log_path = _pick_latest_record_file(_REPO_ROOT / "logger" / "records")

    if log_path is not None and log_path.exists() and log_path.is_file():
        dataset, csv_path, jsonl_path, score_path = build_dataset_from_log(log_path)
    else:
        # 兜底：如果没有 log，就跑一次真实流程再评估
        task = os.getenv("EVAL_TASK") or "请简要总结 2025 年以来 RAG 评估（Ragas）常用指标与使用方式，并给出落地建议。"
        dataset, csv_path, jsonl_path, score_path = build_dataset_from_run(task)

    judge_llm = _get_judge_llm()
    metrics = [faithfulness, answer_relevancy, context_precision]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        experiment_name="deepsearch_rag_eval",
        show_progress=True,
    )

    # 写出逐行得分（不依赖 pandas）
    score_rows: List[Dict[str, Any]] = []
    dataset_rows = dataset.to_list()
    for i, scores in enumerate(result.scores):
        base = dataset_rows[i] if i < len(dataset_rows) else {}
        row = {**base, **scores}
        # contexts 仍保持为 list 时 CSV 不友好，序列化一下
        if isinstance(row.get("contexts"), list):
            row["contexts"] = json.dumps(row["contexts"], ensure_ascii=False)
        score_rows.append(row)
    _write_csv(score_path, score_rows)

    print("Dataset written to:")
    print(f"- {csv_path}")
    print(f"- {jsonl_path}")
    print("Scores written to:")
    print(f"- {score_path}")
    print()
    print("Aggregate:", result)


if __name__ == "__main__":
    main()
