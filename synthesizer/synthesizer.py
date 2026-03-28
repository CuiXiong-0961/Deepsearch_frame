from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from schemas.models import Document


class Synthesizer:
    """子任务摘要 → 大纲 → 最终报告（分层合成）。"""

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    def summarize_subtask(self, subtask: str, docs: List[Document]) -> str:
        parts = []
        for i, d in enumerate(docs):
            parts.append(f"[证据{i+1}] {d.title}\n{d.content[:2000]}")
        body = "\n\n".join(parts) if parts else "(无材料)"
        prompt = f"""你是研究助理。根据下列检索材料，用中文写一段 500-800 字的子任务小结。
子任务：{subtask}

要求：
- 归纳事实与数据，保留可核对性（可带简短来源说明如「来源：网页摘要」）
- 不写空洞套话；若材料不足请明确写出「材料未覆盖：…」

材料：
{body}
"""
        raw = self._llm.invoke([HumanMessage(content=prompt)])
        return raw.content if hasattr(raw, "content") else str(raw)

    def generate_outline(self, task: str, summaries: List[str]) -> str:
        joined = "\n---\n".join(f"【块{i+1}】\n{s}" for i, s in enumerate(summaries))
        prompt = f"""用户研究任务：{task}

下列是各子任务的小结（按研究顺序）：
{joined}

请输出一份中文报告大纲，包含一级标题、二级标题，每节用一句话说明要点。使用 Markdown 编号格式。"""
        raw = self._llm.invoke([HumanMessage(content=prompt)])
        return raw.content if hasattr(raw, "content") else str(raw)

    def generate_report(self, outline: str, summaries: List[str]) -> str:
        joined = "\n---\n".join(summaries)
        prompt = f"""根据大纲与素材撰写完整中文研究报告。结构遵循大纲，正文连贯、有结论与局限性说明。

【大纲】
{outline}

【素材汇总】
{joined}

要求：分节撰写；重要论断尽量对应素材；若素材冲突请并列说明；文末可加「参考要点」列出关键事实。"""
        raw = self._llm.invoke([HumanMessage(content=prompt)])
        return raw.content if hasattr(raw, "content") else str(raw)
