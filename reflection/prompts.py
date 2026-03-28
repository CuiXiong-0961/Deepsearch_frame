REFLECTION_PROMPT = """你是一个研究质量审查员。评估检索结果是否足以回答子任务。
**用户原始总任务（锚点）**：{anchor_task}

**当前子任务**：{subtask}

**检索到的材料**（编号从 0 开始）：
{search_results}

请从三个维度打分（1-5 整数）：
1. 充分性：信息是否足够回答该子任务？
2. 一致性：不同片段是否明显矛盾？
3. 相关性：噪音比例（5=几乎无噪音）

然后决策：
- 若充分性≥4 且相关性≥4 且无明显矛盾 → sufficient=true, need_more=false
- 若充分性<4 或缺关键事实 → need_more=true，在 new_queries 给出 1-3 个**不重复原词**的补搜查询
- 若相关性<3 → need_more=true，new_queries 尝试换角度

只输出一个 JSON（不要 markdown 围栏）：
{{
  "adequacy_score": 1,
  "consistency_score": 1,
  "relevance_score": 1,
  "sufficient": false,
  "need_more": true,
  "new_queries": ["查询1"],
  "filtered_docs": [0, 2],
  "rationale": "一句话说明"
}}

filtered_docs 填写**建议保留**的文档编号列表（基于相关性）；若都可保留则列出全部下标。"""
