# Deep Research

python + langgraph + rag + tools（web等等）+textsql 实现

## 执行流程

- Planner → 生成 Plan

- for subtask:

  -  检索（Retriever）

  - Reflection

  - if need_more → 再搜

  - summary

- Synthesizer → 报告

  

## 模块拆解

### Planner（规划器）

职责:

- Task → Subtasks（任务树）

- 动态调整

接口设计：

```
class Planner:
    def create_plan(self, task: str) -> Plan:
        pass

    def update_plan(self, plan: Plan, context: dict) -> Plan:
        pass
```

数据结构：

```
class SubTask(BaseModel):
    id: str
    content: str
    priority: str  # P0/P1/P2
    status: str    # pending/running/done

class Plan(BaseModel):
    task: str
    subtasks: List[SubTask]
```

### Retriever（检索模块）

职责（统一抽象）

- Web
- RAG
- DB

同一接口：

```
class Retriever(ABC):
    @abstractmethod
    def search(self, query: str) -> List[Document]:
        pass
class WebRetriever(Retriever): ...
class VectorRetriever(Retriever): ...
class HybridRetriever(Retriever): ...
```

### Tool Engine（工具层）

职责

- 统一工具调用（代码执行 / parsing）
- tools = [   "web_search",  "pdf_parser",  "python_executor", "text2sql" ]

### Reflection（反思模块）

职责

- 评估搜索质量
- 决定是否继续

```
class ReflectionResult(BaseModel):
    sufficient: bool
    need_more: bool
    new_queries: List[str]
    filtered_docs: List[int]
class Reflector:
    def evaluate(self, subtask: SubTask, docs: List[Document]) -> ReflectionResult:
        pass
```

### Memory Hub（核心！）

职责（解耦关键）

- 所有中间状态统一存储

### Synthesizer（整合模块）

职责

- 分层总结
- 报告生成

```
class Synthesizer:
    def summarize_subtask(self, docs: List[Document]) -> str:
        pass

    def generate_outline(self, summaries: List[str]) -> str:
        pass

    def generate_report(self, outline: str, summaries: List[str]) -> str:
        pass
```

## Orchestrator（调度器 / 状态机）

状态定义

```
class AgentState(TypedDict):
    task: str	#原始任务
    plan: Plan	#子任务分布
    current_subtask: str #现在的任务
    docs: Dict[str, List[Document]]
    summaries: Dict[str, str]
    iteration: int	#总迭代次数
```