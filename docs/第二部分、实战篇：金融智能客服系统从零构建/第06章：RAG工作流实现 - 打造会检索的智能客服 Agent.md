# 第06章：RAG 工作流实现 - 打造会检索的智能客服 Agent

> **版本信息**
> - **LangGraph**: 1.0.3+
> - **LangChain**: 1.0.7+
> - **编写日期**: 2025-01-16
> - **作者**: LangGraph-RAG Tutorial Team

---

## 本章导读

经过前面章节的铺垫，我们已经完成了：
- ✅ 第03章：项目初始化与环境搭建
- ✅ 第04章：向量数据库构建（ChromaDB）
- ✅ 第05章：工具开发（RAG Tool + LLM Client）

现在，是时候将这些组件编排成一个完整的**智能工作流**了。这就是 LangGraph 的核心价值——让 Agent 能够**自主思考、调用工具、多轮推理**。

**本章核心问题：**
- 🤔 什么是 LangGraph？为什么不直接用 LangChain？
- 🔄 ReAct 模式是什么？如何实现 Reasoning + Acting 循环？
- 🧠 StateGraph vs Chain，状态图和链式调用有何区别？
- 🛠️ ToolNode 和 tools_condition 如何自动路由？
- 💾 MemorySaver 如何实现多轮对话记忆？
- 🎯 如何构建一个完整的 RAG Agent？

**本章将带你实现：**
- ✅ 理解 LangGraph 架构与设计哲学
- ✅ 掌握 StateGraph 的节点、边、条件路由
- ✅ 实现 ReAct Agent（`rag_workflow.py`）
- ✅ 掌握 MessagesState 的状态管理
- ✅ 理解 ToolNode 的自动工具执行
- ✅ 实现多轮对话的检查点机制

**技术栈快速预览：**

```
📦 本章技术栈
├── 🔄 工作流引擎：LangGraph (StateGraph)
├── 📦 状态管理：MessagesState (内置)
├── 🛠️ 工具节点：ToolNode (prebuilt)
├── 🔀 条件路由：tools_condition (prebuilt)
├── 💾 检查点：MemorySaver (内置)
└── 🧠 推理模式：ReAct (Reasoning + Acting)
```

---

## 1. LangGraph 核心概念

### 1.1 为什么需要 LangGraph？

#### **LangChain vs LangGraph**

```mermaid
graph TB
    subgraph "LangChain（链式调用）"
        A1[用户提问] --> B1[LLM]
        B1 --> C1[工具调用]
        C1 --> D1[LLM]
        D1 --> E1[返回结果]
    end

    subgraph "LangGraph（图式循环）"
        A2[用户提问] --> B2[Agent节点]
        B2 --> C2{需要工具?}
        C2 -->|是| D2[工具节点]
        D2 --> B2
        C2 -->|否| E2[返回结果]
        B2 -.多轮循环.-> B2
    end

    style C2 fill:#fff4e1
    style B2 fill:#e1f5ff
    style D2 fill:#f0e1ff
```

**关键区别：**

| 特性 | LangChain (Chain) | LangGraph (Graph) |
|------|-------------------|-------------------|
| **执行模式** | 线性流程（A→B→C） | 循环图（A↔B↔C） |
| **适用场景** | 简单任务（翻译、摘要） | 复杂推理（Agent、多步骤） |
| **状态管理** | 无状态或手动管理 | **内置状态（MessagesState）** |
| **循环支持** | ❌ 不支持 | ✅ **支持（核心特性）** |
| **工具调用** | 手动编排 | ✅ **自动路由（ToolNode）** |
| **多轮对话** | 需手动实现 | ✅ **检查点机制（MemorySaver）** |

---

#### **实际案例对比**

**场景：用户询问 "我想贷款100万买房，月供多少？"**

**LangChain 方式（手动编排）：**

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# 第1步：查询利率
kb_tool = get_naive_rag_tool("financial_products")
rate_result = kb_tool.invoke({"query": "房贷利率"})

# 第2步：手动解析结果
import json
rate_data = json.loads(rate_result)
rate_text = rate_data["已知内容 1"]  # 需要手动提取

# 第3步：调用 LLM 生成回答
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = f"""
已知信息：
{rate_text}

用户问题：我想贷款100万买房，月供多少？

请基于已知信息回答。
"""
response = llm.invoke(prompt)
print(response.content)

# 问题：
# ❌ 需要手动调用工具
# ❌ 需要手动解析结果
# ❌ 无法处理多轮对话
# ❌ LLM 无法自主决策是否需要更多信息
```

---

**LangGraph 方式（自动编排）：**

```python
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 定义工作流
def build_agent():
    tools = [get_naive_rag_tool("financial_products")]
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

    def call_model(state):
        return {"messages": [llm.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")

    return workflow.compile(checkpointer=MemorySaver())

# 使用
agent = build_agent()
response = agent.invoke({
    "messages": [{"role": "user", "content": "我想贷款100万买房，月供多少？"}]
})

# 自动执行流程：
# 1. Agent 识别需要查询利率
# 2. 自动调用 kb_tool("房贷利率")
# 3. 获取结果后，Agent 发现还需要贷款年限信息
# 4. 反问用户："请问您的贷款年限是多少年？"
# 5. 用户回答后，继续计算月供

# 优势：
# ✅ LLM 自主决策何时调用工具
# ✅ 自动处理工具结果
# ✅ 支持多轮对话（检查点）
# ✅ 可循环推理直到问题解决
```

---

### 1.2 LangGraph 核心架构

#### **三大组件：**

```
┌─────────────────────────────────────────────────┐
│            LangGraph 核心架构                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. StateGraph（状态图）                        │
│     - 定义节点（Node）：执行单元                │
│     - 定义边（Edge）：执行顺序                  │
│     - 定义条件边（Conditional Edge）：路由逻辑  │
│                                                 │
│  2. MessagesState（消息状态）                   │
│     - 存储对话历史                              │
│     - 自动合并消息                              │
│     - 提供给所有节点                            │
│                                                 │
│  3. Checkpointer（检查点）                      │
│     - 保存每一步的状态                          │
│     - 支持中断和恢复                            │
│     - 实现多轮对话                              │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

#### **StateGraph 节点与边**

```mermaid
graph LR
    START([START]) --> Agent[Agent节点]
    Agent --> Decide{tools_condition}
    Decide -->|需要工具| Tools[Tools节点]
    Decide -->|不需要工具| END([END])
    Tools --> Agent

    style Agent fill:#e1f5ff
    style Tools fill:#fff4e1
    style Decide fill:#f0e1ff
```

**节点类型：**

1. **普通节点（Node）**
   - 执行函数：`def node_func(state) -> dict`
   - 返回值：更新状态的字典
   - 示例：`agent` 节点、`tools` 节点

2. **条件节点（Conditional Edge）**
   - 路由函数：`def router(state) -> str`
   - 返回值：下一个节点名称
   - 示例：`tools_condition`（判断是否需要调用工具）

3. **特殊节点**
   - `START`：工作流入口
   - `END`：工作流出口

---

### 1.3 MessagesState 深度解析

#### **为什么需要 MessagesState？**

在 LangGraph 0.x 时代，我们需要手动定义状态：

```python
# 0.x 时代（手动定义，约30行代码）
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """手动定义状态结构"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 其他字段...

# 问题：
# ❌ 需要理解 Annotated 和 add_messages
# ❌ 需要手动定义合并逻辑
# ❌ 代码冗长
```

---

**LangGraph 1.x（内置 MessagesState）：**

```python
# 1.x 时代（内置，1行代码）
from langgraph.graph import MessagesState

workflow = StateGraph(MessagesState)

# 优势：
# ✅ 开箱即用
# ✅ 自动处理消息合并
# ✅ 零学习成本
```

---

#### **MessagesState 的自动合并逻辑**

```python
# 初始状态
state = {"messages": [
    HumanMessage(content="房贷利率是多少？")
]}

# 节点1返回
return {"messages": [
    AIMessage(content="让我查询一下...", tool_calls=[...])
]}

# 自动合并后的状态
state = {"messages": [
    HumanMessage(content="房贷利率是多少？"),
    AIMessage(content="让我查询一下...", tool_calls=[...])
]}

# 节点2返回
return {"messages": [
    ToolMessage(content='{"已知内容 1": "..."}', tool_call_id="123")
]}

# 再次自动合并
state = {"messages": [
    HumanMessage(content="房贷利率是多少？"),
    AIMessage(content="让我查询一下...", tool_calls=[...]),
    ToolMessage(content='{"已知内容 1": "..."}', tool_call_id="123")
]}
```

**关键特性：**
- ✅ **自动追加**：新消息自动添加到 `messages` 列表末尾
- ✅ **类型安全**：只接受 `BaseMessage` 子类
- ✅ **不可变性**：每次返回新状态，不修改原状态

---

## 2. ReAct 模式详解

### 2.1 什么是 ReAct？

**ReAct** = **Rea**soning（推理）+ **Act**ing（行动）

这是一种让 LLM 能够**交替进行思考和行动**的推理模式，由 Google 在 2022 年提出。

#### **传统 Prompt vs ReAct**

**传统方式（一次性生成）：**

```
用户：我想贷款100万买房，月供多少？

LLM：根据经验，假设利率4.5%、期限30年，月供约为5000元左右。

问题：
❌ 假设的利率可能不准确（幻觉）
❌ 没有查询实际政策
❌ 用户可能对结果不信任
```

---

**ReAct 方式（思考→行动→观察 循环）：**

```
用户：我想贷款100万买房，月供多少？

【第1轮】
Thought（思考）：需要知道当前的房贷利率，我应该查询知识库。
Action（行动）：调用工具 knowledge_base_search("房贷利率")
Observation（观察）：获取结果 "个人住房贷款利率为LPR+0.5%，当前LPR为3.65%，实际利率为4.15%"

【第2轮】
Thought（思考）：我知道利率了，但还需要贷款年限。用户没有说明，我应该询问。
Action（行动）：生成回复 "请问您的贷款年限是多少年？"
Observation（观察）：等待用户回答

【用户回答】：30年

【第3轮】
Thought（思考）：现在我有了所有信息：本金100万、利率4.15%、年限30年，可以计算了。
Action（行动）：计算月供 = 1000000 * 0.04167% * (1 + 0.04167%)^360 / ((1 + 0.04167%)^360 - 1) = 4859.51元
Observation（观察）：计算完成

【第4轮】
Thought（思考）：我已经得到了答案，可以回复用户了。
Action（行动）：生成最终回复
Final Answer：根据我行最新政策，贷款100万、年限30年，月供约为4859.51元。
```

**优势：**
- ✅ **可解释性**：每一步推理清晰可见
- ✅ **准确性**：基于实际查询，减少幻觉
- ✅ **灵活性**：可根据情况调整策略
- ✅ **可追溯**：用户可看到推理过程

---

### 2.2 LangGraph 中的 ReAct 实现

在 LangGraph 中，ReAct 模式通过**节点循环**实现：

```python
┌─────────────────────────────────────────────────┐
│               ReAct 循环结构                     │
├─────────────────────────────────────────────────┤
│                                                 │
│  Agent 节点（Thought + Action）                 │
│    ├─ 分析当前状态                              │
│    ├─ 决定：需要工具 or 直接回答？             │
│    └─ 生成：tool_calls or 最终回答             │
│                                                 │
│            ↓                                    │
│                                                 │
│  Conditional Edge（路由）                       │
│    └─ 判断：有 tool_calls？                     │
│                                                 │
│       ├─ 是 → Tools 节点                        │
│       └─ 否 → END                               │
│                                                 │
│  Tools 节点（Observation）                      │
│    ├─ 执行所有工具调用                          │
│    ├─ 收集结果                                  │
│    └─ 返回 ToolMessage                          │
│                                                 │
│            ↓                                    │
│                                                 │
│  返回 Agent 节点（继续循环）                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 3. 完整实现：rag_workflow.py

### 3.1 完整源码

在 `core/rag_workflow.py` 中：

```python
from typing import List, Optional

from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage

from core.llm_client import LLMClient
from app_utils.config import get_settings, Settings


def build_rag_graph(tools: List, settings: Optional[Settings] = None):
    """
    构建并编译 RAG 工作流图，返回可调用的应用。

    Args:
        tools: 工具列表（例如：[get_naive_rag_tool("financial_products")]）
        settings: 可选的配置对象

    Returns:
        编译后的 LangGraph 应用
    """
    settings = settings or get_settings()

    # 第1步：创建工具节点
    tool_node = ToolNode(tools)

    # 第2步：定义 Agent 节点
    def call_model(state):
        """代理节点：绑定工具并生成回复"""
        llm = LLMClient(settings)
        llm_with_tools = llm.llm.bind_tools(tools)
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 第3步：构建状态图
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # 第4步：添加边
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")

    # 第5步：编译并启用检查点
    return workflow.compile(checkpointer=MemorySaver())
```

---

### 3.2 逐步解析

#### **第1步：创建工具节点（第 18 行）**

```python
tool_node = ToolNode(tools)
```

**ToolNode 的作用：**

`ToolNode` 是 LangGraph 1.x 的预构建节点，自动执行 LLM 生成的工具调用。

**底层逻辑（简化版）：**

```python
class ToolNode:
    def __init__(self, tools: List):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: MessagesState):
        """执行所有工具调用"""
        messages = state["messages"]
        last_message = messages[-1]  # 获取最后一条消息（AI 消息）

        # 提取工具调用
        tool_calls = last_message.tool_calls  # 例如：[{"name": "kb_tool", "args": {"query": "..."}}]

        # 执行所有工具
        tool_messages = []
        for tool_call in tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])

            # 创建 ToolMessage
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            ))

        return {"messages": tool_messages}
```

**示例执行流程：**

```python
# 输入状态
state = {
    "messages": [
        HumanMessage(content="房贷利率是多少？"),
        AIMessage(
            content="",
            tool_calls=[{
                "id": "call_123",
                "name": "financial_products_knowledge_base_tool",
                "args": {"query": "房贷利率"}
            }]
        )
    ]
}

# ToolNode 执行
tool_node(state)

# 返回值
{
    "messages": [
        ToolMessage(
            content='{"已知内容 1": "个人住房贷款利率为LPR+0.5%..."}',
            tool_call_id="call_123"
        )
    ]
}

# 自动合并后的状态
state = {
    "messages": [
        HumanMessage(content="房贷利率是多少？"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(content='{"已知内容 1": "..."}', tool_call_id="call_123")
    ]
}
```

---

#### **第2步：定义 Agent 节点（第 21-25 行）**

```python
def call_model(state):
    """代理节点：绑定工具并生成回复"""
    llm = LLMClient(settings)
    llm_with_tools = llm.llm.bind_tools(tools)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

**关键代码解析：**

1. **创建 LLM 客户端（第 23 行）**
   ```python
   llm = LLMClient(settings)
   ```
   - 使用第05章封装的 `LLMClient`
   - 自动加载配置（API Key、Model 等）

2. **绑定工具（第 24 行）**
   ```python
   llm_with_tools = llm.llm.bind_tools(tools)
   ```
   - `bind_tools()` 将工具定义转换为 OpenAI Function Calling 格式
   - LLM 可以识别何时需要调用工具

3. **调用 LLM（第 25 行）**
   ```python
   llm_with_tools.invoke(state["messages"])
   ```
   - 输入：完整的对话历史（`state["messages"]`）
   - 输出：
     - **有工具调用**：`AIMessage(content="", tool_calls=[...])`
     - **无工具调用**：`AIMessage(content="最终回答")`

---

**执行示例：**

```python
# 场景1：需要工具
state = {"messages": [HumanMessage(content="房贷利率是多少？")]}

call_model(state)
# 返回：
# {
#   "messages": [
#     AIMessage(
#       content="",
#       tool_calls=[{
#         "id": "call_123",
#         "name": "financial_products_knowledge_base_tool",
#         "args": {"query": "房贷利率"}
#       }]
#     )
#   ]
# }

# 场景2：不需要工具（已有足够信息）
state = {
    "messages": [
        HumanMessage(content="房贷利率是多少？"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(content='{"已知内容 1": "...利率为4.15%"}', tool_call_id="call_123")
    ]
}

call_model(state)
# 返回：
# {
#   "messages": [
#     AIMessage(content="根据我行最新政策，个人住房贷款利率为LPR+0.5%，当前实际利率为4.15%。")
#   ]
# }
```

---

#### **第3步：构建状态图（第 27-30 行）**

```python
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
```

**代码详解：**

1. **创建状态图（第 27 行）**
   ```python
   StateGraph(MessagesState)
   ```
   - `MessagesState`：使用内置的消息状态
   - 自动处理消息合并逻辑

2. **添加节点（第 28-29 行）**
   ```python
   workflow.add_node("agent", call_model)
   workflow.add_node("tools", tool_node)
   ```
   - `"agent"`：节点名称（字符串）
   - `call_model`：节点函数（可调用对象）

---

#### **第4步：添加边（第 32-34 行）**

```python
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
workflow.set_entry_point("agent")
```

**代码详解：**

1. **条件边（第 32 行）**
   ```python
   workflow.add_conditional_edges("agent", tools_condition)
   ```

   **tools_condition 的作用：**
   - 自动检查 AI 消息是否包含 `tool_calls`
   - 如果有 → 路由到 `"tools"` 节点
   - 如果没有 → 路由到 `END`（结束）

   **等价的手动实现：**
   ```python
   def my_tools_condition(state):
       messages = state["messages"]
       last_message = messages[-1]
       if hasattr(last_message, "tool_calls") and last_message.tool_calls:
           return "tools"  # 有工具调用
       else:
           return END      # 无工具调用，结束
   ```

2. **普通边（第 33 行）**
   ```python
   workflow.add_edge("tools", "agent")
   ```
   - 工具执行完成后，**始终**返回 `agent` 节点
   - 形成循环：`agent` → `tools` → `agent` → ...

3. **入口点（第 34 行）**
   ```python
   workflow.set_entry_point("agent")
   ```
   - 工作流从 `agent` 节点开始执行

---

**完整流程图：**

```mermaid
graph TB
    START([START]) --> Agent[Agent 节点]
    Agent --> Condition{tools_condition}

    Condition -->|有 tool_calls| Tools[Tools 节点]
    Condition -->|无 tool_calls| END([END])

    Tools --> Agent

    style Agent fill:#e1f5ff
    style Tools fill:#fff4e1
    style Condition fill:#f0e1ff
```

---

#### **第5步：编译与检查点（第 36-37 行）**

```python
return workflow.compile(checkpointer=MemorySaver())
```

**compile() 的作用：**
- 将图定义转换为可执行的应用
- 验证图的正确性（无死循环、无孤立节点等）
- 返回一个可调用对象

**checkpointer=MemorySaver() 的作用：**
- **保存每一步的状态**（检查点）
- **支持多轮对话**（通过 `thread_id`）
- **支持中断和恢复**

---

## 4. MemorySaver 检查点机制

### 4.1 为什么需要检查点？

**问题场景：**

```python
# 无检查点的对话
agent = build_rag_graph(tools)

# 第1轮
response1 = agent.invoke({
    "messages": [{"role": "user", "content": "房贷利率是多少？"}]
})
print(response1["messages"][-1].content)
# 输出：个人住房贷款利率为LPR+0.5%，当前实际利率为4.15%。

# 第2轮（希望追问）
response2 = agent.invoke({
    "messages": [{"role": "user", "content": "首付比例呢？"}]
})
print(response2["messages"][-1].content)
# 输出：抱歉，我不知道您在问什么。

# 问题：
# ❌ Agent 忘记了之前的对话（无记忆）
# ❌ 每次调用都是全新的会话
```

---

**解决方案：检查点 + thread_id**

```python
# 有检查点的对话
agent = build_rag_graph(tools)  # checkpointer=MemorySaver()

config = {"configurable": {"thread_id": "user_123"}}

# 第1轮
response1 = agent.invoke({
    "messages": [{"role": "user", "content": "房贷利率是多少？"}]
}, config=config)

# 第2轮（使用相同的 thread_id）
response2 = agent.invoke({
    "messages": [{"role": "user", "content": "首付比例呢？"}]
}, config=config)

print(response2["messages"][-1].content)
# 输出：首套房首付比例最低为20%，二套房为30%。

# 优势：
# ✅ Agent 记住了之前的对话
# ✅ 可以进行连续的多轮对话
```

---

### 4.2 MemorySaver 的工作原理

```python
┌─────────────────────────────────────────────────┐
│           MemorySaver 内部结构                   │
├─────────────────────────────────────────────────┤
│                                                 │
│  {                                              │
│    "user_123": {  # thread_id                   │
│      "step_0": {  # 第0步                       │
│        "messages": [                            │
│          HumanMessage(content="房贷利率是多少？")│
│        ]                                        │
│      },                                         │
│      "step_1": {  # 第1步（agent 节点执行后）   │
│        "messages": [                            │
│          HumanMessage(content="房贷利率是多少？"),│
│          AIMessage(content="", tool_calls=[...])│
│        ]                                        │
│      },                                         │
│      "step_2": {  # 第2步（tools 节点执行后）   │
│        "messages": [                            │
│          HumanMessage(...),                     │
│          AIMessage(...),                        │
│          ToolMessage(content="...")             │
│        ]                                        │
│      },                                         │
│      "step_3": {  # 第3步（最终回答）           │
│        "messages": [                            │
│          HumanMessage(...),                     │
│          AIMessage(...),                        │
│          ToolMessage(...),                      │
│          AIMessage(content="利率为4.15%")       │
│        ]                                        │
│      }                                          │
│    },                                           │
│    "user_456": { ... }  # 另一个用户的会话      │
│  }                                              │
│                                                 │
└─────────────────────────────────────────────────┘
```

**关键特性：**
- ✅ **按 thread_id 隔离**：不同用户/会话互不干扰
- ✅ **保存每一步**：可回溯任意历史状态
- ✅ **内存存储**：数据存在内存中（重启后丢失）

---

### 4.3 持久化检查点（生产环境）

在生产环境中，`MemorySaver` 不够用（重启丢失），需要使用持久化存储：

```python
# 方式1：SQLite 存储
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
agent = workflow.compile(checkpointer=checkpointer)

# 方式2：PostgreSQL 存储
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/dbname"
)
agent = workflow.compile(checkpointer=checkpointer)

# 方式3：Redis 存储（需自定义实现）
# 参考：https://langchain-ai.github.io/langgraph/how-tos/persistence_redis/
```

---

## 5. 完整执行流程演示

### 5.1 单轮对话示例

```python
from tools.naive_rag_tool import get_naive_rag_tool
from core.rag_workflow import build_rag_graph

# 第1步：创建工具和 Agent
tools = [get_naive_rag_tool("financial_products")]
agent = build_rag_graph(tools)

# 第2步：执行查询
response = agent.invoke({
    "messages": [{"role": "user", "content": "房贷利率是多少？"}]
})

# 第3步：查看结果
for msg in response["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content[:100]}...")
```

**执行过程（详细日志）：**

```
【第0步：初始状态】
messages: [
  HumanMessage(content="房贷利率是多少？")
]

【第1步：Agent 节点执行】
→ 输入：[HumanMessage(content="房贷利率是多少？")]
→ LLM 推理：用户在询问房贷利率，我需要查询知识库
→ 输出：AIMessage(content="", tool_calls=[{
    "id": "call_abc123",
    "name": "financial_products_knowledge_base_tool",
    "args": {"query": "房贷利率"}
  }])

【状态更新】
messages: [
  HumanMessage(content="房贷利率是多少？"),
  AIMessage(content="", tool_calls=[...])
]

【第2步：Conditional Edge 判断】
→ tools_condition 检测到 tool_calls
→ 路由到 "tools" 节点

【第3步：Tools 节点执行】
→ 提取 tool_calls: [{name: "financial_products_knowledge_base_tool", args: {...}}]
→ 执行工具：get_naive_rag_tool("financial_products").invoke({"query": "房贷利率"})
→ 工具返回：'{"已知内容 1": "个人住房贷款利率为LPR+0.5%，当前LPR为3.65%，实际利率为4.15%"}'
→ 输出：ToolMessage(content='{"已知内容 1": "..."}', tool_call_id="call_abc123")

【状态更新】
messages: [
  HumanMessage(content="房贷利率是多少？"),
  AIMessage(content="", tool_calls=[...]),
  ToolMessage(content='{"已知内容 1": "..."}', tool_call_id="call_abc123")
]

【第4步：返回 Agent 节点】
→ 输入：完整的 messages（包含工具结果）
→ LLM 推理：我已经获取到了利率信息，可以回答用户了
→ 输出：AIMessage(content="根据我行最新政策，个人住房贷款利率为LPR+0.5%，当前LPR为3.65%，实际利率为4.15%。首套房可享受LPR+0.3%的优惠利率。")

【状态更新】
messages: [
  HumanMessage(content="房贷利率是多少？"),
  AIMessage(content="", tool_calls=[...]),
  ToolMessage(content='{"已知内容 1": "..."}', tool_call_id="call_abc123"),
  AIMessage(content="根据我行最新政策，个人住房贷款利率为LPR+0.5%...")
]

【第5步：Conditional Edge 判断】
→ tools_condition 检测到无 tool_calls
→ 路由到 END

【工作流结束】
```

---

### 5.2 多轮对话示例

```python
from tools.naive_rag_tool import get_naive_rag_tool
from core.rag_workflow import build_rag_graph

# 创建 Agent
tools = [get_naive_rag_tool("financial_products")]
agent = build_rag_graph(tools)

# 配置会话 ID
config = {"configurable": {"thread_id": "session_001"}}

# 第1轮对话
print("用户：房贷利率是多少？")
response1 = agent.invoke({
    "messages": [{"role": "user", "content": "房贷利率是多少？"}]
}, config=config)
print(f"AI：{response1['messages'][-1].content}\n")

# 第2轮对话（追问）
print("用户：首付比例呢？")
response2 = agent.invoke({
    "messages": [{"role": "user", "content": "首付比例呢？"}]
}, config=config)
print(f"AI：{response2['messages'][-1].content}\n")

# 第3轮对话（再追问）
print("用户：二套房呢？")
response3 = agent.invoke({
    "messages": [{"role": "user", "content": "二套房呢？"}]
}, config=config)
print(f"AI：{response3['messages'][-1].content}\n")
```

**输出示例：**

```
用户：房贷利率是多少？
AI：根据我行最新政策，个人住房贷款利率为LPR+0.5%，当前实际利率为4.15%。首套房可享受LPR+0.3%的优惠利率。

用户：首付比例呢？
AI：首套房的首付比例最低为20%（商业贷款和公积金贷款均为20%）。

用户：二套房呢？
AI：二套房的首付比例为30%（商业贷款和公积金贷款均为30%），利率为LPR+1.0%。
```

**关键点：**
- ✅ 第2轮对话中，AI 理解"首付比例"指的是房贷首付（因为有上下文）
- ✅ 第3轮对话中，AI 理解"二套房"是对前面问题的延续
- ✅ 所有对话都保存在 `thread_id="session_001"` 的检查点中

---

## 6. 流式输出实现

### 6.1 为什么需要流式输出？

在第05章中，我们将 `streaming=True` 设置在 LLM 客户端中，但工作流默认是**批量输出**（等待所有步骤完成）。

**用户体验对比：**

```python
# 批量输出（默认）
response = agent.invoke({"messages": [...]})
print(response["messages"][-1].content)
# 等待3秒... 然后一次性显示完整回答

# 流式输出
for chunk in agent.stream({"messages": [...]}):
    if "messages" in chunk:
        print(chunk["messages"][-1].content, end="", flush=True)
# 逐字显示：根据... 我行... 最新... 政策...
```

---

### 6.2 实现流式输出

```python
from tools.naive_rag_tool import get_naive_rag_tool
from core.rag_workflow import build_rag_graph

# 创建 Agent
tools = [get_naive_rag_tool("financial_products")]
agent = build_rag_graph(tools)

# 流式执行
print("AI：", end="")
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "房贷利率是多少？"}]
}):
    # chunk 的结构：{"agent": {...}} 或 {"tools": {...}}
    if "agent" in chunk:
        messages = chunk["agent"]["messages"]
        if messages and hasattr(messages[-1], "content"):
            print(messages[-1].content, end="", flush=True)

print()  # 换行
```

**输出效果：**

```
AI：根据我行最新政策，个人住房贷款利率为LPR+0.5%，当前实际利率为4.15%...
```

（逐字显示，类似 ChatGPT 的效果）

---

## 7. 错误处理与重试机制

### 7.1 常见错误场景

1. **工具执行失败**
   ```python
   # 场景：向量库文件被删除
   ToolMessage(content="Error: Collection not found", tool_call_id="...")
   ```

2. **LLM API 限流**
   ```python
   # openai.error.RateLimitError: Rate limit reached
   ```

3. **工具调用参数错误**
   ```python
   # Pydantic 验证失败
   ValidationError: query field required
   ```

---

### 7.2 错误处理最佳实践

#### **方式1：在工具中处理**

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class KBQuery(BaseModel):
    query: str = Field(description="查询字符串")

def _kb_func(query: str) -> str:
    try:
        from langchain_chroma import Chroma
        vectorstore = Chroma(...)
        retriever = vectorstore.as_retriever(...)
        docs = retriever.invoke(query)

        if not docs:
            return "未找到相关信息，请尝试换个问法或联系人工客服。"

        payload = {f"已知内容 {i+1}": doc.page_content for i, doc in enumerate(docs)}
        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        # 返回友好的错误信息
        return f"查询知识库时出错：{str(e)}。请稍后重试或联系人工客服。"

tool = StructuredTool(
    name="kb_search",
    description="搜索知识库",
    args_schema=KBQuery,
    func=_kb_func
)
```

---

#### **方式2：在 Agent 节点中重试**

```python
import time
from openai import RateLimitError

def call_model(state):
    """代理节点：带重试机制"""
    llm = LLMClient(settings)
    llm_with_tools = llm.llm.bind_tools(tools)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避：1s、2s、4s
                print(f"限流错误，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                # 返回错误消息
                return {"messages": [AIMessage(content="系统繁忙，请稍后再试。")]}

        except Exception as e:
            return {"messages": [AIMessage(content=f"发生错误：{str(e)}")]}
```

---

## 8. 本章总结与下一步

### 8.1 本章收获

通过本章学习，我们完成了：

✅ **理论掌握**
- 理解 LangGraph vs LangChain 的核心区别
- 掌握 ReAct 模式（Reasoning + Acting 循环）
- 理解 StateGraph 的节点、边、条件路由
- 掌握 MessagesState 的自动合并机制
- 理解 MemorySaver 的检查点原理

✅ **技术实现**
- 实现完整的 RAG 工作流（`rag_workflow.py`）
- 掌握 ToolNode 的自动工具执行
- 掌握 tools_condition 的自动路由
- 实现多轮对话的检查点机制
- 实现流式输出

✅ **工程实践**
- 错误处理与重试机制
- 生产环境的持久化检查点
- 流式输出提升用户体验

---

### 8.2 关键技术回顾

| 技术点 | LangChain 方式 | LangGraph 方式 | 提升 |
|--------|----------------|----------------|------|
| 工具调用 | 手动编排 | **ToolNode 自动** | 代码减少 ↓80% |
| 条件路由 | 手动判断 | **tools_condition** | 零配置 ✅ |
| 状态管理 | 手动定义 | **MessagesState 内置** | 学习成本 ↓60% |
| 多轮对话 | 手动实现 | **MemorySaver 检查点** | 开箱即用 ✅ |
| 循环推理 | ❌ 不支持 | **StateGraph 循环** | 复杂推理 ✅ |

---

### 8.3 下一章预告

**第 07 章：Streamlit Web 界面开发 - 构建生产级对话系统**

我们将学习：
- 🌐 **Streamlit 核心组件**：st.chat_message、st.chat_input、st.session_state
- 💬 **聊天界面设计**：消息历史、用户输入、实时流式输出
- 🎨 **UI/UX 优化**：加载动画、错误提示、清空会话
- 🔄 **Agent 集成**：将 `rag_workflow.py` 集成到 Web 界面
- 📊 **知识库管理**：上传文档、查看列表、切换知识库
- 🛠️ **完整实现**：`rag_chat_page.py` 详解

**核心代码预览：**

```python
# 第 07 章将实现
import streamlit as st
from core.rag_workflow import build_rag_graph
from tools.naive_rag_tool import get_naive_rag_tool

# 初始化 Agent
if "agent" not in st.session_state:
    tools = [get_naive_rag_tool(st.session_state.get("selected_kb", "default"))]
    st.session_state.agent = build_rag_graph(tools)

# 聊天界面
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent（流式输出）
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in st.session_state.agent.stream({
            "messages": [{"role": "user", "content": prompt}]
        }, config={"configurable": {"thread_id": "user_123"}}):
            if "agent" in chunk:
                delta = chunk["agent"]["messages"][-1].content
                full_response += delta
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
```

---

**版本信息**
- **文档版本**: v1.0
- **最后更新**: 2025-01-16
- **适配项目版本**: langgraph-rag v0.1.0
- **作者**: LangGraph-RAG Tutorial Team
