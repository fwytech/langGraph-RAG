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
    """
    settings = settings or get_settings()

    tool_node = ToolNode(tools)

    def call_model(state):
        """代理节点：绑定工具并生成回复"""
        llm = LLMClient(settings)
        llm_with_tools = llm.llm.bind_tools(tools)
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")

    return workflow.compile(checkpointer=MemorySaver())
