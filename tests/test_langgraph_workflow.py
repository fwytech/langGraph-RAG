import os
import pytest
from langgraph.graph import StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app_utils.config import get_settings


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_langgraph_minimal_workflow():
    """
    构建最小 LangGraph 工作流并验证一次调用流程。
    """
    s = get_settings()

    def call_model(state):
        llm = ChatOpenAI(base_url=s.base_url, api_key=s.api_key, model_name=s.model)
        return {"messages": [llm.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.set_entry_point("agent")
    app = g.compile()

    result = app.invoke({"messages": [HumanMessage(content="请说测试通过")]})
    msg = result["messages"][-1]
    assert hasattr(msg, "content")
    assert isinstance(msg.content, str)
    assert len(msg.content.strip()) > 0
