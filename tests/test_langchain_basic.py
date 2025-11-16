import os
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app_utils.config import get_settings


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_langchain_basic_prompt():
    """
    测试使用 LangChain 基本调用链路：Prompt → LLM → 输出。
    """
    s = get_settings()
    llm = ChatOpenAI(base_url=s.base_url, api_key=s.api_key, model_name=s.model)
    result = llm.invoke([HumanMessage(content="用中文简短回复：测试成功")])
    assert isinstance(result.content, str)
    assert len(result.content.strip()) > 0
