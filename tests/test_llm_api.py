import os
import pytest
from langchain_core.messages import HumanMessage

from app_utils.config import get_settings
from core.llm_client import LLMClient


def _has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_api_key(), reason="OPENAI_API_KEY not set")
def test_llm_connection_and_basic_invoke():
    """
    测试 LLM API 连接与基本调用，断言返回内容非空。
    """
    settings = get_settings()
    client = LLMClient(settings)
    result = client.invoke([HumanMessage(content="请用一句话回答：你好！")])
    assert hasattr(result, "content")
    assert isinstance(result.content, str)
    assert len(result.content.strip()) > 0
