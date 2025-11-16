from typing import Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from app_utils.config import get_settings, Settings


def _build_llm(settings: Settings) -> ChatOpenAI:
    """
    构建并返回统一的 ChatOpenAI 客户端。
    要求通过环境变量安全注入 base_url 与 api_key。
    """
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model_name=settings.model,
        streaming=True,
        temperature=0.1,
    )


class LLMClient:
    """
    统一的 LLM 调用封装，提供简单的 invoke 接口。
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        settings = settings or get_settings()
        self.llm = _build_llm(settings)

    def invoke(self, messages: List[BaseMessage]) -> Any:
        """
        调用底层 LLM，返回模型响应。
        """
        return self.llm.invoke(messages)
