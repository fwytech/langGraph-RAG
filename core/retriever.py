import os
import json
from typing import Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app_utils.config import get_settings, Settings
from app_utils.helpers import to_chroma_collection_name, to_openai_tool_name


def _kb_vectorstore_path(kb_name: str) -> str:
    """
    计算指定知识库的向量库持久化目录。
    """
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", kb_name, "vectorstore")


def create_retriever(kb_name: str, settings: Optional[Settings] = None):
    """
    创建并返回基于 Chroma 的检索器。
    """
    settings = settings or get_settings()
    vectorstore = Chroma(
        collection_name=to_chroma_collection_name(kb_name),
        embedding_function=OpenAIEmbeddings(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.embedding_model,
        ),
        persist_directory=_kb_vectorstore_path(kb_name),
    )
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.15},
    )


def create_tool(kb_name: str, retriever) -> Any:
    class KBQuery(BaseModel):
        query: str = Field(description="查询字符串")

    def _kb_func(query: str) -> str:
        payload = {f"已知内容 {i + 1}": doc.page_content for i, doc in enumerate(retriever.invoke(query))}
        return json.dumps(payload, ensure_ascii=False)
    safe_name = to_openai_tool_name(kb_name)
    return StructuredTool(
        name=f"{safe_name}_knowledge_base_tool",
        description=f"search and return information about {kb_name}",
        args_schema=KBQuery,
        func=_kb_func,
    )
