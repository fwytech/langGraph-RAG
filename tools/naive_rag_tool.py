import os
import json
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 从utils模块中导入get_embedding_model函数，用于获取嵌入模型
from utils import get_embedding_model
from app_utils.helpers import to_chroma_collection_name, to_openai_tool_name

# 定义一个函数get_naive_rag_tool，接收一个参数vectorstore_name，表示向量存储的名称
def get_naive_rag_tool(vectorstore_name):

    class KBQuery(BaseModel):
        query: str = Field(description="查询字符串")

    def _kb_func(query: str) -> str:
        """
        单次查询时按需实例化向量库，避免长时间持有持久连接导致文件锁。
        """
        from langchain_chroma import Chroma
        vectorstore = Chroma(
            collection_name=to_chroma_collection_name(vectorstore_name),
            embedding_function=get_embedding_model(platform_type="OpenAI"),
            persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name, "vectorstore"),
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.15,
            }
        )
        payload = {
            f"已知内容 {inum+1}": doc.page_content.replace(doc.metadata.get("source", "") + "\n\n", "")
            for inum, doc in enumerate(retriever.invoke(query))
        }
        return json.dumps(payload, ensure_ascii=False)

    safe_name = to_openai_tool_name(vectorstore_name)
    return StructuredTool(
        name=f"{safe_name}_knowledge_base_tool",
        description=f"search and return information about {vectorstore_name}",
        args_schema=KBQuery,
        func=_kb_func,
    )

# 如果当前模块是主程序入口
if __name__ == "__main__":
    # 调用get_naive_rag_tool函数，传入"personal_information"作为参数，获取检索工具
    retriever_tool = get_naive_rag_tool("personal_information")
    # 打印检索工具对查询"刘虔"的响应结果
    print(retriever_tool.invoke("刘虔"))
