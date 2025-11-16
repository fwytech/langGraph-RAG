import os
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()


@dataclass
class Settings:
    """
    统一的项目配置项，来源于环境变量。
    """
    base_url: str
    api_key: str
    model: str
    embedding_model: str


def get_settings() -> Settings:
    """
    读取环境变量并返回配置对象；若密钥缺失则抛出异常。
    """
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please set it via environment variables.")
    return Settings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        embedding_model=embedding_model,
    )
