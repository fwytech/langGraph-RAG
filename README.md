# LangGraph+RAG 金融智能客服项目（基于 uv 管理）

本指南面向用户或学员，用于在 Windows 环境下使用 `uv` 管理器配置并运行当前项目。请严格按步骤操作，确保与仓库的 `pyproject.toml` 和源码保持一致。

## 一、前置条件
- 已安装 Python `>=3.12`（本项目 `.python-version` 为 `3.12`）
- 可用的终端：Windows PowerShell
- 网络可访问所需依赖源

## 二、安装与初始化（使用 uv）
1. 安装 uv（如未安装）：
   ```powershell
   pip install uv
   ```
2. 使用 uv 初始化一个空项目（项目名与本仓库一致）：
   ```powershell
   uv init langgraph-rag
   ```
3. 将生成的 `pyproject.toml` 内容替换为本仓库的 `pyproject.toml`（保持依赖为 1.x 生态）：
   ```toml
   [project]
   name = "langgraph-rag"
   version = "0.1.0"
   description = "Add your description here"
   readme = "README.md"
   requires-python = ">=3.12"
   dependencies = [
       "chardet>=5.2.0",
       "chromadb>=1.3.4",
       "langchain>=1.0.7",
       "langchain-chroma>=1.0.0",
       "langchain-community>=0.4.1",
       "langchain-openai>=1.0.3",
       "langchain-text-splitters>=1.0.0",
       "langgraph>=1.0.3",
       "openai>=2.8.0",
       "streamlit>=1.51.0",
       "streamlit-flow-component>=1.6.1",
   ]

   [dependency-groups]
   dev = [
       "pytest>=9.0.1",
   ]
   ```
4. 安装依赖（忽略 `requirements.txt`）：
   ```powershell
   uv sync
   ```

## 三、配置环境变量（使用 .env 文件）
项目通过 `app_utils/config.py` 读取 `.env` 文件（使用 `python-dotenv` 的 `load_dotenv()`）来注入 LLM 相关配置：
- `OPENAI_BASE_URL`：兼容的 OpenAI API Base URL（默认 `https://api.gptsapi.net/v1`）
- `OPENAI_API_KEY`：你的 API Key（必填）
- `OPENAI_MODEL`：对话模型名称（如 `gpt-4o-mini`）
- `OPENAI_EMBEDDING_MODEL`：嵌入模型名称（如 `text-embedding-3-small`）

在项目根目录创建 `.env`：
```env
# OpenAI 兼容接口配置
OPENAI_BASE_URL=https://api.gptsapi.net/v1
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```
> 说明：若未设置 `OPENAI_API_KEY`，程序会抛出异常（参见 `app_utils/config.py:22-35`）。

## 四、运行项目（使用 uv 运行 Streamlit）
在项目根目录执行：
```powershell
uv run streamlit run rag.py
```
- 首页提供“智能客服”和“行业知识库”入口，并在侧边栏支持“清空本地知识库”的运维操作。
- 若首次运行，请先在“行业知识库”页上传金融相关的 `.md` 文件进行索引，或使用已有 `kb` 目录中的示例内容。

## 五、常见问题
- 未配置 API Key：请检查 `.env` 是否包含 `OPENAI_API_KEY`；或在系统环境中设置同名变量。
- 依赖安装缓慢：确保使用 `uv sync`（较快）。必要时切换到稳定的镜像源。
- Windows 文件锁导致清库失败：侧边栏“清空本地知识库”采取两段式执行并包含缓存清理与重试逻辑，确保已按首页流程操作。

## 六、目录速览
- `rag.py`：应用入口与导航
- `webui/`：页面逻辑（聊天页、知识库页等）
- `tools/`：RAG 检索工具（`StructuredTool` + `Pydantic`）
- `core/`：LLM 客户端、工作流等核心模块
- `app_utils/`：配置与辅助方法（含 `.env` 读取）
- `kb/`：本地知识库目录（`files/` 与 `vectorstore/`）
- `docs/`：实战教程（7 章）

## 七、测试（可选）
如需基本连通性测试：
```powershell
uv run pytest -q
```

---

完成以上步骤后，即可在浏览器中使用“LangGraph+RAG 金融智能客服应用”。若需进一步学习，请阅读 `docs/` 目录下的教程系列。 
