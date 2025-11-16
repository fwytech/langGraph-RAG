import os
import base64
from io import BytesIO
import re
import shutil
import time


def get_img_base64(file_name: str) -> str:
    """
    将 img 目录下图片转为 base64 字符串用于 Streamlit。
    """
    image_path = os.path.join(os.path.dirname(__file__), "..", "img", file_name)
    image_path = os.path.abspath(image_path)
    with open(image_path, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base_str}"


def get_kb_names():
    """
    列举 kb 目录下的知识库名称。
    """
    kb_root = os.path.join(os.path.dirname(__file__), "..", "kb")
    kb_root = os.path.abspath(kb_root)
    if not os.path.exists(kb_root):
        os.mkdir(kb_root)
    return [f for f in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, f))]


def to_chroma_collection_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    s = re.sub(r"^[^a-zA-Z0-9]+", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+$", "", s)
    if len(s) < 3:
        base = re.sub(r"[^a-zA-Z0-9]", "", name)
        s = base if len(base) >= 3 else f"kb-{abs(hash(name))%100000}"
    return s[:512]


def to_openai_tool_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    s = re.sub(r"^([^a-zA-Z0-9_])+", "", s)
    s = re.sub(r"([^a-zA-Z0-9_])+$", "", s)
    if len(s) < 3:
        base = re.sub(r"[^a-zA-Z0-9_]", "", name)
        s = base if len(base) >= 3 else f"kb_{abs(hash(name))%100000}"
    return s[:64]


def clear_all_kb() -> int:
    """
    清空本地知识库：严格删除 kb 根目录下的所有内容并重建空目录。

    返回值为成功删除的知识库文件夹数量（不含根目录下的散落文件）。
    """
    kb_root = os.path.join(os.path.dirname(__file__), "..", "kb")
    kb_root = os.path.abspath(kb_root)
    try:
        import chromadb.api
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    def _force_remove_dir(path: str):
        for _ in range(5):
            try:
                shutil.rmtree(path)
            except Exception:
                try:
                    for root, dirs, files in os.walk(path, topdown=False):
                        for name in files:
                            fp = os.path.join(root, name)
                            try:
                                os.chmod(fp, 0o666)
                                os.remove(fp)
                            except Exception:
                                pass
                        for name in dirs:
                            dp = os.path.join(root, name)
                            try:
                                os.chmod(dp, 0o777)
                                shutil.rmtree(dp, ignore_errors=True)
                            except Exception:
                                pass
                    os.chmod(path, 0o777)
                    shutil.rmtree(path, ignore_errors=True)
                except Exception:
                    pass
            if not os.path.exists(path):
                break
            time.sleep(0.2)

    removed_dirs = 0
    if os.path.exists(kb_root):
        for name in os.listdir(kb_root):
            p = os.path.join(kb_root, name)
            if os.path.isdir(p):
                _force_remove_dir(p)
                if not os.path.exists(p):
                    removed_dirs += 1
            else:
                try:
                    os.remove(p)
                except Exception:
                    pass
        _force_remove_dir(kb_root)
    os.makedirs(kb_root, exist_ok=True)
    return removed_dirs
