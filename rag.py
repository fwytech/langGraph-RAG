# 导入streamlit库并简写为st，streamlit是一个用于创建数据应用的Python库
import streamlit as st
st.set_page_config(layout="wide")
from dotenv import load_dotenv
load_dotenv()
from webui import rag_chat_page, knowledge_base_page  # , platforms_page
from utils import get_img_base64
from app_utils.helpers import clear_all_kb

# 检查当前模块是否是主程序入口
if __name__ == "__main__":
    if st.session_state.get("kb_pending_clear"):
        n = clear_all_kb()
        st.session_state["kb_pending_clear"] = False
        st.toast(f"已清空 {n} 个知识库")
    # 使用streamlit的sidebar上下文管理器，在侧边栏中添加内容
    with st.sidebar:
        # 在侧边栏中显示一个logo，使用get_img_base64函数获取图片的base64编码
        st.logo(
            get_img_base64("chatchat_lite_logo.png"),  # 获取大图标的base64编码
            size="large",  # 设置logo的大小为large
            icon_image=get_img_base64("chatchat_lite_small_logo.png"),  # 获取小图标的base64编码
        )
        with st.popover(":wastebasket: 清空本地知识库", use_container_width=True):
            confirm = st.checkbox("确认清空所有知识库")
            if st.button("执行清空", disabled=not confirm, use_container_width=True):
                st.session_state["kb_pending_clear"] = True
                st.rerun()

    # 创建一个导航对象pg，定义应用的页面结构
    pg = st.navigation({
        "对话": [  # 定义一个名为“对话”的页面组
            st.Page(rag_chat_page, title="智能客服", icon=":material/chat:"),
        ],
        "设置": [  # 定义一个名为“设置”的页面组
            st.Page(knowledge_base_page, title="行业知识库", icon=":material/library_books:"),
            # 添加一个页面，使用knowledge_base_page函数，标题为“知识库管理”，图标为图书馆
        ]
    })
    # 运行导航对象pg，显示定义的页面
    pg.run()
