from pathlib import Path

import streamlit as st

from core.qa_logic import answer_literature_question
from core.query_logic import answer_query_question


BASE_DIR = Path(__file__).resolve().parent


# ----------------------------
# 初始化会话状态
# ----------------------------
def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


# ----------------------------
# 侧边栏
# ----------------------------
def render_sidebar() -> str:
    st.sidebar.header("系统说明")

    st.sidebar.markdown("**功能模式**")
    mode = st.sidebar.selectbox(
        "选择功能模式",
        ["文献问答", "实验结果查数"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**提问示例**")
    if mode == "文献问答":
        st.sidebar.code("ESM-2 和 AlphaFold2 的核心区别是什么？")
    else:
        st.sidebar.code("把 CAFA3 的 micro-F1 从高到低排序")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**数据来源**\n"
        "- 文献知识库：`papers.json`\n"
        "- 实验结果表：`experiments.csv`"
    )

    st.sidebar.markdown(
        "⚠ 当前为课程 Demo，回答基于示例数据与预设逻辑"
    )

    return mode


# ----------------------------
# 主聊天区域
# ----------------------------
def render_chat_area(mode: str) -> str:
    st.markdown("### 蛋白质预测领域智能问答系统")
    st.caption(
        "基于示例文献与实验数据的领域问答 Demo（支持华为云大模型扩展）"
    )

    st.markdown(
        f"**当前模式：{'📄 文献问答' if mode == '文献问答' else '📊 实验结果查数'}**"
    )

    st.markdown("---")

    if not st.session_state.messages:
        st.info("请输入问题开始对话。可在左侧切换功能模式。")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    return st.chat_input("请输入你的问题，例如：ESM-2 的优势是什么？")


# ----------------------------
# 处理提问
# ----------------------------

def handle_question(mode: str, user_input: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 关键：立刻把用户问题渲染出来（否则要等下一次 rerun 才会出现在历史区）
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("模型正在思考中，请稍候..."):
            if mode == "文献问答":
                answer = answer_literature_question(user_input, base_dir=BASE_DIR)
            else:
                answer = answer_query_question(user_input, base_dir=BASE_DIR)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ----------------------------
# 主入口
# ----------------------------
def main() -> None:
    st.set_page_config(
        page_title="蛋白质预测领域问答 & 查数 Demo",
        page_icon="🧬",
        layout="wide",
    )

    init_state()
    mode = render_sidebar()
    user_input = render_chat_area(mode)

    if user_input:
        handle_question(mode, user_input)


if __name__ == "__main__":
    main()
