from pathlib import Path

import streamlit as st

from core.qa_logic import answer_literature_question
from core.query_logic import answer_query_question


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    st.title("蛋白质预测领域智能问答 & 查数 Demo")

    mode = st.sidebar.selectbox("选择功能", ["文献问答", "实验结果查数"])
    user_input = st.text_area("请输入你的问题：")

    if st.button("提交"):
        if not user_input.strip():
            st.warning("问题不能为空。")
            return

        if mode == "文献问答":
            answer = answer_literature_question(user_input, base_dir=BASE_DIR)
        else:
            answer = answer_query_question(user_input, base_dir=BASE_DIR)

        st.markdown("**回答：**")
        st.write(answer)


if __name__ == "__main__":
    main()
