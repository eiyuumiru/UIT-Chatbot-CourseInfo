import os

os.environ["TIKTOKEN_CACHE_DIR"] = os.path.expanduser("~/.cache/tiktoken")
os.makedirs(os.environ["TIKTOKEN_CACHE_DIR"], exist_ok=True)

import streamlit as st
from rag_pipeline import init_rag_engine, query_hybrid, DEFAULT_CSV

st.set_page_config(layout="centered", page_title="Chatbot môn học UIT", page_icon="🤖")
st.title("🤖 Chatbot môn học UIT")

@st.cache_resource(show_spinner=False)
def load_engine():
    return init_rag_engine(
        csv=DEFAULT_CSV,
        chunk_size=150,
        chunk_overlap=20,
        top_k=10,
        device="cpu",
    )

engine = load_engine()

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if prompt := st.chat_input("Nhập câu hỏi về UIT (vd. Cho tôi thông tin về môn IT003?)"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm câu trả lời…"):
            answer = query_hybrid(engine, prompt)
            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
