import os
import streamlit as st
from llama_index.core import QueryBundle
from rag_pipeline import init_rag_engine, query_hybrid, DEFAULT_CSV

os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache"
os.makedirs(os.environ["TIKTOKEN_CACHE_DIR"], exist_ok=True)

os.environ["NLTK_DATA"] = "/tmp/nltk_data"
os.makedirs(os.environ["NLTK_DATA"], exist_ok=True)

import nltk
nltk.download("stopwords", download_dir=os.environ["NLTK_DATA"], quiet=True)
nltk.download("punkt",     download_dir=os.environ["NLTK_DATA"], quiet=True)

st.set_page_config(layout="centered", page_title="Chatbot môn học UIT", page_icon="🤖")
st.title("🤖 Chatbot môn học UIT")

show_chunks = st.checkbox("Hiển thị các chunk được retrieve")

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
        if show_chunks:
            qb = QueryBundle(query_str=prompt)
            nodes = engine._retriever._retrieve(qb)
            with st.expander(f"Retrieved {len(nodes)} chunks"):
                for n in nodes:
                    st.write(f"**Score:** {n.score:.3f}")
                    content = n.node.get_content()
                    st.text(content[:200] + ("…" if len(content) > 200 else ""))

        with st.spinner("Đang tìm câu trả lời…"):
            answer = query_hybrid(engine, prompt)
            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
