# app.py  –  Streamlit UI cho RAG + Gemini
import streamlit as st
from pathlib import Path
import torch, time

import rag_pipeline as rag                              # file gốc của bạn
from rag_pipeline import EMBED_MODEL, QA_PROMPT, load_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# ---------- hằng số ----------
STORAGE_DIR = Path("storage")         # đã commit vào repo
DEVICE      = "cpu"                   # Streamlit Cloud chỉ có CPU

# ---------- khởi tạo FAISS + embedding (cache) ----------
@st.cache_resource(show_spinner="⚙️ Đang nạp FAISS index …")
def init_engine():
    """Load FAISS index và tạo query_engine, chỉ chạy 1 lần/session."""
    # Đặt embedding model (intfloat/multilingual-e5-small)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL,
                                                device=DEVICE)

    # Nạp index từ thư mục storage/ đã có sẵn
    index = load_index()
    return index.as_query_engine(text_qa_template=QA_PROMPT,
                                 similarity_top_k=40)

# ---------- UI ----------
st.set_page_config(page_title="Chatbot môn học UIT", page_icon="🤖")
st.title("🤖 Chatbot môn học UIT (RAG + Gemini)")

# Lịch sử hội thoại
if "history" not in st.session_state:
    st.session_state.history = []     # list[(role, msg)]

# Hiển thị lịch sử
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

# Hộp nhập liệu kiểu ChatGPT
if prompt := st.chat_input("Nhập câu hỏi (vd: IT003 học gì?)"):
    # Ghi câu hỏi
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    # Lấy engine (cache) & truy vấn
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm câu trả lời …"):
            engine  = init_engine()            # cache_resource
            answer  = str(engine.query(prompt))
            st.markdown(answer)

    # Ghi câu trả lời
    st.session_state.history.append(("assistant", answer))
