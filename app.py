# app.py
import streamlit as st
from pathlib import Path
import zipfile
import torch
import time

import rag_pipeline as rag
from rag_pipeline import EMBED_MODEL, QA_PROMPT, load_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# ---------------------------------------------------------------
# Đường dẫn
STORAGE_DIR = Path("storage")
ZIP_PATH     = Path("storage.zip")     # file bạn đã commit

# ---------------------------------------------------------------
@st.cache_resource(show_spinner="⚙️ Khởi tạo FAISS index…")
def init_engine(device: str = "auto", top_k: int = 40):
    """Đảm bảo có storage/ (unzip nếu cần) rồi tạo query_engine."""
    if not STORAGE_DIR.exists():
        if ZIP_PATH.exists():
            with st.spinner("📦 Đang giải nén storage.zip…"):
                with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                    zf.extractall(".")
        else:
            st.error("Không tìm thấy storage.zip và storage/.")
            st.stop()
        # đợi I/O flush (nhất là trên Streamlit Cloud)
        time.sleep(0.1)

    # ---------- init embedding ----------
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL,
                                                device=device)

    # ---------- load index ----------
    index = load_index()  # từ rag_pipeline.py
    return index.as_query_engine(text_qa_template=QA_PROMPT,
                                 similarity_top_k=top_k)

# ---------------------------------------------------------------
# UI đơn giản
st.title("🤖 Chatbot môn học UIT (RAG + Gemini)")

device_sel = st.sidebar.selectbox("Thiết bị embedding",
                                  ["auto", "cpu", "cuda"], index=0)

engine = init_engine(device_sel)

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if prompt := st.chat_input("Nhập câu hỏi (vd: IT003 là gì?)"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm câu trả lời…"):
            resp    = engine.query(prompt)
            answer  = str(resp)
            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
