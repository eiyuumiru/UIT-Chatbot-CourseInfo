# app.py
import streamlit as st
from pathlib import Path
import torch

# ---- import các hàm / hằng số có sẵn trong rag_pipeline.py ----
import rag_pipeline as rag
from rag_pipeline import (
    QA_PROMPT,
    EMBED_MODEL,
    DEFAULT_DEVICE,
    load_index,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# --------------------------------- Helper -----------------------------------
@st.cache_resource(show_spinner="Đang nạp FAISS index …")
def init_engine(device: str = "auto", top_k: int = 40):
    """Load FAISS index và tạo query_engine."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)
    index = load_index()                          # từ rag_pipeline.py
    return index.as_query_engine(
        text_qa_template=QA_PROMPT,
        similarity_top_k=top_k,
    )

# --------------------------------- UI ---------------------------------------
st.title("Chatbot hỏi đáp môn học UIT")

# chọn thiết bị (tùy bạn giữ nguyên default)
device_opt = st.sidebar.selectbox("Thiết bị embedding", ["auto", "cpu", "cuda"], index=0)
engine = init_engine(device_opt)

# khởi tạo session history
if "history" not in st.session_state:
    st.session_state.history = []

# hiển thị lịch sử
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

# ô nhập ChatGPT-style
if prompt := st.chat_input("Hỏi gì về môn học? (ví dụ: IT003 là môn gì?)"):
    # lưu & hiển thị câu hỏi
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    # trả lời
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm câu trả lời…"):
            resp = engine.query(prompt)
            answer = str(resp)
            st.markdown(answer)

    # lưu lịch sử
    st.session_state.history.append(("assistant", answer))
