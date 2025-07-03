import streamlit as st
from pathlib import Path
import os
import torch

st.set_page_config(page_title="Chatbot môn học UIT", page_icon="🤖")

import rag_pipeline as rag
from rag_pipeline import load_index, QA_PROMPT, EMBED_MODEL

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from google.api_core.exceptions import ResourceExhausted

DEVICE = "cpu"                         
STORAGE_DIR = Path("storage")         

FALLBACK_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite-preview-06-17",
]

API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

@st.cache_resource(show_spinner="⚙️ Nạp FAISS index & embedding…")
def init_index_and_embedding():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device=DEVICE,
    )
    return load_index()

index = init_index_and_embedding()

def query_with_fallback(prompt: str) -> str:
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            Settings.llm = Gemini(api_key=API_KEY, model_name=model_name)
            engine = index.as_query_engine(
                text_qa_template=QA_PROMPT,
                similarity_top_k=40,
            )
            response = engine.query(prompt)
            return str(response)
        except ResourceExhausted:
            st.warning(f"⚠️ Quota cho `{model_name.split('/')[-1]}` đã hết, thử model khác…")
            last_error = ResourceExhausted(f"Quota hết cho {model_name}")
        except Exception as e:
            st.error(f"❌ Lỗi khi truy vấn model `{model_name}`: {e}")
            raise
    raise last_error or RuntimeError("Tất cả model Gemini đều hết quota.")

st.title("🤖 Chatbot môn học UIT")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if prompt := st.chat_input("Nhập câu hỏi (vd: IT003 học gì?)"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm câu trả lời…"):
            answer = query_with_fallback(prompt)
            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))