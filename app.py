# app.py — Streamlit UI cho RAG + Gemini với fallback tự động
import streamlit as st
from pathlib import Path
import os, torch

# Import từ rag_pipeline của bạn
import rag_pipeline as rag
from rag_pipeline import load_index, QA_PROMPT, EMBED_MODEL

# Thiết lập embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
# Import Gemini LLM và exception
from llama_index.llms.gemini import Gemini
from google.api_core.exceptions import ResourceExhausted

# ---------- Cấu hình chung ----------
DEVICE = "cpu"  # Chạy trên CPU để phù hợp Streamlit Cloud
STORAGE_DIR = Path("storage")  # Thư mục index đã commit

# Danh sách model fallback theo thứ tự ưu tiên
FALLBACK_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite-preview-06-17",
]

# Lấy API key
API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# ---------- Khởi tạo một lần: embedding + FAISS index ----------
@st.cache_resource(show_spinner="⚙️ Nạp embedding & FAISS index…")
def init_index_and_embedding():
    # Cấu hình embedding model từ rag_pipeline
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device=DEVICE,
    )
    # Load index từ storage/
    index = load_index()
    return index

# Khởi tạo index & embedding
index = init_index_and_embedding()

# ---------- Hàm query với fallback ----------
def query_with_fallback(prompt: str) -> str:
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            # Cập nhật LLM trong Settings
            Settings.llm = Gemini(api_key=API_KEY, model_name=model_name)
            # Tạo query_engine trên cùng index
            engine = index.as_query_engine(
                text_qa_template=QA_PROMPT,
                similarity_top_k=40,
            )
            # Thực hiện truy vấn
            response = engine.query(prompt)
            return str(response)
        except ResourceExhausted:
            # Quota model hiện tại đã hết, thử model kế
            st.warning(f"⚠️ Quota cho `{model_name.split('/')[-1]}` đã hết, chuyển model khác…")
            last_error = ResourceExhausted(f"Quota hết cho {model_name}")
            continue
        except Exception as e:
            # Lỗi khác thì hiển thị và dừng
            st.error(f"❌ Lỗi khi truy vấn model `{model_name}`: {e}")
            raise
    # Nếu cả 3 model đều hết quota
    raise last_error or RuntimeError("Tất cả model Gemini đều hết quota.")

# ---------- Giao diện chat ----------
st.set_page_config(page_title="Chatbot môn học UIT", page_icon="🤖")
st.title("🤖 Chatbot môn học UIT")

# Khởi tạo lịch sử chat
if "history" not in st.session_state:
    st.session_state.history = []

# Hiển thị lịch sử
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

# Nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi (vd: IT003 học gì?)"):
    # Lưu và hiển thị câu hỏi
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    # Trả lời
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm câu trả lời…"):
            answer = query_with_fallback(prompt)
            st.markdown(answer)

    # Lưu câu trả lời vào lịch sử
    st.session_state.history.append(("assistant", answer))
