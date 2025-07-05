import streamlit as st
from pathlib import Path
import os
import subprocess

st.set_page_config(page_title="Chatbot môn học UIT", page_icon="🤖")

DEVICE = "cuda"
DEFAULT_TOP_K     = 10
DEFAULT_CHUNK_SZ  = 150
DEFAULT_CHUNK_OL  = 20
STORAGE_DIR = Path("storage")
API_KEY     = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

st.sidebar.title("Cài đặt RAG")
top_k = st.sidebar.number_input("Max kết quả (top_k)", min_value=1, max_value=50, value=DEFAULT_TOP_K)
device = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)

hybrid = st.sidebar.checkbox("Bật hybrid search", value=False)
chunk_size    = st.sidebar.slider("Chunk size (keyword)", 50, 500, DEFAULT_CHUNK_SZ, 10)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 200, DEFAULT_CHUNK_OL, 5)

st.title("🤖 Chatbot môn học UIT")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if prompt := st.chat_input("Nhập câu hỏi (vd: Cho tôi thông tin về IT003?)"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    cmd = [
        "python", "rag_pipeline.py", "query",
        prompt,
        "--top_k", str(top_k),
        "--device", device,
    ]
    if hybrid:
        cmd += [
            "--hybrid",
            "--chunk_size",    str(chunk_size),
            "--chunk_overlap", str(chunk_overlap),
        ]

    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm câu trả lời…"):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                answer = result.stdout.strip()
            except subprocess.CalledProcessError as e:
                answer = (
                    "**Lỗi khi truy vấn RAG:**\n"
                    f"```bash\n{e.stderr.strip()}\n```"
                )
        st.markdown(answer)

    # Lưu lịch sử trả lời
    st.session_state.history.append(("assistant", answer))
