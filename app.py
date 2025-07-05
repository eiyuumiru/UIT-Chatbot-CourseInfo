import streamlit as st
from pathlib import Path
import os
import torch

st.set_page_config(page_title="Chatbot môn học UIT", page_icon="🤖")

import rag_pipeline as rag
from rag_pipeline import load_index, QA_PROMPT, DEFAULT_CSV, EMBED_MODEL

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.indices.keyword_table.simple_base import SimpleKeywordTableIndex
from llama_index.core.indices.keyword_table.retrievers import KeywordTableSimpleRetriever
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer

DEVICE = "cpu"
API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

FALLBACK_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite-preview-06-17",
]

st.sidebar.title("Cấu hình RAG")
top_k        = st.sidebar.number_input("Max kết quả (top_k)", 1, 50, 10)
device       = st.sidebar.selectbox("Device", ["auto","cuda","cpu"], index=0)
hybrid       = st.sidebar.checkbox("Bật hybrid search", value=False)
chunk_size   = st.sidebar.slider("Chunk size (keyword)", 50, 500, 150, 10)
chunk_overlap= st.sidebar.slider("Chunk overlap", 0, 200, 20, 5)

@st.cache_resource(show_spinner="⚙️ Đang khởi tạo RAG engine")
def init_rag_engine(
    hybrid: bool,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> RetrieverQueryEngine:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device=device if device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    index = load_index()

    if hybrid:
        docs  = rag.make_docs(Path(DEFAULT_CSV))
        nodes = rag.chunk_docs(docs, chunk_size, chunk_overlap)
        keyword_index = SimpleKeywordTableIndex(nodes, storage_context=index.storage_context)

        vec_ret = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        kw_ret  = KeywordTableSimpleRetriever(index=keyword_index)

        class CustomRetriever(BaseRetriever):
            def __init__(self, vec, kw, mode="OR"):
                super().__init__()
                self.vec, self.kw, self.mode = vec, kw, mode
            def _retrieve(self, query_bundle):
                vec_nodes = self.vec.retrieve(query_bundle)
                kw_nodes  = self.kw.retrieve(query_bundle)
                combined = {n.node.node_id: n for n in vec_nodes}
                for n in kw_nodes:
                    combined.setdefault(n.node.node_id, NodeWithScore(n.node, 0.0))
                v = {n.node.node_id for n in vec_nodes}
                k = {n.node.node_id for n in kw_nodes}
                ids = v&k if self.mode=="AND" else v|k
                return sorted([combined[i] for i in ids],
                              key=lambda x: x.score, reverse=True)

        custom_ret = CustomRetriever(vec_ret, kw_ret, mode="OR")

        synth = get_response_synthesizer()
        engine = RetrieverQueryEngine.from_args(
            retriever=custom_ret,
            response_synthesizer=synth,
            text_qa_template=QA_PROMPT,
        )

    else:
        engine = index.as_query_engine(
            text_qa_template=QA_PROMPT,
            similarity_top_k=top_k,
        )

    return engine

engine = init_rag_engine(hybrid, chunk_size, chunk_overlap, top_k)

def query_with_fallback(prompt: str) -> str:
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            Settings.llm = Gemini(api_key=API_KEY, model_name=model_name)
            resp = engine.query(prompt)
            return str(resp)
        except Exception as e:
            last_error = e
    raise last_error

st.title("🤖 Chatbot môn học UIT")
if "history" not in st.session_state:
    st.session_state.history = []
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if prompt := st.chat_input("Nhập câu hỏi về UIT…"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm câu trả lời…"):
            answer = query_with_fallback(prompt)
            st.markdown(answer)
    st.session_state.history.append(("assistant", answer))
