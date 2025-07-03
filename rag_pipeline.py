#!/usr/bin/env python3
"""
rag_pipeline.py — Build & query UIT‑Course RAG index (FAISS + Gemini-1.5‑Flash)
=============================================================================
Giả định môi trường đã đầy đủ (faiss, llama‑index, Gemini plugin, …).
Bạn CHỈ cần đặt biến môi trường:
    export GEMINI_API_KEY="<your-key>"
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List

import faiss
import pandas as pd
from llama_index.core import (
    Document,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore

# ----------------------------- GLOBAL CONFIG --------------------------------
STORAGE_DIR: Path = Path("storage")
DEFAULT_CSV = "qa_full.csv"
# EMBED_MODEL = "AITeamVN/Vietnamese_Embedding" // heavy embedding
EMBED_MODEL = "intfloat/multilingual-e5-small"
DIM = 384
DEFAULT_DEVICE = "auto"  # "auto" → dùng CUDA nếu có
COURSE_CODE_RE = re.compile(r"[A-Z]{2}\d{3}")  # eg. IT003, IE406

Settings.llm = Gemini(
    api_key=os.environ["GEMINI_API_KEY"],
    model_name="models/gemini-2.5-flash-lite-preview-06-17",
)

# ----------------------------- HELPERS --------------------------------------

def make_docs(csv_path: Path) -> List[Document]:
    """Chuyển CSV thành list Document có header (mã + tên đầy đủ)."""
    df = pd.read_csv(csv_path)
    docs: List[Document] = []

    for _, row in df.iterrows():
        code = str(row["course_code"]).strip().upper()
        name = str(row.get("course_name", "")).strip()
        ctx = str(row["context"])

        header_lines = [f"Mã môn: {code}"]
        if name:
            header_lines.append(f"Tên đầy đủ: {name}")
        header = "\n".join(header_lines)

        docs.append(
            Document(
                text=f"{header}\n\n{ctx}",
                metadata={"course_code": code, "course_name": name},
            )
        )
    return docs


def chunk_docs(docs: List[Document], size: int, overlap: int):
    return TokenTextSplitter(chunk_size=size, chunk_overlap=overlap).get_nodes_from_documents(docs)

# ----------------------------- BUILD ----------------------------------------

def build(args):
    docs = make_docs(Path(args.csv))
    nodes = chunk_docs(docs, args.chunk_size, args.chunk_overlap)

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        embed = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)
    except RuntimeError:
        embed = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cpu")

    Settings.embed_model = embed

    dim = len(embed.get_text_embedding("test"))
    vs = FaissVectorStore(faiss_index=faiss.IndexFlatL2(dim))
    index = VectorStoreIndex(nodes, vector_store=vs)

    STORAGE_DIR.mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print("✅ Rebuilt index → storage/")

# ----------------------------- LOAD -----------------------------------------

def load_index():
    sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(sc)

# ----------------------------- QUERY ----------------------------------------

QA_PROMPT = PromptTemplate(
    """Bạn là trợ lý tư vấn môn học của UIT (Trường Đại học Công nghệ Thông tin, ĐHQG-HCM).
Bạn sẽ trả lời câu hỏi của sinh viên về các môn học tại UIT dựa trên thông tin có sẵn trong cơ sở dữ liệu.
Hãy kết hợp các thông tin trong cơ sở dữ liệu, cùng với khả năng suy luận của bạn để trả lời câu hỏi một cách đầy đủ và chính xác nhất.
Nếu bạn có khả năng suy nghĩ (thinking), hãy suy nghĩ thật kỹ và cẩn thận, duyệt qua tất cả các thông tin trong cơ sở dữ liệu và đưa ra câu trả lời chính xác nhất.
Bạn không được tự ý suy diễn, chỉ sử dụng thông tin trong cơ sở dữ liệu để trả lời. Tuy nhiên, bạn có thể dựa vào thông tin trong cơ sở dữ liệu, kết hợp với việc tra cứu từ trang web student.uit.edu.vn và khả năng suy luận của bản thân để cung cấp thông tin đầy đủ hơn. Nếu gặp môn học không có trong cơ sở dữ liệu, bạn có thể tra cứu từ trang web student.uit.edu.vn để cung cấp thông tin đầy đủ hơn. Tuy nhiên, bạn không được tra cứu từ các nguồn khác ngoài cơ sở dữ liệu và trang web student.uit.edu.vn.

Câu hỏi: {query_str}

Thông tin tham khảo:
{context_str}

Yêu cầu:
Bạn chỉ được trả lời bằng tiếng Việt, không giải thích dài dòng.
Nếu câu hỏi không liên quan đến môn học, hãy từ chối trả lời.
Khi gặp câu hỏi liên quan đến mã môn học, hãy sử dụng mã môn để tìm kiếm thông tin trong cơ sở dữ liệu, dựa vào các nội dung liên quan đến mã môn đó để suy luận và tìm ra câu trả lời.
Khi gặp câu hỏi liên quan đến việc liệt kê các môn học, hãy tìm kiếm tất cả các môn học trong cơ sở dữ liệu có mã môn phù hợp với câu hỏi và trả lời đầy đủ nhất có thể.

Trả lời:"""
)


def query(args):
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)

    index = load_index()

    # Lấy query_engine với top_k cao (cho truy vấn chứa mã môn)
    top_k = max(args.top_k, 30)
    engine = index.as_query_engine(text_qa_template=QA_PROMPT,
                                   similarity_top_k=top_k)

    answer = engine.query(args.question)
    print("\n---\n" + str(answer) + "\n---")

# ----------------------------- CLI ------------------------------------------

def cli():
    ap = argparse.ArgumentParser(description="UIT‑Course RAG (Gemini)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--csv", default=DEFAULT_CSV)
    b.add_argument("--chunk_size", type=int, default=150)
    b.add_argument("--chunk_overlap", type=int, default=20)
    b.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cuda", "cpu"])
    b.set_defaults(func=build)

    q = sub.add_parser("query")
    q.add_argument("question")
    q.add_argument("--top_k", type=int, default=10)
    q.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cuda", "cpu"])
    q.set_defaults(func=query)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    cli()
