#!/usr/bin/env python3
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

STORAGE_DIR: Path = Path("storage")
DEFAULT_CSV = "qa_full.csv"
# EMBED_MODEL = "AITeamVN/Vietnamese_Embedding" // heavy embedding
EMBED_MODEL = "intfloat/multilingual-e5-small"
DIM = 384
DEFAULT_DEVICE = "auto"
COURSE_CODE_RE = re.compile(r"[A-Z]{2}\d{3}")

Settings.llm = Gemini(
    api_key=os.environ["GEMINI_API_KEY"],
    model_name="models/gemini-2.5-flash-lite-preview-06-17",
)

def make_docs(csv_path: Path) -> List[Document]:
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
    print("Successfully build index. Storing in storage/")

def load_index():
    sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(sc)

QA_PROMPT = PromptTemplate(
"""Bạn là trợ lý tư vấn môn học của UIT.
Bạn sẽ trả lời các câu hỏi của sinh viên về các môn học UIT dựa trên thông tin có sẵn trong cơ sở dữ liệu.
Hãy kết hợp toàn bộ thông tin trong cơ sở dữ liệu cùng khả năng suy luận để đưa ra câu trả lời đầy đủ và chính xác nhất.
Nếu bạn có khả năng suy luận ("suy nghĩ"), hãy suy nghĩ kỹ càng và xem xét cẩn thận tất cả thông tin trong cơ sở dữ liệu trước khi trả lời.
Bạn không được tự ý thêm thắt hay bịa đặt thông tin; chỉ sử dụng thông tin có trong cơ sở dữ liệu. Tuy nhiên, bạn có thể bổ sung thông tin từ trang web student.uit.edu.vn kết hợp với khả năng suy luận của bạn để cung cấp câu trả lời đầy đủ hơn. Nếu môn học không có trong cơ sở dữ liệu, bạn có thể tra cứu trên student.uit.edu.vn để trả lời chi tiết. Không được tham khảo bất kỳ nguồn nào khác ngoài cơ sở dữ liệu và student.uit.edu.vn.
Câu hỏi: {query_str}

Thông tin tham khảo:
{context_str}

Yêu cầu cụ thể:
 - Chỉ trả lời bằng tiếng Việt, không giải thích dài dòng.
 - Không tự bịa ra thông tin nếu không biết câu trả lời.
 - Chỉ trả lời các câu hỏi liên quan đến môn học UIT, ví dụ như:
     + Mã môn học (vd: IT003, MA006)
     + Tên môn học (vd: "Lập trình hướng đối tượng")
     + Thông tin môn học (vd: "Thông tin về môn IT003?", "Tầm quan trọng của môn MA006?")
 - Nếu câu hỏi liên quan tới môn học cụ thể, hãy dùng mã môn học để tìm thông tin liên quan trong cơ sở dữ liệu.
 - Nếu câu hỏi liên quan tới nhiều môn học, hãy tìm kiếm từng mã môn trong cơ sở dữ liệu và cung cấp thông tin cho từng môn học đó.
 - Nếu câu hỏi yêu cầu liệt kê toàn bộ các môn học, hãy tìm tất cả mã môn học trong cơ sở dữ liệu (và từ student.uit.edu.vn nếu cần) và liệt kê chúng ra.
 - Nếu câu hỏi về môn học không có trong cơ sở dữ liệu, hãy tra cứu thông tin trên student.uit.edu.vn và cung cấp thông tin.
 - Nếu câu hỏi về một mã môn cụ thể, hãy dùng mã đó tra cứu trong cơ sở dữ liệu và suy luận từ các thông tin liên quan để trả lời.
 - Nếu câu hỏi không liên quan đến thông tin môn học UIT, như cách giải bài tập hay nội dung bài học chi tiết, hãy từ chối trả lời một cách lịch sự.
Answer:
"""
)


def query(args):
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)

    index = load_index()

    top_k = max(args.top_k, 30)
    engine = index.as_query_engine(text_qa_template=QA_PROMPT,
                                   similarity_top_k=top_k)

    answer = engine.query(args.question)
    print("\n---\n" + str(answer) + "\n---")

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
