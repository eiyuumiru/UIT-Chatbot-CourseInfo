#!/usr/bin/env python3
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List

import faiss
import pandas as pd
import torch

from llama_index.core import (
    Document,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
    QueryBundle,
    get_response_synthesizer,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.core.indices.keyword_table.simple_base import SimpleKeywordTableIndex
from llama_index.core.indices.keyword_table.retrievers import KeywordTableSimpleRetriever
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine

STORAGE_DIR    = Path("storage")
DEFAULT_CSV    = "course_info.csv"
EMBED_MODEL    = "intfloat/multilingual-e5-small"
COURSE_CODE_RE = re.compile(r"[A-Z]{2}\d{3}")

Settings.llm = Gemini(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    model_name="models/gemini-2.5-flash",
)

QA_PROMPT = PromptTemplate(
"""Bạn là trợ lý tư vấn môn học của UIT.
Bạn sẽ trả lời các câu hỏi của sinh viên về các môn học UIT dựa trên thông tin có sẵn trong cơ sở dữ liệu.
Hãy kết hợp toàn bộ thông tin trong cơ sở dữ liệu cùng khả năng suy luận để đưa ra câu trả lời đầy đủ và chính xác nhất.
Nếu bạn có khả năng suy luận (thinking), hãy suy nghĩ kỹ càng và xem xét cẩn thận tất cả thông tin trong cơ sở dữ liệu trước khi trả lời, cùng với việc tránh trả lời các câu hỏi không có trong cơ sở dữ liệu.
Bạn không được tự ý thêm thắt hay bịa đặt thông tin; chỉ sử dụng thông tin có trong cơ sở dữ liệu.

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
 - Nếu câu hỏi về một mã môn cụ thể, hãy dùng mã đó tra cứu trong cơ sở dữ liệu và suy luận từ các thông tin liên quan để trả lời.
 - Nếu câu hỏi không liên quan đến thông tin môn học UIT, như cách giải bài tập hay nội dung bài học chi tiết, hãy từ chối trả lời một cách lịch sự.
Trả lời:
"""
)

def make_docs(csv_path: Path) -> List[Document]:
    df = pd.read_csv(csv_path)
    docs: List[Document] = []
    for _, row in df.iterrows():
        code = str(row["course_code"]).strip().upper()
        name = str(row.get("course_name", "")).strip()
        ctx = str(row["context"])
        header = f"Mã môn: {code}" + (f"\nTên đầy đủ: {name}" if name else "")
        docs.append(Document(text=header + "\n\n" + ctx,
                             metadata={"course_code": code, "course_name": name}))
    return docs

def chunk_docs(docs: List[Document], size: int, overlap: int):
    return TokenTextSplitter(chunk_size=size, chunk_overlap=overlap).get_nodes_from_documents(docs)

def build_index(csv: str, chunk_size: int, chunk_overlap: int, device: str):
    docs  = make_docs(Path(csv))
    nodes = chunk_docs(docs, chunk_size, chunk_overlap)

    dev = "cuda" if device=="auto" and torch.cuda.is_available() else device
    try:
        embed = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=dev)
    except RuntimeError:
        embed = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cpu")
    Settings.embed_model = embed

    dim = len(embed.get_text_embedding("test"))
    vs  = FaissVectorStore(faiss.IndexFlatL2(dim))
    from llama_index.core.indices.vector_store import VectorStoreIndex
    idx = VectorStoreIndex(nodes, vector_store=vs)

    STORAGE_DIR.mkdir(exist_ok=True)
    idx.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print("Index built and saved to storage/")

def load_index():
    ctx = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(ctx)

class HybridRetriever(BaseRetriever):
    def __init__(self, index, keyword_index, top_k: int):
        super().__init__()
        self.vec_ret = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        self.kw_ret  = KeywordTableSimpleRetriever(index=keyword_index)
    def _retrieve(self, query: QueryBundle):
        vec = self.vec_ret.retrieve(query)
        kw  = self.kw_ret.retrieve(query)
        merged = {n.node.node_id: n for n in vec}
        for n in kw:
            merged.setdefault(n.node.node_id, NodeWithScore(node=n.node, score=0.0))
        return sorted(merged.values(), key=lambda n: n.score, reverse=True)

def init_rag_engine(
    csv: str       = DEFAULT_CSV,
    chunk_size: int = 150,
    chunk_overlap: int = 20,
    top_k: int      = 10,
    device: str     = "auto",
):
    dev = "cuda" if device=="auto" and torch.cuda.is_available() else device
    try:
        embed = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=dev)
    except RuntimeError:
        embed = HuggingFaceEmbedding(model_name=EMBED_MODEL, device="cpu")
    Settings.embed_model = embed

    idx = load_index()

    docs  = make_docs(Path(csv))
    nodes = chunk_docs(docs, chunk_size, chunk_overlap)
    kw_idx = SimpleKeywordTableIndex(nodes, storage_context=idx.storage_context)

    retriever = HybridRetriever(index=idx, keyword_index=kw_idx, top_k=top_k)
    synth     = get_response_synthesizer()
    engine    = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=synth,
        text_qa_template=QA_PROMPT,
    )
    return engine

def query_hybrid(engine, question: str):
    return engine.query(question)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--csv",           default=DEFAULT_CSV)
    b.add_argument("--chunk_size",    type=int, default=150)
    b.add_argument("--chunk_overlap", type=int, default=20)
    b.add_argument("--device",        choices=["auto","cpu","cuda"], default="auto")
    b.set_defaults(func=lambda a: build_index(a.csv, a.chunk_size, a.chunk_overlap, a.device))

    q = sub.add_parser("query")
    q.add_argument("question")
    q.add_argument("--top_k",         type=int, default=10)
    q.add_argument("--chunk_size",    type=int, default=150)
    q.add_argument("--chunk_overlap", type=int, default=20)
    q.add_argument("--device",        choices=["auto","cpu","cuda"], default="auto")
    def _run(a):
        eng = init_rag_engine(
            csv=DEFAULT_CSV,
            chunk_size=a.chunk_size,
            chunk_overlap=a.chunk_overlap,
            top_k=a.top_k,
            device=a.device,
        )
        print(str(query_hybrid(eng, a.question)))
    q.set_defaults(func=_run)

    args = ap.parse_args()
    args.func(args)
