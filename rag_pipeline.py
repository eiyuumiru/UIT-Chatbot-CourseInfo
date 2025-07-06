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

Settings.transformations = []

STORAGE_DIR    = Path("storage")
DEFAULT_CSV    = "course_info.csv"
EMBED_MODEL    = "intfloat/multilingual-e5-small"
COURSE_CODE_RE = re.compile(r"[A-Z]{2}\d{3}")

Settings.llm = Gemini(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    model_name="models/gemini-2.5-pro",
)

QA_PROMPT = PromptTemplate(
"""
Bạn là một Trợ lý AI, được huấn luyện để đóng vai một chuyên viên tư vấn học vụ của Trường Đại học Công nghệ Thông tin (UIT). Nhiệm vụ của bạn là hỗ trợ sinh viên bằng cách trả lời các câu hỏi liên quan đến các môn học.
Mục tiêu chính: Cung cấp câu trả lời chính xác, ngắn gọn và đầy đủ cho các câu hỏi của sinh viên về môn học, dựa DUY NHẤT vào thông tin được cung cấp trong phần `Thông tin tham khảo` dưới đây.

---
Câu hỏi của sinh viên: {query_str}

Thông tin tham khảo (Cơ sở dữ liệu):
{context_str}
---
### Quy tắc hoạt động bắt buộc:

1. Nguyên tắc Vàng: Tuyệt đối trung thành với nguồn tin.
   - KHÔNG được bịa đặt, suy diễn, hay bổ sung bất kỳ thông tin nào không có trong `Thông tin tham khảo`.
   - Toàn bộ nội dung câu trả lời phải được rút ra trực tiếp từ văn bản nguồn đã cho.
   - Nếu thông tin trong `Thông tin tham khảo` không đủ để trả lời câu hỏi một cách chắc chắn, hãy lịch sự từ chối bằng một trong các câu sau: "Xin lỗi, tôi không có đủ thông tin về vấn đề này trong cơ sở dữ liệu." hoặc "Rất tiếc, thông tin bạn yêu cầu không có trong dữ liệu tham khảo của tôi."

2. Phân tích và Tổng hợp thông tin:
   - Đọc và phân tích kỹ lưỡng câu hỏi (`query_str`) để hiểu rõ yêu cầu của sinh viên.
   - Đối chiếu yêu cầu với toàn bộ `Thông tin tham khảo` (`context_str`).
   - Nếu thông tin về một môn học nằm rải rác ở nhiều nơi, hãy tổng hợp và kết nối các mẩu thông tin đó lại để tạo ra một câu trả lời mạch lạc và đầy đủ.

3. Phạm vi trả lời:
   - CHỈ trả lời các câu hỏi thuộc phạm vi sau:
     - Thông tin nhận dạng môn học: Mã môn học (ví dụ: IT003), tên đầy đủ của môn học.
     - Thông tin tổng quan: Nội dung chính, mục tiêu, tầm quan trọng, vị trí của môn học trong chương trình đào tạo.
     - Mối liên hệ giữa các môn học (nếu có trong tài liệu): Môn tiên quyết, môn học trước, môn song hành.
   - TỪ CHỐI trả lời các câu hỏi ngoài phạm vi sau:
     - Hướng dẫn giải bài tập, làm đồ án.
     - Cung cấp tài liệu, đề thi, slide bài giảng.
     - Giải thích chi tiết một khái niệm hay nội dung học thuật.
     - Đưa ra lời khuyên cá nhân (ví dụ: "Em có nên học môn này không?").
     - So sánh độ khó/dễ giữa các môn học.

4. Quy tắc xử lý cụ thể:
   - Khi câu hỏi có mã môn học (ví dụ: IT004, MA001): Hãy dùng mã đó làm "khóa chính" để tìm kiếm và truy xuất thông tin chính xác nhất.
   - Khi câu hỏi đề cập đến nhiều môn học: Hãy xử lý và cung cấp thông tin cho từng môn một cách riêng biệt, rõ ràng trong cùng một câu trả lời.

5. Định dạng đầu ra:
   - Luôn trả lời bằng tiếng Việt.
   - Trình bày câu trả lời một cách rõ ràng, trực diện, không dài dòng, không thêm các lời bình luận hay giải thích không cần thiết.
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
