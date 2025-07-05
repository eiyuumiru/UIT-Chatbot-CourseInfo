#!/usr/bin/env python3
from __future__ import annotations
import argparse
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
    VectorStoreIndex,
    load_index_from_storage,
    QueryBundle,
    get_response_synthesizer,
)

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore

# Hybrid search imports
from llama_index.core.indices.keyword_table.simple_base import SimpleKeywordTableIndex
from llama_index.core.indices.keyword_table.retrievers import KeywordTableSimpleRetriever
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine

STORAGE_DIR: Path = Path("storage")
DEFAULT_CSV = "course_info.csv"
EMBED_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_DEVICE = "auto"
COURSE_CODE_RE = re.compile(r"[A-Z]{2}\d{3}")

Settings.llm = Gemini(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    model_name="models/gemini-2.5-pro",
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

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
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
    print("Successfully built index. Stored in storage/")

def load_index():
    sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(sc)

QA_PROMPT = PromptTemplate(
    """
Bạn là trợ lý tư vấn môn học của UIT.
Bạn sẽ trả lời các câu hỏi của sinh viên về các môn học UIT dựa trên thông tin có sẵn trong cơ sở dữ liệu.
Hãy kết hợp toàn bộ thông tin trong cơ sở dữ liệu cùng khả năng suy luận để đưa ra câu trả lời đầy đủ và chính xác nhất.
Nếu bạn có khả năng suy luận ("suy nghĩ"), hãy nghĩ kỹ càng và xem xét cẩn thận tất cả thông tin trong cơ sở dữ liệu trước khi trả lời.
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

Trả lời:
"""
)

class CustomRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "OR",
    ) -> None:
        super().__init__()
        if mode not in ("AND", "OR"):
            raise ValueError("mode must be 'AND' or 'OR'.")
        self._mode = mode
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vec_nodes = self._vector_retriever.retrieve(query_bundle)
        kw_nodes  = self._keyword_retriever.retrieve(query_bundle)

        combined: dict[str, NodeWithScore] = {}
        for n in vec_nodes:
            combined[n.node.node_id] = n  
        for n in kw_nodes:
            combined.setdefault(n.node.node_id, NodeWithScore(node=n.node, score=0.0))

        vec_ids = {n.node.node_id for n in vec_nodes}
        kw_ids  = {n.node.node_id for n in kw_nodes}
        if self._mode == "AND":
            keep_ids = vec_ids & kw_ids
        else:
            keep_ids = vec_ids | kw_ids

        reranked = sorted(
            (combined[i] for i in keep_ids),
            key=lambda n: n.score,
            reverse=True,
        )
        return reranked


def query(args):
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)

    index = load_index()
    top_k = max(args.top_k, 30)

    if getattr(args, "hybrid", False):
        docs = make_docs(Path(DEFAULT_CSV))
        nodes = chunk_docs(docs, args.chunk_size, args.chunk_overlap)
        keyword_index = SimpleKeywordTableIndex(nodes, storage_context=index.storage_context)

        vec_ret    = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        kw_ret     = KeywordTableSimpleRetriever(index=keyword_index)
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

    response = engine.query(args.question)
    print(str(response))

def cli():
    ap = argparse.ArgumentParser(description="UIT-Course RAG (Gemini)")
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
    q.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid search combining semantic and keyword",
    )
    q.add_argument(
        "--chunk_size",
        type=int,
        default=150,
        help="Chunk size for keyword index building",
    )
    q.add_argument(
        "--chunk_overlap",
        type=int,
        default=20,
        help="Chunk overlap for keyword index building",
    )
    q.set_defaults(func=query)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    cli()
