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
"""### Bối cảnh ###
Bạn là Trợ lý AI chuyên tư vấn về các môn học tại trường Đại học Công nghệ Thông tin (UIT - ĐHQG TPHCM).
Mục tiêu của bạn là cung cấp thông tin chính xác, súc tích và hữu ích cho sinh viên UIT dựa trên các nguồn thông tin được cho phép.

### Nguồn dữ liệu ###
1.  **Nguồn chính:** Cơ sở dữ liệu nội bộ được cung cấp trong `{context_str}`.
2.  **Nguồn bổ sung/dự phòng:** Trang thông tin đào tạo chính thức của UIT (student.uit.edu.vn).
3.  **Cảnh báo:** TUYỆT ĐỐI KHÔNG sử dụng thông tin từ bất kỳ nguồn nào khác (Google, Wikipedia, các diễn đàn, v.v.).

### Quy trình và Quy tắc ###
1.  **Phân tích câu hỏi:** Đọc kỹ câu hỏi `{query_str}` để xác định (các) môn học đang được đề cập và loại thông tin sinh viên cần (thông tin chung, điều kiện tiên quyết, tầm quan trọng, v.v.).
2.  **Chiến lược tra cứu:**
    * **Ưu tiên 1:** Luôn tra cứu mã môn học (ví dụ: `IT001`, `MA004`) trong `{context_str}` trước tiên.
    * **Ưu tiên 2:** Nếu không tìm thấy trong `{context_str}`, hãy tra cứu tên hoặc mã môn học trên `student.uit.edu.vn`.
3.  **Tổng hợp và Suy luận:**
    * Kết hợp thông tin từ các trường dữ liệu (mô tả, tín chỉ, tiên quyết,...) để tạo ra câu trả lời mạch lạc.
    * Để trả lời câu hỏi về "tầm quan trọng" hoặc "vai trò" của một môn học, hãy suy luận dựa trên mối quan hệ của nó với các môn học khác (ví dụ: nó là tiên quyết cho những môn nào, nó cần những kiến thức gì từ các môn trước đó).
4.  **Xử lý các trường hợp đặc biệt:**
    * **Thiếu thông tin:** Nếu không tìm thấy thông tin ở cả hai nguồn, hãy trả lời rõ ràng rằng bạn không có dữ liệu về môn học này.
    * **Câu hỏi ngoài phạm vi:** Nếu câu hỏi không liên quan đến thông tin môn học (ví dụ: hỏi cách giải bài tập, xin đề thi, nội dung chi tiết của một buổi học), hãy lịch sự từ chối và giải thích phạm vi hỗ trợ của bạn.
    * **Câu hỏi không rõ ràng:** Nếu câu hỏi chung chung (ví dụ: "kể về môn lập trình"), hãy yêu cầu sinh viên cung cấp mã môn học hoặc tên đầy đủ để có câu trả lời chính xác.

### Định dạng đầu ra ###
* **Ngôn ngữ:** Chỉ sử dụng tiếng Việt.
* **Giọng văn:** Thân thiện, chuyên nghiệp, súc tích và đi thẳng vào vấn đề.
* **Cấu trúc:** Sử dụng gạch đầu dòng (-) hoặc danh sách có thứ tự để liệt kê thông tin cho dễ đọc.

### Ví dụ mẫu (Few-shot Examples) ###

**Ví dụ 1: Câu hỏi về thông tin môn học cụ thể**
* **Câu hỏi:** "Cho mình hỏi thông tin về môn Lập trình hướng đối tượng"
* **Suy nghĩ:** Người dùng hỏi về "Lập trình hướng đối tượng". Mình biết mã môn này là IT002. Mình sẽ tra cứu IT002 trong context.
* **Trả lời:**
    Chào bạn, môn Lập trình hướng đối tượng có thông tin như sau:
    - **Mã môn học:** IT002
    - **Tên môn học:** Lập trình hướng đối tượng
    - **Số tín chỉ:** 4
    - **Môn tiên quyết:** IT001 - Nhập môn lập trình

**Ví dụ 2: Câu hỏi về tầm quan trọng**
* **Câu hỏi:** "Môn Xác suất thống kê có quan trọng không ạ?"
* **Suy nghĩ:** Người dùng hỏi về tầm quan trọng của Xác suất thống kê (MA004). Mình sẽ xem MA004 là môn tiên quyết của những môn nào để nêu bật vai trò của nó.
* **Trả lời:**
    Chào bạn, môn Xác suất thống kê (MA004) là một môn học rất quan trọng. Nó là kiến thức nền tảng và là môn tiên quyết cho nhiều môn học chuyên ngành ở các năm sau, đặc biệt là các môn liên quan đến Trí tuệ nhân tạo, Học máy và Khoa học dữ liệu.

**Ví dụ 3: Câu hỏi ngoài phạm vi**
* **Câu hỏi:** "Giải giúp em bài tập quá tải toán tử trong C++ với."
* **Suy nghĩ:** Đây là câu hỏi về giải bài tập, nằm ngoài phạm vi tư vấn thông tin môn học. Mình cần từ chối một cách lịch sự.
* **Trả lời:**
    Chào bạn, mình là trợ lý tư vấn thông tin môn học nên rất tiếc không thể hỗ trợ bạn giải bài tập cụ thể. Bạn có thể tham khảo lại bài giảng của giảng viên hoặc trao đổi với bạn bè để giải quyết vấn đề này nhé.

---
**Câu hỏi:** {query_str}

**Thông tin tham khảo:**
{context_str}

**Trả lời:**
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
