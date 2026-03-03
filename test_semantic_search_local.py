import os
import json
import torch
import pandas as pd
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ================================
# CONFIG
# ================================

CSV_PATH = "vprint_products_clean.csv" # Đổi tên thành file CSV của bạn nếu khác
PERSIST_DIR = "vprint_agentic_db_local"
COLLECTION_NAME = "vprint_products_local"

# Cùng mô hình tiếng việt đã dùng ở file Ingest
EMBED_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
TOP_K = 5

DEFAULT_COLUMNS = [
    "name", "source_url", "category_id", "product_url", "sku", "price", 
    "view_count", "summary", "description", "features", "specs_json", "unused", "image_urls"
]

# ================================
# 1. LOAD TỪ KHÓA (BM25) TỪ CSV
# ================================
def build_bm25_retriever():
    print("Đang tải dữ liệu từ vựng (BM25 Keyword Search)...")
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"Không tìm thấy file {CSV_PATH}. Tính năng Hybrid cần đọc CSV.")
        
    df = pd.read_csv(CSV_PATH)
    
    # Gắn đúng tên cột
    rename_map = {col: DEFAULT_COLUMNS[i] if i < len(DEFAULT_COLUMNS) else f"extra_{i}" 
                  for i, col in enumerate(df.columns)}
    df = df.rename(columns=rename_map).fillna("")

    docs = []
    for idx, row in df.iterrows():
        # Lắp ghép text giống hệt lúc ingest để so khớp chuẩn nhất
        text_blocks = []
        name = str(row.get("name")).strip()
        summary = str(row.get("summary")).strip()
        
        if name: text_blocks.append(f"Product: {name}")
        if summary: text_blocks.append(f"Application: {summary}")
        
        page_content = "\n\n".join(text_blocks)
        if not page_content: continue

        metadata = {
            "name": name,
            "summary": summary,
            "price": str(row.get("price")),
            "product_url": str(row.get("product_url")),
        }
        docs.append(Document(page_content=page_content, metadata=metadata))

    # Khởi tạo bộ máy tìm kiếm từ khóa BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = TOP_K
    return bm25_retriever


# ================================
# 2. LOAD NGỮ NGHĨA (CHROMADB)
# ================================
def build_vector_retriever():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Đang tải Vector DB (Semantic Search) trên {device.upper()}...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not os.path.exists(PERSIST_DIR):
        print(f"\n[CẢNH BÁO] Không tìm thấy DB '{PERSIST_DIR}'. Hãy chạy file ingest_bkai.py --rebuild trước!\n")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    
    # Chuyển đổi vectorstore thành retriever chuẩn của Langchain
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ================================
# FORMAT & PRINT RESULT
# ================================
def print_results(results):
    print("\n" + "=" * 60)
    print(f"KẾT QUẢ TÌM KIẾM HYBRID (Từ khóa + Ngữ nghĩa) - Top {len(results)}")
    print("=" * 60)

    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        name = meta.get("name", "Unknown Name")
        summary = meta.get("summary", "")
        price = meta.get("price", "")
        url = meta.get("product_url", "")
        
        print(f"\n[{i}] {name}")
        
        if price: print(f"Giá bán       : {price}")
        if summary: print(f"Tóm tắt       : {summary[:150]}...")
        if url: print(f"Link sản phẩm : {url}")
        print("-" * 60)


# ================================
# MAIN LOOP
# ================================
def main():
    # Khởi tạo 2 bộ tìm kiếm
    bm25_retriever = build_bm25_retriever()
    vector_retriever = build_vector_retriever()

    # Trộn 2 kết quả lại (Trọng số: 50% cho từ khóa, 50% cho Ngữ nghĩa AI)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    print("\n" + "*"*40)
    print("HỆ THỐNG HYBRID SEARCH SẴN SÀNG")
    print("Gõ 'exit' hoặc 'quit' để thoát.")
    print("*"*40)

    while True:
        try:
            query = input("\nNhập câu hỏi: ").strip()
        except KeyboardInterrupt:
            break

        if not query: continue
        if query.lower() in ["exit", "quit", "thoat"]: break

        # Thực thi Hybrid Search
        results = ensemble_retriever.invoke(query)
        print_results(results)

if __name__ == "__main__":
    main()