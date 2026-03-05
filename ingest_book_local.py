import argparse
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", type=str, required=True, help="Đường dẫn đến file PDF sách")
    parser.add_argument("--persist-dir", default="vprint_agentic_db_local")
    parser.add_argument("--collection", default="vprint_knowledge_base") 
    
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-large")
    
    # THAY ĐỔI QUAN TRỌNG 1: Batch size mặc định giảm xuống 2 cho GPU 2GB
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rebuild", action="store_true")
    
    return parser.parse_args()

def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

def main():
    args = parse_args()
    pdf_path = Path(args.pdf_path)
    persist_dir = Path(args.persist_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

    if args.rebuild and persist_dir.exists():
        print(f"Xóa database cũ tại: {persist_dir}...")
        shutil.rmtree(persist_dir)

    print(f"Đang đọc file PDF: {pdf_path.name}...")
    loader = PyMuPDFLoader(str(pdf_path))
    raw_documents = loader.load()
    print(f"Đã tải {len(raw_documents)} trang.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    print("Đang chia nhỏ văn bản (Chunking)...")
    docs = text_splitter.split_documents(raw_documents)
    
    for i, doc in enumerate(docs):
        doc.metadata["source_book"] = pdf_path.name
        doc.metadata["chunk_id"] = i

    print(f"Đã tạo {len(docs)} chunks.")

    # Xác định thiết bị
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Đang tải Embedding Model: {args.embedding_model} trên {device.upper()}...")
    
    # THAY ĐỔI QUAN TRỌNG 2: Ép chạy FP16 để tiết kiệm VRAM
    model_kwargs = {"device": device}
    if device == "cuda":
        print("Kích hoạt chế độ FP16 cho GPU để tiết kiệm VRAM...")
        model_kwargs["torch_dtype"] = torch.float16

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    print("Đang nạp dữ liệu vào ChromaDB...")
    for chunk in tqdm(
        list(batched(docs, args.batch_size)),
        desc="Embedding",
        unit="batch",
    ):
        vectorstore.add_documents(chunk)

    print("\nHoàn tất! Dữ liệu sách đã được nạp thành công.")

if __name__ == "__main__":
    main()