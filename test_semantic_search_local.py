import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
#from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

def parse_args():
    parser = argparse.ArgumentParser(description="Test Hybrid Search (Semantic + BM25)")
    parser.add_argument("--persist-dir", default="vprint_agentic_db_local", help="Thư mục chứa ChromaDB")
    parser.add_argument("--collection", default="vprint_products_local", help="Tên Collection cần tìm") 
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Tên model OpenAI")
    
    # Cho phép nhập query tay hoặc chạy chế độ test suite
    parser.add_argument("--query", type=str, help="Câu hỏi tìm kiếm thủ công")
    parser.add_argument("--run-tests", action="store_true", help="Chạy bộ câu hỏi test có sẵn")
    parser.add_argument("--k", type=int, default=3, help="Số lượng kết quả trả về")
    
    return parser.parse_args()

def get_hybrid_retriever(vectorstore, k):
    """Tạo bộ thu thập lai (Hybrid Retriever) kết hợp Vector và BM25"""
    # 1. Vector Retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # 2. BM25 Retriever (Cần load toàn bộ docs từ Chroma để build index từ khóa)
    db_data = vectorstore.get()
    docs = [
        Document(page_content=content, metadata=meta) 
        for content, meta in zip(db_data['documents'], db_data['metadatas'])
    ]
    
    if not docs:
        print("⚠️ Database trống, không thể tạo BM25 Retriever.")
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    
    # 3. Ensemble Retriever (Trọng số: 50% Ngữ nghĩa - 50% Từ khóa)
    # EnsembleRetriever sử dụng Reciprocal Rank Fusion (RRF) để xếp hạng lại
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def run_query(query, collection_name, persist_dir, embeddings, k):
    """Hàm thực thi tìm kiếm và in kết quả cho 1 câu hỏi"""
    print(f"\n" + "="*80)
    print(f"🔍 ĐANG TÌM KIẾM: '{query}'")
    print(f"📂 Kho dữ liệu: '{collection_name}'")
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    
    # Lấy chỉ số Similarity (Vector distance) để kiểm tra sim (Chỉ dùng Vector Search)
    print("\n[BƯỚC 1] 📊 Phân tích Chỉ số Vector thuần (Càng gần 0 càng tốt):")
    vector_results = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(vector_results):
        print(f"   - Vector Top {i+1} | K/c: {score:.4f} | Nguồn: {doc.metadata.get('name', doc.metadata.get('source_book', 'N/A'))}")

    # Chạy Hybrid Search (BM25 + Vector kết hợp bằng RRF)
    print("\n[BƯỚC 2] 🚀 KẾT QUẢ HYBRID SEARCH CHÍNH THỨC:")
    hybrid_retriever = get_hybrid_retriever(vectorstore, k=k)
    hybrid_results = hybrid_retriever.invoke(query)

    if not hybrid_results:
        print("⚠️ Không tìm thấy kết quả nào.")
        return

    for i, doc in enumerate(hybrid_results):
        print(f"\n🏆 TOP {i + 1} HYBRID RANKING") 
        print("-" * 60)
        
        # In Metadata tùy theo Collection
        if collection_name == "vprint_products_local":
            print(f"Tên máy : {doc.metadata.get('name', 'N/A')}")
            print(f"Link    : {doc.metadata.get('product_url', 'N/A')}")
        else:
            print(f"Nguồn   : {doc.metadata.get('source_book', 'N/A')} (Chunk: {doc.metadata.get('chunk_id', 'N/A')})")
            
        print("\n📄 Nội dung (Content):")
        # Rút gọn nội dung để dễ nhìn khi test nhiều câu
        snippet = doc.page_content[:400] + "\n[... đã thu gọn ...]" if len(doc.page_content) > 400 else doc.page_content
        print(snippet)
        print("-" * 60)

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ LỖI: Không tìm thấy OPENAI_API_KEY trong file .env!")
        return

    args = parse_args()
    persist_dir = Path(args.persist_dir)

    if not persist_dir.exists():
        print(f"❌ LỖI: Không tìm thấy thư mục Database tại '{persist_dir}'.")
        return

    embeddings = OpenAIEmbeddings(model=args.embedding_model)

    # Chạy chế độ Test Suite
    if args.run_tests:
        print("🔥 BẮT ĐẦU CHẠY BỘ TEST SUITE ĐÁNH GIÁ...")
        test_queries = [
            # Test tìm kiếm Máy móc
            ("Cho tôi các máy có thể làm ly giấy. [Xác định dung lượng sản xuất cần thiết (cái/phút) và loại ly (độ dày, hình dạng) để chọn máy phù hợp.]", "vprint_products_local"),
            ("OK, tôi cần toàn bộ dây chuyền máy để làm ly 20oz, sản lượng 1triệu/tháng", "vprint_products_local"),
            ("Tôi đang cần đầu tư máy ghi bản kẽm ctp cho máy in offset 72*102. Hãy tư vấn cho tôi nên chọn đầu tư loại nào", "vprint_products_local"),
            # Test tìm kiếm Kiến thức Handbook
            ("phân biệt công nghệ flexo , digital, và offset", "vprint_knowledge_base"),
            ("Chế bản điện tử là gì?", "vprint_knowledge_base")
        ]
        
        for q, col in test_queries:
            run_query(query=q, collection_name=col, persist_dir=persist_dir, embeddings=embeddings, k=args.k)
            
    # Chạy bằng cách truyền tham số thủ công
    elif args.query:
        run_query(query=args.query, collection_name=args.collection, persist_dir=persist_dir, embeddings=embeddings, k=args.k)
    else:
        print("⚠️ Vui lòng cung cấp --query hoặc chạy với tham số --run-tests để kiểm tra bộ câu hỏi mẫu.")

if __name__ == "__main__":
    main()