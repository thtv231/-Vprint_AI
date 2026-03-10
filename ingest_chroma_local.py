# # import argparse
# # import shutil
# # import json
# # from pathlib import Path
# # from typing import List

# # import pandas as pd
# # import torch
# # from dotenv import load_dotenv
# # from tqdm import tqdm

# # from langchain_chroma import Chroma
# # from langchain_core.documents import Document

# # try:
# #     from langchain_huggingface import HuggingFaceEmbeddings
# # except Exception:
# #     from langchain_community.embeddings import HuggingFaceEmbeddings


# # # ================================
# # # CSV STRUCTURE
# # # ================================

# # DEFAULT_COLUMNS = [
# #     "name",
# #     "source_url",
# #     "category_id",
# #     "product_url",
# #     "sku",
# #     "price",
# #     "view_count",
# #     "summary",
# #     "description",
# #     "features",
# #     "specs_json",
# #     "unused",
# #     "image_urls",
# # ]


# # # ================================
# # # ARGUMENTS
# # # ================================

# # def resolve_default_csv_path():

# #     for p in ["vprint_products_clean.csv", "vprint_product_clean.csv"]:
# #         if Path(p).exists():
# #             return p

# #     return "vprint_products_clean.csv"


# # def parse_args():

# #     parser = argparse.ArgumentParser()

# #     parser.add_argument("--csv-path", default=resolve_default_csv_path())
# #     # Bạn có thể đổi tên thư mục/collection để không đè lên cái cũ, hoặc dùng --rebuild
# #     parser.add_argument("--persist-dir", default="vprint_agentic_db_local")
# #     parser.add_argument("--collection", default="vprint_products_local")

# #     # ĐÃ SỬA: Thay đổi sang mô hình chuyên dụng cho Tiếng Việt
# #     parser.add_argument(
# #         "--embedding-model",
# #         default="bkai-foundation-models/vietnamese-bi-encoder",
# #     )

# #     parser.add_argument("--batch-size", type=int, default=64)
# #     parser.add_argument("--rebuild", action="store_true")

# #     return parser.parse_args()


# # # ================================
# # # LOAD CSV
# # # ================================

# # def load_csv(csv_path: Path):

# #     if not csv_path.exists():
# #         raise FileNotFoundError(f"CSV not found: {csv_path}")

# #     # File CSV có header
# #     df = pd.read_csv(csv_path)

# #     rename_map = {}

# #     for idx, col in enumerate(df.columns):
# #         rename_map[col] = (
# #             DEFAULT_COLUMNS[idx]
# #             if idx < len(DEFAULT_COLUMNS)
# #             else f"extra_{idx}"
# #         )

# #     df = df.rename(columns=rename_map)

# #     df = df.fillna("")

# #     return df


# # # ================================
# # # CLEAN TEXT
# # # ================================

# # def clean_text(text):

# #     if not text:
# #         return ""

# #     text = str(text).strip()

# #     if text.lower() in ["mô tả chung", "nan", "none"]:
# #         return ""

# #     return text


# # # ================================
# # # BUILD EMBEDDING TEXT
# # # ================================

# # def build_document_text(row):

# #     name = clean_text(row.get("name"))
# #     summary = clean_text(row.get("summary"))
# #     description = clean_text(row.get("description"))
# #     features = clean_text(row.get("features"))
# #     specs = clean_text(row.get("specs_json"))

# #     text_blocks = []

# #     if name:
# #         text_blocks.append(f"Product: {name}")

# #     if summary:
# #         text_blocks.append(f"Application: {summary}")

# #     if description:
# #         text_blocks.append(f"Description:\n{description}")

# #     if features:
# #         text_blocks.append(f"Features:\n{features}")

# #     if specs:
# #         text_blocks.append(f"Specifications:\n{specs}")

# #     return "\n\n".join(text_blocks)


# # # ================================
# # # IMAGE PARSER
# # # ================================

# # def parse_image_urls(value):

# #     if not value:
# #         return []

# #     value = str(value).strip()

# #     if not value:
# #         return []

# #     # JSON list
# #     if value.startswith("["):
# #         try:
# #             arr = json.loads(value)
# #             return [str(x).strip() for x in arr if str(x).strip()]
# #         except:
# #             pass

# #     for sep in [";", ",", "|"]:
# #         if sep in value:
# #             return [x.strip() for x in value.split(sep) if x.strip()]

# #     return [value]


# # # ================================
# # # CREATE DOCUMENTS
# # # ================================

# # def create_documents(df):

# #     docs: List[Document] = []

# #     for idx, row in df.iterrows():

# #         page_content = build_document_text(row)

# #         if not page_content:
# #             continue

# #         product_url = str(row.get("product_url", "")).strip()

# #         # Tạo ID chống trùng lặp
# #         doc_id = f"{product_url}_{idx}" if product_url else f"row-{idx}"

# #         images = parse_image_urls(row.get("image_urls"))

# #         # Ép kiểu dữ liệu chuẩn chỉnh cho ChromaDB
# #         metadata = {
# #             "row_index": int(idx),
# #             "name": str(row.get("name", "")),
# #             "summary": str(row.get("summary", "")),
# #             "price": str(row.get("price", "")),
# #             "product_url": str(product_url),
# #             "source_url": str(row.get("source_url", "")),
# #             "category_id": str(row.get("category_id", "")),
# #             "sku": str(row.get("sku", "")),
# #             "images": json.dumps(images),
# #             "image_main": str(images[0]) if images else "",
# #             "image_count": int(len(images)),
# #         }

# #         docs.append(
# #             Document(
# #                 page_content=page_content,
# #                 metadata=metadata,
# #                 id=doc_id,
# #             )
# #         )

# #     return docs


# # # ================================
# # # BATCH
# # # ================================

# # def batched(items, batch_size):

# #     for i in range(0, len(items), batch_size):
# #         yield items[i : i + batch_size]


# # # ================================
# # # MAIN
# # # ================================

# # def main():

# #     load_dotenv()

# #     args = parse_args()

# #     csv_path = Path(args.csv_path)
# #     persist_dir = Path(args.persist_dir)

# #     if args.rebuild and persist_dir.exists():
# #         print(f"Rebuilding database (removing old dir: {persist_dir})...")
# #         shutil.rmtree(persist_dir)

# #     df = load_csv(csv_path)

# #     docs = create_documents(df)

# #     if not docs:
# #         raise ValueError("No documents generated")

# #     print("Documents:", len(docs))

# #     device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# #     print(f"Using device for embedding: {device.upper()}")
# #     print(f"Loading model: {args.embedding_model}")

# #     embeddings = HuggingFaceEmbeddings(
# #         model_name=args.embedding_model,
# #         model_kwargs={"device": device},
# #         encode_kwargs={"normalize_embeddings": True},
# #     )

# #     vectorstore = Chroma(
# #         collection_name=args.collection,
# #         embedding_function=embeddings,
# #         persist_directory=str(persist_dir),
# #     )

# #     for chunk in tqdm(
# #         list(batched(docs, args.batch_size)),
# #         desc="Embedding",
# #         unit="batch",
# #     ):
# #         vectorstore.add_documents(chunk)

# #     print(
# #         f"\nDone. Ingested {len(docs)} documents\n"
# #         f"Collection : {args.collection}\n"
# #         f"Persist dir: {persist_dir}\n"
# #         f"Model      : {args.embedding_model}"
# #     )


# # # ================================
# # # ENTRY
# # # ================================

# # if __name__ == "__main__":
# #     main()


# import argparse
# import json
# from pathlib import Path
# from typing import List

# import pandas as pd
# import torch
# from dotenv import load_dotenv
# from tqdm import tqdm
# import chromadb

# from langchain_chroma import Chroma
# from langchain_core.documents import Document

# try:
#     from langchain_huggingface import HuggingFaceEmbeddings
# except Exception:
#     from langchain_community.embeddings import HuggingFaceEmbeddings

# DEFAULT_COLUMNS = [
#     "name", "source_url", "category_id", "product_url", "sku", 
#     "price", "view_count", "summary", "description", "features", 
#     "specs_json", "unused", "image_urls"
# ]

# def resolve_default_csv_path():
#     for p in ["vprint_products_clean.csv", "vprint_product_clean.csv"]:
#         if Path(p).exists(): return p
#     return "vprint_products_clean.csv"

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv-path", default=resolve_default_csv_path())
#     parser.add_argument("--persist-dir", default="vprint_agentic_db_local")
#     parser.add_argument("--collection", default="vprint_products_local")
#     parser.add_argument("--embedding-model", default="bkai-foundation-models/vietnamese-bi-encoder")
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument("--rebuild", action="store_true")
#     return parser.parse_args()

# def load_csv(csv_path: Path):
#     if not csv_path.exists(): raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")
#     df = pd.read_csv(csv_path)
#     rename_map = {col: (DEFAULT_COLUMNS[idx] if idx < len(DEFAULT_COLUMNS) else f"extra_{idx}") for idx, col in enumerate(df.columns)}
#     df = df.rename(columns=rename_map).fillna("")
#     return df

# def clean_text(text):
#     if not text: return ""
#     text = str(text).strip()
#     return "" if text.lower() in ["mô tả chung", "nan", "none"] else text

# def format_specs(specs_str):
#     """Dịch chuỗi JSON thông số thành ngôn ngữ tự nhiên (Bullet points)"""
#     if not specs_str: return ""
#     try:
#         specs_dict = json.loads(specs_str)
#         if isinstance(specs_dict, dict):
#             lines = [f"- {k}: {v}" for k, v in specs_dict.items() if str(v).strip() and str(v).lower() != "nan"]
#             return "\n".join(lines)
#     except: pass
#     return str(specs_str)

# def parse_image_urls(value):
#     if not value: return []
#     value = str(value).strip()
#     if value.startswith("["):
#         try:
#             arr = json.loads(value)
#             return [str(x).strip() for x in arr if str(x).strip()]
#         except: pass
#     for sep in [";", ",", "|"]:
#         if sep in value: return [x.strip() for x in value.split(sep) if x.strip()]
#     return [value]

# def create_documents(df):
#     docs: List[Document] = []
#     for idx, row in df.iterrows():
#         name = clean_text(row.get("name"))
#         summary = clean_text(row.get("summary"))
#         description = clean_text(row.get("description"))
#         features = clean_text(row.get("features"))
#         specs_natural = format_specs(clean_text(row.get("specs_json")))

#         if not name: continue

#         product_url = str(row.get("product_url", "")).strip()
#         images = parse_image_urls(row.get("image_urls"))

#         # Metadata chuẩn xác cho ChromaDB (Chỉ nhận chuỗi, số, boolean)
#         metadata = {
#             "row_index": int(idx),
#             "name": name,
#             "product_url": product_url,
#             "category_id": str(row.get("category_id", "")),
#             "images": json.dumps(images),
#         }

#         # CHUNK 1: TỔNG QUAN & TÍNH NĂNG
#         content_1 = f"Tên máy: {name}\n"
#         if summary: content_1 += f"Ứng dụng: {summary}\n"
#         if description: content_1 += f"Mô tả: {description}\n"
#         if features: content_1 += f"Tính năng nổi bật:\n{features}\n"
        
#         docs.append(Document(page_content=content_1.strip(), metadata=metadata, id=f"doc_{idx}_overview"))

#         # CHUNK 2: THÔNG SỐ KỸ THUẬT (Chống tràn Token)
#         if specs_natural:
#             content_2 = f"Tên máy: {name}\nThông số kỹ thuật chi tiết:\n{specs_natural}"
#             docs.append(Document(page_content=content_2.strip(), metadata=metadata, id=f"doc_{idx}_specs"))

#     return docs

# def batched(items, batch_size):
#     for i in range(0, len(items), batch_size): yield items[i : i + batch_size]

# def main():
#     load_dotenv()
#     args = parse_args()
#     persist_dir = Path(args.persist_dir)

#     # CƠ CHẾ DỌN DẸP AN TOÀN
#     if args.rebuild and persist_dir.exists():
#         print(f"🔄 Đang dọn dẹp Collection: '{args.collection}'...")
#         try:
#             client = chromadb.PersistentClient(path=str(persist_dir))
#             client.delete_collection(name=args.collection)
#             print("✅ Đã xóa sạch dữ liệu cũ. Các Collection khác vẫn an toàn!")
#         except Exception as e:
#             print(f"⚠️ Bỏ qua bước xóa (Collection có thể chưa tồn tại). Chi tiết: {e}")

#     df = load_csv(Path(args.csv_path))
#     docs = create_documents(df)
#     print(f"📦 Đã xử lý {len(df)} máy thành {len(docs)} chunks.")

#     device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#     embeddings = HuggingFaceEmbeddings(
#         model_name=args.embedding_model, 
#         model_kwargs={"device": device}, 
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     vectorstore = Chroma(
#         collection_name=args.collection, 
#         embedding_function=embeddings, 
#         persist_directory=str(persist_dir)
#     )

#     for chunk in tqdm(list(batched(docs, args.batch_size)), desc="Nhúng Vector (Embedding)"):
#         vectorstore.add_documents(chunk)

#     print("\n✅ Hoàn tất nạp dữ liệu MÁY MÓC!")

# if __name__ == "__main__":
#     main()


import os
import shutil
import stat
import json
import torch
import chromadb
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# --- 1. KẾT NỐI GOOGLE DRIVE ---
from google.colab import drive
drive.mount('/content/drive')

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 2. CẤU HÌNH HỆ THỐNG ---
WORK_DIR = "/content/drive/MyDrive/DoA/chatbot/VPrint_RAG"
CSV_PATH = f"{WORK_DIR}/vprint_products_clean.csv" # Dùng file đã làm sạch
PDF_PATH = f"{WORK_DIR}/Handbook_of_Print_Media.pdf"

# Đường dẫn DB
DRIVE_PERSIST_DIR = f"{WORK_DIR}/vprint_agentic_db_local"
LOCAL_PERSIST_DIR = "/content/vprint_agentic_db_local"

# Cấu hình Collection & Model (GỘP VỀ 1 MODEL DUY NHẤT)
COLLECTION_MACHINES = "vprint_products_local"
COLLECTION_BOOK = "vprint_knowledge_base"
EMBED_MODEL = "intfloat/multilingual-e5-large" # Dùng chung cho cả Máy và Sách

BATCH_SIZE_MACHINES = 32
BATCH_SIZE_BOOK = 16
REBUILD_ALL = True # True = Xóa sạch DB cũ làm lại từ đầu

# --- 3. HÀM HỖ TRỢ ---
def clean_text(text):
    text = str(text).strip()
    return "" if text.lower() in ["mô tả chung", "nan", "none", ""] else text

def format_specs(specs_str):
    try:
        specs_dict = json.loads(specs_str)
        if isinstance(specs_dict, dict):
            return "\n".join([f"- {k}: {v}" for k, v in specs_dict.items() if str(v).strip() and str(v).lower() != "nan"])
    except: pass
    return str(specs_str)

def batched(items, batch_size):
    for i in range(0, len(items), batch_size): yield items[i : i + batch_size]

# =====================================================================
# BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG CỤC BỘ (LOCAL SSD) - SẠCH SẼ 100%
# =====================================================================
print("="*60)
print("📥 BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG CỤC BỘ (LOCAL SSD)")
print("="*60)

if not os.path.exists(CSV_PATH) or not os.path.exists(PDF_PATH):
    raise FileNotFoundError("⚠️ Thiếu file CSV hoặc PDF trên Drive. Vui lòng kiểm tra lại đường dẫn!")

# 1. Quét sạch tàn dư Local từ các phiên chạy lỗi trước
if os.path.exists(LOCAL_PERSIST_DIR):
    print("🧹 Đang dọn rác Local từ phiên làm việc cũ...")
    for root, dirs, files in os.walk(LOCAL_PERSIST_DIR):
        for d in dirs: os.chmod(os.path.join(root, d), 0o777)
        for f in files: os.chmod(os.path.join(root, f), 0o777)
    shutil.rmtree(LOCAL_PERSIST_DIR)

# 2. Kéo DB từ Drive xuống (nếu có) và bẻ khóa Read-only
if os.path.exists(DRIVE_PERSIST_DIR):
    print("⏳ Đang copy Database từ Drive xuống ổ cứng Colab...")
    shutil.copytree(DRIVE_PERSIST_DIR, LOCAL_PERSIST_DIR)

    print("🔓 Đang cấp quyền Write cho file SQLite từ Drive...")
    for root, dirs, files in os.walk(LOCAL_PERSIST_DIR):
        for d in dirs: os.chmod(os.path.join(root, d), 0o777)
        for f in files: os.chmod(os.path.join(root, f), 0o777)
else:
    print("✨ Chưa có Database trên Drive, sẽ tạo thư mục mới tinh.")
    os.makedirs(LOCAL_PERSIST_DIR, exist_ok=True)

# Khởi tạo Client DUY NHẤT để tránh lỗi Lock Database
chroma_client = chromadb.PersistentClient(path=LOCAL_PERSIST_DIR)
if REBUILD_ALL:
    for col_name in [COLLECTION_MACHINES, COLLECTION_BOOK]:
        try:
            chroma_client.delete_collection(name=col_name)
            print(f"🧹 Đã xóa Collection cũ để làm mới: {col_name}")
        except: pass

# Tải Model Embedding 1 LẦN DUY NHẤT cho cả hệ thống
print(f"🚀 Tải model Embedding chung: {EMBED_MODEL}...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cuda"})


# =====================================================================
# BƯỚC 2: NẠP DỮ LIỆU MÁY MÓC (PRODUCTS DB)
# =====================================================================
print("\n" + "="*60)
print("⚙️ BƯỚC 2: NẠP DỮ LIỆU MÁY MÓC (PRODUCTS DB)")
print("="*60)

print("📦 Đọc & tạo Chunk cho CSV...")
df = pd.read_csv(CSV_PATH).fillna("")
docs_machines = []

for idx, row in df.iterrows():
    name = clean_text(row.get("name"))
    if not name: continue

    metadata = {"name": name, "product_url": str(row.get("product_url", ""))}

    content_1 = f"Tên máy: {name}\nỨng dụng: {clean_text(row.get('summary'))}\nMô tả: {clean_text(row.get('description'))}\nTính năng: {clean_text(row.get('features'))}"
    docs_machines.append(Document(page_content=content_1.strip(), metadata=metadata, id=f"doc_{idx}_overview"))

    specs_natural = format_specs(clean_text(row.get("specs_json")))
    if specs_natural:
        content_2 = f"Tên máy: {name}\nThông số kỹ thuật chi tiết:\n{specs_natural}"
        docs_machines.append(Document(page_content=content_2.strip(), metadata=metadata, id=f"doc_{idx}_specs"))

print(f"✅ Đã tạo {len(docs_machines)} chunks máy móc.")

# Truyền chroma_client vào để nạp dữ liệu máy
vectorstore_machine = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_MACHINES,
    embedding_function=embeddings
)

for chunk in tqdm(list(batched(docs_machines, BATCH_SIZE_MACHINES)), desc="Nhúng Vector Máy móc"):
    vectorstore_machine.add_documents(chunk)


# =====================================================================
# BƯỚC 3: NẠP DỮ LIỆU CẨM NANG (HANDBOOK DB)
# =====================================================================
print("\n" + "="*60)
print("📖 BƯỚC 3: NẠP DỮ LIỆU CẨM NANG (HANDBOOK DB)")
print("="*60)

print(f"Đọc file PDF: {Path(PDF_PATH).name}...")
loader = PyMuPDFLoader(PDF_PATH)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500, separators=["\n\n\n", "\n\n", "\n", ". ", " "])
docs_book = text_splitter.split_documents(raw_documents)

for i, doc in enumerate(docs_book):
    doc.metadata["source_book"] = Path(PDF_PATH).name
    doc.metadata["chunk_id"] = i
    doc.metadata["page"] = doc.metadata.get("page", 0)

print(f"✅ Đã tạo {len(docs_book)} chunks sách (Bảo toàn bảng biểu).")

# Truyền chroma_client vào để nạp dữ liệu sách
vectorstore_book = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_BOOK,
    embedding_function=embeddings
)

for chunk in tqdm(list(batched(docs_book, BATCH_SIZE_BOOK)), desc="Nhúng Vector Sách"):
    vectorstore_book.add_documents(chunk)


# =====================================================================
# BƯỚC 4: ĐỒNG BỘ LÊN CLOUD
# =====================================================================
print("\n" + "="*60)
print("📤 BƯỚC 4: ĐỒNG BỘ TRỞ LẠI GOOGLE DRIVE")
print("="*60)

if os.path.exists(DRIVE_PERSIST_DIR):
    for root, dirs, files in os.walk(DRIVE_PERSIST_DIR):
        for d in dirs: os.chmod(os.path.join(root, d), 0o777)
        for f in files: os.chmod(os.path.join(root, f), 0o777)
    shutil.rmtree(DRIVE_PERSIST_DIR)

shutil.copytree(LOCAL_PERSIST_DIR, DRIVE_PERSIST_DIR)

drive.flush_and_unmount()
drive.mount('/content/drive')

print("🎉 XUẤT SẮC! HỆ THỐNG VECTOR DATABASE TỔNG HỢP (1 MODEL) ĐÃ HOÀN THIỆN!")