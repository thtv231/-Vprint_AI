import argparse
import shutil
import json
from pathlib import Path
from typing import List

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ================================
# CSV STRUCTURE
# ================================

DEFAULT_COLUMNS = [
    "name",
    "source_url",
    "category_id",
    "product_url",
    "sku",
    "price",
    "view_count",
    "summary",
    "description",
    "features",
    "specs_json",
    "unused",
    "image_urls",
]


# ================================
# ARGUMENTS
# ================================

def resolve_default_csv_path():

    for p in ["vprint_products_clean.csv", "vprint_product_clean.csv"]:
        if Path(p).exists():
            return p

    return "vprint_products_clean.csv"


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv-path", default=resolve_default_csv_path())
    # Bạn có thể đổi tên thư mục/collection để không đè lên cái cũ, hoặc dùng --rebuild
    parser.add_argument("--persist-dir", default="vprint_agentic_db_local")
    parser.add_argument("--collection", default="vprint_products_local")

    # ĐÃ SỬA: Thay đổi sang mô hình chuyên dụng cho Tiếng Việt
    parser.add_argument(
        "--embedding-model",
        default="bkai-foundation-models/vietnamese-bi-encoder",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rebuild", action="store_true")

    return parser.parse_args()


# ================================
# LOAD CSV
# ================================

def load_csv(csv_path: Path):

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # File CSV có header
    df = pd.read_csv(csv_path)

    rename_map = {}

    for idx, col in enumerate(df.columns):
        rename_map[col] = (
            DEFAULT_COLUMNS[idx]
            if idx < len(DEFAULT_COLUMNS)
            else f"extra_{idx}"
        )

    df = df.rename(columns=rename_map)

    df = df.fillna("")

    return df


# ================================
# CLEAN TEXT
# ================================

def clean_text(text):

    if not text:
        return ""

    text = str(text).strip()

    if text.lower() in ["mô tả chung", "nan", "none"]:
        return ""

    return text


# ================================
# BUILD EMBEDDING TEXT
# ================================

def build_document_text(row):

    name = clean_text(row.get("name"))
    summary = clean_text(row.get("summary"))
    description = clean_text(row.get("description"))
    features = clean_text(row.get("features"))
    specs = clean_text(row.get("specs_json"))

    text_blocks = []

    if name:
        text_blocks.append(f"Product: {name}")

    if summary:
        text_blocks.append(f"Application: {summary}")

    if description:
        text_blocks.append(f"Description:\n{description}")

    if features:
        text_blocks.append(f"Features:\n{features}")

    if specs:
        text_blocks.append(f"Specifications:\n{specs}")

    return "\n\n".join(text_blocks)


# ================================
# IMAGE PARSER
# ================================

def parse_image_urls(value):

    if not value:
        return []

    value = str(value).strip()

    if not value:
        return []

    # JSON list
    if value.startswith("["):
        try:
            arr = json.loads(value)
            return [str(x).strip() for x in arr if str(x).strip()]
        except:
            pass

    for sep in [";", ",", "|"]:
        if sep in value:
            return [x.strip() for x in value.split(sep) if x.strip()]

    return [value]


# ================================
# CREATE DOCUMENTS
# ================================

def create_documents(df):

    docs: List[Document] = []

    for idx, row in df.iterrows():

        page_content = build_document_text(row)

        if not page_content:
            continue

        product_url = str(row.get("product_url", "")).strip()

        # Tạo ID chống trùng lặp
        doc_id = f"{product_url}_{idx}" if product_url else f"row-{idx}"

        images = parse_image_urls(row.get("image_urls"))

        # Ép kiểu dữ liệu chuẩn chỉnh cho ChromaDB
        metadata = {
            "row_index": int(idx),
            "name": str(row.get("name", "")),
            "summary": str(row.get("summary", "")),
            "price": str(row.get("price", "")),
            "product_url": str(product_url),
            "source_url": str(row.get("source_url", "")),
            "category_id": str(row.get("category_id", "")),
            "sku": str(row.get("sku", "")),
            "images": json.dumps(images),
            "image_main": str(images[0]) if images else "",
            "image_count": int(len(images)),
        }

        docs.append(
            Document(
                page_content=page_content,
                metadata=metadata,
                id=doc_id,
            )
        )

    return docs


# ================================
# BATCH
# ================================

def batched(items, batch_size):

    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ================================
# MAIN
# ================================

def main():

    load_dotenv()

    args = parse_args()

    csv_path = Path(args.csv_path)
    persist_dir = Path(args.persist_dir)

    if args.rebuild and persist_dir.exists():
        print(f"Rebuilding database (removing old dir: {persist_dir})...")
        shutil.rmtree(persist_dir)

    df = load_csv(csv_path)

    docs = create_documents(df)

    if not docs:
        raise ValueError("No documents generated")

    print("Documents:", len(docs))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device for embedding: {device.upper()}")
    print(f"Loading model: {args.embedding_model}")

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    for chunk in tqdm(
        list(batched(docs, args.batch_size)),
        desc="Embedding",
        unit="batch",
    ):
        vectorstore.add_documents(chunk)

    print(
        f"\nDone. Ingested {len(docs)} documents\n"
        f"Collection : {args.collection}\n"
        f"Persist dir: {persist_dir}\n"
        f"Model      : {args.embedding_model}"
    )


# ================================
# ENTRY
# ================================

if __name__ == "__main__":
    main()