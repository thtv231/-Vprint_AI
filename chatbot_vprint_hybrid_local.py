import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
#from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ================================
# IMAGE PARSER (ADD ONLY)
# ================================

def parse_images(value: str) -> List[str]:
    if not value:
        return []

    value = str(value).strip()

    for sep in [";", ",", "|"]:
        if sep in value:
            return [x.strip() for x in value.split(sep) if x.strip()]

    return [value]


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

SYSTEM_PROMPT = """Ban la sales ky thuat cua VPRINT MACHINERY.
- Tu van may in cong nghiep, may be, may dan hop, giai phap bao bi.
- Uu tien de xuat cu thu, khong hoi qua nhieu.
- Neu user dang tim may, dua ra 3 lua chon phu hop + diem noi bat + truong hop nen dung.
- Chi su dung thong tin co trong ngu canh truy xuat. Khong bịa thong so.
- Luon tra loi bang tieng Viet. Khong dung ngon ngu khac.
- Khong tu quy doi tien te. Neu du lieu la "Gia: Lien he" thi giu nguyen.
- Neu chua du du lieu, chi hoi them toi da 1 cau hoi chot.
- Nho lich su hoi thoai de khong hoi lap lai.
"""

DIRECT_CHAT_SYSTEM_PROMPT = """Ban la tro ly sales VPRINT.
- Dung cho cau chao, cam on, hoi cach dung.
- Tra loi ngan gon, than thien.
- Neu user chuyen sang nhu cau may moc, moi ho mo ta nhu cau de tu van.
"""

WELCOME_MESSAGE = (
    "Chao anh/chi, em la tro ly Sales Ky thuat cua VPRINT MACHINERY.\n"
    "Em ho tro tu van may in cong nghiep va giai phap bao bi theo nhu cau san xuat.\n"
    "Anh/chi co the mo ta nhanh nhu cau (vat lieu, san luong, yeu cau chat luong, ngan sach) de em de xuat may phu hop."
)

ROUTER_SYSTEM_PROMPT = """Ban la bo dinh tuyen (router) cho chatbot sales ky thuat VPRINT.
Nhiem vu: voi moi cau hoi, quyet dinh co can truy xuat RAG hay co the tra loi truc tiep.

Tra ve JSON dung schema sau:
{
  "intent": "direct_chat|find_machine|spec_query|price_query|compare_query|rag_general",
  "use_rag": true|false,
  "reset_focus": true|false,
  "direct_answer": "string"
}

Quy tac:
- Neu user chao hoi/cam on/hoi cach dung: intent=direct_chat, use_rag=false, co the tra loi ngan trong direct_answer.
- Neu user tim may/de xuat may: intent=find_machine, use_rag=true, reset_focus=true.
- Neu user hoi thong so/gia/so sanh model: use_rag=true.
- Neu co the tra loi ngan gon ma khong can du lieu san pham: use_rag=false.
- direct_answer chi dung khi use_rag=false.
- Luon dung tieng Viet.
"""


@dataclass
class RouterDecision:
    intent: str
    use_rag: bool
    reset_focus: bool = False
    direct_answer: str = ""


def is_find_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(x in q for x in ["tim may", "muon tim", "can tim", "de xuat may", "goi y may"])


def extract_query_terms(query: str) -> List[str]:
    stop_words = {
        "toi", "muon", "tim", "can", "de", "xuat", "goi", "y", "may", "va", "la", "cho", "cua", "ve", "nhu", "nao", "gi",
    }
    return [t for t in simple_tokenize(query) if len(t) >= 2 and t not in stop_words]


def filter_docs_by_query_terms(docs: List[Document], query: str) -> List[Document]:
    terms = extract_query_terms(query)
    if not terms:
        return docs

    scored: List[Tuple[int, Document]] = []
    for doc in docs:
        haystack = normalize_for_match(f"{doc.metadata.get('name', '')} {doc.page_content}")
        hits = sum(1 for t in terms if t in haystack)
        scored.append((hits, doc))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] > 0:
        return [doc for score, doc in scored if score > 0]
    return docs


def filter_docs_by_need_profile(docs: List[Document], query: str) -> List[Document]:
    q = normalize_for_match(query)
    if not docs:
        return docs

    positive_terms: List[str] = []
    negative_terms: List[str] = []

    if "gap" in q:
        positive_terms = ["gap", "fold", "folder"]
        negative_terms = ["be nhan", "kts", "flexo", "in "]
    elif "be" in q:
        positive_terms = ["be", "cat", "kts", "dao", "die"]
        negative_terms = ["gap giay"]
    elif "in" in q:
        positive_terms = ["in", "printer", "flexo", "offset", "phun"]

    strict_mode = False
    if "gap" in q:
        strict_mode = True

    if not positive_terms and not negative_terms:
        return docs

    scored: List[Tuple[int, Document]] = []
    for doc in docs:
        hay = normalize_for_match(f"{doc.metadata.get('name', '')} {doc.page_content}")
        score = 0
        for t in positive_terms:
            if t in hay:
                score += 3
        for t in negative_terms:
            if t in hay:
                score -= 2
        scored.append((score, doc))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    kept = [d for s, d in scored if s > 0]
    if kept:
        return kept
    if strict_mode:
        return []
    return docs


def extract_labeled_value(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*(.*?)(?=\s*[A-Za-z ]+:|$)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return normalize_text(m.group(1))


def is_spec_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(k in q for k in ["thong so", "thong so ky thuat", "spec", "cau hinh"])


def is_price_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(k in q for k in ["gia", "bao nhieu", "chi phi", "gia tham khao"])


def is_compare_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(k in q for k in ["so sanh", "khac nhau", "nen chon", "chon may nao"])


def is_followup_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(k in q for k in ["may do", "model do", "loai do", "cai do", "no", "mau do"])


def is_direct_chat_intent(query: str) -> bool:
    q = normalize_for_match(query)
    direct_keywords = ["chao", "hello", "hi", "cam on", "ban la ai", "huong dan", "cach dung", "giup toi voi"]
    return any(k in q for k in direct_keywords)


def route_query(user_query: str, history: List[Tuple[str, str]]) -> RouterDecision:
    q = normalize_for_match(user_query)
    if not q:
        return RouterDecision(intent="empty", use_rag=False)

    has_machine_context = any(
        k in q for k in ["may", "in", "be", "cat", "gap", "dan", "carton", "bao bi", "tem", "nhan", "model", "thong so", "gia", "ky thuat"]
    )

    if is_find_intent(q):
        return RouterDecision(intent="find_machine", use_rag=True, reset_focus=True)
    if is_spec_intent(q):
        return RouterDecision(intent="spec_query", use_rag=True)
    if is_compare_intent(q):
        return RouterDecision(intent="compare_query", use_rag=True)
    if is_price_intent(q):
        return RouterDecision(intent="price_query", use_rag=True)
    if is_followup_intent(q) and len(history) > 0:
        return RouterDecision(intent="followup_query", use_rag=True)
    if is_direct_chat_intent(q) and not has_machine_context:
        return RouterDecision(intent="direct_chat", use_rag=False)
    if has_machine_context:
        return RouterDecision(intent="rag_general", use_rag=True)
    return RouterDecision(intent="direct_chat", use_rag=False)


# def llm_route_query(
#     router_llm: ChatOllama,
#     user_query: str,
#     history: List[Tuple[str, str]],
#     fallback_history_limit: int = 4,
# ) -> RouterDecision:
#     recent = history[-fallback_history_limit:]
#     history_lines = []
#     for u, a in recent:
#         history_lines.append(f"User: {u}")
#         history_lines.append(f"Bot: {a}")
#     history_text = "\n".join(history_lines) if history_lines else "(khong co lich su)"

#     messages = [
#         SystemMessage(content=ROUTER_SYSTEM_PROMPT),
#         HumanMessage(
#             content=(
#                 f"Lich su gan day:\n{history_text}\n\n"
#                 f"Cau hoi hien tai:\n{user_query}\n\n"
#                 "Chi tra JSON, khong them text."
#             )
#         ),
#     ]

#     try:
#         res = router_llm.invoke(messages)
#         raw = res.content if isinstance(res.content, str) else str(res.content)
#         start = raw.find("{")
#         end = raw.rfind("}")
#         if start >= 0 and end > start:
#             raw = raw[start : end + 1]
#         data = json.loads(raw)
#         return RouterDecision(
#             intent=str(data.get("intent", "rag_general")),
#             use_rag=bool(data.get("use_rag", True)),
#             reset_focus=bool(data.get("reset_focus", False)),
#             direct_answer=str(data.get("direct_answer", "")).strip(),
#         )
#     except Exception:
#         return route_query(user_query, history)


def apply_router_guards(user_query: str, decision: RouterDecision) -> RouterDecision:
    q = normalize_for_match(user_query)
    if any(k in q for k in ["thong so", "thong so ky thuat", "spec", "cau hinh"]):
        decision.intent = "spec_query"
        decision.use_rag = True
        decision.reset_focus = False
        decision.direct_answer = ""
    if any(k in q for k in ["tim may", "muon tim", "can tim", "de xuat may", "goi y may"]):
        decision.intent = "find_machine"
        decision.use_rag = True
        decision.reset_focus = True
        decision.direct_answer = ""
    return decision


def match_score_doc(doc: Document, query: str) -> int:
    terms = extract_query_terms(query)
    if not terms:
        return 0
    haystack = normalize_for_match(f"{doc.metadata.get('name', '')} {doc.page_content}")
    return sum(2 if t in normalize_for_match(str(doc.metadata.get("name", ""))) else 1 for t in terms if t in haystack)


def pick_best_doc_for_query(query: str, docs: List[Document]) -> Document | None:
    if not docs:
        return None
    ranked = sorted(((match_score_doc(d, query), d) for d in docs), key=lambda x: x[0], reverse=True)
    if ranked[0][0] <= 0:
        return docs[0]
    return ranked[0][1]


def parse_specs_to_lines(specs_text: str) -> List[str]:
    raw = specs_text.strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            lines = [f"- {k}: {v}" for k, v in data.items() if str(v).strip()]
            return lines[:12]
    except Exception:
        pass

    cleaned = normalize_text(raw).replace("{", "").replace("}", "")
    if not cleaned:
        return []
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return [f"- {p}" for p in parts[:12]]


def build_specs_answer(query: str, doc: Document | None) -> str:
    if doc is None:
        return (
            "Em chua xac dinh duoc dung model tu cau hoi. "
            "Anh/chi vui long gui dung ten model (vi du: Ares-350A, FD 970x550) de em tra thong so chinh xac."
        )

    name = doc.metadata.get("name", "Khong ro ten")
    price = doc.metadata.get("price", "Gia: Lien he")
    url = doc.metadata.get("product_url", "")

    summary = extract_labeled_value(doc.page_content, "Summary")
    features = extract_labeled_value(doc.page_content, "Features")
    specs = extract_labeled_value(doc.page_content, "Specifications")
    spec_lines = parse_specs_to_lines(specs)

    lines = [f"Thong so ky thuat tham khao cho: {name}"]
    if summary:
        lines.append(f"- Tong quan: {summary}")
    if features:
        lines.append(f"- Diem noi bat: {features}")
    lines.append(f"- Gia tham khao: {price}")
    lines.append(f"- Link: {url}")

    if spec_lines:
        lines.append("\nCau hinh/Thong so trong du lieu:")
        lines.extend(spec_lines)
    else:
        lines.append("\nHien tai du lieu chua co block thong so chi tiet theo cau truc ro rang cho model nay.")

    lines.append("\nNeu anh/chi can, em se doi chieu them theo vat lieu va san luong de de xuat cau hinh phu hop nhat.")
    return "\n".join(lines)


def build_find_answer(query: str, docs: List[Document]) -> str:
    if not docs:
        q = normalize_for_match(query)
        if "gap" in q:
            return (
                "Em chua tim thay model may gap giay chuyen dung duoc ghi ro trong du lieu hien tai. "
                "Anh/chi muon em de xuat nhom may cat/hoan thien giay gan nhu cau, hay em loc lai theo hang/model cu thu?"
            )
        return (
            "Em chua tim thay du lieu phu hop trong kho san pham hien tai. "
            "Anh/chi cho em biet them vat lieu va khoi luong san xuat de em de xuat chinh xac hon."
        )

    top = docs[:3]
    terms = extract_query_terms(query)
    need_disclaimer = bool(terms)
    if terms:
        joined_terms = " ".join(terms)
        for d in top:
            haystack = normalize_for_match(f"{d.metadata.get('name', '')} {d.page_content}")
            if joined_terms in haystack:
                need_disclaimer = False
                break

    lines: List[str] = []
    if need_disclaimer:
        lines.append(
            "Em chua thay model ghi ro dung tu khoa ban yeu cau trong du lieu, "
            "duoi day la 3 may gan nhat de anh/chi tham khao:"
        )
    else:
        lines.append("Em de xuat 3 may phu hop de anh/chi tham khao:")

    for i, doc in enumerate(top, start=1):
        name = doc.metadata.get("name", "Khong ro ten")
        price = doc.metadata.get("price", "Gia: Lien he")
        url = doc.metadata.get("product_url", "")
        summary = extract_labeled_value(doc.page_content, "Summary")
        features = extract_labeled_value(doc.page_content, "Features")
        highlights = summary if summary else features
        if not highlights:
            highlights = normalize_text(doc.page_content)[:220]

        lines.append(f"\n{i}) {name}")
        lines.append(f"- Diem noi bat: {highlights}")
        lines.append("- Truong hop nen dung: phu hop cac don hang can on dinh, toi uu nang suat va chat luong.")
        lines.append(f"- Gia tham khao: {price}")
        lines.append(f"- Link: {url}")

    lines.append(
        "\nAnh/chi cho em 1 thong tin de chot nhanh: vat lieu chinh anh/chi dang xu ly la gi "
        "(giay, carton, nhan decal, hay vat lieu khac)?"
    )
    return "\n".join(lines)


def resolve_default_csv_path() -> str:
    for p in ["vprint_products_clean.csv", "vprint_product_clean.csv"]:
        if Path(p).exists():
            return p
    return "vprint_products_clean.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VPRINT local hybrid RAG chatbot (Chroma + BM25 + Ollama).")
    parser.add_argument("--csv-path", default=resolve_default_csv_path())
    parser.add_argument("--persist-dir", default="vprint_agentic_db_local")
    parser.add_argument("--collection", default="vprint_products_local")
    
    # Cập nhật thành model Tiếng Việt bạn đã tạo Database
    parser.add_argument(
        "--embedding-model",
        default="bkai-foundation-models/vietnamese-bi-encoder",
    )
    parser.add_argument("--ollama-model", default="gemma3:270m")
    parser.add_argument("--k-vector", type=int, default=6)
    parser.add_argument("--k-bm25", type=int, default=6)
    parser.add_argument("--k-final", type=int, default=4)
    parser.add_argument("--max-history-turns", type=int, default=5)
    return parser.parse_args()


def setup_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_for_match(text: str) -> str:
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


def simple_tokenize(text: str) -> List[str]:
    normalized = normalize_for_match(text)
    return re.findall(r"[a-z0-9_]+", normalized)


def load_csv_docs(csv_path: str) -> List[Document]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(path, header=0) # Sửa header về 0 nếu file bạn có cột tiêu đề
    rename_map = {}
    for idx, col in enumerate(df.columns):
        rename_map[col] = DEFAULT_COLUMNS[idx] if idx < len(DEFAULT_COLUMNS) else f"extra_{idx}"
    df = df.rename(columns=rename_map).fillna("")

    docs: List[Document] = []
    for idx, row in df.iterrows():
        text = "\n".join(
            [
                f"Product name: {row.get('name', '')}",
                f"Price: {row.get('price', '')}",
                f"Summary: {row.get('summary', '')}",
                f"Description: {row.get('description', '')}",
                f"Features: {row.get('features', '')}",
                f"Specifications: {row.get('specs_json', '')}",
            ]
        ).strip()
        if not text:
            continue

        product_url = str(row.get("product_url", "")).strip()
        doc_id = product_url if product_url else f"row-{idx}"
        images = str(row.get("image_urls", "")).strip()

        metadata = {
            "row_index": int(idx),
            "name": str(row.get("name", "")),
            "product_url": product_url,
            "price": str(row.get("price", "")),
            "images": images
        }
        docs.append(Document(page_content=text, metadata=metadata, id=doc_id))
    return docs


def format_context(docs: List[Document]) -> str:
    if not docs:
        return "Khong tim thay tai lieu phu hop."

    blocks: List[str] = []
    for i, doc in enumerate(docs, start=1):
        name = doc.metadata.get("name", "")
        price = doc.metadata.get("price", "")
        url = doc.metadata.get("product_url", "")
        text = normalize_text(doc.page_content)
        blocks.append(
            f"[Tai lieu {i}]\nTen: {name}\nGia: {price}\nURL: {url}\nNoi dung: {text}"
        )
    return "\n\n".join(blocks)


def build_rag_messages(user_query, context, history, max_history=5):
    system_prompt = f"""
    Bạn là VPRINT Sales AI, chuyên viên tư vấn thiết bị in ấn, bao bì.
    Dưới đây là thông tin kỹ thuật được trích xuất từ tài liệu của công ty:
    {context}
    
    🎯 NHIỆM VỤ CỦA BẠN:
    1. Trả lời chính xác, ngắn gọn dựa trên thông tin được cung cấp.
    
    🤝 QUY TẮC XỬ LÝ CÂU HỎI "VƯỢT TẦM" VÀ CHỐT SALE:
    - Nếu câu hỏi hỏi về GIÁ CẢ cụ thể, CHÍNH SÁCH BẢO HÀNH phức tạp, hoặc yêu cầu mà thông tin trên không có đủ để trả lời chắc chắn: TUYỆT ĐỐI KHÔNG BỊA RA SỐ LIỆU.
    - Thay vào đó, hãy khéo léo thông báo rằng vấn đề này cần chuyên viên tư vấn chi tiết hơn và CHỦ ĐỘNG XIN THÔNG TIN.
    - Ví dụ: "Dạ, đối với dòng máy này, để có báo giá và cấu hình chính xác nhất cho xưởng của mình, anh/chị vui lòng để lại **Số điện thoại/Zalo** nhé. Chuyên viên Sales của VPRINT sẽ liên hệ hỗ trợ anh/chị ngay ạ."
    
    💡 QUY TẮC ĐỀ XUẤT CÂU HỎI TƯƠNG TÁC (BẮT BUỘC):
    - Dựa vào nội dung khách hàng vừa hỏi, hãy suy luận xem họ có thể quan tâm đến điều gì tiếp theo (ví dụ: thông số khác của máy, máy cùng phân khúc, vật tư đi kèm...).
    - Ở phần CUỐI CÙNG của câu trả lời, luôn luôn đề xuất 2-3 câu hỏi gợi ý.
    - Bắt buộc dùng định dạng sau (không thay đổi format):
    
    ---
    💡 **Có thể bạn quan tâm:**
    - [Gợi ý 1]
    - [Gợi ý 2]
    - [Gợi ý 3]
    """
    
    # ... code xử lý history và trả về mảng messages (giữ nguyên logic của bạn)
    messages = [("system", system_prompt)]
    for role, msg in history[-max_history:]:
        messages.append((role, msg))
    messages.append(("user", user_query))
    
    return messages


def build_direct_messages(
    user_query: str, history: List[Tuple[str, str]], max_history_turns: int
):
    messages = [SystemMessage(content=DIRECT_CHAT_SYSTEM_PROMPT)]
    recent = history[-max_history_turns:]
    for user_msg, bot_msg in recent:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=bot_msg))
    messages.append(HumanMessage(content=user_query))
    return messages


def main() -> None:
    setup_utf8_stdout()
    load_dotenv()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading embeddings on {device.upper()}...")

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.persist_dir,
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": args.k_vector})

    llm = ChatOllama(model=args.ollama_model, temperature=0.2)
    router_llm = ChatOllama(model=args.ollama_model, temperature=0)

    bm25_docs = load_csv_docs(args.csv_path)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = args.k_bm25

    # Sử dụng bộ Hybrid Search tiêu chuẩn của Langchain thay cho hàm thủ công
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    print("VPRINT Local Hybrid RAG Chatbot")
    print("Mode: HuggingFace embedding + Chroma + BM25 + Ollama")
    print("Nhap cau hoi. Go 'exit' de thoat.\n")
    print(f"VPRINT: {WELCOME_MESSAGE}\n")

    history: List[Tuple[str, str]] = []
    last_suggested_docs: List[Document] = []

    while True:
        try:
            user_query = input("Ban: ").strip()
        except KeyboardInterrupt:
            print("\nTam biet.")
            break
            
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("Tam biet.")
            break

        decision = llm_route_query(router_llm, user_query, history)
        decision = apply_router_guards(user_query, decision)

        if decision.reset_focus:
            last_suggested_docs = []

        if decision.use_rag:
            # GỌI TRỰC TIẾP ENSEMBLE RETRIEVER ĐỂ LẤY KẾT QUẢ RRF
            fused_docs = ensemble_retriever.invoke(user_query)
            
            # Cắt lấy số lượng tài liệu mong muốn
            fused_docs = fused_docs[:args.k_final]

            filtered_docs = filter_docs_by_query_terms(fused_docs, user_query)
            filtered_docs = filter_docs_by_need_profile(filtered_docs, user_query)
            context_text = format_context(filtered_docs)

            if decision.intent == "find_machine":
                answer = build_find_answer(user_query, filtered_docs)
                print(f"\nVPRINT: {answer}\n")
                
                # ===== SHOW IMAGES (ADD ONLY) =====
                for doc in filtered_docs[:3]:
                    images = parse_images(doc.metadata.get("images", ""))
                    if images:
                        print("Hinh anh san pham:")
                        for img in images[:3]:
                            print(img)
                        print()
                history.append((user_query, answer))
                last_suggested_docs = filtered_docs[:3]
                continue

            if decision.intent == "spec_query":
                candidate_docs = last_suggested_docs + filtered_docs
                best_doc = pick_best_doc_for_query(user_query, candidate_docs)
                answer = build_specs_answer(user_query, best_doc)
                print(f"\nVPRINT: {answer}\n")
                
                # ===== SHOW IMAGES (ADD ONLY) =====
                if best_doc:
                    images = parse_images(best_doc.metadata.get("images", ""))
                    if images:
                        print("Hinh anh san pham:")
                        for img in images[:3]:
                            print(img)
                        print()
                history.append((user_query, answer))
                continue

            messages = build_rag_messages(
                user_query=user_query,
                context_text=context_text,
                history=history,
                max_history_turns=args.max_history_turns,
            )
        else:
            if decision.direct_answer:
                answer = decision.direct_answer
                print(f"\nVPRINT: {answer}\n")
                history.append((user_query, answer))
                continue
            messages = build_direct_messages(user_query, history, args.max_history_turns)

        response = llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        print(f"\nVPRINT: {answer}\n")
        history.append((user_query, answer))

if __name__ == "__main__":
    main()
