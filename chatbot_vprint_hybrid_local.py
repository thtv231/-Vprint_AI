


#NÂNG CẤP CHATBOT VỚI KNOWEDGE BASE 

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
#from langchain.retrievers import EnsembleRetriever

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# ================================
# IMAGE PARSER
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
    "name", "source_url", "category_id", "product_url", "sku", 
    "price", "view_count", "summary", "description", "features", 
    "specs_json", "unused", "image_urls",
]

# ================================
# INTENT DETECTORS
# ================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def normalize_for_match(text: str) -> str:
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip()

def simple_tokenize(text: str) -> List[str]:
    normalized = normalize_for_match(text)
    return re.findall(r"[a-z0-9_]+", normalized)

def is_find_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(x in q for x in ["tim may", "muon tim", "can tim", "de xuat may", "goi y may"])

def is_spec_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(k in q for k in ["thong so", "thong so ky thuat", "spec", "cau hinh"])

def is_price_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(k in q for k in ["gia", "bao nhieu", "chi phi", "gia tham khao"])

def is_compare_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(k in q for k in ["so sanh", "khac nhau", "nen chon", "chon may nao"])

def is_book_knowledge_intent(query: str) -> bool:
    q = normalize_for_match(query)
    book_keywords = ["cong nghe", "la gi", "quy trinh", "su khac biet", "tai sao", "nguyen ly", "kien thuc", "giai thich"]
    return any(k in q for k in book_keywords)

def is_followup_intent(query: str) -> bool:
    q = normalize_for_match(query)
    return any(k in q for k in ["may do", "model do", "loai do", "cai do", "no", "mau do"])

def is_direct_chat_intent(query: str) -> bool:
    q = normalize_for_match(query)
    direct_keywords = ["chao", "hello", "hi", "cam on", "ban la ai", "huong dan", "cach dung", "giup toi voi"]
    return any(k in q for k in direct_keywords)

@dataclass
class RouterDecision:
    intent: str
    use_rag: bool
    reset_focus: bool = False

def apply_router_guards(user_query: str, decision: RouterDecision) -> RouterDecision:
    q = normalize_for_match(user_query)
    if any(k in q for k in ["thong so", "thong so ky thuat", "spec", "cau hinh"]):
        decision.intent = "spec_query"
        decision.use_rag = True
        decision.reset_focus = False
    elif any(k in q for k in ["tim may", "muon tim", "can tim", "de xuat may", "goi y may"]):
        decision.intent = "find_machine"
        decision.use_rag = True
        decision.reset_focus = True
    elif is_book_knowledge_intent(q):
        decision.intent = "book_knowledge"
        decision.use_rag = True
    return decision


# ================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ================================
class MachineSummary(BaseModel):
    description: str = Field(description="1 câu mô tả chức năng chính của máy")
    performance: str = Field(description="Thông tin tốc độ hoặc hiệu suất nổi bật")
    technology: str = Field(description="Công nghệ hoặc cấu hình đáng chú ý")
    advantage: str = Field(description="Điểm ưu việt lớn nhất của máy")

class MachineSummaryResponse(BaseModel):
    summaries: List[MachineSummary] = Field(description="Danh sách tóm tắt theo đúng thứ tự máy đầu vào")

class QueryRewriteResponse(BaseModel):
    queries: List[str] = Field(description="Danh sách truy vấn viết lại ngắn gọn, cùng ý nghĩa chuyên môn")

class MachineRerankResponse(BaseModel):
    indices: List[int] = Field(description="Danh sách chỉ số máy phù hợp nhất, theo thứ tự ưu tiên")

class MachineQueryProfile(BaseModel):
    include_terms: List[str] = Field(description="Từ khóa/cụm từ phải ưu tiên khớp với máy")
    exclude_terms: List[str] = Field(description="Từ khóa/cụm từ cần loại trừ nếu khác công đoạn")

# ================================
# UTILS & FORMATTERS
# ================================

def extract_labeled_value(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*(.*?)(?=\s*[A-Za-z ]+:|$)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return normalize_text(m.group(1)) if m else ""

def pick_best_doc_for_query(query: str, docs: List[Document]) -> Document | None:
    # Bỏ hàm lọc hardcore, tin tưởng vào top 1 của Retriever
    return docs[0] if docs else None

def load_csv_docs(csv_path: str) -> List[Document]:
    df = pd.read_csv(Path(csv_path), header=0).fillna("")
    rename_map = {col: DEFAULT_COLUMNS[idx] if idx < len(DEFAULT_COLUMNS) else f"extra_{idx}" for idx, col in enumerate(df.columns)}
    df = df.rename(columns=rename_map)

    docs = []
    for idx, row in df.iterrows():
        text = "\n".join([
            f"Product name: {row.get('name', '')}", f"Price: {row.get('price', '')}",
            f"Summary: {row.get('summary', '')}", f"Description: {row.get('description', '')}",
            f"Features: {row.get('features', '')}", f"Specifications: {row.get('specs_json', '')}",
        ]).strip()
        if text:
            docs.append(Document(
                page_content=text, 
                metadata={"row_index": int(idx), "name": str(row.get("name", "")), "product_url": str(row.get("product_url", "")), "price": str(row.get("price", "")), "images": str(row.get("image_urls", ""))}
            ))
    return docs

def format_context(docs: List[Document]) -> str:
    if not docs: return "Khong tim thay tai lieu phu hop."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[Tai lieu {i}]\nTen: {doc.metadata.get('name', '')}\nGia: {doc.metadata.get('price', '')}\nURL: {doc.metadata.get('product_url', '')}\nNoi dung: {normalize_text(doc.page_content)}")
    return "\n\n".join(blocks)

def format_book_context(docs: List[Document]) -> str:
    if not docs: return "No relevant information found in the book."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get('source_book', 'Handbook')
        blocks.append(f"[Nguồn: {source} - Đoạn {i}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)

def deduplicate_docs(docs: List[Document]) -> List[Document]:
    """Lọc các tài liệu bị trùng lặp dựa trên URL sản phẩm hoặc nội dung."""
    seen, unique_docs = set(), []
    for d in docs:
        url = d.metadata.get("product_url", d.page_content[:20])
        if url not in seen:
            seen.add(url); unique_docs.append(d)
    return unique_docs

def format_specs_to_table(spec_text: str) -> str:
    """Chuyển đổi chuỗi JSON/dict thông số kỹ thuật thành bảng Markdown."""
    if not spec_text: return ""
    try:
        data = json.loads(spec_text)
        if not isinstance(data, dict): return spec_text.replace(r'\n', '\n')

        # Xử lý từ điển phẳng (một model)
        if all(not isinstance(v, dict) for v in data.values()):
            header = "| **Thông số** | **Giá trị** |"
            separator = "|---|---|"
            rows = [f"| {k} | {v} |" for k, v in data.items()]
            return "\n".join([header, separator] + rows)

        # Xử lý từ điển lồng nhau (nhiều model)
        models = list(data.keys())
        features = sorted(list({k for md in data.values() if isinstance(md, dict) for k in md.keys()}))
        if not features: return spec_text.replace(r'\n', '\n')
        header = "| **Thông số** | " + " | ".join([f"**{m}**" for m in models]) + " |"
        separator = "|---|" + "|".join(["---"] * len(models)) + "|"
        rows = [f"| {feat} | " + " | ".join([str(data[m].get(feat, "-")) if isinstance(data[m], dict) else "-" for m in models]) + " |" for feat in features]
        return "\n".join([header, separator] + rows)
    except:
        # Trả về text gốc nếu không parse được JSON
        return spec_text.replace(r'\n', '\n')

def parse_specs_to_lines(specs_text: str) -> List[str]:
    raw = specs_text.strip()
    if not raw: return []
    try:
        data = json.loads(raw)
        if isinstance(data, dict): return [f"- {k}: {v}" for k, v in data.items() if str(v).strip()][:12]
    except Exception: pass
    cleaned = normalize_text(raw).replace("{", "").replace("}", "")
    return [f"- {p.strip()}" for p in cleaned.split(",") if p.strip()][:12] if cleaned else []

def build_specs_answer(query: str, doc: Document | None) -> str:
    if not doc: return "Xin lỗi, tôi chưa xác định được máy bạn đang hỏi. Vui lòng cung cấp tên model cụ thể."
    lines = [
        f"Thông số kỹ thuật tham khảo cho: **{doc.metadata.get('name', 'Không rõ')}**",
        f"- Tóm tắt: {extract_labeled_value(doc.page_content, 'Summary')}",
        f"- Đặc điểm: {extract_labeled_value(doc.page_content, 'Features')}",
        f"- Link: {doc.metadata.get('product_url', '')}",
        "\n**Thông số chi tiết:**"
    ]
    lines.extend(parse_specs_to_lines(extract_labeled_value(doc.page_content, "Specifications")))
    return "\n".join(lines)

# ================================
# PROMPT BUILDERS
# ================================
import re

def get_optimized_history(history, max_history=5, max_bot_chars=200):
    """Nén lịch sử chat để tiết kiệm Input Token"""
    recent_history = history[-max_history:] if max_history > 0 else []
    optimized_msgs = []
    
    for role, msg in recent_history:
        if role == "assistant":
            if len(msg) > max_bot_chars:
                important_keywords = " ".join(re.findall(r'\*\*(.*?)\*\*', msg))[:100]
                short_msg = msg[:max_bot_chars] + f"... [Đã thu gọn]. Máy đang đề cập: {important_keywords}"
                optimized_msgs.append((role, short_msg))
            else:
                optimized_msgs.append((role, msg))
        else:
            optimized_msgs.append((role, msg))
            
    return optimized_msgs
def build_rag_messages(user_query, context, history, max_history=5):
    sys_prompt = f"""Bạn là Chuyên gia AI của VPRINT.
    Sử dụng thông tin trong [KHO DỮ LIỆU] dưới đây để trả lời khách hàng.
    Nếu không có thông tin, hãy nói không biết, TUYỆT ĐỐI KHÔNG BỊA ĐẶT.
    
    [KHO DỮ LIỆU]:
    {context}
    
    QUAN TRỌNG: Ở dưới cùng của câu trả lời, BẮT BUỘC cung cấp 3 gợi ý theo đúng định dạng sau:
    💡 **Có thể bạn quan tâm:**
    - [Gợi ý 1]
    - [Gợi ý 2]
    - [Gợi ý 3]
    """
    opt_history = get_optimized_history(history, max_history)
    return [("system", sys_prompt)] + opt_history + [("user", user_query)]

def build_book_rag_messages(user_query, context, history, max_history=5):
    sys_prompt = f"""Bạn là Kỹ sư In ấn của VPRINT.
    Dựa vào [CẨM NANG NGÀNH IN] dưới đây để giải đáp thắc mắc kỹ thuật của khách hàng.
    
    [CẨM NANG NGÀNH IN]:
    {context}
    
    QUAN TRỌNG: Ở dưới cùng của câu trả lời, BẮT BUỘC cung cấp 3 gợi ý theo đúng định dạng sau:
    💡 **Có thể bạn quan tâm:**
    - [Gợi ý 1]
    - [Gợi ý 2]
    - [Gợi ý 3]
    """
    opt_history = get_optimized_history(history, max_history)
    return [("system", sys_prompt)] + opt_history + [("user", user_query)]

def build_direct_messages(user_query, history, max_history=5):
    sys_prompt = """Bạn là AI Sales vui vẻ, nhiệt tình của VPRINT.
    Khách hàng đang trò chuyện xã giao hoặc chào hỏi. 
    Hãy đáp lại lịch sự, ngắn gọn và khéo léo điều hướng họ hỏi về các dòng máy in, máy bế.
    Trả lời không quá 3-4 câu.
    
    QUAN TRỌNG: Ở dưới cùng của câu trả lời, BẮT BUỘC cung cấp 3 gợi ý theo đúng định dạng sau:
    💡 **Có thể bạn quan tâm:**
    - [Gợi ý 1]
    - [Gợi ý 2]
    - [Gợi ý 3]
    """
    opt_history = get_optimized_history(history, max_history, max_bot_chars=100)
    return [("system", sys_prompt)] + opt_history + [("user", user_query)]