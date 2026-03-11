import streamlit as st
import time
import re
import json
import os
import torch
import threading
import smtplib
import unicodedata
from contextlib import contextmanager
import gspread
import pandas as pd
import plotly.graph_objects as go
from html import escape, unescape
from dataclasses import dataclass
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from email.message import EmailMessage
from google.oauth2.service_account import Credentials
from datetime import datetime
from dotenv import load_dotenv
import pytz 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Import từ file logic chung ---
from chatbot_vprint_hybrid_local import (
    MachineSummary,
    MachineSummaryResponse,
    QueryRewriteResponse,
    MachineRerankResponse,
    MachineQueryProfile,
    deduplicate_docs,
    format_specs_to_table,
)

# ==========================================
# 1. CẤU HÌNH TRANG VÀ KHỞI TẠO SESSION STATE
# ==========================================
st.set_page_config(page_title="VPRINT AI", page_icon="img/logo_2.jpg", layout="wide")

if "initialized" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_docs = []
    st.session_state.qa_cache = {}
    st.session_state.initialized = True
    st.session_state.intent_counts = {
        "find_machine": 0,
        "book_knowledge": 0,
        "solution_consulting": 0,
        "direct_chat": 0,
    }
    st.session_state.api_tokens = 0 
    st.session_state.total_session_tokens = 0
    st.session_state.sent_booking_keys = set()
    st.session_state.viewed_machines = []

# Backward-compatible guard for existing Streamlit sessions
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}
if "sent_booking_keys" not in st.session_state:
    st.session_state.sent_booking_keys = set()
if "viewed_machines" not in st.session_state:
    st.session_state.viewed_machines = []

st.markdown("""
<style>
/* CSS cho Nút gợi ý */
div[data-testid="stVerticalBlock"] div.stButton > button {
    background-color: #f3f4f6; color: #374151; border-radius: 20px; 
    border: 1px solid #e5e7eb; padding: 8px 16px; font-size: 14.5px; 
    font-weight: 500; transition: all 0.2s ease-in-out; text-align: left; 
    width: 100%; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
div[data-testid="stVerticalBlock"] div.stButton > button:hover {
    background-color: #e5e7eb; border-color: #d1d5db; color: #000000;
    transform: translateY(-1px); box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.stChatMessage { font-size: 15px; }

/* CSS cho Spinner */
.spinner {
    width: 18px;
    height: 18px;
    border: 2.5px solid rgba(0, 0, 0, 0.15);
    border-top-color: #3b82f6; /* Màu xanh dương giống welcome box */
    border-radius: 50%;
    animation: spin 1.2s linear infinite;
    margin-right: 12px;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* CSS TINH CHỈNH AVATAR CHATBOT */
div[data-testid="stChatMessageAvatar"] {
    background-color: white !important;
    border-radius: 8px !important; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
}
div[data-testid="stChatMessageAvatar"] img {
    object-fit: contain !important; 
    border-radius: 0px !important; 
    padding: 2px !important; 
    transform: scale(1.1); 
}

.thinking-shell {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    padding: 4px 0 10px 0;
}

.thinking-text {
    font-size: 16px;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: rgba(55, 65, 81, 0.42);
    background: linear-gradient(
        110deg,
        rgba(55, 65, 81, 0.30) 0%,
        rgba(55, 65, 81, 0.52) 35%,
        rgba(31, 41, 55, 0.96) 48%,
        rgba(55, 65, 81, 0.52) 61%,
        rgba(55, 65, 81, 0.30) 100%
    );
    background-size: 220% 100%;
    background-position: 120% 0;
    -webkit-background-clip: text;
    background-clip: text;
    animation: thinking-shimmer 1.7s ease-in-out infinite;
}

@keyframes thinking-shimmer {
    0% {
        background-position: 120% 0;
    }
    100% {
        background-position: -30% 0;
    }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIG 
# ==========================================
load_dotenv()

CSV_PATH = "vprint_products_clean.csv"
PERSIST_DIR = "vprint_agentic_db_local"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1GvR6uMGIT0J1MHJplPsepbWDALntCgLAT9ENrhqidEc/edit"

COLLECTION_MACHINES = "vprint_products_local"
#EMBED_MODEL_MACHINES = "intfloat/multilingual-e5-small"
EMBED_MODEL_MACHINES = "intfloat/multilingual-e5-base"

COLLECTION_BOOK = "vprint_knowledge_base"
#EMBED_MODEL_BOOK = "intfloat/multilingual-e5-small"
EMBED_MODEL_BOOK = "intfloat/multilingual-e5-base"

K_VECTOR, K_BM25, K_FINAL, MAX_HISTORY = 20, 20, 5, 5
SEARCH_POOL_K = 24
VECTOR_WEIGHT, BM25_WEIGHT = 0.65, 0.35
MAX_CONTEXT_DOCS = 6
MAX_CONTEXT_CHARS_PER_DOC = 1200
INTENT_QUERY_MAX_CHARS = 220

DEFAULT_COLUMNS = [
    "name", "source_url", "category_id", "product_url", "sku",
    "price", "view_count", "summary", "description", "features",
    "specs_json", "unused", "image_urls",
]

WELCOME = """
<div style="background-color: #f0f8ff; border-radius: 8px; padding: 16px; border-left: 5px solid #3b82f6; line-height: 1.6; font-size: 15.5px; color: #1e293b; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
    <span style="font-size: 24px;">👋</span> <b>Xin chào! Tôi là trợ lý ảo của VPRINT</b><br>
    Tôi có thể hỗ trợ bạn:<br>
    🔍 <b>Tìm kiếm máy móc</b> (Máy in, máy bế, dán thùng...)<br>
    📐 <b>Tra cứu thông số</b> (Tốc độ, kích thước, công suất...)<br>
    📚 <b>Tư vấn kỹ thuật in</b> (Offset, Flexo, Lem mực, Chế bản...)<br>
    ⚙️ <b>Tư vấn dây chuyền</b> (Setup xưởng, tính toán sản lượng...)
</div>
"""

# ==========================================
# 2.1 TEXT NORMALIZATION + RAG HELPERS
# ==========================================
def parse_images(value: str):
    if not value:
        return []
    value = str(value).strip()
    for sep in [";", ",", "|"]:
        if sep in value:
            return [x.strip() for x in value.split(sep) if x.strip()]
    return [value]

def normalize_text(text: str) -> str:
    if not text: return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text.strip())

def normalize_for_match(text: str) -> str:
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip()

def is_book_knowledge_intent(query: str) -> bool:
    q = normalize_for_match(query)
    book_keywords = ["cong nghe", "la gi", "quy trinh", "su khac biet", "tai sao", "nguyen ly", "kien thuc", "giai thich"]
    return any(k in q for k in book_keywords)

def has_machine_code(query: str) -> bool:
    """Detects common machine code patterns like VPX-1200, LM-360, CRON H, etc."""
    # Pattern for codes like VPX-1200, LM360, VP-S54
    if re.search(r'\b[A-Z]{2,}[-]?\d{3,}\b', query, re.IGNORECASE):
        return True
    # Pattern for codes like CRON H 36
    if re.search(r'\b[A-Z]{3,}\s+[A-Z]\s+\d{2,}\b', query, re.IGNORECASE):
        return True
    return False

def is_ctp_knowledge_query(query: str) -> bool:
    q = normalize_for_match(query)
    ctp_terms = ["ctp", "thermal plate", "violet plate", "ban thermal", "ban violet", "computer-to-plate"]
    compare_terms = ["khac nhau", "su khac biet", "phan biet", "so sanh", "phu hop", "moi truong san xuat", "uu nhuoc diem"]
    machine_buy_terms = ["may", "model", "gia", "bao gia", "dau tu", "xem may", "dat lich", "tu van mua"]
    has_ctp = any(t in q for t in ctp_terms)
    has_compare = any(t in q for t in compare_terms)
    has_buy_intent = any(t in q for t in machine_buy_terms)
    return has_ctp and (has_compare or not has_buy_intent)

@dataclass
class RouterDecision:
    intent: str = "direct_chat"
    use_rag: bool = True
    reset_focus: bool = False

class ConsultingRequirement(BaseModel):
    is_clear: bool = Field(description="Đánh giá xem yêu cầu đã đủ rõ ràng để lên cấu hình máy chưa. Trả về False nếu quá mập mờ.")
    missing_info: List[str] = Field(description="Thông tin cần hỏi lại.")
    product_type: str = Field(description="Loại sản phẩm (VD: hộp mỹ phẩm, thùng sóng, tem nhãn).")
    
    # 2 THUỘC TÍNH MỚI ĐỂ CHỐNG CHỌN SAI LOẠI MÁY
    material_format: str = Field(description="Định dạng vật liệu in: 'tờ rời' (sheet-fed) hay 'cuộn' (web/roll). VD: Hộp Ivory là tờ rời, decal cuộn là cuộn.")
    production_scale: str = Field(description="Quy mô sản xuất: 'công nghiệp/tự động' hay 'nhỏ/thủ công'. VD: Hộp cao cấp/sản lượng lớn là công nghiệp tự động.")
    
    suggested_processes: List[str] = Field(description="Các công đoạn sản xuất. CHÚ Ý: Nếu là hộp mỹ phẩm/cao cấp, phải có Ép kim (Hot stamping), Dập nổi (Emboss) hoặc Spot UV.")
    search_keywords: List[str] = Field(description="Từ khóa TỐI ƯU. Phải bao gồm cả định dạng và quy mô để tránh nhầm máy. VD: ['máy bế phẳng tờ rời tự động', 'máy cán màng tự động tốc độ cao', 'máy ép kim tờ rời'].")

def parse_consulting_request(user_query: str, turn_history: list, llm_main) -> ConsultingRequirement:
    history_text = "\n".join([f"{r}: {m}" for r, m in turn_history[-4:]]) if turn_history else "Không có"
    prompt = f"""Bạn là Kỹ sư trưởng thiết kế dây chuyền sản xuất ngành in bao bì.
    
    Lịch sử chat: {history_text}
    Yêu cầu hiện tại của khách: "{user_query}"
    
    Quy tắc TỐI QUAN TRỌNG:
    1. KẾ THỪA NGỮ CẢNH: Nếu "Yêu cầu hiện tại" là một câu nói lửng hoặc câu tinh chỉnh (VD: "loại có thể đựng nước nóng", "kích thước nhỏ hơn"), bạn BẮT BUỘC phải đối chiếu với "Lịch sử chat" để biết khách đang nói về sản phẩm/máy móc nào. 
       - Ví dụ: Lịch sử là "Máy làm ly giấy", yêu cầu hiện tại là "loại đựng nước nóng" -> Bạn phải hiểu là khách đang tìm "máy làm ly giấy đựng nước nóng". KHÔNG ĐƯỢC tự suy diễn sang máy in.
    2. Xác định đúng Định dạng vật liệu (material_format). Các loại hộp giấy (Ivory, Duplex, Bristol) LUÔN LUÔN là "Tờ rời" (Sheet-fed), không bao giờ dùng máy bế cuộn.
    3. Xác định Quy mô (production_scale). Nếu có chữ "cao cấp", "dây chuyền", "nhà máy", ưu tiên chọn máy "công nghiệp", "tự động" (Automatic).
    4. Gia công sau in (Finishing): Nếu sản phẩm là hộp mỹ phẩm, hộp cao cấp, BẮT BUỘC thêm công đoạn Ép kim (Hot foil stamping) hoặc UV định hình (Spot UV) vào danh sách quy trình.
    5. Từ khóa tìm kiếm (search_keywords): Phải gộp cả thông tin từ Lịch sử và Yêu cầu hiện tại thành một cụm từ khóa hoàn chỉnh. (VD: "máy làm ly giấy tự động tráng màng PE chịu nhiệt"). Ngay cả khi yêu cầu chưa rõ ràng (is_clear = False), BẮT BUỘC phải phỏng đoán 1-2 từ khóa chung chung nhất để hệ thống có thể rút ra 1-2 máy ví dụ.
    """
    try:
        # GPT-4o, GPT-3.5 hoặc Llama 3.1 trên Groq đều hỗ trợ structured output rất tốt
        structured_llm = llm_main.with_structured_output(ConsultingRequirement)
        res = structured_llm.invoke([("system", "Bạn là chuyên gia phân tích yêu cầu ngành in."), ("user", prompt)])
        return res
    except Exception as e:
        # Fallback an toàn nếu LLM parse lỗi
        return ConsultingRequirement(is_clear=True, missing_info=[], product_type="Chưa rõ", material_format="Chưa rõ", production_scale="Chưa rõ", suggested_processes=["Sản xuất in ấn"], search_keywords=[user_query])

def smart_truncate(text: str, max_len: int = 260) -> str:
    """Hàm cắt chuỗi thông minh, không cắt ngang từ và thêm dấu ..."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    
    # Cắt nháp trước
    truncated = text[:max_len]
    # Tìm khoảng trắng cuối cùng để không làm đứt từ
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        return truncated[:last_space] + "..."
    return truncated + "..."

def build_fallback_machine_summary(doc: Document) -> MachineSummary:
    raw = str(doc.page_content or "")
    name = str(doc.metadata.get("name", "")).strip() 
    summary = extract_labeled_value(raw, "Summary")
    desc = extract_labeled_value(raw, "Description")
    features = extract_labeled_value(raw, "Features")
    specs = extract_labeled_value(raw, "Specifications")
    
    return MachineSummary(
        description=smart_truncate(summary or desc or f"Thiết bị {name}.", 260),
        performance=smart_truncate(specs or "Vui lòng liên hệ VPRINT để nhận thông số hiệu suất chi tiết.", 260),
        technology=smart_truncate(features or desc or "Cấu hình công nghiệp, vận hành ổn định.", 260),
        advantage=smart_truncate(summary or features or "Phù hợp cho nhu cầu sản xuất bao bì.", 260),
    )

def apply_router_guards(user_query: str, decision: RouterDecision) -> RouterDecision:
    q = normalize_for_match(user_query)

    # --- NEW GUARD for Machine Comparison ---
    is_comparison = any(k in q for k in ["so sanh", "khac nhau", "khac biet"])
    if is_comparison and has_machine_code(user_query):
        # If user is comparing specific machine models, it's a consulting task, not a general knowledge question.
        decision.intent = "solution_consulting"
        decision.use_rag = True
        decision.reset_focus = True # Treat as a new query context
        return decision # Return early to avoid other guards

    machine_request_phrases = [
        "may nao",
        "dong may",
        "cac may",
        "may co the",
        "tim may",
        "chon may",
        "goi y may",
        "tu van may",
        "thiet bi",
        "model",
        "thong so may",
        "bao gia may",
        "may lam",
    ]
    knowledge_markers = [
        "la gi",
        "nguyen ly",
        "tai sao",
        "khac nhau",
        "phan biet",
        "so sanh",
        "quy trinh",
    ]

    has_machine_anchor = any(t in q for t in [" may ", "may ", " may", "thiet bi", "model"])
    has_machine_phrase = any(p in q for p in machine_request_phrases)
    has_knowledge_marker = any(k in q for k in knowledge_markers)

    if decision.intent in ["direct_chat", "book_knowledge"] and (has_machine_phrase or (has_machine_anchor and "lam" in q)) and not has_knowledge_marker:
        decision.intent = "find_machine"
        decision.use_rag = True
        decision.reset_focus = True
    return decision

def extract_labeled_value(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*(.*?)(?=\s*[A-Za-z ]+:|$)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return normalize_text(m.group(1)) if m else ""

def format_specs_to_json_table(specs_str: str) -> str:
    """Chuyển đổi chuỗi JSON thông số thành bảng Markdown"""
    if not specs_str or specs_str.strip() in ["", "{}", "None"]:
        return "*(Đang cập nhật)*"
        
    try:
        # Cố gắng đọc chuỗi thành Dictionary
        data = json.loads(specs_str)
        if not data:
            return "*(Đang cập nhật)*"

        # Kiểm tra xem có phải JSON lồng nhau (Multiple Models) không
        # Ví dụ: {"INVICTA 430": {"Speed": "220"}, "INVICTA 630": {"Speed": "200"}}
        is_nested = any(isinstance(v, dict) for v in data.values())

        if is_nested:
            # Lấy danh sách các model và tất cả các thông số
            models = [str(k) for k in data.keys() if str(k).strip() != ""]
            if not models: return "*(Đang cập nhật)*"
            
            all_params = []
            for model in models:
                if isinstance(data[model], dict):
                    for k in data[model].keys():
                        if k not in all_params:
                            all_params.append(k)
            
            # Khởi tạo Header của bảng
            header = "| Thông số | " + " | ".join(models) + " |"
            separator = "|---| " + " | ".join(["---" for _ in models]) + " |"
            
            # Điền dữ liệu vào từng hàng
            rows = []
            for param in all_params:
                row = f"| **{param}** |"
                for model in models:
                    val = data[model].get(param, "-") if isinstance(data[model], dict) else "-"
                    row += f" {val} |"
                rows.append(row)
                
            return "\n".join([header, separator] + rows)
            
        else:
            # JSON phẳng (Single Model)
            # Ví dụ: {"Tốc độ": "220", "Khổ giấy": "A4"}
            header = "| Thông số | Giá trị |"
            separator = "|---|---|"
            rows = [f"| **{k}** | {v} |" for k, v in data.items()]
            return "\n".join([header, separator] + rows)

    except json.JSONDecodeError:
        # Fallback: Nếu lỗi parse JSON, cắt gọn và làm sạch chuỗi thô
        clean_str = specs_str.replace("{", "").replace("}", "").replace('"', '').replace(":", " - ")
        return f"*(Dữ liệu thô)*: {clean_str[:250]}..."

def derive_machine_metadata(row: pd.Series):
    text = normalize_for_match(
        f"{row.get('name', '')} {row.get('summary', '')} {row.get('description', '')} {row.get('features', '')}"
    )
    category = "general"
    if any(k in text for k in ["ctp", "ghi ban", "ban kem"]):
        category = "ctp"

    size_tag = ""
    if any(k in text for k in ["72x102", "72*102", "72 102", "60/72"]):
        size_tag = "72x102"

    return category, size_tag

def load_csv_docs(csv_path: str):
    df = pd.read_csv(Path(csv_path), header=0).fillna("")
    rename_map = {col: DEFAULT_COLUMNS[idx] if idx < len(DEFAULT_COLUMNS) else f"extra_{idx}" for idx, col in enumerate(df.columns)}
    df = df.rename(columns=rename_map)

    docs = []
    for idx, row in df.iterrows():
        category, size_tag = derive_machine_metadata(row)
        text = "\n".join([
            f"Product name: {row.get('name', '')}",
            f"Price: {row.get('price', '')}",
            f"Summary: {row.get('summary', '')}",
            f"Description: {row.get('description', '')}",
            f"Features: {row.get('features', '')}",
            f"Specifications: {row.get('specs_json', '')}",
        ]).strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "row_index": int(idx),
                        "name": str(row.get("name", "")),
                        "product_url": str(row.get("product_url", "")),
                        "price": str(row.get("price", "")),
                        "images": str(row.get("image_urls", "")),
                        "category": category,
                        "size_tag": size_tag,
                    },
                )
            )
    return docs

def format_context(docs):
    if not docs:
        return "Khong tim thay tai lieu phu hop."
    blocks = []
    for i, doc in enumerate(docs[:MAX_CONTEXT_DOCS], start=1):
        content = normalize_text(doc.page_content)[:MAX_CONTEXT_CHARS_PER_DOC]
        blocks.append(
            f"[Tai lieu {i}]\nTen: {doc.metadata.get('name', '')}\nGia: {doc.metadata.get('price', '')}\nURL: {doc.metadata.get('product_url', '')}\nNoi dung: {content}"
        )
    return "\n\n".join(blocks)

def format_book_context(docs):
    if not docs:
        return "Không tìm thấy thông tin phù hợp trong cẩm nang."

    blocks = []

    for i, doc in enumerate(docs[:MAX_CONTEXT_DOCS], start=1):
        content = normalize_text(doc.page_content)[:MAX_CONTEXT_CHARS_PER_DOC]

        blocks.append(content)

    return "\n\n============================\n\n".join(blocks)

def get_optimized_history(history, max_history=5, max_bot_chars=200):
    recent_history = history[-max_history:] if max_history > 0 else []
    optimized_msgs = []
    for role, msg in recent_history:
        if role == "assistant" and len(msg) > max_bot_chars:
            important_keywords = " ".join(re.findall(r"\*\*(.*?)\*\*", msg))[:100]
            short_msg = msg[:max_bot_chars] + f"... [Đã thu gọn]. Máy đang đề cập: {important_keywords}"
            optimized_msgs.append((role, short_msg))
        else:
            optimized_msgs.append((role, msg))
    return optimized_msgs

def build_rag_messages(user_query, context, history, max_history=5):
    sys_prompt = f"""Bạn là Chuyên gia AI của VPRINT.
    Sử dụng thông tin trong [KHO DỮ LIỆU] dưới đây để trả lời khách hàng.
    Nếu không có thông tin, hãy nói không biết, TUYỆT ĐỐI KHÔNG BỊA ĐẶT.
    Không lặp lại câu hỏi của người dùng.

    [KHO DỮ LIỆU]:
    {context}
    """
    return [("system", sys_prompt)] + get_optimized_history(history, max_history) + [("user", user_query)]

def build_book_rag_messages(user_query, context, history, max_history=5):
    sys_prompt = f"""Bạn là Chuyên gia Cấp cao về Công nghệ In ấn & Bao bì của VPRINT, với hơn 20 năm kinh nghiệm nghiên cứu, vận hành máy và giảng dạy ngành in.
    
Nhiệm vụ của bạn là giải đáp các câu hỏi kỹ thuật ngành in dựa trên [CẨM NANG NGÀNH IN].

✨ TIÊU CHUẨN CỦA MỘT CHUYÊN GIA:
1. Sâu sắc & Chính xác: Không chỉ trả lời "Cái gì" (What) mà phải giải thích "Tại sao" (Why) và "Như thế nào" (How).
2. Thuật ngữ chuyên ngành: Sử dụng đúng thuật ngữ tiếng Việt kèm theo tiếng Anh gốc trong ngoặc đơn nếu cần (VD: trạm màu (color station), hiện tượng tram hóa (dot gain), trục anilox...).
3. Khách quan: Khi nói về công nghệ, luôn nhìn nhận 2 mặt (ưu/nhược điểm) và đặt vào bối cảnh sản xuất thực tế.

📝 CẤU TRÚC CÂU TRẢ LỜI BẮT BUỘC (Hãy linh hoạt áp dụng tùy câu hỏi):
- 🎯 Tóm tắt/Định nghĩa: 1-2 câu ngắn gọn đi thẳng vào bản chất vấn đề.
- ⚙️ Nguyên lý / Phân tích chuyên sâu: Cấu trúc hóa bằng gạch đầu dòng giải thích cách thức hoạt động hoặc các đặc điểm kỹ thuật cốt lõi.
- ⚖️ So sánh / Ưu nhược điểm (Nếu câu hỏi mang tính chọn lựa): Đối chiếu các thông số kỹ thuật rõ ràng.
- 💡 Góc nhìn chuyên môn (Thực tiễn): Lời khuyên ứng dụng trong môi trường xưởng in thực tế (VD: công nghệ này hợp với in bao bì số lượng lớn, hay in tem nhãn ngắn ngày?).

⚠️ QUY TẮC RÀNG BUỘC:
- CHỈ sử dụng dữ liệu từ [CẨM NANG NGÀNH IN] để làm nền tảng.
- Tuyệt đối không bịa đặt thông số kỹ thuật.
- Nếu Cẩm nang không có thông tin chi tiết, hãy nói rõ: "Theo tài liệu kỹ thuật hiện tại của VPRINT chưa đi sâu vào phần này, tuy nhiên..." và vận dụng kiến thức chung để trả lời ở mức độ tổng quan.

[CẨM NANG NGÀNH IN]:
{context}
"""

    return (
        [("system", sys_prompt)]
        + get_optimized_history(history, max_history)
        + [("user", user_query)]
    )

def is_comparison_query(user_query: str) -> bool:
    q = normalize_for_match(user_query)
    comparison_markers = [
        "khac nhau",
        "so sanh",
        "phan biet",
        "am screening",
        "fm screening",
        "thermal",
        "violet",
        "flexo",
        "gravure",
    ]
    return any(marker in q for marker in comparison_markers)

def should_expand_book_answer(user_query: str, raw_answer: str) -> bool:
    answer = normalize_text(raw_answer)
    if not answer:
        return False
    if len(answer) < 260:
        return True
    if is_comparison_query(user_query):
        lower = normalize_for_match(answer)
        has_table = "|" in answer
        lacks_sections = not any(
            phrase in lower for phrase in ["uu diem", "han che", "ung dung", "nguyen ly", "co che", "su khac nhau"]
        )
        return len(answer) < 520 or (has_table and lacks_sections)
    return False

def expand_book_answer(user_query: str, context: str, raw_answer: str, llm_main):
    prompt = f"""
Bạn là biên tập viên kỹ thuật ngành in.
Nhiệm vụ: Tối ưu hóa câu trả lời dưới đây để **ngắn gọn, súc tích và dễ đọc hơn**.

Câu hỏi: {user_query}
Dữ liệu nguồn: {context}
Câu trả lời gốc: {raw_answer}

Yêu cầu biên tập:
1. **Cấu trúc lại:** Chuyển các đoạn văn thành danh sách gạch đầu dòng (bullet points).
2. **Làm rõ sự khác biệt (Nếu là so sánh):** Đối chiếu trực tiếp các thông số (VD: Dot size, Dot spacing, LPI...).
3. **Cắt bỏ:** Các từ nối rườm rà, câu dẫn nhập không cần thiết.
4. **Giữ nguyên:** Các thuật ngữ chuyên ngành và thông số kỹ thuật chính xác từ dữ liệu nguồn.
"""
    try:
        res = llm_main.invoke(
            [("system", "Bạn là kỹ sư công nghệ in, chuyên viết lại câu trả lời kỹ thuật cho đầy đủ và bám nguồn."), ("user", prompt)]
        )
        expanded = getattr(res, "content", "") or ""
        return expanded.strip() or raw_answer
    except Exception:
        return raw_answer

def is_short_ambiguous_query(user_query: str) -> bool:
    q = normalize_text(user_query)
    word_count = len([w for w in q.split(" ") if w.strip()])
    return len(q) <= 60 or word_count <= 8

def expand_book_queries(user_query: str, llm_main, max_queries: int = 3) -> List[str]:
    base_q = normalize_text(user_query)
    # TỐI ƯU: Nếu câu hỏi đã dài (> 10 từ), không cần LLM viết lại để tiết kiệm token
    if len(base_q.split()) > 10:
        return [base_q]

    # Keep retrieval robust for short questions by enriching terms, but avoid long generations.
    prompt = (
        "Viết lại câu hỏi ngành in thành các biến thể truy vấn ngắn để tìm kiếm tài liệu chuyên sâu.\n"
        f"Câu hỏi gốc: {base_q}\n"
        "Yêu cầu:\n"
        "- Tạo ra 3 truy vấn để quét toàn diện vấn đề này trong sách kỹ thuật.\n"
        "- Truy vấn 1: Tập trung vào định nghĩa / khái niệm.\n"
        "- Truy vấn 2: Tập trung vào nguyên lý hoạt động / thông số kỹ thuật.\n"
        "- Truy vấn 3: Tập trung vào ưu nhược điểm / ứng dụng thực tế.\n"
        "- Ưu tiên thêm thuật ngữ tiếng Anh tương đương (VD: prepress, CTP, flexographic, anilox, dot gain).\n"
        "- Mỗi truy vấn <= 12 từ."
    )
    try:
        structured_llm = llm_main.with_structured_output(QueryRewriteResponse)
        res = structured_llm.invoke([("system", "Bạn tối ưu truy vấn retrieval."), ("user", prompt)])
        candidates = [base_q]
        q_norm = normalize_for_match(base_q)
        if "che ban dien tu" in q_norm:
            candidates.extend([
                "chế bản điện tử là gì trong prepress",
                "chế bản điện tử khác in kỹ thuật số như thế nào",
                "computer to plate ctp trong chế bản offset",
            ])
        if res and getattr(res, "queries", None):
            candidates.extend([normalize_text(x) for x in res.queries if str(x).strip()])
        # unique, giữ thứ tự
        dedup = []
        seen = set()
        for q in candidates:
            k = normalize_for_match(q)
            if k and k not in seen:
                seen.add(k)
                dedup.append(q)
        return dedup[: 1 + max_queries]
    except Exception:
        return [base_q]

def expand_machine_queries(user_query: str, llm_main, max_queries: int = 3) -> List[str]:
    base_q = normalize_text(user_query)
    if len(base_q.split()) > 15:
        return [base_q]

    # PROMPT MỚI: ÉP LLM CẮT BỎ TỪ NHIỄU
    prompt = (
        "Bạn là chuyên gia tra cứu thiết bị ngành in.\n"
        f"Câu hỏi gốc: {base_q}\n"
        "Nhiệm vụ: Lọc sạch các từ ngữ cảnh, CHỈ giữ lại TÊN MÁY hoặc CÔNG ĐOẠN cốt lõi để làm từ khóa tìm kiếm.\n"
        "Ví dụ:\n"
        "- Khách: 'Máy nào dùng để cán màng BOPP cho bao bì hộp giấy?' -> Kết quả: 'máy cán màng BOPP', 'máy cán màng'\n"
        "- Khách: 'Tư vấn cho tôi máy bế hộp tốc độ cao' -> Kết quả: 'máy bế hộp', 'máy bế'\n"
        "Yêu cầu:\n"
        "- TUYỆT ĐỐI bỏ các từ gây nhiễu: 'máy nào', 'dùng để', 'cho', 'tư vấn', 'tốc độ cao', 'bao bì', 'hộp giấy'.\n"
        "- Trả về 2-3 cụm từ khóa cực kỳ ngắn gọn."
    )
    try:
        structured_llm = llm_main.with_structured_output(QueryRewriteResponse)
        res = structured_llm.invoke([("system", "Bạn tối ưu truy vấn tìm máy."), ("user", prompt)])
        candidates = [base_q]
        if res and getattr(res, "queries", None):
            candidates.extend([normalize_text(x) for x in res.queries if str(x).strip()])
        dedup, seen = [], set()
        for q in candidates:
            k = normalize_for_match(q)
            if k and k not in seen:
                seen.add(k)
                dedup.append(q)
        return dedup[: 1 + max_queries]
    except Exception:
        return [base_q]

def build_machine_query_profile(user_query: str, llm_main) -> MachineQueryProfile:
    prompt = f"""Phân tích câu hỏi tìm máy và trích xuất profile truy hồi.
Câu hỏi: {user_query}
Yêu cầu:
- include_terms: 3-8 cụm từ ngắn mô tả đúng công đoạn/chức năng/vật liệu.
- exclude_terms: 0-5 cụm từ thể hiện công đoạn dễ nhầm nhưng không phải mục tiêu.
- Không thêm tên máy không có trong câu hỏi.
"""
    try:
        profiler = llm_main.with_structured_output(MachineQueryProfile)
        res = profiler.invoke([("system", "Bạn trích xuất profile truy hồi tìm máy."), ("user", prompt)])
        include_terms = [normalize_for_match(t) for t in (getattr(res, "include_terms", []) or []) if str(t).strip()]
        exclude_terms = [normalize_for_match(t) for t in (getattr(res, "exclude_terms", []) or []) if str(t).strip()]
        if include_terms:
            return MachineQueryProfile(include_terms=include_terms[:8], exclude_terms=exclude_terms[:5])
    except Exception:
        pass

    toks = re.findall(r"[a-z0-9]+", normalize_for_match(user_query))
    stop = {"may", "nao", "dung", "de", "cho", "toi", "anh", "chi", "la", "gi", "vprint"}
    include = [t for t in toks if t not in stop and len(t) >= 2][:8]
    return MachineQueryProfile(include_terms=include, exclude_terms=[])

def score_doc_with_profile(doc: Document, profile: MachineQueryProfile) -> int:
    text = normalize_for_match(f"{doc.metadata.get('name', '')} {doc.page_content}")
    include_hits = sum(1 for t in profile.include_terms if t and t in text)
    exclude_hits = sum(1 for t in profile.exclude_terms if t and t in text)
    return include_hits * 3 - exclude_hits * 2

def extract_query_ngrams(user_query: str):
    toks = re.findall(r"[a-z0-9]+", normalize_for_match(user_query))
    stop = {"may", "nao", "dung", "de", "cho", "toi", "anh", "chi", "la", "gi", "vprint", "tren", "voi"}
    toks = [t for t in toks if t not in stop and len(t) >= 2]
    unigrams = toks[:12]
    bigrams = [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]
    trigrams = [f"{toks[i]} {toks[i+1]} {toks[i+2]}" for i in range(len(toks) - 2)]
    return unigrams, bigrams, trigrams

def get_operation_bucket(text: str) -> str:
    q = normalize_for_match(text)
    # Lamination
    if re.search(r'\b(can mang|laminat|bopp|phu uv|trang phu)\b', q): return "lamination"
    # Gluing (Dán)
    if re.search(r'\b(dan hop|dan carton|dan cua so|thu hop|gluer|folder gluer)\b', q): return "gluing"
    # Die cut (Bế/Ép kim)
    if re.search(r'\b(be|cat be|die cut|ep kim|hot foil|dap noi|be phang|be cuon|thut nep)\b', q): return "die_cut"
    # CTP (Chế bản)
    if re.search(r'\b(ctp|ghi ban|ban kem|cron|ban in|phoi ban)\b', q): return "ctp"
    # Printing (In)
    if re.search(r'\b(in|offset|flexo|digital|printer|kts|ong dong|gravure)\b', q): return "printing"
    # Cutting (Xén giấy)
    if re.search(r'\b(xen|cat giay|guillotine)\b', q): return "cutting"
    
    return "unknown"

def infer_query_operation_bucket(user_query: str) -> str:
    return get_operation_bucket(user_query)

def doc_operation_bucket(doc: Document) -> str:
    # Quét Tên Máy trước, vì Tên Máy đại diện chính xác nhất cho công năng
    name_bucket = get_operation_bucket(doc.metadata.get('name', ''))
    if name_bucket != "unknown": 
        return name_bucket
    # Nếu Tên không rõ, mới quét vào Nội dung Mô tả
    return get_operation_bucket(doc.page_content)

def lexical_match_score(user_query: str, doc: Document) -> int:
    text = normalize_for_match(f"{doc.metadata.get('name', '')} {doc.page_content}")
    unigrams, bigrams, trigrams = extract_query_ngrams(user_query)
    score = 0

    # 1. Điểm Ngram (Từ khóa liền kề)
    for t in trigrams:
        if t in text: score += 10
    for t in bigrams:
        if t in text: score += 5
    for t in unigrams:
        if t in text: score += 1

    # 2. ÉP LUẬT DANH MỤC (HARD BUCKETING)
    q_bucket = infer_query_operation_bucket(user_query)
    doc_bucket = doc_operation_bucket(doc)
    
    # Nếu hệ thống nhận diện được câu hỏi đang hỏi về công đoạn nào đó
    if q_bucket != "unknown":
        if doc_bucket == q_bucket:
            score += 100  # Thưởng điểm tuyệt đối cho máy ĐÚNG công đoạn
        else:
            score -= 500  # PHẠT CỰC NẶNG (-500) nếu máy bị lệch công đoạn hoặc unknown
            
    return score

def pre_rank_machine_candidates(user_query: str, docs, llm_main, top_n: int = 20):
    if not docs:
        return []
    profile = build_machine_query_profile(user_query, llm_main)
    scored = []
    for d in docs:
        p_score = score_doc_with_profile(d, profile)
        l_score = lexical_match_score(user_query, d)
        scored.append((p_score + l_score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    positives = [d for s, d in scored if s >= 3]
    if positives:
        return positives[:top_n]
    return [d for _, d in scored[:top_n]]

def retrieve_book_with_rrf(user_query: str, book_retriever, llm_main, top_k: int = 8) -> List[Document]:
    # RAG Fusion + RRF path for short/ambiguous knowledge queries (e.g. "chế bản điện tử là gì").
    queries = [normalize_text(user_query)]
    if is_short_ambiguous_query(user_query):
        queries = expand_book_queries(user_query, llm_main, max_queries=3)

    rrf_k = 60.0
    scored = {}
    for q in queries:
        try:
            docs = book_retriever.invoke(q)
        except Exception:
            docs = []
        for rank, doc in enumerate(docs, start=1):
            doc_id = f"{doc.metadata.get('source_book', 'book')}::{normalize_for_match(doc.page_content[:120])}"
            entry = scored.setdefault(doc_id, {"doc": doc, "score": 0.0})
            entry["score"] += 1.0 / (rrf_k + rank)

    merged = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in merged[:top_k]]

def get_history_limit_for_model(model_name: str) -> int:
    if model_name == "gpt-3.5-turbo":
        return 3
    return MAX_HISTORY

def make_query_cache_key(query: str) -> str:
    return f"v20::{normalize_for_match(query)}"

def is_context_dependent_query(query: str) -> bool:
    q = normalize_for_match(query)
    
    # 1. Các cụm từ nối trực tiếp rõ ràng
    followup_markers = [
        "y tren", "cau tren", "nhu tren", "o tren", "vua noi",
        "giai thich them", "noi ro hon", "chi tiet hon",
        "vi sao vay", "tai sao vay", "truong hop nay",
        "van de nay", "phan nay", "phan do", "cai nay", "cai do",
        "nhu vay", "tiep theo", "them nua",
        "gia bao nhieu", "bao nhieu tien", "thong so the nao", "cau hinh ra sao"
    ]
    if any(m in q for m in followup_markers):
        return True

    # 2. Bắt các câu yêu cầu lọc/tinh chỉnh (Refine markers)
    refine_markers = [
        "loai co", "loai dung", "loai chay", "loai lam", "loai khac", 
        "dong may", "mau may", "mau khac",
        "to hon", "nho hon", "nhanh hon", "re hon", "dat hon", "cao cap hon",
        "tu dong", "thu cong", "co the", "vay thi", "neu the", "con loai"
    ]
    if any(m in q for m in refine_markers):
        return True

    # Nếu câu hỏi chứa từ khóa tìm kiếm máy rõ ràng, coi như là context mới (Reset)
    new_search_signals = [
        "tim may", "cho xem", "can mua", "bao gia may",
        "may in", "may be", "may dan", "may cat", "may ep", "may trang", "may gap", "may lam"
    ]
    if any(s in q for s in new_search_signals):
        # Trừ khi có từ chỉ định trỏ về cái cũ rõ ràng
        if not any(p in q for p in ["nay", "do", "vua roi", "tren", "cu", "vua xem"]):
            return False

    # 3. Phân tích độ dài và đại từ (Tăng lên 12 từ để bắt được các câu dài hơn)
    tokens = [w for w in re.findall(r"[a-z0-9]+", q) if w.strip()]
    if len(tokens) <= 12:
        pronouns = {"nay", "do", "no", "vay", "them", "con", "roi", "loai", "dong", "mau", "cai"}
        if any(t in pronouns for t in tokens):
            return True
            
    return False

def quick_intent_classify(user_query: str):
    # TỐI ƯU: Sử dụng quy tắc từ khóa để bỏ qua LLM Router cho các trường hợp rõ ràng
    q = normalize_for_match(user_query)
    
    # 1. Direct Chat (Chào hỏi, cảm ơn)
    direct_keywords = ["xin chao", "hello", "hi bot", "cam on", "tam biet", "ban la ai", "giup gi duoc"]
    if any(k == q for k in direct_keywords) or len(q) < 3:
        return "direct_chat"
        
    # 2. Find Machine (Tìm máy rõ ràng)
    # Nếu có từ khóa "máy" + hành động cụ thể, và không hỏi "tại sao/là gì"
    machine_actions = ["tim may", "can mua", "bao gia", "thong so", "may in", "may be", "may dan", "may cat", "may nao", "dong may", "loai may"]
    knowledge_markers = ["la gi", "tai sao", "nguyen ly", "khac nhau", "phan biet", "khac gi", "uu diem", "nhuoc diem", "cau tao", "cong dung"]
    consulting_markers = ["tu van", "giai phap", "day chuyen", "xuong in", "ke hoach", "dau tu"]
    
    if any(k in q for k in machine_actions) and not any(k in q for k in knowledge_markers):
        # Nếu có từ khóa tư vấn/giải pháp, hãy để LLM quyết định (thường là solution_consulting) thay vì ép về find_machine
        if any(c in q for c in consulting_markers):
            return None
            
        # Đảm bảo không phải là câu hỏi tư vấn phức tạp (quá dài)
        if len(q.split()) < 15: 
            return "find_machine"
            
    return None # Nếu không chắc chắn, để LLM quyết định

def build_direct_messages(user_query, history, max_history=5):
    sys_prompt = """Bạn là trợ lý AI của VPRINT.
    Đây là cuộc trò chuyện thông thường, chào hỏi hoặc câu hỏi không cần truy hồi dữ liệu.
    Hãy trả lời tự nhiên, lịch sự, ngắn gọn.
    Nếu phù hợp, có thể gợi ý khách hỏi thêm về máy móc hoặc công nghệ ngành in.
    """
    return [("system", sys_prompt)] + get_optimized_history(history, max_history, max_bot_chars=100) + [("user", user_query)]

def build_general_knowledge_fallback_messages(user_query, history, max_history=5):
    sys_prompt = """Bạn là Kỹ sư Trưởng chuyên xử lý sự cố và vận hành xưởng in của VPRINT.
Hiện tại, hệ thống Cẩm nang nội bộ không chứa tài liệu trực tiếp về câu hỏi này. Tuy nhiên, với kinh nghiệm uyên thâm của mình, bạn hãy tư vấn cho khách hàng dựa trên tiêu chuẩn chung của ngành công nghiệp in (ISO 12647, FOGRA, G7...).

📝 Hướng dẫn trả lời:
1. Mở đầu bằng một câu rào đón lịch sự: "Dữ liệu cẩm nang nội bộ của VPRINT hiện chưa có tài liệu cụ thể cho vấn đề này. Tuy nhiên, dưới góc độ chuyên môn ngành in, tôi xin chia sẻ như sau:"
2. Đưa ra các giả thuyết kỹ thuật (Nếu khách hỏi về lỗi in ấn: Liệt kê các nguyên nhân có thể do mực, do giấy, do áp lực lô, do chế bản...).
3. Đề xuất hướng kiểm tra hoặc khắc phục từng bước (Troubleshooting steps).
4. Giữ phong thái chuyên nghiệp, dùng từ vựng kỹ thuật chuẩn xác (VD: overprinting, trapping, tack mực, pH nước máng...).
5. Khéo léo gợi ý khách hàng có thể liên hệ kỹ thuật viên VPRINT để được hỗ trợ chuyên sâu hơn.
"""
    return [("system", sys_prompt)] + get_optimized_history(history, max_history) + [("user", user_query)]

@contextmanager
def thinking_indicator():
    placeholder = st.empty()
    placeholder.markdown(
        '<div class="thinking-shell"><div class="spinner"></div><div class="thinking-text">Thinking ...</div></div>',
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()

def build_solution_consulting_messages(user_query, machine_context, book_context, history, max_history=5):
    sys_prompt = f"""Bạn là kỹ sư tư vấn giải pháp của VPRINT.
Hãy phân tích yêu cầu sản xuất và tư vấn giải pháp chặt chẽ dựa trên 2 nguồn:
1. [KHO MÁY VPRINT]
2. [CẨM NANG NGÀNH IN]

Quy tắc:
- **QUAN TRỌNG**: Nếu yêu cầu của khách hàng rất chung chung, không rõ ràng (VD: 'tìm máy in', 'tư vấn máy bế'), hãy **BẮT ĐẦU** bằng việc hỏi các câu hỏi để làm rõ nhu cầu (sản phẩm là gì, sản lượng, vật liệu, khổ in...) trước khi đề xuất bất kỳ máy nào. Đừng cố gắng đoán ý và gợi ý máy ngay.
- Chỉ dùng máy có trong [KHO MÁY VPRINT].
- TUYỆT ĐỐI KHÔNG ĐỀ XUẤT MÁY KHÔNG CÓ TRONG [KHO MÁY VPRINT]. Nếu cần máy mà kho không có, hãy nói là 'cần trang bị thêm máy X (hiện kho VPRINT chưa có dữ liệu)'.
- Có thể dùng [CẨM NANG NGÀNH IN] để giải thích lập luận kỹ thuật.
- Không bịa tên máy, không bịa thông số.
- Nếu thiếu dữ liệu máy cho một công đoạn, nói rõ là hiện chưa có đủ dữ liệu máy trong kho VPRINT.
- Nếu câu hỏi hiện tại đang hỏi về tiêu chí chọn máy, dữ liệu đầu vào cần xác định, hoặc các yếu tố ảnh hưởng đến chọn máy:
  không được nhảy ngay sang đề xuất model máy.
  Hãy ưu tiên liệt kê các dữ liệu cần xác định, giải thích vì sao từng dữ liệu quan trọng, rồi mới nêu hướng chọn máy.
  Không lặp lại câu hỏi.

Đầu ra mong muốn:
1. Phân tích yêu cầu và các ràng buộc kỹ thuật chính.
2. Đề xuất công nghệ/quy trình phù hợp.
3. Đề xuất các máy VPRINT phù hợp nhất cho từng công đoạn quan trọng.
4. Nêu rõ lý do chọn máy dựa trên dữ liệu đã truy hồi.
5. Nếu còn thiếu thông tin đầu vào, kết thúc bằng 2-3 câu hỏi làm rõ.

[KHO MÁY VPRINT]:
{machine_context}

[CẨM NANG NGÀNH IN]:
{book_context}
"""
    return [("system", sys_prompt)] + get_optimized_history(history, max_history) + [("user", user_query)]

# ==========================================
# 3. UTIL FUNCTIONS
# ==========================================
def plot_intent_radar():
    intents = list(st.session_state.intent_counts.keys())
    counts = list(st.session_state.intent_counts.values())
    max_val = max(counts) if counts else 0
    chart_max = max_val + 1 if max_val > 0 else 2
    
    fig = go.Figure(data=go.Scatterpolar(
        r=counts + [counts[0]] if counts else [0], 
        theta=intents + [intents[0]] if intents else [""], 
        fill='toself',
        fillcolor='rgba(255, 75, 75, 0.4)', line=dict(color='#ff4b4b', width=2),
        marker=dict(symbol='circle', size=4, color='#ff4b4b'), mode='lines+markers'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-0.8, chart_max], gridcolor='#e5e7eb', linecolor='#d1d5db', tickfont=dict(size=10, color='gray')),
            angularaxis=dict(gridcolor='#e5e7eb', linecolor='#d1d5db', tickfont=dict(size=11, color='#374151')),
            bgcolor='rgba(0,0,0,0)' 
        ),
        showlegend=False, margin=dict(l=40, r=40, t=30, b=30), height=280,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@st.cache_resource
def get_gsheet_client():
    try:
        credentials_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets", 
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        print(f"Lỗi xác thực chứng chỉ Google: {e}")
        return None

def log_chat_to_gsheet(user_query, bot_response, intent, response_time, tokens, model_name):
    try:
        client = get_gsheet_client()
        if not client: return False
        sheet = client.open_by_url(SPREADSHEET_URL).sheet1
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        timestamp = datetime.now(vietnam_tz).strftime("%Y-%m-%d %H:%M:%S")
        bot_response_json = json.dumps({"response": bot_response}, ensure_ascii=False)
        row_data = [timestamp, user_query, intent, bot_response_json, round(response_time, 2), tokens, model_name]
        sheet.append_row(row_data)
        return True
    except Exception as e:
        print(f"Lỗi bắn log GSheet: {e}")
        return False

def log_chat_to_gsheet_async(user_query, bot_response, intent, response_time, tokens, model_name):
    thread = threading.Thread(
        target=log_chat_to_gsheet,
        args=(user_query, bot_response, intent, response_time, tokens, model_name),
        daemon=True,
    )
    thread.start()

def get_safe_api_key(key_name="GROQ_API_KEY"):
    def _clean(v: str):
        if v is None:
            return None
        s = str(v).strip()
        # Remove surrounding quotes from .env values like "abc"
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        # Handle accidental inline comment in .env, e.g. VALUE  # comment
        if " #" in s:
            s = s.split(" #", 1)[0].strip()
        return s

    try:
        if key_name in st.secrets:
            return _clean(st.secrets[key_name])
    except Exception: pass 
    return _clean(os.getenv(key_name))

def detect_booking_lead(text: str):
    q = normalize_for_match(text)
    booking_keywords = [
        "dat lich", "đặt lịch", "xem may", "xem máy", 
        "tu van", "tư vấn", "hen gap", "hẹn gặp", "lien he", "liên hệ",
        "bao gia", "báo giá", "mua may", "mua máy"
    ]
    has_booking_intent = any(k in q for k in booking_keywords)

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(?:\+84|0)\d{9,10}", re.sub(r"[^\d\+]", "", text))
    name_match = re.search(r"(?:ten|tên)\s*[:\-]\s*([^\n,;]+)", text, flags=re.IGNORECASE)
    if not name_match:
        # Capture natural introductions: "toi la Tom", "tôi là Tom", "em là Tom", ...
        name_match = re.search(
            r"(?:toi|tôi|em|anh|chi|chị)\s+la\s+([A-Za-zÀ-ỹĐđ\s]{2,40})",
            normalize_for_match(text),
            flags=re.IGNORECASE,
        )

    raw_name = name_match.group(1).strip() if name_match else ""
    payload = {
        "name": " ".join([w.capitalize() for w in raw_name.split()]) if raw_name else "",
        "phone": phone_match.group(0) if phone_match else "",
        "email": email_match.group(0) if email_match else "",
        "content": text.strip(),
    }
    # Trigger when user expresses booking intent and provides at least one contact method
    is_lead = has_booking_intent and bool(payload["phone"] or payload["email"])
    return is_lead, payload

def infer_interest_area(text: str, last_docs):
    q = normalize_for_match(text)
    if any(k in q for k in ["ctp", "ghi ban", "ban kem"]):
        return "CTP / chế bản"
    if any(k in q for k in ["offset"]):
        return "In offset"
    if any(k in q for k in ["flexo"]):
        return "In flexo"
    if any(k in q for k in ["ky thuat so", "kts", "digital"]):
        return "In kỹ thuật số"
    if any(k in q for k in ["bao bi", "carton", "hop giay"]):
        return "Bao bì / carton"

    if last_docs:
        names = " ".join([str(d.metadata.get("name", "")).lower() for d in last_docs[:5]])
        if "ctp" in names or "cron" in names:
            return "CTP / chế bản"
        if "flexo" in names:
            return "In flexo"
        if "offset" in names:
            return "In offset"
        if "digital" in names or "kts" in names:
            return "In kỹ thuật số"
    return "Thiết bị in công nghiệp (chưa xác định rõ nhóm)"

def extract_viewed_machines(last_docs):
    viewed = []
    for d in (last_docs or [])[:5]:
        viewed.append(
            {
                "name": d.metadata.get("name", ""),
                "url": d.metadata.get("product_url", ""),
                "price": d.metadata.get("price", ""),
            }
        )
    return viewed

def track_viewed_machines(docs):
    if not docs:
        return
    seen_urls = {m.get("url", "") for m in st.session_state.viewed_machines}
    for d in docs:
        url = d.metadata.get("product_url", "")
        if url and url in seen_urls:
            continue
        st.session_state.viewed_machines.append(
            {
                "name": d.metadata.get("name", ""),
                "url": url,
                "price": d.metadata.get("price", ""),
            }
        )
        if url:
            seen_urls.add(url)
    # Keep recent viewed list bounded
    st.session_state.viewed_machines = st.session_state.viewed_machines[-10:]

def collect_recent_user_questions(current_query: str, max_items: int = 6):
    user_msgs = [msg for role, msg in st.session_state.history if role == "user"]
    if not user_msgs or user_msgs[-1] != current_query:
        user_msgs.append(current_query)
    user_msgs = [m.strip() for m in user_msgs if str(m).strip()]
    return user_msgs[-max_items:]

def build_recent_chat_excerpt(max_turns: int = 8):
    excerpt = []
    for role, msg in st.session_state.history[-max_turns:]:
        role_label = "Khách" if role == "user" else "Bot"
        clean_msg = re.sub(r"<[^>]+>", "", str(msg)).strip()
        if clean_msg:
            excerpt.append(f"{role_label}: {clean_msg[:220]}")
    return "\n".join(excerpt)

def extract_machine_codes(viewed_machines):
    codes = []
    for m in viewed_machines or []:
        name = str(m.get("name", ""))
        # Common machine-code pattern (e.g., 60/72, HDI-1200, VP-S54, 26 Models)
        found = re.findall(r"\b[A-Z0-9]{1,6}(?:[-/][A-Z0-9]{1,8})+\b|\b\d{1,4}(?:/\d{1,4})?\b", name, flags=re.IGNORECASE)
        if found:
            codes.extend(found[:2])
        else:
            codes.append(name[:40])
    # Unique while preserving order
    unique = []
    seen = set()
    for c in codes:
        k = c.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        unique.append(c.strip())
    return unique[:8]

def build_interest_summary(recent_user_questions, current_content, interest_area):
    points = []
    if interest_area:
        points.append(f"Lĩnh vực: {interest_area}")
    if recent_user_questions:
        points.append(f"Câu hỏi gần đây: {' | '.join(recent_user_questions[-3:])}")
    if current_content:
        points.append(f"Yêu cầu hiện tại: {current_content.strip()[:220]}")
    return " ; ".join(points)

def send_sale_email(payload: dict):
    smtp_host = get_safe_api_key("SMTP_HOST")
    smtp_port_raw = get_safe_api_key("SMTP_PORT") or "587"
    smtp_port = int(str(smtp_port_raw).strip())
    smtp_user = get_safe_api_key("SMTP_USER")
    smtp_pass = get_safe_api_key("SMTP_PASS")
    sale_notify = get_safe_api_key("SALE_NOTIFY_EMAIL")
    if smtp_pass:
        smtp_pass = smtp_pass.replace(" ", "").strip()

    if not (smtp_host and smtp_user and smtp_pass and sale_notify):
        return False, "Missing SMTP config"

    msg = EmailMessage()
    msg["Subject"] = f"[VPRINT AI] 🔥 Lead Mới: {payload.get('name', 'Khách')} - {payload.get('phone', '')}"
    msg["From"] = smtp_user
    msg["To"] = sale_notify
    machine_codes = payload.get("viewed_machine_codes", [])
    machine_codes_text = ", ".join(machine_codes) if machine_codes else "(chưa có dữ liệu)"
    
    # Xử lý format nội dung tóm tắt cho dễ đọc
    raw_summary = payload.get("interest_summary", "")
    # Chuyển đổi các dấu phân cách thành xuống dòng gạch đầu dòng
    formatted_plain = raw_summary.replace(" ; ", "\n- ").replace(" | ", "\n  * ")
    if formatted_plain: formatted_plain = "- " + formatted_plain
    
    plain_body = (
        "--------------------------------------------------\n"
        "🚀 THÔNG BÁO LEAD MỚI TỪ CHATBOT VPRINT\n"
        "--------------------------------------------------\n\n"
        "1. THÔNG TIN LIÊN HỆ:\n"
        f"- Họ tên: {payload.get('name', 'Unknown')}\n"
        f"- SĐT:    {payload.get('phone', '')}\n"
        f"- Email:  {payload.get('email', '')}\n\n"
        "2. MỐI QUAN TÂM:\n"
        f"- Lĩnh vực: {payload.get('interest_area', '')}\n"
        f"- Máy đã xem: {machine_codes_text}\n\n"
        "3. NGỮ CẢNH HỘI THOẠI:\n"
        f"{formatted_plain}\n\n"
        "--------------------------------------------------\n"
        "Hệ thống VPRINT AI Sales Assistant"
    )
    msg.set_content(plain_body)

    # HTML Body - Định dạng đẹp chuyên nghiệp
    html_summary = escape(raw_summary).replace(" ; ", "<br>• ").replace(" | ", "<br>&nbsp;&nbsp;- ")
    if html_summary: html_summary = "• " + html_summary

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="background-color: #2563eb; padding: 20px; text-align: center;">
                <h2 style="margin: 0; color: #ffffff; font-size: 20px;">🚀 LEAD KHÁCH HÀNG MỚI</h2>
            </div>
            <div style="padding: 25px;">
                <h3 style="color: #1e40af; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; margin-top: 0;">👤 Thông Tin Liên Hệ</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr>
                        <td style="padding: 8px 0; width: 120px; color: #6b7280; font-weight: bold;">Họ tên:</td>
                        <td style="padding: 8px 0; font-weight: bold; color: #111827;">{escape(payload.get('name', 'Unknown'))}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280; font-weight: bold;">Điện thoại:</td>
                        <td style="padding: 8px 0; font-weight: bold; color: #dc2626; font-size: 16px;">{escape(payload.get('phone', ''))}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #6b7280; font-weight: bold;">Email:</td>
                        <td style="padding: 8px 0; color: #2563eb;">
                            <a href="mailto:{escape(payload.get('email', ''))}" style="text-decoration: none; color: #2563eb;">{escape(payload.get('email', ''))}</a>
                        </td>
                    </tr>
                </table>

                <h3 style="color: #1e40af; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px;">🎯 Nhu Cầu & Quan Tâm</h3>
                <p style="margin: 10px 0;"><b>Lĩnh vực:</b> {escape(payload.get('interest_area', ''))}</p>
                <p style="margin: 10px 0;"><b>Máy đã xem:</b> {escape(machine_codes_text)}</p>
                
                <div style="background-color: #f3f4f6; border-left: 4px solid #2563eb; padding: 15px; margin-top: 20px; border-radius: 4px;">
                    <p style="margin-top: 0; font-weight: bold; color: #374151; margin-bottom: 10px;">📝 Tóm tắt ngữ cảnh hội thoại:</p>
                    <div style="color: #4b5563; font-size: 14px; line-height: 1.5;">
                        {html_summary}
                    </div>
                </div>
            </div>
            <div style="background-color: #f9fafb; padding: 15px; text-align: center; font-size: 12px; color: #9ca3af; border-top: 1px solid #e5e7eb;">
                Email tự động từ VPRINT AI Sales Assistant<br>
                {datetime.now().strftime("%d/%m/%Y %H:%M")}
            </div>
        </div>
      </body>
    </html>
    """
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, ""
    except Exception as e:
        print(f"Loi gui email lead: {e}")
        return False, str(e)

def send_sale_email_async(payload: dict):
    thread = threading.Thread(target=send_sale_email, args=(payload,), daemon=True)
    thread.start()

def build_booking_ack(payload: dict, sent_ok: bool, err: str = "") -> str:
    customer_name = payload.get("name", "").strip() or "anh/chị"
    phone = payload.get("phone", "").strip()
    email = payload.get("email", "").strip()
    contact_line = []
    if phone:
        contact_line.append(f"SĐT: **{phone}**")
    if email:
        contact_line.append(f"Email: **{email}**")
    contact_text = " | ".join(contact_line) if contact_line else "Thông tin liên hệ đã được ghi nhận."

    if sent_ok:
        return (
            f"Cảm ơn {customer_name}, VPRINT đã nhận thông tin đặt lịch xem máy của bạn.\n\n"
            f"{contact_text}\n"
            "✅ Thông tin đã được chuyển đến nhân viên sales và sẽ liên hệ trực tiếp sớm để xác nhận lịch hẹn.\n\n"
            "Nếu cần bổ sung khung giờ mong muốn, bạn có thể gửi lại ngay trong khung chat này."
        )

    return (
        f"Cảm ơn {customer_name}, VPRINT đã ghi nhận yêu cầu đặt lịch của bạn.\n\n"
        f"{contact_text}\n"
        "⚠️ Hệ thống email nội bộ đang lỗi tạm thời nên chưa chuyển tự động cho sales.\n"
        "Vui lòng gửi lại một lần nữa hoặc để lại khung giờ mong muốn, mình sẽ ưu tiên chuyển tiếp ngay."
        + (f"\n\n(Chi tiết lỗi: {err})" if err else "")
    )

def generate_chat_export():
    export_text = "LỊCH SỬ TƯ VẤN - VPRINT SALES AI\n"
    export_text += "="*40 + "\n\n"
    for role, msg in st.session_state.history:
        sender = "Khách hàng" if role == "user" else "VPRINT Bot"
        clean_msg = re.sub(r'<[^>]+>', '', msg) 
        export_text += f"[{sender}]: {clean_msg}\n\n"
    return export_text

# ==========================================
# 4. HÀM HỨNG TOKEN VÀ XỬ LÝ TEXT
# ==========================================
def stream_response(messages, llm):
    for chunk in llm.stream(messages):
        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
            st.session_state.api_tokens += chunk.usage_metadata.get('total_tokens', 0)
        yield chunk.content

def custom_write_stream(stream_generator):
    """Renders a stream with a Gemini-style typing cursor for a smoother feel."""
    placeholder = st.empty()
    full_response = ""
    for chunk in stream_generator:
        if chunk:
            full_response += chunk
            placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
    # Final render without the cursor
    placeholder.markdown(full_response, unsafe_allow_html=True)
    return full_response

def summarize_machines_structured(user_query: str, suggested_docs, llm_main):
    # TỐI ƯU: Thay vì gọi LLM để tóm tắt (tốn rất nhiều token context),
    # ta sử dụng hàm Regex `build_fallback_machine_summary` đã viết sẵn.
    # Hàm này trích xuất Description, Features, Specs từ text rất tốt và MIỄN PHÍ token.
    return [build_fallback_machine_summary(doc) for doc in suggested_docs]

def rerank_machine_candidates(user_query: str, candidate_docs, llm_main, top_k: int = 3):
    if not candidate_docs:
        return []

    # TỐI ƯU: Sử dụng thuật toán chấm điểm cục bộ (Local Scoring) thay vì gọi LLM.
    # Hàm `lexical_match_score` và `score_doc_with_profile` đã có sẵn logic so khớp từ khóa và ngữ cảnh.
    
    profile = build_machine_query_profile(user_query, llm_main) # Vẫn dùng LLM nhỏ để trích xuất profile (ít token)
    scored = []
    for d in candidate_docs:
        # Kết hợp điểm Profile (Semantic) và điểm Lexical (Keyword)
        p_score = score_doc_with_profile(d, profile) if profile else 0
        l_score = lexical_match_score(user_query, d)
        total_score = p_score + l_score
        scored.append((total_score, d))
    
    # Sắp xếp giảm dần theo điểm
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

def is_garbage_response(text: str) -> bool:
    t = text.lower()
    if t.count("platemaker") >= 5:
        return True
    if len(re.findall(r"platemaker\s*\d+", t)) >= 5:
        return True
    return False

# ==========================================
# 5. LOAD SYSTEM & LLM ROUTER MỚI
# ==========================================
@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    machine_embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_MACHINES,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    docs = load_csv_docs(CSV_PATH)
    machine_store = Chroma.from_documents(documents=docs, embedding=machine_embedder, collection_name=COLLECTION_MACHINES)
    vector_retriever = machine_store.as_retriever(search_kwargs={"k": K_VECTOR})

    book_embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_BOOK,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    book_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=book_embedder, collection_name=COLLECTION_BOOK)
    book_retriever = book_store.as_retriever(search_kwargs={"k": 8})

    return vector_retriever, book_retriever, machine_embedder

def semantic_machine_search(
    user_query: str, 
    vector_retriever, 
    top_k: int, 
    llm_main=None, 
    target_format: str = "", 
    target_scale: str = ""
):
    queries = [normalize_text(user_query)]
    if llm_main is not None:
        queries = expand_machine_queries(user_query, llm_main, max_queries=3)

    semantic_docs = []
    for q in queries:
        semantic_docs.extend(vector_retriever.invoke(q))

    rrf_k = 60.0
    scored = {}
    
    # Hàm phụ để chuẩn hóa text dễ check keyword
    def get_doc_text(doc):
        return normalize_for_match(f"{doc.metadata.get('name', '')} {doc.page_content}")

    # Chuẩn hóa target để dễ so sánh
    t_format = normalize_for_match(target_format)
    t_scale = normalize_for_match(target_scale)

    # Từ khóa nhận diện
    sheet_keywords = ["to roi", "sheet fed", "sheet", "to"]
    web_keywords = ["cuon", "web fed", "roll"]
    auto_keywords = ["tu dong", "automatic", "cong nghiep", "toc do cao"]
    manual_keywords = ["thu cong", "ban tu dong", "mini", "nho"]

    # 1. Chấm điểm RRF cơ bản cho Semantic Docs
    for rank, doc in enumerate(semantic_docs, start=1):
        doc_id = str(doc.metadata.get("row_index")) if "row_index" in doc.metadata else doc.metadata.get("product_url", doc.page_content[:80])
        entry = scored.setdefault(doc_id, {"doc": doc, "score": 0.0})
        entry["score"] += 1.0 / (rrf_k + rank)

    # 3. HEURISTIC BOOSTING & PENALTY (Can thiệp điểm tuyệt đối)
    for doc_id, entry in scored.items():
        doc_text = get_doc_text(entry["doc"])
        
        # Áp dụng Luật FORMAT (Định dạng vật liệu)
        if "to roi" in t_format or "sheet" in t_format:
            # Khách cần "Tờ rời" -> Thấy "Máy cuộn" là phạt cực nặng (hoặc loại luôn bằng cách chia điểm cho 10)
            if any(k in doc_text for k in web_keywords) and not any(k in doc_text for k in sheet_keywords):
                entry["score"] = entry["score"] * 0.1 
            # Ưu tiên cộng điểm lớn nếu ghi rõ máy tờ rời
            if any(k in doc_text for k in sheet_keywords):
                entry["score"] += 5.0 # Cộng 5 điểm RRF là một con số khổng lồ (thường RRF < 1.0)
                
        elif "cuon" in t_format or "web" in t_format:
            # Khách cần "Cuộn" -> Phạt máy tờ rời
            if any(k in doc_text for k in sheet_keywords) and not any(k in doc_text for k in web_keywords):
                entry["score"] = entry["score"] * 0.1
            if any(k in doc_text for k in web_keywords):
                entry["score"] += 5.0

        # Áp dụng Luật SCALE (Quy mô/Công nghiệp)
        if "cong nghiep" in t_scale or "tu dong" in t_scale:
            if any(k in doc_text for k in auto_keywords):
                entry["score"] += 3.0 # Cộng 3 điểm ưu tiên
            if any(k in doc_text for k in manual_keywords):
                entry["score"] = entry["score"] * 0.5 # Giảm một nửa điểm nếu là máy mini

    # Sắp xếp và trả về
    merged = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in merged[:top_k]]

def build_decision_llm(groq_api_key, openai_api_key, fallback_llm=None):
    # Ưu tiên 1: Dùng GPT-3.5-Turbo vì nó rẻ, nhanh và có structured output tốt
    try:
        if openai_api_key:
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.0,
                openai_api_key=openai_api_key,
                max_tokens=50, # Chỉ cần lấy nhãn intent
            )
    except Exception:
        pass
    
    # Ưu tiên 2: Dùng Groq Llama 8B nếu có
    try:
        if groq_api_key:
            return ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.0,
                groq_api_key=groq_api_key,
            )
    except Exception:
        pass
        
    # Fallback: Dùng model chính
    return fallback_llm

def llm_classify_intent(user_query, llm_main, history=None):
    # TỐI ƯU: Kiểm tra nhanh bằng quy tắc trước
    quick_intent = quick_intent_classify(user_query)
    if quick_intent:
        return quick_intent

    recent_history = ""
    if history:
        recent_turns = []
        for role, msg in history[-6:]:
            role_label = "user" if role == "user" else "assistant"
            clean_msg = normalize_text(re.sub(r"<[^>]+>", "", str(msg)))[:220]
            if clean_msg:
                recent_turns.append(f"{role_label}: {clean_msg}")
        recent_history = "\n".join(recent_turns)

    system_prompt = """Bạn là bộ phân loại intent cho chatbot VPRINT.
Chỉ trả về đúng 1 nhãn trong danh sách:
find_machine,book_knowledge,solution_consulting,direct_chat

Tiêu chí:
- find_machine: khách hỏi tìm một loại máy cụ thể, có đủ chi tiết để tìm kiếm (VD: 'máy bế hộp carton', 'máy in offset 4 màu khổ nhỏ').
- book_knowledge: khách hỏi kiến thức ngành in, nguyên lý, quy trình, so sánh công nghệ, thuật ngữ kỹ thuật.
- solution_consulting: khách mô tả bài toán sản xuất có đơn hàng, vật liệu, kích thước, sản lượng, năng suất, quy trình hoặc nhiều ràng buộc kỹ thuật và cần tư vấn giải pháp/chọn công nghệ/chọn tổ hợp máy.
- direct_chat: chào hỏi, cảm ơn, trò chuyện thông thường hoặc câu hỏi không cần tra cứu dữ liệu.

Ưu tiên:
- **QUAN TRỌNG**: Nếu khách hỏi tìm máy chung chung, không rõ ràng (VD: 'tôi cần máy in', 'tư vấn máy bế') và thiếu các thông tin quan trọng (sản lượng, vật liệu, khổ giấy), hãy chọn **solution_consulting** để bot hỏi lại thông tin, không chọn `find_machine`.
- **QUAN TRỌNG**: Nếu câu hỏi so sánh **công nghệ** (VD: 'offset vs flexo'), hãy chọn **book_knowledge**. Nếu so sánh **model máy cụ thể** (VD: 'VPX-1200 vs VPX-800'), hãy chọn **solution_consulting**. Đừng chọn `find_machine` cho các câu so sánh.
- Nếu câu hỏi nhắc rõ "máy nào", "dòng máy", "model", "thiết bị", "thông số máy" VÀ có đủ chi tiết thì mới chọn `find_machine`.
- Nếu câu hỏi là "là gì", "tại sao", "vai trò", "nguyên lý" trong ngữ cảnh ngành in thì chọn book_knowledge. Với câu hỏi "khác nhau", hãy xem quy tắc so sánh ở trên.
- Nếu câu hỏi có nhiều ràng buộc như sản lượng, vật liệu, khổ in, cấu trúc đơn hàng, công đoạn, workflow, đầu tư dây chuyền, chọn solution_consulting.
- Nếu câu hiện tại là câu làm rõ như "cần xác định gì", "để chọn máy phù hợp", "cần thông tin gì", "xác định dung lượng sản xuất", thì chọn solution_consulting, không chọn find_machine.
- QUAN TRỌNG: Nếu câu hiện tại có chủ đề KHÁC BIỆT hoàn toàn với lịch sử (VD: đang hỏi kiến thức in Offset sang tìm máy Bế), hãy ưu tiên phân loại theo CÂU HIỆN TẠI và BỎ QUA lịch sử cũ.
- Phải xét cả lịch sử hội thoại gần đây, không chỉ nhìn một câu đơn lẻ.
- Nếu không chắc giữa direct_chat và book_knowledge, ưu tiên direct_chat.
Chỉ trả về nhãn, không giải thích."""
    try:
        short_query = user_query[:INTENT_QUERY_MAX_CHARS]
        classifier_input = short_query
        if recent_history:
            classifier_input = f"Lịch sử gần đây:\n{recent_history}\n\nCâu hiện tại:\n{short_query}"
        response = llm_main.invoke([("system", system_prompt), ("user", classifier_input)])
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            st.session_state.api_tokens += response.usage_metadata.get('total_tokens', 0)
        intent = response.content.strip().lower()
        valid_intents = ["find_machine", "book_knowledge", "solution_consulting", "direct_chat"]
        for valid in valid_intents:
            if valid in intent: return valid
        return "direct_chat"
    except Exception:
        return "direct_chat"

try:
    machine_vector_retriever, book_retriever, sys_embedder = load_system()
except Exception as e:
    st.error(f"⚠️ Không thể khởi tạo hệ thống retrieval: {e}")
    st.stop()

# ==========================================
# 6. UI RENDER (SIDEBAR & MAIN)
# ==========================================
with st.sidebar:
    st.image("img/logo_2.jpg", width=200)
    st.header("⚙️ Model Settings")
    
    # Đã thêm GPT-4o vào danh sách lựa chọn
    selected_model = st.selectbox("Chọn Model", [
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        
        "gpt-4o",
        "gpt-3.5-turbo",
        "qwen/qwen3-32b", 
        "llama-3.1-8b-instant",
        
    ])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    
    st.subheader("📊 Phân tích (Intent)")
    st.plotly_chart(plot_intent_radar(), use_container_width=True)
    
    st.metric(label="🪙 Tổng Token đã dùng", value=f"{st.session_state.total_session_tokens:,}")
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.history, st.session_state.last_docs = [], []
            st.session_state.qa_cache = {}
            st.session_state.sent_booking_keys = set()
            st.session_state.viewed_machines = []
            st.session_state.intent_counts = {k: 0 for k in st.session_state.intent_counts}
            st.session_state.api_tokens = 0
            st.session_state.total_session_tokens = 0
            st.rerun()
    with col2:
        chat_data = generate_chat_export()
        st.download_button(label="📥 Tải Lịch sử", data=chat_data, file_name=f"VPRINT_Chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain", use_container_width=True)

st.title("🤖 VPRINT Sales AI")

for role, msg in st.session_state.history:
    avatar = "👤" if role == "user" else "img/logo_2.jpg" 
    with st.chat_message(role, avatar=avatar): st.markdown(msg, unsafe_allow_html=True)

if len(st.session_state.history) == 0: 
    st.markdown(WELCOME, unsafe_allow_html=True)

# ==========================================
# 7. LOGIC XỬ LÝ CHÍNH
# ==========================================
user_query = st.chat_input("Nhập câu hỏi...")

if user_query:
    with st.chat_message("user", avatar="👤"): st.markdown(user_query)
    st.session_state.history.append(("user", user_query))
    st.session_state.api_tokens = 0 

    # Auto notify sales via email when user sends booking/contact lead info.
    is_lead, lead_payload = detect_booking_lead(user_query)
    if is_lead:
        viewed_machines = st.session_state.viewed_machines or extract_viewed_machines(st.session_state.last_docs)
        recent_user_questions = collect_recent_user_questions(user_query)
        interest_area = infer_interest_area(" | ".join(recent_user_questions), st.session_state.last_docs)
        lead_payload["interest_area"] = interest_area
        lead_payload["viewed_machine_codes"] = extract_machine_codes(viewed_machines)
        lead_payload["interest_summary"] = build_interest_summary(recent_user_questions, user_query, interest_area)

        lead_key = make_query_cache_key(
            f"{lead_payload.get('phone', '')}|{lead_payload.get('email', '')}|{lead_payload.get('content', '')[:120]}"
        )
        ok, err = (True, "")
        if lead_key not in st.session_state.sent_booking_keys:
            ok, err = send_sale_email(lead_payload)
            if ok:
                st.session_state.sent_booking_keys.add(lead_key)
                st.toast("Đã gửi thông tin đặt lịch cho nhân viên sales.", icon="✅")
            else:
                st.toast(f"Gửi email lead thất bại: {err}", icon="⚠️")
        booking_ack = build_booking_ack(lead_payload, ok, err)
        with st.chat_message("assistant", avatar="img/logo_2.jpg"):
            st.markdown(booking_ack, unsafe_allow_html=True)
        st.session_state.history.append(("assistant", booking_ack))
        st.caption("⏱ Phản hồi: **0.00s** | 🎯 Phân tích: `booking_lead` | 🪙 Token: **0**")
        st.stop()

    has_context_dependency = is_context_dependent_query(user_query)
    turn_history = st.session_state.history[:-1] if has_context_dependency else []
    use_contextual_cache = not has_context_dependency
    cache_key = make_query_cache_key(user_query)
    cached_item = st.session_state.qa_cache.get(cache_key) if use_contextual_cache else None
    if cached_item:
        cached_answer = cached_item.get("answer", "")
        cached_intent = cached_item.get("intent", "normal_rag")
        with st.chat_message("assistant", avatar="img/logo_2.jpg"):
            st.markdown(cached_answer, unsafe_allow_html=True)
        st.session_state.history.append(("assistant", cached_answer))
        if cached_intent in st.session_state.intent_counts:
            st.session_state.intent_counts[cached_intent] += 1
        st.caption(f"⏱ Phản hồi: **0.00s** | 🎯 Phân tích: `{cached_intent}` | 🪙 Token: **0** | ⚡ Cache hit")
        st.stop()

    # Lấy API Key động dựa trên Model
    groq_api_key = get_safe_api_key("GROQ_API_KEY")
    openai_api_key = get_safe_api_key("OPENAI_API_KEY")

    # Setup LLM Main (Chuyển đổi linh hoạt giữa OpenAI và Groq)
    if selected_model in ["gpt-4o", "gpt-3.5-turbo"]:
        if not openai_api_key:
            st.error("⚠️ Không tìm thấy OPENAI_API_KEY. Vui lòng thêm vào file .env!")
            st.stop()
        llm_main = ChatOpenAI(
            model_name=selected_model,
            temperature=temperature,
            openai_api_key=openai_api_key,
            stream_usage=True,
            max_tokens=3000,
        )
    else:
        if not groq_api_key:
            st.error("⚠️ Không tìm thấy GROQ_API_KEY cho model Groq đã chọn.")
            st.stop()
        llm_main = ChatGroq(
            model_name=selected_model,
            temperature=temperature,
            groq_api_key=groq_api_key,
            max_tokens=3000,
        )
    
    start = time.perf_counter()
    history_limit = get_history_limit_for_model(selected_model)
    llm_decision = build_decision_llm(groq_api_key, openai_api_key, llm_main)
    intent_label = llm_classify_intent(user_query, llm_decision or llm_main, turn_history)
    decision = RouterDecision(
        intent=intent_label,
        use_rag=(intent_label in ["find_machine", "book_knowledge", "solution_consulting"]),
        reset_focus=(intent_label == "find_machine")
    )
    decision = apply_router_guards(user_query, decision)

    if decision.reset_focus:
        st.session_state.last_docs = []
    if decision.intent in st.session_state.intent_counts:
        st.session_state.intent_counts[decision.intent] += 1

    try:
        raw_answer = ""
        if decision.intent == "book_knowledge":
            with thinking_indicator():
                # NÂNG CẤP: Sử dụng RRF Retrieval để lấy kiến thức chính xác hơn,
                # tránh LLM bịa thông tin khi context rỗng.
                docs = retrieve_book_with_rrf(user_query, book_retriever, llm_main, top_k=10)
            
            if not docs:
                # FALLBACK: Nếu không tìm thấy tài liệu trong cẩm nang, dùng kiến thức chung của LLM
                messages = build_general_knowledge_fallback_messages(user_query, turn_history, history_limit)
                with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                    raw_answer = custom_write_stream(stream_response(messages, llm_main))
            else:
                # RAG BÌNH THƯỜNG: Nếu có tài liệu
                context = format_book_context(docs)
                messages = build_book_rag_messages(user_query, context, turn_history, history_limit)
                with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                    raw_answer = custom_write_stream(stream_response(messages, llm_main))

        elif decision.intent == "solution_consulting":
            with thinking_indicator():
                # --- BƯỚC 1: Bóc tách yêu cầu ---
                requirement = parse_consulting_request(user_query, turn_history, llm_main)
                
                # --- BƯỚC 2: Truy hồi kiến thức Cẩm nang (Luôn cần để làm nền) ---
                book_docs = retrieve_book_with_rrf(user_query, book_retriever, llm_main, top_k=4)
                book_context = format_book_context(book_docs)
                
                # --- BƯỚC 3: LUÔN LUÔN TÌM MÁY DÙ THÔNG TIN MỜ MỊT HAY RÕ RÀNG ---
                machine_docs = []
                machine_context = ""
                
                # TÍNH TOÁN DYNAMIC TOP-K
                # Nếu đã rõ ràng -> Chỉ chốt 1 máy tối ưu nhất cho mỗi công đoạn.
                # Nếu còn mù mờ -> Đưa ra 2-3 máy để khách tham khảo các phân khúc.
                dynamic_top_k = 1 if requirement.is_clear else 3
                
                for keyword in requirement.search_keywords:
                    process_docs = semantic_machine_search(
                        user_query=keyword,
                        vector_retriever=machine_vector_retriever,
                        top_k=6, # Vẫn lấy pool rộng để lọc
                        llm_main=None,
                        target_format=requirement.material_format,
                        target_scale=requirement.production_scale
                    )
                    process_docs = deduplicate_docs(process_docs)
                    
                    # Truyền dynamic_top_k vào hàm Rerank
                    top_process_docs = rerank_machine_candidates(keyword, process_docs, llm_main, top_k=dynamic_top_k)
                    
                    if top_process_docs:
                        machine_docs.extend(top_process_docs)
                        machine_context += f"\n--- MÁY TÌM ĐƯỢC CHO: {keyword.upper()} ---\n"
                        machine_context += format_context(top_process_docs)

                # --- BƯỚC 4: Rẽ nhánh Prompt dựa trên độ rõ ràng của yêu cầu ---
                if not requirement.is_clear:
                    # TÌNH HUỐNG A: Yêu cầu mù mờ -> Hỏi lại, nhưng có thể dùng máy đã tìm được làm VÍ DỤ
                    sys_prompt = f"""Bạn là chuyên gia tư vấn giải pháp của VPRINT.
                    Khách hàng đang nhờ tư vấn, nhưng thông tin quá chung chung.
                    Phân tích hệ thống cho thấy ta đang thiếu: {', '.join(requirement.missing_info)}
                    
                    Nhiệm vụ:
                    1. Tuyệt đối KHÔNG đề xuất bừa một model máy cụ thể nào ở bước này.
                    2. Dựa vào [CẨM NANG] và [VÍ DỤ MÁY], giải thích nhẹ nhàng cho khách hiểu tại sao việc xác định các yếu tố trên (như vật liệu, sản lượng) lại quan trọng để chọn đúng máy.
                    3. Đặt 2-3 câu hỏi ngắn gọn, lịch sự để khách hàng cung cấp thêm thông tin.
                    
                    [VÍ DỤ MÁY]:
                    {machine_context if machine_context else "Chưa tìm thấy máy ví dụ phù hợp."}

                    [CẨM NANG NGÀNH IN]:
                    {book_context}
                    """
                    messages = [("system", sys_prompt)] + get_optimized_history(turn_history, history_limit) + [("user", user_query)]
                
                else:
                    # TÌNH HUỐNG B: Yêu cầu đã rõ -> Trình bày giải pháp
                    # Lọc trùng lặp nếu có máy đa năng (VD: vừa in vừa bế)
                    machine_docs = deduplicate_docs(machine_docs)
                    
                    # Prompt tổng hợp cuối cùng
                    sys_prompt = f"""Bạn là Kỹ sư trưởng tư vấn giải pháp dây chuyền của VPRINT.
                    Khách hàng muốn làm sản phẩm: {requirement.product_type}.
                    Hệ thống đề xuất quy trình gồm: {', '.join(requirement.suggested_processes)}.
                    
                    Nhiệm vụ:
                    1. Nêu tóm tắt workflow.
                    2. Đề xuất máy. TUYỆT ĐỐI KHÔNG chọn máy cuộn (web-fed) cho vật liệu tờ rời (sheet-fed) và ngược lại.
                    3. QUAN TRỌNG: Lập một bảng đánh giá sự đồng bộ về Tốc độ/Năng suất của dây chuyền (Dựa vào thông số tốc độ của các máy đề xuất). Chỉ ra đâu có thể là nút thắt cổ chai (bottleneck) nếu chạy thực tế.
                    4. Chỉ dùng máy trong kho, nếu kho thiếu máy cho công đoạn nào, hãy báo rõ "VPRINT hiện chưa cập nhật dòng máy này trên hệ thống online".
                    5. Dùng [CẨM NANG] để giải thích thêm (nếu cần thiết) tại sao công nghệ/máy đó lại phù hợp với sản phẩm của khách.
                    
                    [KHO MÁY VPRINT LỌC THEO CÔNG ĐOẠN]:
                    {machine_context}
                    
                    [CẨM NANG NGÀNH IN]:
                    {book_context}
                    """
                    messages = [("system", sys_prompt)] + get_optimized_history(turn_history, history_limit) + [("user", user_query)]

                # --- BƯỚC 4: Generate Output ---
                with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                    raw_answer = custom_write_stream(stream_response(messages, llm_main))
                
                if machine_docs:
                    st.session_state.last_docs = machine_docs
                    track_viewed_machines(machine_docs[:3])

        elif decision.intent == "find_machine":
            with thinking_indicator():
                # Xử lý tham chiếu: Nếu đang hỏi tiếp về máy cũ (VD: "Máy này...") -> Gắn tên máy vào query
                search_query = user_query
                if has_context_dependency and st.session_state.last_docs:
                    last_machine_name = st.session_state.last_docs[0].metadata.get("name", "")
                    q_norm = normalize_for_match(user_query)
                    
                    # Chỉ nối tên máy cũ vào nếu câu hỏi HIỆN TẠI KHÔNG có từ khóa tìm loại máy MỚI
                    new_machine_keywords = ["may in", "may be", "may dan", "may can", "may ep", "may ghi"]
                    is_asking_new_machine = any(k in q_norm for k in new_machine_keywords)
                    
                    if last_machine_name and not is_asking_new_machine:
                        search_query = f"{user_query} ({last_machine_name})"

                filtered_docs = semantic_machine_search(
                    search_query,
                    machine_vector_retriever,
                    SEARCH_POOL_K,
                    llm_main,
                )
                filtered_docs = deduplicate_docs(filtered_docs)[:SEARCH_POOL_K]

            # TÍNH TOÁN DYNAMIC TOP-K CHO TÌM KIẾM TRỰC TIẾP
            q_norm = normalize_for_match(user_query)
            # Kiểm tra xem có số (thể hiện model/thông số) hoặc câu hỏi mô tả dài không
            has_specs = bool(re.search(r'\d+', q_norm)) 
            is_detailed = len(user_query.split()) > 7
            
            dynamic_top_k = 1 if (has_specs or is_detailed) else 3

            # Gọi Rerank với Top-K động
            suggested_docs = rerank_machine_candidates(
                user_query=user_query,
                candidate_docs=filtered_docs,
                llm_main=llm_main,
                top_k=dynamic_top_k, 
            )

            if not suggested_docs:
                raw_answer = "Xin lỗi, tôi chưa tìm thấy máy phù hợp trong kho dữ liệu hiện tại của VPRINT."
                with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                    st.markdown(raw_answer, unsafe_allow_html=True)
            else:
                track_viewed_machines(suggested_docs)
                
                # Sửa câu chào dựa trên số lượng máy
                if len(suggested_docs) == 1:
                    raw_answer = "👋 Dựa trên thông số chi tiết bạn cung cấp, đây là cấu hình máy tối ưu nhất:\n\n"
                else:
                    raw_answer = "👋 Để bạn dễ hình dung, VPRINT xin gợi ý một vài dòng máy phù hợp với các phân khúc khác nhau:\n\n"
                
                is_spec_request = any(k in normalize_for_match(user_query) for k in ["thong so", "cau hinh", "spec", "ky thuat", "chi tiet"])

                with thinking_indicator():
                    machine_summaries = summarize_machines_structured(user_query, suggested_docs, llm_main)

                with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                    display_container = st.empty()
                    for i, doc in enumerate(suggested_docs):
                        name = doc.metadata.get("name", f"Sản phẩm {i+1}")
                        url = doc.metadata.get("product_url", "")
                        imgs = parse_images(doc.metadata.get("images", ""))
                        if len(suggested_docs) > 1:
                            raw_answer += f"### 🏆 Top {i+1}: **[{name}]({url})**\n"
                        else:
                            raw_answer += f"### 🏆 **[{name}]({url})**\n"
                        if imgs:
                            img_html = "".join(
                                [f'<img src="{img}" style="height:140px;margin-right:8px;border-radius:8px;border:1px solid #ddd;object-fit:contain;">' for img in imgs[:3]]
                            )
                            raw_answer += f"<div>{img_html}</div><br>\n\n"

                        if i < len(machine_summaries):
                            s = machine_summaries[i]
                            
                            # Trích xuất chuỗi JSON gốc từ page_content thay vì dùng s.performance đã bị cắt cụt
                            specs_json_str = extract_labeled_value(doc.page_content, "Specifications")
                            
                            # Render thành bảng Markdown
                            formatted_performance = format_specs_to_json_table(specs_json_str)

                            raw_answer += (
                                f"- **Mô tả:** {s.description}\n"
                                f"**📊 Tốc độ / Hiệu suất:**\n\n{formatted_performance}\n\n"
                                f"- **Công nghệ:** {s.technology}\n"
                                f"- **Điểm ưu việt:** {s.advantage}\n\n"
                            )

                        raw_answer += "---\n\n"
                        # Cập nhật UI ngay lập tức sau mỗi máy để người dùng không phải chờ
                        display_container.markdown(raw_answer, unsafe_allow_html=True)
                st.session_state.last_docs = suggested_docs

        else:
            with thinking_indicator():
                messages = build_direct_messages(user_query, turn_history, history_limit)
            with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                raw_answer = custom_write_stream(stream_response(messages, llm_main))

        if raw_answer:
            if not raw_answer.startswith("⚠️"):
                st.session_state.qa_cache[cache_key] = {
                    "answer": raw_answer,
                    "intent": decision.intent,
                }
            st.session_state.history.append(("assistant", raw_answer))
            
            tokens_used = st.session_state.api_tokens
            if tokens_used == 0: tokens_used = (len(str(user_query)) + len(str(raw_answer))) // 3
            
            st.session_state.total_session_tokens += tokens_used
            process_time = time.perf_counter() - start
            
            log_chat_to_gsheet_async(user_query, raw_answer, decision.intent, process_time, tokens_used, selected_model)

    except Exception as e:
        error_msg = f"⚠️ Hệ thống AI đang bận. Vui lòng thử lại! (Chi tiết: {str(e)})"
        with st.chat_message("assistant", avatar="img/logo_2.jpg"): st.markdown(error_msg)
        st.session_state.history.append(("assistant", error_msg))
        log_chat_to_gsheet_async(user_query, error_msg, decision.intent, 0, 0, selected_model)

    st.caption(f"⏱ Phản hồi: **{round(time.perf_counter() - start, 2)}s** | 🎯 Phân tích: `{decision.intent}` | 🪙 Token: **{st.session_state.api_tokens}**")
