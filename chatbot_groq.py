import streamlit as st
import streamlit.components.v1 as components
import time
import re
import json
import os
import base64
import torch
import threading
import smtplib
import unicodedata
import yaml
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
        "troubleshooting": 0,
        "direct_chat": 0,
        "out_of_scope": 0,
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

@st.cache_data
def load_bot_rules(filepath="bot_rules.yaml"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Lỗi đọc file YAML: {e}")
        return {}

BOT_RULES = load_bot_rules()

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

logo_watermark_b64 = ""
logo_watermark_path = Path("img/logo_2.jpg")
if logo_watermark_path.exists():
    logo_watermark_b64 = base64.b64encode(logo_watermark_path.read_bytes()).decode("ascii")
logo_watermark_css = f'url("data:image/jpeg;base64,{logo_watermark_b64}")' if logo_watermark_b64 else "none"

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

def add_e5_passage_prefix(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return "passage:"
    return clean if clean.lower().startswith("passage:") else f"passage: {clean}"

def add_e5_query_prefix(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return "query:"
    return clean if clean.lower().startswith("query:") else f"query: {clean}"

def strip_e5_prefix(text: str) -> str:
    return re.sub(r"^(?:passage|query):\s*", "", str(text or "").strip(), flags=re.IGNORECASE)

def detect_query_language(user_query: str) -> str:
    q = str(user_query or "").strip()
    if not q:
        return "vi"

    lowered = q.lower()
    ascii_q = q.encode("ascii", errors="ignore").decode("ascii").lower()

    vi_diacritics = bool(re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", lowered))
    vi_markers = BOT_RULES.get("language_detection", {}).get("vi_markers", [])
    en_markers = BOT_RULES.get("language_detection", {}).get("en_markers", [])

    vi_hits = sum(1 for marker in vi_markers if marker in lowered)
    en_hits = sum(1 for marker in en_markers if marker in ascii_q)

    if vi_diacritics or vi_hits > en_hits:
        return "vi"
    if en_hits > vi_hits:
        return "en"

    english_word_hits = len(re.findall(r"\b(the|and|for|with|machine|printing|price|spec|help|show|recommend|what|how|why)\b", ascii_q))
    vietnamese_word_hits = len(re.findall(r"\b(may|gia|thong|so|tu|van|giup|bao|nhieu|xuong|in|ly|giay)\b", normalize_for_match(q)))
    return "en" if english_word_hits > vietnamese_word_hits else "vi"

def get_response_language_instruction(user_query: str) -> str:
    if detect_query_language(user_query) == "en":
        return "Answer in English. Keep the full response in English unless the user asks to switch language."
    return "Trả lời bằng tiếng Việt. Nếu người dùng chuyển sang tiếng Anh thì trả lời bằng tiếng Anh."

def get_machine_intro_prompt(user_query: str) -> str:
    """Tạo phần intro mở đầu động theo ngôn ngữ của người dùng"""
    if detect_query_language(user_query) == "en":
        return "Start by providing a warm greeting, then present the recommended machines. Example intro: 'Based on your requirements, here are the top machines VPRINT recommends...'"
    return "Bắt đầu bằng lời chào ấm áp bằng tiếng Việt, rồi giới thiệu các máy được đề xuất. Ví dụ: 'Dựa trên nhu cầu của bạn, dưới đây là những máy tốt nhất VPRINT gợi ý...'"

def ui_copy_for_language(user_query: str) -> dict:
    if detect_query_language(user_query) == "en":
        return {
            "no_result": "Sorry, I could not find a suitable machine in the current VPRINT database.",
            "single_intro": "Based on the detailed requirements you provided, this is the most suitable machine configuration:\n\n",
            "multi_intro": "To help you compare options more easily, VPRINT suggests several suitable machine lines across different segments:\n\n",
            "top_single": "### 🏆 **{name}**\n",
            "top_multi": "### 🏆 Top {rank}: **{name}**\n",
            "desc": "- **Description:** {value}\n",
            "perf": "**📊 Speed / Performance:**\n\n{value}\n\n",
            "tech": "- **Technology:** {value}\n",
            "adv": "- **Key advantage:** {value}\n\n",
        }
    return {
        "no_result": "Xin lỗi, tôi chưa tìm thấy máy phù hợp trong kho dữ liệu hiện tại của VPRINT.",
        "single_intro": "👋 Dựa trên thông số chi tiết bạn cung cấp, đây là cấu hình máy tối ưu nhất:\n\n",
        "multi_intro": "👋 Để bạn dễ hình dung, VPRINT xin gợi ý một vài dòng máy phù hợp với các phân khúc khác nhau:\n\n",
        "top_single": "### 🏆 **{name}**\n",
        "top_multi": "### 🏆 Top {rank}: **{name}**\n",
        "desc": "- **Mô tả:** {value}\n",
        "perf": "**📊 Tốc độ / Hiệu suất:**\n\n{value}\n\n",
        "tech": "- **Công nghệ:** {value}\n",
        "adv": "- **Điểm ưu việt:** {value}\n\n",
    }

def is_book_knowledge_intent(query: str) -> bool:
    q = normalize_for_match(query)
    book_keywords = BOT_RULES.get("intent_routing", {}).get("book_knowledge", [])
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
    ctp_terms = BOT_RULES.get("ctp_knowledge_rules", {}).get("ctp_terms", [])
    compare_terms = BOT_RULES.get("ctp_knowledge_rules", {}).get("compare_terms", [])
    machine_buy_terms = BOT_RULES.get("ctp_knowledge_rules", {}).get("machine_buy_terms", [])
    has_ctp = any(t in q for t in ctp_terms)
    has_compare = any(t in q for t in compare_terms)
    has_buy_intent = any(t in q for t in machine_buy_terms)
    return has_ctp and (has_compare or not has_buy_intent)

@dataclass
class RouterDecision:
    intent: str = "direct_chat"
    use_rag: bool = True
    reset_focus: bool = False

class IntentRoute(BaseModel):
    intent: str = Field(
        description="Chỉ được chọn 1 trong 6 nhãn: find_machine, book_knowledge, solution_consulting, troubleshooting, direct_chat, out_of_scope"
    )
    reasoning: str = Field(
        description="Giải thích ngắn sau khi đã chọn intent"
    )
    # intent trước → model commit nhãn trước khi bị reasoning kéo lệch

class DocGrade(BaseModel):
    doc_index: int = Field(description="Số thứ tự của tài liệu")
    is_relevant: bool = Field(description="True nếu tài liệu liên quan đến câu hỏi, False nếu rác")

class BatchGradeDocuments(BaseModel):
    grades: List[DocGrade] = Field(description="Danh sách điểm của tất cả tài liệu")

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

def is_out_of_scope_machine_query(user_query: str) -> bool:
    q = normalize_for_match(user_query)
    if not q:
        return False

    consumer_markers = BOT_RULES.get("out_of_scope_rules", {}).get("consumer_markers", [])
    industrial_markers = BOT_RULES.get("out_of_scope_rules", {}).get("industrial_markers", [])

    has_consumer_marker = any(marker in q for marker in consumer_markers)
    has_industrial_marker = any(marker in q for marker in industrial_markers)
    return has_consumer_marker and not has_industrial_marker

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
        text = add_e5_passage_prefix("\n".join([
            f"Product name: {row.get('name', '')}",
            f"Price: {row.get('price', '')}",
            f"Summary: {row.get('summary', '')}",
            f"Description: {row.get('description', '')}",
            f"Features: {row.get('features', '')}",
            f"Specifications: {row.get('specs_json', '')}",
        ]).strip())
        if text.strip() != "passage:":
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
        content = normalize_text(strip_e5_prefix(doc.page_content))[:MAX_CONTEXT_CHARS_PER_DOC]
        blocks.append(
            f"[Tai lieu {i}]\nTen: {doc.metadata.get('name', '')}\nGia: {doc.metadata.get('price', '')}\nURL: {doc.metadata.get('product_url', '')}\nNoi dung: {content}"
        )
    return "\n\n".join(blocks)

def format_book_context(docs):
    if not docs:
        return "Không tìm thấy thông tin phù hợp trong cẩm nang."

    blocks = []

    for i, doc in enumerate(docs[:MAX_CONTEXT_DOCS], start=1):
        content = normalize_text(strip_e5_prefix(doc.page_content))[:MAX_CONTEXT_CHARS_PER_DOC]

        blocks.append(content)

    return "\n\n============================\n\n".join(blocks)

def filter_relevant_docs_batch(user_query: str, docs: List[Document], llm_main) -> List[Document]:
    """[Self-RAG Batch] Lọc toàn bộ tài liệu rác chỉ với 1 API Call duy nhất."""
    if not docs:
        return []
    
    # Giới hạn tối đa 20 docs để tránh prompt quá dài
    docs_to_filter = docs[:20]
    
    # Gộp tất cả tài liệu thành 1 chuỗi dài có đánh số thứ tự
    combined_docs_text = ""
    for idx, doc in enumerate(docs_to_filter):
        # Cắt mỗi doc ngắn lại khoảng 250 ký tự để không tràn token
        content = str(doc.page_content)[:250].replace('\n', ' ')
        doc_name = doc.metadata.get('name', doc.metadata.get('source_book', 'Tài liệu'))
        combined_docs_text += f"\n[Tài liệu {idx}]: {doc_name} - {content}"
    
    system_prompt = """Bạn là giám khảo chấm điểm tài liệu nghiêm khắc.
Nhiệm vụ: Đánh giá xem từng tài liệu trong danh sách có chứa thông tin để trả lời câu hỏi của khách hàng không.
Hãy duyệt qua TẤT CẢ các tài liệu được cung cấp và trả về danh sách kết quả (True/False).
QUAN TRỌNG: Chỉ đánh giá dựa trên nội dung tài liệu, không cần suy diễn quá mức."""

    user_msg = f"Câu hỏi của khách: {user_query}\n\nDanh sách tài liệu:\n{combined_docs_text}"

    try:
        structured_llm_grader = llm_main.with_structured_output(BatchGradeDocuments)
        res = structured_llm_grader.invoke([
            ("system", system_prompt),
            ("user", user_msg)
        ])
        
        # Bóc tách kết quả để lọc mảng docs ban đầu
        filtered_docs = []
        for grade in res.grades:
            # Lấy đúng index và kiểm tra xem có pass không
            idx = grade.doc_index
            if grade.is_relevant and 0 <= idx < len(docs_to_filter):
                filtered_docs.append(docs_to_filter[idx])
            elif 0 <= idx < len(docs_to_filter):
                print(f"🗑️ [Self-RAG] Đã loại: {docs_to_filter[idx].metadata.get('name', 'Cẩm nang')}")
        
        # Fallback giữ nguyên nếu lọc nhầm hết
        return filtered_docs if filtered_docs else docs_to_filter
        
    except Exception as e:
        print(f"⚠️ Lỗi Batch Grader: {e}")
        return docs_to_filter  # Fallback an toàn

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
    language_instruction = get_response_language_instruction(user_query)
    sys_prompt = f"""Bạn là Chuyên gia AI của VPRINT.
    Sử dụng thông tin trong [KHO DỮ LIỆU] dưới đây để trả lời khách hàng.
    Nếu không có thông tin, hãy nói không biết, TUYỆT ĐỐI KHÔNG BỊA ĐẶT.
    Không lặp lại câu hỏi của người dùng.
    {language_instruction}

    [KHO DỮ LIỆU]:
    {context}
    """
    return [("system", sys_prompt)] + get_optimized_history(history, max_history) + [("user", user_query)]

def build_book_rag_messages(user_query, context, history, max_history=5):
    language_instruction = get_response_language_instruction(user_query)
    is_english = detect_query_language(user_query) == "en"
    
    # Chuyển đổi cấu trúc Prompt tùy theo ngôn ngữ
    if is_english:
        structure_rules = """
📝 MANDATORY RESPONSE STRUCTURE (Be flexible based on the question):
- 🎯 Summary/Definition: 1-2 brief sentences getting straight to the core.
- ⚙️ Principles / In-depth Analysis: Use bullet points to explain mechanisms or core technical features.
- ⚖️ Comparison / Pros & Cons (If applicable): Directly compare clear technical parameters.
- 💡 Expert Advice (Practical): Application advice in real-world factory environments (e.g., best for high-volume packaging vs. short-run labels).

⚠️ CRITICAL RULES:
- ONLY use the provided [PRINTING KNOWLEDGE BASE] as your foundation.
- Do not fabricate technical specifications.
- Translate all concepts and technical terms smoothly into professional English.
"""
    else:
        structure_rules = """
📝 CẤU TRÚC CÂU TRẢ LỜI BẮT BUỘC (Hãy linh hoạt áp dụng tùy câu hỏi):
- 🎯 Tóm tắt/Định nghĩa: 1-2 câu ngắn gọn đi thẳng vào bản chất vấn đề.
- ⚙️ Nguyên lý / Phân tích chuyên sâu: Cấu trúc hóa bằng gạch đầu dòng giải thích cách thức hoạt động hoặc các đặc điểm kỹ thuật cốt lõi.
- ⚖️ So sánh / Ưu nhược điểm (Nếu câu hỏi mang tính chọn lựa): Đối chiếu các thông số kỹ thuật rõ ràng.
- 💡 Góc nhìn chuyên môn (Thực tiễn): Lời khuyên ứng dụng trong môi trường xưởng in thực tế (VD: công nghệ này hợp với in bao bì số lượng lớn, hay in tem nhãn ngắn ngày?).

⚠️ QUY TẮC RÀNG BUỘC:
- CHỈ sử dụng dữ liệu từ [CẨM NANG NGÀNH IN] để làm nền tảng.
- Tuyệt đối không bịa đặt thông số kỹ thuật.
"""

    sys_prompt = f"""Bạn là Chuyên gia Cấp cao về Công nghệ In ấn & Bao bì của VPRINT, với hơn 20 năm kinh nghiệm nghiên cứu, vận hành máy và giảng dạy ngành in.
    
Nhiệm vụ của bạn là giải đáp các câu hỏi kỹ thuật ngành in dựa trên [CẨM NANG NGÀNH IN].
{language_instruction}

✨ TIÊU CHUẨN CỦA MỘT CHUYÊN GIA:
1. Sâu sắc & Chính xác: Không chỉ trả lời "Cái gì" (What) mà phải giải thích "Tại sao" (Why) và "Như thế nào" (How).
2. Thuật ngữ chuyên ngành: Sử dụng đúng thuật ngữ chuyên môn.

{structure_rules}

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


# ==========================================
# QUERY EXPANSION V2 — 3 LỚP (0 LLM CALL)
# ==========================================
# Lớp 1: Original query
# Lớp 2: Rule-based templates (deterministic)
# Lớp 3: Alias-driven từ VectorDB (tự tìm tên máy chính thức)

EXPANSION_TEMPLATES = [
    "máy sản xuất {q}",
    "thiết bị {q}",
]

QUERY_STOP_WORDS = {
    "may", "nao", "dung", "de", "cho", "toi", "anh", "chi",
    "la", "gi", "vprint", "tren", "voi", "muon", "tim", "co",
    "nhu", "the", "nao", "biet", "hoi", "xin", "ban", "minh",
    "chung", "toi", "gia", "bao", "nhieu", "loai", "dong",
}

def _extract_core_keywords(query: str) -> str:
    """Bóc tách keyword cốt lõi, bỏ stop words."""
    normalized = normalize_for_match(query)
    tokens = [t for t in re.findall(r"[a-z0-9]+", normalized)
              if t not in QUERY_STOP_WORDS and len(t) >= 2]
    return " ".join(tokens[:6]) if tokens else normalized

def _rule_based_expand(query: str) -> List[str]:
    """Lớp 2: Sinh queries bằng template cố định. 0 LLM call."""
    core = _extract_core_keywords(query)
    results = []
    for tmpl in EXPANSION_TEMPLATES:
        expanded = tmpl.format(q=core)
        if expanded != core:
            results.append(expanded)
    return results

def _alias_driven_expand(query: str, vector_retriever, top_k_alias: int = 3, min_score: float = 0.50) -> List[str]:
    """
    Lớp 3: Tìm tên máy chính thức từ alias trong VectorDB. 0 LLM call.
    "tô giấy" → similarity search → doc "Máy làm ly giấy" → dùng tên chính thức làm query mới.
    """
    try:
        if hasattr(vector_retriever, "vectorstore"):
            store = vector_retriever.vectorstore
            pairs = store.similarity_search_with_relevance_scores(
                add_e5_query_prefix(query), k=top_k_alias
            )
        else:
            docs = vector_retriever.invoke(add_e5_query_prefix(query))
            pairs = [(d, 1.0) for d in docs[:top_k_alias]]

        alias_queries = []
        seen_names = set()
        for doc, score in pairs:
            if score < min_score:
                continue
            machine_name = str(doc.metadata.get("name", "")).strip()
            if not machine_name:
                continue
            name_key = normalize_for_match(machine_name)
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            alias_queries.append(machine_name)
            # Nếu tên dài, thêm phiên bản rút gọn 4 tokens
            tokens = machine_name.split()
            if len(tokens) > 4:
                short_name = " ".join(tokens[:4])
                if normalize_for_match(short_name) not in seen_names:
                    alias_queries.append(short_name)
                    seen_names.add(normalize_for_match(short_name))
        return alias_queries
    except Exception as e:
        print(f"[QueryExpand] Alias-driven failed: {e}")
        return []

def expand_machine_queries_v2(
    user_query: str,
    vector_retriever,
    llm_main=None,
    max_queries: int = 6,
) -> List[str]:
    """
    Query expansion 3 lớp — thay thế expand_machine_queries().
    Lớp 1: Original query (luôn có)
    Lớp 2: Rule-based templates (0 LLM)
    Lớp 3: Alias-driven từ VectorDB (0 LLM)
    """
    base = normalize_text(user_query)
    candidates = [base]

    rule_queries = _rule_based_expand(user_query)
    candidates.extend(rule_queries)

    if vector_retriever is not None:
        alias_queries = _alias_driven_expand(user_query, vector_retriever, top_k_alias=3, min_score=0.50)
        candidates.extend(alias_queries)

    seen = set()
    deduped = []
    for q in candidates:
        k = normalize_for_match(q)
        if k and k not in seen:
            seen.add(k)
            deduped.append(q)

    result = deduped[:max_queries]
    print(f"[QueryExpand] '{user_query[:50]}' → {len(result)} queries:")
    for i, q in enumerate(result):
        layer = "orig" if i == 0 else ("rule" if i <= len(rule_queries) else "alias")
        print(f"   [{layer}] {q}")
    return result

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

    lamination = BOT_RULES.get("operation_buckets", {}).get("lamination", [])
    gluing = BOT_RULES.get("operation_buckets", {}).get("gluing", [])
    die_cut = BOT_RULES.get("operation_buckets", {}).get("die_cut", [])
    ctp = BOT_RULES.get("operation_buckets", {}).get("ctp", [])
    printing = BOT_RULES.get("operation_buckets", {}).get("printing", [])
    cutting = BOT_RULES.get("operation_buckets", {}).get("cutting", [])
    cup_making = BOT_RULES.get("operation_buckets", {}).get("cup_making", [])

    # Lamination
    if lamination and re.search(r"\b(" + "|".join(re.escape(x) for x in lamination) + r")\b", q):
        return "lamination"
    # Gluing (Dán)
    if gluing and re.search(r"\b(" + "|".join(re.escape(x) for x in gluing) + r")\b", q):
        return "gluing"
    # Die cut (Bế/Ép kim)
    if die_cut and re.search(r"\b(" + "|".join(re.escape(x) for x in die_cut) + r")\b", q):
        return "die_cut"
    # CTP (Chế bản)
    if ctp and re.search(r"\b(" + "|".join(re.escape(x) for x in ctp) + r")\b", q):
        return "ctp"
    # Printing (In)
    if printing and re.search(r"\b(" + "|".join(re.escape(x) for x in printing) + r")\b", q):
        return "printing"
    # Cutting (Xén giấy)
    if cutting and re.search(r"\b(" + "|".join(re.escape(x) for x in cutting) + r")\b", q):
        return "cutting"
    # Cup making (Ly giấy)
    if cup_making and re.search(r"\b(" + "|".join(re.escape(x) for x in cup_making) + r")\b", q):
        return "cup_making"

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
    queries = [normalize_text(user_query)]
    if is_short_ambiguous_query(user_query):
        queries = expand_book_queries(user_query, llm_main, max_queries=3)

    rrf_k = 60.0
    scored = {}
    for q in queries:
        try:
            docs = book_retriever.invoke(add_e5_query_prefix(q))
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
    return f"v21::{normalize_for_match(query)}"

def is_context_dependent_query(query: str) -> bool:
    q = normalize_for_match(query)
    
    # 1. Các cụm từ nối trực tiếp rõ ràng
    followup_markers = BOT_RULES.get("context_dependency", {}).get("followup_markers", [])
    if any(m in q for m in followup_markers):
        return True

    # 2. Bắt các câu yêu cầu lọc/tinh chỉnh (Refine markers)
    refine_markers = BOT_RULES.get("context_dependency", {}).get("refine_markers", [])
    if any(m in q for m in refine_markers):
        return True

    # Nếu câu hỏi chứa từ khóa tìm kiếm máy rõ ràng, coi như là context mới (Reset)
    new_search_signals = BOT_RULES.get("context_dependency", {}).get("new_search_signals", [])
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

def build_direct_messages(user_query, history, max_history=5):
    language_instruction = get_response_language_instruction(user_query)
    sys_prompt = f"""Bạn là Trợ lý AI Chuyên gia Bán hàng của VPRINT.
Khách hàng đang chào hỏi hoặc hỏi câu ngoài lề (chit-chat, đố vui, trêu chọc).

{language_instruction}

🎯 QUY TẮC BẮT BUỘC — KHÔNG ĐƯỢC VI PHẠM:
1. Trả lời TỐI ĐA 1 câu ngắn. Không kể chuyện, không giải thích dài.
2. Câu thứ 2 BẮT BUỘC bẻ lái về máy móc/ngành in của VPRINT.
3. TUYỆT ĐỐI không kể chuyện ma, chuyện cười, thơ, hay bất kỳ nội dung giải trí nào dù chỉ 1 câu.

--- VÍ DỤ BẮT BUỘC PHẢI FOLLOW ---
Khách: "Kể chuyện ma đi"
Bot: "Chuyện ma thì mình không rành, nhưng công nghệ UV hay ép kim trong ngành in thì mình khá chắc tay. Xưởng của bạn đang cần tư vấn dòng máy hay công nghệ nào?"

Khách: "1+1 bằng mấy"
Bot: "Dạ bằng 2 ạ. Nếu bạn cần tính chi phí đầu tư hoặc sản lượng cho dây chuyền in bao bì, mình hỗ trợ rất nhanh."

Khách: "Bạn ăn cơm chưa?"
Bot: "Dạ mình là AI nên chạy bằng điện ạ. Nếu bạn đang tìm máy in công nghiệp, mình hỗ trợ ngay."
---

LƯU Ý CUỐI: Dù khách hỏi bao nhiêu lần hay theo cách nào, KHÔNG BAO GIỜ kể chuyện, làm thơ, hay tạo nội dung giải trí."""

    # KHÔNG truyền history vào direct_chat — stateless, không cần nhớ context chit-chat
    return [("system", sys_prompt), ("user", user_query)]

def build_general_knowledge_fallback_messages(user_query, history, max_history=5):
    language_instruction = get_response_language_instruction(user_query)
    is_english = detect_query_language(user_query) == "en"
    
    if is_english:
        intro_line = "Our internal handbook currently does not contain specific documentation for this question. However, drawing on my extensive professional expertise, I will advise you based on common industry standards (ISO 12647, FOGRA, G7...)."
        guidelines = """📝 RESPONSE GUIDELINES:
1. Open with a courteous acknowledgment: "Our VPRINT knowledge base doesn't have specific documentation for this topic. However, from a professional printing industry perspective, here's my advice:"
2. Provide technical hypotheses (If customer asks about printing issues: List possible causes—ink, paper, pressure, plate composition...).
3. Suggest troubleshooting steps or solutions (Troubleshooting steps).
4. Use professional language with accurate technical terminology (e.g., overprinting, trapping, ink tack, pan pH...).
5. Tactfully suggest the customer can contact VPRINT technical staff for more in-depth support."""
    else:
        intro_line = "Hiện tại, hệ thống Cẩm nang nội bộ không chứa tài liệu trực tiếp về câu hỏi này. Tuy nhiên, với kinh nghiệm uyên thâm của mình, bạn hãy tư vấn cho khách hàng dựa trên tiêu chuẩn chung của ngành công nghiệp in (ISO 12647, FOGRA, G7...)."
        guidelines = """📝 Hướng dẫn trả lời:
1. Mở đầu bằng một câu rào đón lịch sự: "Dữ liệu cẩm nang nội bộ của VPRINT hiện chưa có tài liệu cụ thể cho vấn đề này. Tuy nhiên, dưới góc độ chuyên môn ngành in, tôi xin chia sẻ như sau:"
2. Đưa ra các giả thuyết kỹ thuật (Nếu khách hỏi về lỗi in ấn: Liệt kê các nguyên nhân có thể do mực, do giấy, do áp lực lô, do chế bản...).
3. Đề xuất hướng kiểm tra hoặc khắc phục từng bước (Troubleshooting steps).
4. Giữ phong thái chuyên nghiệp, dùng từ vựng kỹ thuật chuẩn xác (VD: overprinting, trapping, tack mực, pH nước máng...).
5. Khéo léo gợi ý khách hàng có thể liên hệ kỹ thuật viên VPRINT để được hỗ trợ chuyên sâu hơn."""
    
    sys_prompt = f"""Bạn là Kỹ sư Trưởng chuyên xử lý sự cố và vận hành xưởng in của VPRINT.
{intro_line}

{guidelines}
{language_instruction}
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
    language_instruction = get_response_language_instruction(user_query)
    is_english = detect_query_language(user_query) == "en"
    
    if is_english:
        role_intro = "You are a solution consulting engineer at VPRINT. Analyze the production requirements and provide a tight consulting solution based on 2 sources: [VPRINT MACHINE DATABASE] and [PRINTING INDUSTRY HANDBOOK]."
        rules_section = """RULES:
- **IMPORTANT**: If the customer's request is vague or unclear (e.g., 'find a printing machine', 'recommend a die-cutter'), START by asking clarifying questions about their needs (product type, production volume, material, sheet size...) before proposing any machines. Don't guess and jump to recommendations.
- Only use machines available in [VPRINT MACHINE DATABASE].
- ABSOLUTELY DO NOT recommend machines not in [VPRINT MACHINE DATABASE]. If they need a machine not available, state clearly: 'Need to equip machine X (VPRINT database currently has no data for this)'.
- You can use [PRINTING INDUSTRY HANDBOOK] to explain technical reasoning.
- Do not fabricate machine names or specifications.
- If machine data is missing for a process step, clearly state that sufficient machine data is not yet available in VPRINT's database.
- If the current question asks about machine selection criteria, required input data, or factors affecting machine choice:
  do NOT jump directly to recommending a model.
  Prioritize listing the required data, explaining why each piece is important, then suggest the direction for machine selection.
  Don't repeat the question.

EXPECTED OUTPUT:
1. Analyze requirements and key technical constraints.
2. Propose suitable technology/process.
3. Recommend VPRINT machines best fit for each important process step.
4. Clearly explain why each machine is chosen based on retrieved data.
5. If input information is still missing, end with 2-3 clarifying questions."""
    else:
        role_intro = "Bạn là kỹ sư tư vấn giải pháp của VPRINT. Hãy phân tích yêu cầu sản xuất và tư vấn giải pháp chặt chẽ dựa trên 2 nguồn: [KHO MÁY VPRINT] và [CẨM NANG NGÀNH IN]."
        rules_section = """Quy tắc:
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
5. Nếu còn thiếu thông tin đầu vào, kết thúc bằng 2-3 câu hỏi làm rõ."""
    
    sys_prompt = f"""{role_intro}
{language_instruction}

{rules_section}
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
def translate_machine_details_to_english(text_block: str, llm_main) -> str:
    """Dịch thông số và mô tả máy sang tiếng Anh, bắt buộc giữ nguyên format Markdown (Bảng)."""
    prompt = f"""You are an expert technical translator in the printing industry.
Translate the following Vietnamese machine details into professional English.

CRITICAL RULES:
1. You MUST preserve the exact Markdown formatting, especially the table structure (|---|---|), boldings, and bullet points.
2. Translate all technical terms accurately (e.g., "Mực" -> "Ink", "Đầu in" -> "Printhead").
3. Output ONLY the translated text, without any conversational introduction.

Text to translate:
{text_block}
"""
    try:
        # Lợi dụng tốc độ siêu nhanh của Groq để dịch
        res = llm_main.invoke([("system", "You are a professional translator."), ("user", prompt)])
        return res.content.strip()
    except Exception as e:
        print(f"⚠️ Lỗi dịch thuật: {e}")
        return text_block  # Fallback: Giữ nguyên tiếng Việt nếu đứt mạng

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
    valid_candidates = [d for score, d in scored if score >= 0]
    return valid_candidates[:top_k]

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

    # 1. Chỉ khởi tạo 1 Embedder duy nhất (Tiết kiệm một nửa thời gian load và RAM)
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_MACHINES, 
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2. Xử lý Machine Store: Lưu xuống ổ cứng để không phải embed lại từ file CSV
    MACHINE_PERSIST_DIR = "vprint_machines_db_local" # Thư mục lưu DB máy móc
    
    if os.path.exists(MACHINE_PERSIST_DIR) and os.listdir(MACHINE_PERSIST_DIR):
        # Nếu đã tạo DB trước đó, chỉ cần load lên (Tốc độ tính bằng mili-giây)
        machine_store = Chroma(
            persist_directory=MACHINE_PERSIST_DIR, 
            embedding_function=embedder, 
            collection_name=COLLECTION_MACHINES
        )
    else:
        # Nếu chưa có, mới đọc CSV và nhúng vector (Chỉ chậm ở LẦN CHẠY ĐẦU TIÊN)
        docs = load_csv_docs(CSV_PATH)
        machine_store = Chroma.from_documents(
            documents=docs, 
            embedding=embedder, 
            collection_name=COLLECTION_MACHINES,
            persist_directory=MACHINE_PERSIST_DIR # Ghi xuống ổ cứng
        )
        
    vector_retriever = machine_store.as_retriever(search_kwargs={"k": K_VECTOR})

    # 3. Xử lý Book Store (Dùng chung embedder)
    book_store = Chroma(
        persist_directory=PERSIST_DIR, 
        embedding_function=embedder, 
        collection_name=COLLECTION_BOOK
    )
    book_retriever = book_store.as_retriever(search_kwargs={"k": 8})

    return vector_retriever, book_retriever, embedder

def semantic_machine_search(
    user_query: str,
    vector_retriever,
    top_k: int,
    llm_main=None,
    target_format: str = "",
    target_scale: str = "",
) -> List[Document]:
    """
    Semantic search với Query Expansion V2 (3 lớp, 0 LLM call).
    Lớp 1: Original query
    Lớp 2: Rule-based templates
    Lớp 3: Alias-driven từ VectorDB — tự tìm tên máy chính thức
    """
    # Expansion v2 — luôn chạy, không cần LLM
    queries = expand_machine_queries_v2(
        user_query,
        vector_retriever=vector_retriever,
        llm_main=llm_main,
        max_queries=6,
    )

    semantic_docs = []
    for q in queries:
        try:
            semantic_docs.extend(vector_retriever.invoke(add_e5_query_prefix(q)))
        except Exception as e:
            print(f"[SemanticSearch] Query '{q[:40]}' failed: {e}")

    if not semantic_docs:
        return []

    rrf_k = 60.0
    scored = {}

    def get_doc_text(doc):
        return normalize_for_match(f"{doc.metadata.get('name', '')} {doc.page_content}")

    t_format = normalize_for_match(target_format)
    t_scale  = normalize_for_match(target_scale)

    sheet_keywords  = ["to roi", "sheet fed", "sheet", "to"]
    web_keywords    = ["cuon", "web fed", "roll"]
    auto_keywords   = ["tu dong", "automatic", "cong nghiep", "toc do cao"]
    manual_keywords = ["thu cong", "ban tu dong", "mini", "nho"]

    # RRF scoring
    for rank, doc in enumerate(semantic_docs, start=1):
        doc_id = (str(doc.metadata.get("row_index"))
                  if "row_index" in doc.metadata
                  else doc.metadata.get("product_url", doc.page_content[:80]))
        entry = scored.setdefault(doc_id, {"doc": doc, "score": 0.0})
        entry["score"] += 1.0 / (rrf_k + rank)

    # Heuristic boosting & penalty
    for doc_id, entry in scored.items():
        doc_text = get_doc_text(entry["doc"])

        if "to roi" in t_format or "sheet" in t_format:
            if any(k in doc_text for k in web_keywords) and not any(k in doc_text for k in sheet_keywords):
                entry["score"] *= 0.1
            if any(k in doc_text for k in sheet_keywords):
                entry["score"] += 5.0

        elif "cuon" in t_format or "web" in t_format:
            if any(k in doc_text for k in sheet_keywords) and not any(k in doc_text for k in web_keywords):
                entry["score"] *= 0.1
            if any(k in doc_text for k in web_keywords):
                entry["score"] += 5.0

        if "cong nghiep" in t_scale or "tu dong" in t_scale:
            if any(k in doc_text for k in auto_keywords):
                entry["score"] += 3.0
            if any(k in doc_text for k in manual_keywords):
                entry["score"] *= 0.5

    merged = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in merged[:top_k]]

def build_decision_llm(groq_api_key, fallback_llm=None):
    # Ưu tiên 2: Dùng Groq z 8B nếu có
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

# Few-shot examples tĩnh cho router — được chọn động dựa trên cosine similarity keyword
_FEWSHOT_EXAMPLES = [
    # find_machine
    ("find_machine",        "Cho mình xin thông số máy in offset 4 màu"),
    ("find_machine",        "Báo giá máy dán nhãn tự động"),
    ("find_machine",        "Tìm máy bế hộp giấy carton"),
    ("find_machine",        "tôi muốn tìm máy làm tô giấy đựng nước nóng"),
    ("find_machine",        "máy làm ly giấy dùng 1 lần"),
    # book_knowledge
    ("book_knowledge",      "Compare Flexo Printing with Digital Printing"),
    ("book_knowledge",      "CTP và ghi bản kẽm khác gì nhau?"),
    ("book_knowledge",      "when is choice flexo or digital?"),
    ("book_knowledge",      "UV coating là gì, dùng khi nào?"),
    # solution_consulting
    ("solution_consulting", "Mình muốn mở xưởng làm hộp giấy, bắt đầu từ đâu?"),
    ("solution_consulting", "Tư vấn dây chuyền in nhãn chai nước 5000 sp/ngày"),
    # troubleshooting — bao phủ máy in phun công nghiệp, UV, flexo
    ("troubleshooting",     "Máy in bị lệch màu, sọc ngang thì sửa thế nào?"),
    ("troubleshooting",     "Đầu phun máy in phun công nghiệp bị tắc"),
    ("troubleshooting",     "máy in phun carton của tôi mực ra không đều phải làm sao"),
    ("troubleshooting",     "máy in phun UV bị nghẹt đầu phun"),
    ("troubleshooting",     "máy flexo in bị lem mực và dot gain tăng mạnh"),
    # out_of_scope — CHỈ khi có tên model văn phòng rõ ràng
    ("out_of_scope",        "Máy in Canon A4 nhà mình bị kẹt giấy"),
    ("out_of_scope",        "Máy Epson L3150 bị báo lỗi"),
    ("out_of_scope",        "Canon 2900 của tôi in bị mờ"),
    # direct_chat
    ("direct_chat",         "Bạn ơi, VPRINT có hỗ trợ kỹ thuật không?"),
    ("direct_chat",         "Xin chào, tôi cần tư vấn"),
]

def get_relevant_history_for_router(history: list, max_turns: int = 4) -> list:
    """
    Chỉ giữ lại các turn có nội dung liên quan đến ngành in/máy móc.
    Loại bỏ chit-chat, chuyện ma, toán học... khỏi context của router.
    """
    industry_signals = [
        "may", "in", "bao bi", "carton", "offset", "flexo",
        "ctp", "khuon", "muc", "giay", "may be", "may dan",
        "xuong", "cong nghe", "thong so", "gia", "machine", "print",
        "label", "packaging", "laminating", "coating", "die"
    ]
    relevant = []
    last_user_kept = False
    for role, msg in reversed(history):
        if role == "user":
            q = normalize_for_match(str(msg))
            last_user_kept = any(k in q for k in industry_signals)
            if last_user_kept:
                relevant.append((role, msg))
        elif role == "assistant" and last_user_kept:
            # Chỉ giữ assistant turn nếu user turn ngay trước đó được giữ
            relevant.append((role, msg))
            last_user_kept = False  # reset sau khi đã ghép cặp

        if len(relevant) >= max_turns * 2:
            break

    return list(reversed(relevant))

def get_dynamic_fewshot(user_query: str, top_k: int = 3) -> str:
    """Chọn top_k ví dụ few-shot gần nhất với câu hỏi dựa trên word overlap đơn giản."""
    q_norm = normalize_for_match(user_query)
    q_tokens = set(q_norm.split())
    
    scored = []
    for label, example in _FEWSHOT_EXAMPLES:
        ex_tokens = set(normalize_for_match(example).split())
        overlap = len(q_tokens & ex_tokens)
        scored.append((overlap, label, example))
    
    # Lấy top_k ví dụ có overlap cao nhất, không trùng label quá nhiều
    scored.sort(key=lambda x: -x[0])
    seen_labels = {}
    selected = []
    for overlap, label, example in scored:
        if seen_labels.get(label, 0) < 2:  # Tối đa 2 ví dụ mỗi nhãn
            selected.append((label, example))
            seen_labels[label] = seen_labels.get(label, 0) + 1
        if len(selected) >= top_k:
            break
    
    if not selected:
        return ""
    
    lines = ["Ví dụ:"]
    for label, example in selected:
        lines.append(f'  "{example}" → {label}')
    return "\n".join(lines)

def llm_classify_intent(user_query, llm_main, history=None, example_store=None):
    dynamic_examples = get_dynamic_fewshot(user_query, top_k=3)
    
    recent_history = ""
    if history:
        recent_turns = [f"{role}: {str(msg)[:80]}" for role, msg in history[-3:]]
        recent_history = "Lịch sử:\n" + "\n".join(recent_turns) + "\n\n"

    system_prompt = f"""Phân loại câu hỏi vào 1 trong 6 nhãn. Ưu tiên ngữ nghĩa, không dựa từ khoá đơn lẻ.

troubleshooting    : báo lỗi, sự cố, hỏi cách sửa máy — KỂ CẢ máy in phun công nghiệp
out_of_scope       : chỉ khi có TÊN MODEL văn phòng rõ ràng (Canon, Epson, HP, Brother)
find_machine       : tìm mua, báo giá máy công nghiệp
book_knowledge     : lý thuyết, nguyên lý kỹ thuật
solution_consulting: tư vấn mở xưởng, thiết kế dây chuyền
direct_chat        : chào hỏi, câu hỏi ngoài ngành in

{dynamic_examples}

Trả về JSON: {{"intent": "<nhãn>", "reasoning": "<1 câu giải thích>"}}"""

    user_msg = f"{recent_history}Câu hỏi: {user_query[:250]}"

    try:
        # JSON mode — ổn định hơn tool calling với model 8B
        llm_json = llm_main.bind(
            response_format={"type": "json_object"}
        )
        res = llm_json.invoke([
            ("system", system_prompt),
            ("user", user_msg)
        ])

        parsed = json.loads(res.content)
        intent = parsed.get("intent", "").strip()
        reasoning = parsed.get("reasoning", "")

        print(f"🧐 Router: {reasoning} → {intent}")

        valid = {"find_machine", "book_knowledge", "solution_consulting",
                 "troubleshooting", "direct_chat", "out_of_scope"}
        return intent if intent in valid else "direct_chat"

    except Exception as e:
        print(f"Lỗi Router: {e}")
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
    
    selected_model = st.selectbox("Chọn Model", [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
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

# ==========================================
# 7. LOGIC XỬ LÝ CHÍNH
# ==========================================
# 1. TẠO CONTAINER HỨNG CHAT (Luôn nằm trên thanh chat)
# 1. TẠO CONTAINER HỨNG CHAT (Luôn nằm trên thanh chat)
chat_container = st.container()

with chat_container:
    # Hiển thị lịch sử chat vào trong container này
    for role, msg in st.session_state.history:
        avatar = "👤" if role == "user" else "img/logo_2.jpg" 
        with st.chat_message(role, avatar=avatar): 
            st.markdown(msg, unsafe_allow_html=True)
            
    if len(st.session_state.history) == 0: 
        st.markdown(WELCOME, unsafe_allow_html=True)

# ==========================================
# 2. INPUT NGƯỜI DÙNG (CUSTOM GEMINI UI - WHITE & VOICE)
# ==========================================

groq_api_key = get_safe_api_key("GROQ_API_KEY")

st.markdown(
    f"""
<style>
    :root {{
        --chat-font: "Source Sans Pro", sans-serif;
    }}
    .stApp {{
        background: #fcfcfc;
        font-family: var(--chat-font) !important;
    }}
    .main .block-container {{
        background: #fcfcfc;
        padding-bottom: 140px;
        position: relative;
        z-index: 0;
        font-family: var(--chat-font) !important;
    }}
    .main .block-container > * {{
        position: relative;
        z-index: 1;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #d7dee8 0%, #c6d0dc 100%) !important;
        border-right: 1px solid #b9c5d3;
    }}
    section[data-testid="stSidebar"] > div {{
        background: transparent !important;
    }}
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {{
        background: #fcfcfc !important;
        font-family: var(--chat-font) !important;
    }}
    .stChatMessage,
    .stChatMessage *,
    .stMarkdown,
    .stMarkdown *,
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] *,
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] *,
    label,
    input,
    textarea,
    button {{
        font-family: var(--chat-font) !important;
    }}
    div[data-testid="stChatInput"] {{
        display: none !important;
    }}
    @media (max-width: 768px) {{
        .main .block-container {{
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
            padding-bottom: 150px !important;
        }}
        .stChatMessage {{
            font-size: 14px !important;
        }}
        div[data-testid="stChatMessage"] {{
            padding: 0.15rem 0 0.85rem 0 !important;
            gap: 0.55rem !important;
        }}
        div[data-testid="stChatMessageContent"] {{
            width: 100% !important;
            max-width: 100% !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
            line-height: 1.55 !important;
            font-size: 14px !important;
        }}
        div[data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessageContent"] li,
        div[data-testid="stChatMessageContent"] span {{
            font-size: 14px !important;
            line-height: 1.55 !important;
        }}
        div[data-testid="stChatMessageContent"] img {{
            max-width: 100% !important;
            height: auto !important;
        }}
        div[data-testid="stChatMessageAvatar"] {{
            width: 2rem !important;
            height: 2rem !important;
            min-width: 2rem !important;
            min-height: 2rem !important;
        }}
        .stMarkdown table {{
            width: 100% !important;
            display: table !important;
            table-layout: fixed !important;
            border-collapse: collapse !important;
        }}
        .stMarkdown th,
        .stMarkdown td,
        div[data-testid="stChatMessageContent"] table th,
        div[data-testid="stChatMessageContent"] table td {{
            white-space: normal !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
            font-size: 12px !important;
            line-height: 1.4 !important;
            padding: 6px 8px !important;
            vertical-align: top !important;
        }}
        div[data-testid="stChatMessageContent"] table {{
            width: 100% !important;
            table-layout: fixed !important;
        }}
        .stCaption {{
            font-size: 12px !important;
            line-height: 1.45 !important;
        }}
    }}
    .main .block-container::before {{
        content: "";
        position: fixed;
        left: 19rem;
        top: 10rem;
        width: 70vw;
        height: 70vh;
        background-image: {logo_watermark_css};
        background-repeat: no-repeat;
        background-position: left center;
        background-size: contain;
        opacity: 0.055;
        pointer-events: none;
        z-index: 0;
        filter: grayscale(1);
    }}
</style>
""",
    unsafe_allow_html=True,
)

text_query = st.chat_input("Hỏi bất kỳ điều gì với VprintAI")

custom_chat_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: "Source Sans Pro", sans-serif;
        }}
        .chat-container {{
            background-color: #ffffff;
            border-radius: 36px;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            box-sizing: border-box;
            width: 100%;
            height: 72px;
            border: 1px solid #dfdfdf;
        }}
        .chat-input {{
            background: transparent;
            border: none;
            color: #111827;
            font-size: 17px;
            outline: none;
            flex-grow: 1;
            width: 100%;
        }}
        .chat-input::placeholder {{ color: #9ca3af; }}
        .right-section {{ display: flex; align-items: center; gap: 12px; }}
        .icon-btn {{ cursor: pointer; display: flex; align-items: center; justify-content: center; width: 38px; height: 38px; border-radius: 50%; transition: all 0.2s; }}
        .icon-btn svg {{ fill: #6b7280; width: 28px; height: 28px; transition: fill 0.2s; }}
        #mic-icon {{ width: 40px; height: 40px; }}
        #mic-icon svg {{ width: 30px; height: 30px; }}
        .icon-btn:hover {{ background-color: #f3f4f6; }}
        .icon-btn:hover svg {{ fill: #111827; }}
        .recording svg {{ fill: #ef4444 !important; animation: pulse 1.5s infinite; }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.15); }}
            100% {{ transform: scale(1); }}
        }}
        #send-icon {{ display: none; }}
        @media (max-width: 768px) {{
            .chat-container {{
                height: 60px;
                border-radius: 28px;
                padding: 10px 16px;
            }}
            .chat-input {{
                font-size: 16px;
            }}
            .right-section {{
                gap: 8px;
            }}
            .icon-btn {{
                width: 34px;
                height: 34px;
            }}
            .icon-btn svg {{
                width: 24px;
                height: 24px;
            }}
            #mic-icon {{
                width: 36px;
                height: 36px;
            }}
            #mic-icon svg {{
                width: 26px;
                height: 26px;
            }}
        }}
    </style>
</head>
<body>
    <div class="chat-container">
        <input type="text" id="gemini-input" class="chat-input" placeholder="Hỏi bất cứ điều gì với VprintAI" autocomplete="off">
        <div class="right-section">
            <div id="mic-icon" class="icon-btn" title="Nhấn để nói">
                <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
            </div>

            <div id="send-icon" class="icon-btn">
                <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
            </div>
        </div>
    </div>

    <script>
        const GROQ_API_KEY = "{groq_api_key}";

        const frame = window.frameElement;
        function layoutChatFrame() {{
            if (!frame) return;
            const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
            const sidebarWidth = sidebar ? sidebar.getBoundingClientRect().width : 0;
            const visualViewport = window.parent.visualViewport || window.visualViewport;
            const viewportWidth = visualViewport ? visualViewport.width : (window.parent.innerWidth || window.innerWidth);
            const isMobile = viewportWidth <= 768;
            const horizontalPadding = isMobile ? 12 : 32;
            const usableSidebarWidth = isMobile ? 0 : sidebarWidth;
            const maxWidth = isMobile ? Math.max(280, viewportWidth - horizontalPadding * 2) : 1040;
            const minWidth = isMobile ? 0 : 640;
            const availableWidth = Math.max(minWidth, Math.min(maxWidth, viewportWidth - usableSidebarWidth - horizontalPadding * 2));
            const centerOffset = usableSidebarWidth / 2;

            frame.style.position = 'fixed';
            frame.style.bottom = isMobile ? '72px' : '25px';
            frame.style.left = isMobile ? '50%' : `calc(50% + ${{centerOffset}}px)`;
            frame.style.transform = 'translateX(-50%)';
            frame.style.width = `${{availableWidth}}px`;
            frame.style.maxWidth = isMobile ? 'calc(100vw - 24px)' : `${{maxWidth}}px`;
            frame.style.minWidth = isMobile ? '280px' : '640px';
            frame.style.zIndex = '9999';
            frame.style.border = 'none';
            frame.style.height = isMobile ? '68px' : '85px';
            frame.style.background = 'transparent';
        }}
        layoutChatFrame();
        window.addEventListener('resize', layoutChatFrame);
        if (window.parent.visualViewport) {{
            window.parent.visualViewport.addEventListener('resize', layoutChatFrame);
        }}
        if (window.ResizeObserver) {{
            const observer = new ResizeObserver(() => layoutChatFrame());
            observer.observe(window.parent.document.body);
            const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) observer.observe(sidebar);
        }}

        const input = document.getElementById('gemini-input');
        const micIcon = document.getElementById('mic-icon');
        const sendIcon = document.getElementById('send-icon');

        input.addEventListener('input', () => {{
            if (input.value.trim().length > 0) {{
                micIcon.style.display = 'none';
                sendIcon.style.display = 'flex';
            }} else {{
                micIcon.style.display = 'flex';
                sendIcon.style.display = 'none';
            }}
        }});

        function submitToStreamlit() {{
            const text = input.value.trim();
            if (!text) return;
            const stTextArea = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (stTextArea) {{
                let nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(stTextArea, text);
                stTextArea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                stTextArea.dispatchEvent(new KeyboardEvent('keydown', {{
                    key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true
                }}));
                input.value = '';
                micIcon.style.display = 'flex';
                sendIcon.style.display = 'none';
            }}
        }}

        sendIcon.addEventListener('click', submitToStreamlit);
        input.addEventListener('keydown', (e) => {{
            if (e.key === 'Enter') submitToStreamlit();
        }});

        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        micIcon.addEventListener('click', async () => {{
            if (!isRecording) {{
                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();
                    isRecording = true;
                    micIcon.classList.add('recording');
                    input.placeholder = "Đang lắng nghe... (Nhấn lại mic để dừng)";

                    mediaRecorder.addEventListener("dataavailable", event => {{
                        audioChunks.push(event.data);
                    }});

                    mediaRecorder.addEventListener("stop", async () => {{
                        input.placeholder = "Đang dịch giọng nói sang văn bản...";
                        const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
                        audioChunks = [];
                        stream.getTracks().forEach(track => track.stop());

                        const formData = new FormData();
                        formData.append('file', audioBlob, 'audio.webm');
                        formData.append('model', 'whisper-large-v3-turbo');

                        try {{
                            const response = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {{
                                method: 'POST',
                                headers: {{ 'Authorization': `Bearer ${{GROQ_API_KEY}}` }},
                                body: formData
                            }});
                            const data = await response.json();

                            if (data.text) {{
                                input.value = data.text;
                                input.dispatchEvent(new Event('input'));
                            }}
                        }} catch (e) {{
                            console.error("Lỗi Whisper API:", e);
                            alert("Có lỗi xảy ra khi nhận diện giọng nói.");
                        }} finally {{
                            input.placeholder = "Ask Gemini 3";
                        }}
                    }});
                }} catch (err) {{
                    console.error("Lỗi truy cập Micro:", err);
                    alert("Vui lòng cấp quyền truy cập Micro trên trình duyệt để sử dụng tính năng này!");
                }}
            }} else {{
                mediaRecorder.stop();
                isRecording = false;
                micIcon.classList.remove('recording');
            }}
        }});
    </script>
</body>
</html>
"""
components.html(custom_chat_html, height=85)

user_query = None

if text_query:
    user_query = text_query

# === TIẾP TỤC LUỒNG RAG HIỆN TẠI ===
# === TIẾP TỤC LUỒNG RAG HIỆN TẠI ===
if user_query:
    # [THÊM DÒNG NÀY]: Ép mọi câu trả lời mới xuất hiện ở phía trên thanh chat
    with chat_container: 
        
        with st.chat_message("user", avatar="👤"): 
            st.markdown(user_query)
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

        # Setup LLM Main (Chuyển đổi linh hoạt giữa OpenAI và Groq)
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
        llm_decision = build_decision_llm(groq_api_key, llm_main)
        clean_history = get_relevant_history_for_router(st.session_state.history[:-1])
        intent_label = llm_classify_intent(user_query, llm_decision or llm_main, clean_history)
        decision = RouterDecision(
            intent=intent_label,
            use_rag=(intent_label in ["find_machine", "book_knowledge", "solution_consulting"]),
            reset_focus=(intent_label == "find_machine")
        )
        #decision = apply_router_guards(user_query, decision)

        if decision.reset_focus:
            st.session_state.last_docs = []
        if decision.intent in st.session_state.intent_counts:
            st.session_state.intent_counts[decision.intent] += 1

        try:
            raw_answer = ""
            language_instruction = get_response_language_instruction(user_query)
            
            # [THÊM NHÁNH XỬ LÝ NÀY VÀO ĐẦU TIÊN]
            if decision.intent == "out_of_scope":
                raw_answer = "VPRINT là đơn vị chuyên cung cấp các giải pháp và thiết bị in ấn **công nghiệp** (như máy in offset, in flexo bao bì, máy bế, dán thùng...). \n\nHiện tại, chúng tôi **không phân phối** các dòng máy in văn phòng, máy in gia đình cỡ nhỏ hoặc máy in cá nhân.\n\nNếu bạn có nhu cầu mở rộng quy mô sản xuất công nghiệp hoặc đầu tư xưởng in, hãy cho VPRINT biết nhé!"
                with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                    st.markdown(raw_answer)

            elif decision.intent == "book_knowledge":
                with thinking_indicator():
                    # NÂNG CẤP: Sử dụng RRF Retrieval để lấy kiến thức chính xác hơn,
                    # tránh LLM bịa thông tin khi context rỗng.
                    docs = retrieve_book_with_rrf(user_query, book_retriever, llm_main, top_k=10)
                    
                    # [SELF-RAG FILTERING] Lọc docs không liên quan chỉ 1 call
                    if docs and llm_main:
                        docs = filter_relevant_docs_batch(user_query, docs, llm_main)
                
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
                    
                    # [SELF-RAG FILTERING] Lọc book docs không liên quan
                    if book_docs and llm_main:
                        book_docs = filter_relevant_docs_batch(user_query, book_docs, llm_main)
                    
                    book_context = format_book_context(book_docs)
                    
                    # --- BƯỚC 3: LUÔN LUÔN TÌM MÁY DÙ THÔNG TIN MỜ MỊT HAY RÕ RÀNG ---
                    machine_docs = []
                    machine_context = ""
                    
                    # TÍNH TOÁN DYNAMIC TOP-K
                    # Nếu đã rõ ràng -> Chỉ chốt 1 máy tối ưu nhất cho mỗi công đoạn.
                    # Nếu còn mù mờ -> Đưa ra 2-3 máy để khách tham khảo các phân khúc.
                    dynamic_top_k = 3
                    
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
                        {language_instruction}
                        
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
                        {language_instruction}
                        
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
                lang_ui = ui_copy_for_language(user_query)
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
                        llm_main=None,      # Không cần LLM — v2 tự expand qua alias
                    )
                    filtered_docs = deduplicate_docs(filtered_docs)[:SEARCH_POOL_K]

                # TÍNH TOÁN DYNAMIC TOP-K CHO TÌM KIẾM TRỰC TIẾP
                q_norm = normalize_for_match(user_query)
                # Kiểm tra xem có số (thể hiện model/thông số) hoặc câu hỏi mô tả dài không
                has_specs = bool(re.search(r'\d+', q_norm)) 
                is_detailed = len(user_query.split()) > 7
                
                dynamic_top_k = 3

                # Gọi Rerank với Top-K động
                suggested_docs = rerank_machine_candidates(
                    user_query=search_query,
                    candidate_docs=filtered_docs,
                    llm_main=llm_main,
                    top_k=dynamic_top_k, 
                )

                # [SELF-RAG FILTERING] Ép Giám khảo Llama check lại lần cuối xem máy tìm được có đúng ý không
                if suggested_docs and llm_main:
                    suggested_docs = filter_relevant_docs_batch(search_query, suggested_docs, llm_main)

                if not suggested_docs:
                    raw_answer = lang_ui["no_result"]
                    with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                        st.markdown(raw_answer, unsafe_allow_html=True)
                else:
                    track_viewed_machines(suggested_docs)
                    
                    # 1. Gom toàn bộ dữ liệu máy thô (Bao gồm URL ảnh) thành Text
                    machine_context_raw = ""
                    for i, doc in enumerate(suggested_docs):
                        name = doc.metadata.get("name", "")
                        imgs = parse_images(doc.metadata.get("images", ""))
                        # Lấy tối đa 3 ảnh để gửi cho LLM
                        img_urls = imgs[:3] if imgs else []
                        specs = extract_labeled_value(doc.page_content, "Specifications")
                        
                        machine_context_raw += f"--- [MÁY TOP {i+1}] ---\n"
                        machine_context_raw += f"- Tên máy gốc (VN): {name}\n"
                        machine_context_raw += f"- URL Ảnh: {', '.join(img_urls)}\n"
                        machine_context_raw += f"- Mô tả & Tính năng: {doc.page_content}\n"
                        machine_context_raw += f"- Thông số kỹ thuật (JSON): {specs}\n\n"

                    # 2. Xây dựng Prompt giao toàn quyền cho gpt-oss-20b
                    language_instruction = get_response_language_instruction(user_query)
                    intro_instruction = get_machine_intro_prompt(user_query)
                    
                    sys_prompt = f"""Bạn là Chuyên gia Tư vấn Máy In Công Nghiệp cấp cao của VPRINT.
Khách hàng đang tìm kiếm máy móc. Dựa vào [DANH SÁCH MÁY ĐỀ XUẤT] bên dưới, hãy giới thiệu máy một cách chuyên nghiệp, hấp dẫn.

{language_instruction}

BẮT BUỘC: Nếu yêu cầu trả lời bằng tiếng Anh, bạn phải dịch MƯỢT MÀ toàn bộ Tên máy, Mô tả và cấu trúc lại bảng Thông số kỹ thuật sang tiếng Anh chuẩn ngành in.

QUY TẮC TRÌNH BÀY BẮT BUỘC:
1. {intro_instruction}
2. Đối với mỗi máy, trình bày theo đúng format sau:
   ### 🏆 Top [Số thứ tự]: **[Tên máy đã dịch]**
   
   <div style="margin-bottom: 10px;">
   [CHÈN ẢNH VÀO ĐÂY]
   </div>

   - **Description:** [Mô tả ngắn gọn khoảng 2-3 câu]
   - **Key Advantages:** [Gạch đầu dòng 1-2 ưu điểm nổi bật]
   - **Specifications:** [Chuyển đổi dữ liệu JSON thành một bảng Markdown gọn gàng. Dịch tên các cột/dòng sang ngôn ngữ đích].

3. CÁCH CHÈN ẢNH: Sử dụng thẻ HTML sau (TUYỆT ĐỐI không dùng Markdown `![]()`). Nếu có nhiều ảnh, chèn cạnh nhau:
   <img src="https://www.pinterest.com/anh5327/h%C3%ACnh-%E1%BA%A3nh/" style="height:140px;margin-right:8px;border-radius:8px;border:1px solid #ddd;object-fit:contain;">

[DANH SÁCH MÁY ĐỀ XUẤT]:
{machine_context_raw}
"""
                    messages = [("system", sys_prompt)] + get_optimized_history(turn_history, history_limit) + [("user", user_query)]
                    
                    # 3. Stream kết quả mượt mà ra giao diện
                    with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                        raw_answer = custom_write_stream(stream_response(messages, llm_main))
                        
                    st.session_state.last_docs = suggested_docs

            elif decision.intent == "troubleshooting":
                with thinking_indicator():
                    docs = retrieve_book_with_rrf(user_query, book_retriever, llm_main, top_k=8)
                    if docs and llm_main:
                        docs = filter_relevant_docs_batch(user_query, docs, llm_main)
                
                if not docs:
                    messages = build_general_knowledge_fallback_messages(user_query, turn_history, history_limit)
                else:
                    context = format_book_context(docs)
                    messages = build_book_rag_messages(user_query, context, turn_history, history_limit)
                with st.chat_message("assistant", avatar="img/logo_2.jpg"):
                    raw_answer = custom_write_stream(stream_response(messages, llm_main))

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
