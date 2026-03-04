


# NÂNG CẤP BỘ ĐỊNH TUYẾN


# NÂNG CẤP BỘ ĐỊNH TUYẾN & XỬ LÝ NGỮ CẢNH BÁN HÀNG - CÓ UI BUTTONS GỢI Ý

import streamlit as st
import time
import re
import torch
import json
import ast
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
#from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from chatbot_vprint_hybrid_local import (
    HuggingFaceEmbeddings,
    load_csv_docs,
    filter_docs_by_need_profile,
    build_specs_answer,
    pick_best_doc_for_query,
    format_context,
    build_rag_messages,
    build_direct_messages,
    apply_router_guards,
)

# ==========================================
# 1. CẤU HÌNH TRANG VÀ KHỞI TẠO SESSION STATE
# ==========================================
st.set_page_config(page_title="VPRINT AI", page_icon="🤖", layout="wide")

# FIX LỖI ATTRIBUTE ERROR: Khởi tạo biến môi trường ngay trên cùng
if "initialized" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_docs = []
    st.session_state.suggestion_clicked = None 
    st.session_state.current_suggestions = []  
    st.session_state.initialized = True

# ==========================================
# 2. CSS TÙY CHỈNH (NÚT BẤM CLAUDE/CHATGPT STYLE)
# ==========================================
st.markdown("""
<style>
/* CSS Tùy chỉnh cho các nút bấm gợi ý (st.button) */
div[data-testid="stVerticalBlock"] div.stButton > button {
    background-color: #f3f4f6; /* Màu nền xám nhạt */
    color: #374151; /* Màu chữ tối */
    border-radius: 20px; /* Bo góc tròn mạnh như Claude/GPT */
    border: 1px solid #e5e7eb; /* Viền mỏng */
    padding: 8px 16px;
    font-size: 14.5px;
    font-weight: 500;
    transition: all 0.2s ease-in-out; /* Hiệu ứng mượt mà */
    text-align: left; /* Căn trái text */
    width: 100%; /* Giãn đầy cột */
    box-shadow: 0 1px 2px rgba(0,0,0,0.05); /* Đổ bóng nhẹ */
}
div[data-testid="stVerticalBlock"] div.stButton > button:hover {
    background-color: #e5e7eb; /* Đậm hơn khi hover */
    border-color: #d1d5db;
    color: #000000;
    transform: translateY(-1px); /* Hơi nẩy lên khi đưa chuột vào */
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
div[data-testid="stVerticalBlock"] div.stButton > button:active {
    background-color: #d1d5db;
    transform: translateY(0);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CONFIG & DATA
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CSV_PATH = "vprint_products_clean.csv"
COLLECTION = "vprint_products_local"
EMBED_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

K_VECTOR = 10
K_BM25 = 10
K_FINAL = 5
MAX_HISTORY = 5

WELCOME = """
👋 Xin chào! Tôi là **VPRINT Sales AI**

Tôi có thể hỗ trợ:
- Tìm máy in / máy bế
- Tư vấn giải pháp bao bì
- Tra cứu thông số kỹ thuật

Hãy mô tả nhu cầu của bạn.
"""

ROUTING_SAMPLES = {
    "find_machine": [
        "Giới thiệu cho tôi các dòng máy bế hộp tự động.",
        "Xưởng mình mới mở, cần tìm máy in offset 4 màu loại nhỏ.",
        "Bên VPRINT có bán máy dán thùng carton sóng E không?",
        "Tư vấn giúp tôi hệ thống máy in flexo cuộn.",
        "Mình đang tìm mua máy cán màng nhiệt tốc độ cao.",
    ],
    "spec_query": [
        "Tốc độ tối đa của máy này là bao nhiêu tờ/giờ?",
        "Khổ giấy lớn nhất mà máy số 1 có thể chạy là bao nhiêu?",
        "Máy đó tiêu thụ điện năng như thế nào? Dùng điện 3 pha à?",
        "Kích thước tổng thể nặng nhẹ ra sao, tốn bao nhiêu diện tích?",
        "Thông số kỹ thuật chi tiết của máy này?"
    ],
    "normal_rag": [
        "Sự khác biệt lớn nhất giữa in offset và in flexo là gì?",
        "Làm thế nào để khắc phục lỗi lem mực khi in màng mạ kim loại?",
        "Quy trình tiêu chuẩn sản xuất hộp giấy mỹ phẩm gồm những bước nào?",
        "Tại sao khi bế hộp carton sóng hay bị nứt nếp gấp?",
        "Giải thích công nghệ in dữ liệu biến đổi VDP.",
    ],
    "direct_chat": [
        "Chào VPRINT, chúc một ngày tốt lành.",
        "Alo, có tư vấn viên ở đó không?",
        "Cảm ơn bạn đã hỗ trợ nhé, thông tin rất hữu ích.",
        "Tuyệt vời, để mình xem xét thêm rồi báo lại.",
        "Bạn là bot tự động hay người thật vậy?",
    ],
    "out_of_domain": [
        "Cách nấu món thịt kho tàu như thế nào?",
        "Thời tiết hôm nay ở Sài Gòn ra sao?",
        "Bạn có biết ai là tỷ phú giàu nhất thế giới không?",
        "Viết cho tôi một bài thơ tình.",
        "Giá vàng hôm nay bao nhiêu?",
        "Ai vô địch World Cup năm ngoái?"
    ]
}

# ==========================================
# 4. UTIL FUNCTIONS
# ==========================================
def get_groq_models():
        return [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

def parse_images(img_field):
    try:
        if isinstance(img_field, str):
            imgs = json.loads(img_field)
        else:
            imgs = img_field
    except:
        imgs = str(img_field).split(",")
    return [i.strip() for i in imgs if i][:4]

def get_field(text, label):
    pattern = rf"{label}:\s*(.*?)(?=\n\S+?:|$)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def deduplicate_docs(docs):
    seen = set()
    unique_docs = []
    for d in docs:
        url = d.metadata.get("product_url")
        if url not in seen:
            seen.add(url)
            unique_docs.append(d)
    return unique_docs

def format_specs_to_table(spec_text):
    if not spec_text:
        return ""
    try:
        try:
            data = json.loads(spec_text)
        except json.JSONDecodeError:
            data = ast.literal_eval(spec_text)

        if not isinstance(data, dict): return spec_text.replace(r'\n', '\n')

        models = list(data.keys())
        features = []
        for model_data in data.values():
            if isinstance(model_data, dict):
                for k in model_data.keys():
                    if k not in features: features.append(k)
        
        if not features: return spec_text.replace(r'\n', '\n')

        header = "| **Thông số / Model** | " + " | ".join([f"**{m}**" for m in models]) + " |"
        separator = "|---|" + "|".join(["---"] * len(models)) + "|"
        rows = []
        for feat in features:
            row_vals = [str(data[m].get(feat, "-")) for m in models]
            rows.append(f"| {feat} | " + " | ".join(row_vals) + " |")

        return "\n".join([header, separator] + rows)
    except Exception:
        return spec_text.replace(r'\n', '\n')

def parse_and_clean_suggestions(full_text):
    """Tách phần text trả lời chính và danh sách câu hỏi gợi ý."""
    markers = ["💡 **Có thể bạn quan tâm:**", "💡 Có thể bạn quan tâm:"]
    for marker in markers:
        if marker in full_text:
            parts = full_text.split(marker)
            main_text = parts[0].strip()
            sug_text = parts[1].strip()
            
            sugs = []
            for line in sug_text.split('\n'):
                # Xóa các ký tự thừa ở đầu dòng để làm nút cho đẹp
                clean_line = line.strip().lstrip("-*•1234567890. ")
                if clean_line: sugs.append(clean_line)
            return main_text, sugs
    return full_text, []

# ==========================================
# 5. LOAD SYSTEM & ROUTER
# ==========================================
@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    docs = load_csv_docs(CSV_PATH)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=COLLECTION)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": K_VECTOR})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = K_BM25
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
    return ensemble_retriever, docs, embeddings, device

@st.cache_resource
def init_vector_router(_embedder, device):
    intent_labels, corpus_texts = [], []
    for intent, questions in ROUTING_SAMPLES.items():
        for q in questions:
            corpus_texts.append(q)
            intent_labels.append(intent)
    corpus_embeddings = _embedder.embed_documents(corpus_texts)
    corpus_tensor = torch.tensor(corpus_embeddings, dtype=torch.float32).to(device)
    return corpus_tensor, intent_labels

class RouterDecision:
    def __init__(self):
        self.intent = "normal_rag"
        self.use_rag = True
        self.reset_focus = False
        self.score = 0.0

def fast_vector_route_query(user_query, embedder, corpus_tensor, intent_labels, device, threshold=0.45):
    query_emb = embedder.embed_query(user_query)
    query_tensor = torch.tensor([query_emb], dtype=torch.float32).to(device)
    cos_scores = torch.mm(query_tensor, corpus_tensor.transpose(0, 1))[0]
    best_score, best_idx = torch.max(cos_scores, dim=0)
    best_intent = intent_labels[best_idx.item()]
    
    decision = RouterDecision()
    decision.score = round(best_score.item() * 100, 2)
    
    if best_score.item() < threshold:
        decision.intent, decision.use_rag = "normal_rag", True
    else:
        decision.intent = best_intent
        decision.use_rag = False if best_intent in ["direct_chat", "out_of_domain"] else True
        if best_intent == "find_machine": decision.reset_focus = True
    return decision

ensemble_retriever, all_docs, embedder, system_device = load_system()
router_tensor, router_labels = init_vector_router(embedder, system_device)

# ==========================================
# 6. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Model Settings")
    selected_model = st.selectbox("Chọn Groq model", get_groq_models())
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    st.divider()
    if st.button("🔄 Reset Chat"):
        st.session_state.history = []
        st.session_state.last_docs = []
        st.session_state.current_suggestions = []
        st.rerun()

# ==========================================
# 7. RENDER HISTORY CHAT & GỢI Ý
# ==========================================
st.title("🤖 VPRINT Sales AI")

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg, unsafe_allow_html=True)

if len(st.session_state.history) == 0:
    st.info(WELCOME)

# Render các nút bấm UI Pills nếu có gợi ý
if len(st.session_state.history) > 0 and st.session_state.history[-1][0] == "assistant":
    if st.session_state.current_suggestions:
        st.markdown("<br>💡 <i>Gợi ý câu hỏi cho bạn:</i>", unsafe_allow_html=True)
        # Tạo hàng ngang các nút (Tối đa 3-4 nút cho đẹp)
        cols = st.columns(len(st.session_state.current_suggestions))
        for idx, sug in enumerate(st.session_state.current_suggestions):
            with cols[idx]:
                if st.button(sug, key=f"btn_sug_{idx}", use_container_width=True):
                    st.session_state.suggestion_clicked = sug
                    st.rerun()

# ==========================================
# 8. XỬ LÝ LOGIC NGƯỜI DÙNG NHẬP
# ==========================================
user_query = st.chat_input("Nhập câu hỏi hoặc chọn gợi ý phía trên...")

# Ưu tiên lấy câu hỏi từ nút bấm nếu người dùng click
if st.session_state.suggestion_clicked:
    user_query = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None 

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.history.append(("user", user_query))
    st.session_state.current_suggestions = [] # Xóa gợi ý cũ

    llm = ChatGroq(model_name=selected_model, temperature=temperature, groq_api_key=GROQ_API_KEY)
    start = time.perf_counter()

    decision = fast_vector_route_query(user_query, embedder, router_tensor, router_labels, system_device, threshold=0.5)
    decision = apply_router_guards(user_query, decision)

    if decision.reset_focus:
        st.session_state.last_docs = []

    try:
        raw_answer = "" # Biến hứng toàn bộ câu trả lời

        if decision.use_rag:
            with st.spinner("🔍 Đang tìm kiếm thông tin..."):
                fused_docs = ensemble_retriever.invoke(user_query)
                filtered_docs = filter_docs_by_need_profile(deduplicate_docs(fused_docs)[:K_FINAL], user_query)
            context = format_context(filtered_docs)

            # LUỒNG 1: TÌM MÁY
            # LUỒNG 1: TÌM MÁY
            if decision.intent == "find_machine":
                suggested_docs = filtered_docs[:3]
                if not suggested_docs:
                    raw_answer = "Xin lỗi, tôi không tìm thấy máy phù hợp."
                    with st.chat_message("assistant"): st.markdown(raw_answer)
                else:
                    raw_answer = "👋 Hệ thống đã tìm thấy các dòng máy phù hợp:\n\n"
                    for i, doc in enumerate(suggested_docs):
                        machine_name = doc.metadata.get("name", f"Sản phẩm {i+1}")
                        raw_answer += f"### 🏆 Top {i+1}: **{machine_name}**\n"
                        
                        content = doc.page_content
                        
                        # --- 1. MÔ TẢ ---
                        summary = get_field(content, "Summary")
                        description = get_field(content, "Description")
                        desc_text = description if description else summary
                        if desc_text:
                            raw_desc = desc_text.replace(r'\n', '\n')
                            desc_list = [d.strip() for d in raw_desc.split('\n') if d.strip()] if '\n' in raw_desc else [d.strip() + '.' for d in raw_desc.split('.') if len(d.strip()) > 1]
                            MAX_DESC_LINES = 3
                            formatted_desc_lines = []
                            for d in desc_list[:MAX_DESC_LINES]:
                                clean_line = d.lstrip('-*• ')
                                if clean_line:
                                    formatted_desc_lines.append(f"- {clean_line[0].upper() + clean_line[1:]}")
                            
                            formatted_desc = "\n".join(formatted_desc_lines)
                            if len(desc_list) > MAX_DESC_LINES: formatted_desc += "\n- *... (Xem thêm tại link chi tiết)*"
                            raw_answer += f"**Mô tả:**\n{formatted_desc}\n\n"

                        # --- 2. ĐẶC ĐIỂM NỔI BẬT (ĐÃ KHÔI PHỤC) ---
                        features = get_field(content, "Features")
                        if features:
                            raw_features = features.replace(r'\n', '\n')
                            feature_list = [f.strip() for f in raw_features.split('\n') if f.strip()] if '\n' in raw_features else [f.strip() for f in raw_features.split(',') if f.strip()]
                            MAX_FEAT_LINES = 3
                            formatted_feat_lines = []
                            for f in feature_list[:MAX_FEAT_LINES]:
                                clean_feat = f.lstrip('-*• ')
                                if clean_feat:
                                    formatted_feat_lines.append(f"- {clean_feat[0].upper() + clean_feat[1:]}")
                            
                            formatted_features = "\n".join(formatted_feat_lines)
                            if len(feature_list) > MAX_FEAT_LINES: formatted_features += "\n- *... (Xem chi tiết thông số bên dưới)*"
                            raw_answer += f"**Đặc điểm nổi bật:**\n{formatted_features}\n\n"
                        
                        # --- 3. LINK & HÌNH ẢNH ---
                        product_url = doc.metadata.get("product_url", "")
                        if product_url:
                            raw_answer += f"🔗 **Xem chi tiết:** [Nhấn vào đây]({product_url})\n\n"

                        images = parse_images(doc.metadata.get("images", ""))
                        if images:
                            img_html = "".join([f'<img src="{img}" style="height:140px;margin-right:8px;border-radius:8px;border:1px solid #ddd;object-fit:contain;">' for img in images])
                            raw_answer += f"<div>{img_html}</div><br>\n\n"
                        
                        raw_answer += "---\n\n"
                    
                    # Tự động gán thêm gợi ý tĩnh cho luồng này để UI có nút bấm
                    raw_answer += "💡 **Có thể bạn quan tâm:**\n- Thông số kỹ thuật chi tiết của máy\n- Yêu cầu báo giá dòng máy này\n- Tư vấn các dòng máy khác"
                    
                    with st.chat_message("assistant"):
                        st.markdown(raw_answer, unsafe_allow_html=True)
                    st.session_state.last_docs = suggested_docs

            # LUỒNG 2: THÔNG SỐ
            elif decision.intent == "spec_query":
                best_doc = pick_best_doc_for_query(user_query, st.session_state.last_docs + filtered_docs)
                raw_answer = build_specs_answer(user_query, best_doc)
                if best_doc:
                    table_specs = format_specs_to_table(get_field(best_doc.page_content, "Specifications"))
                    if table_specs: raw_answer += f"\n\n---\n**🔧 Bảng Thông Số Kỹ Thuật:**\n\n{table_specs}"
                
                # Gợi ý tĩnh cho luồng thông số
                raw_answer += "\n\n💡 **Có thể bạn quan tâm:**\n- Máy này bảo hành bao lâu?\n- Có dòng máy nào rẻ hơn không?\n- Xin báo giá chi tiết"
                
                with st.chat_message("assistant"):
                    st.markdown(raw_answer, unsafe_allow_html=True)

            # LUỒNG 3: NORMAL RAG (Dùng LLM sinh streaming)
            else:
                if not filtered_docs:
                    raw_answer = "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
                    with st.chat_message("assistant"): st.markdown(raw_answer)
                else:
                    messages = build_rag_messages(user_query, context, st.session_state.history, MAX_HISTORY)
                    with st.chat_message("assistant"):
                        raw_answer = st.write_stream((chunk.content for chunk in llm.stream(messages) if hasattr(chunk, "content")))

        # LUỒNG 4: CHAT TRỰC TIẾP / NGOÀI LỀ (Dùng LLM sinh streaming)
        else:
            messages = build_direct_messages(user_query, st.session_state.history, MAX_HISTORY)
            with st.chat_message("assistant"):
                raw_answer = st.write_stream((chunk.content for chunk in llm.stream(messages) if hasattr(chunk, "content")))

        # ==========================================
        # 9. TÁCH GỢI Ý, LƯU LỊCH SỬ VÀ RERUN APP
        # ==========================================
        if raw_answer:
            clean_answer, suggestions = parse_and_clean_suggestions(raw_answer)
            st.session_state.history.append(("assistant", clean_answer))
            st.session_state.current_suggestions = suggestions
            
            # Chỉ Rerun khi có gợi ý (để Streamlit xóa dòng text và render nút bấm)
            if suggestions:
                st.rerun()

    except Exception as e:
        error_msg = f"⚠️ Rất xin lỗi, hệ thống đang gặp chút sự cố kết nối. (Lỗi: {str(e)})"
        with st.chat_message("assistant"): st.markdown(error_msg)
        st.session_state.history.append(("assistant", error_msg))

    end = time.perf_counter()
    st.caption(f"⏱ Thời gian: **{round(end - start, 2)}s** | Router: `{decision.intent}`")