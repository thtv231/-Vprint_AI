

import streamlit as st
import time
import re
import torch
import json
import ast
import os
import gspread
import plotly.graph_objects as go
from google.oauth2.service_account import Credentials
from datetime import datetime
from dotenv import load_dotenv
import pytz
from langchain_chroma import Chroma
#from langchain_classic.retrievers import EnsembleRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Import các hàm từ file xử lý local (ĐÃ XÓA TẬN GỐC filter_docs_by_need_profile)
from chatbot_vprint_hybrid_local import (
    load_csv_docs,
    build_specs_answer,
    pick_best_doc_for_query,
    format_context,
    format_book_context,
    build_rag_messages,
    build_book_rag_messages,
    build_direct_messages,
    apply_router_guards,
    parse_images,
    extract_labeled_value
)

# ==========================================
# 1. CẤU HÌNH TRANG VÀ KHỞI TẠO SESSION STATE
# ==========================================
st.set_page_config(page_title="VPRINT AI", page_icon="img/logo.png", layout="wide")

if "initialized" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_docs = []
    st.session_state.suggestion_clicked = None 
    st.session_state.current_suggestions = []  
    st.session_state.initialized = True
    st.session_state.intent_counts = {
        "find_machine": 0, "spec_query": 0, "book_knowledge": 0, 
        "normal_rag": 0, "direct_chat": 0, "out_of_domain": 0
    }
    st.session_state.api_tokens = 0 
    st.session_state.total_session_tokens = 0 # Tổng token cả phiên

st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIG & LÀM GIÀU DỮ LIỆU ĐỊNH TUYẾN
# ==========================================
load_dotenv()

CSV_PATH = "vprint_products_clean.csv"
PERSIST_DIR = "vprint_agentic_db_local"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1GvR6uMGIT0J1MHJplPsepbWDALntCgLAT9ENrhqidEc/edit"

COLLECTION_MACHINES = "vprint_products_local"
EMBED_MODEL_MACHINES = "bkai-foundation-models/vietnamese-bi-encoder"

COLLECTION_BOOK = "vprint_knowledge_base"
EMBED_MODEL_BOOK = "intfloat/multilingual-e5-large" 

K_VECTOR, K_BM25, K_FINAL, MAX_HISTORY = 10, 10, 5, 5

WELCOME = """👋 Xin chào! Tôi là **Chuyên gia AI của VPRINT**
Tôi có thể hỗ trợ bạn:
🔍 **Tìm kiếm máy móc** (Máy in, máy bế, dán thùng...)
📐 **Tra cứu thông số** (Tốc độ, kích thước, công suất...)
📚 **Tư vấn kỹ thuật in** (Offset, Flexo, Lem mực, Chế bản...)
"""

ROUTING_SAMPLES = {
    "find_machine": [
        "Giới thiệu cho tôi các dòng máy bế.", 
        "Xưởng cần tìm mua máy in offset 4 màu.",
        "Bên bạn có bán máy in flexo không?",
        "Tư vấn cho tôi các dòng máy dán thùng carton.",
        "Tôi muốn đầu tư máy cán màng nhiệt tốc độ cao.",
        "Tìm máy in nhãn mác giá rẻ cho doanh nghiệp nhỏ.",
        "Có máy bế decal cuộn nào tốt không?"
    ],
    "spec_query": [
        "Tốc độ của máy này là bao nhiêu?", 
        "Khổ giấy lớn nhất máy có thể chạy?", 
        "Cho tôi xem thông số kỹ thuật chi tiết.",
        "Máy này dùng điện mấy pha? Công suất bao nhiêu?",
        "Kích thước và trọng lượng của máy in này?",
        "Máy có hệ thống tự động bù mực không?",
        "Độ phân giải bản in tối đa là bao nhiêu dpi?"
    ],
    "book_knowledge": [
        "Sự khác biệt lớn nhất giữa in offset và in flexo là gì?", 
        "Cách khắc phục lỗi lem mực trên màng nilon?",
        "Tôi muốn tìm hiểu về kỹ thuật in flexo.",
        "Nguyên lý hoạt động của hệ thống sấy UV là gì?",
        "Công nghệ in dữ liệu biến đổi VDP hoạt động ra sao?",
        "Tiêu chuẩn màu ISO 12647 trong in ấn?",
        "Giải thích các bước trong quy trình chế bản (prepress)."
    ],
    "normal_rag": [
        "Chính sách bảo hành và bảo trì máy ra sao?", 
        "Công ty VPRINT nằm ở đâu?",
        "Thời gian giao hàng và lắp đặt mất bao lâu?",
        "Bên bạn có hỗ trợ chuyển giao công nghệ không?",
        "Dịch vụ sau bán hàng của VPRINT thế nào?"
    ],
    "direct_chat": [
        "Chào bạn", 
        "Cảm ơn sự tư vấn của bạn", 
        "Bạn là người thật hay bot?",
        "Ok, để mình thảo luận lại với sếp",
        "Tạm biệt nhé",
        "Rất tuyệt vời"
    ],
    "out_of_domain": [
        "Thời tiết ở Sài Gòn hôm nay thế nào?", 
        "Hướng dẫn cách nấu món thịt kho tàu.",
        "Ai đang là tỷ phú giàu nhất thế giới?",
        "Giá vàng sjc hôm nay bao nhiêu?",
        "Viết cho tôi một bài thơ vui."
    ]
}

# ==========================================
# 3. UTIL FUNCTIONS
# ==========================================
def plot_intent_radar():
    intents = list(st.session_state.intent_counts.keys())
    counts = list(st.session_state.intent_counts.values())
    max_val = max(counts)
    chart_max = max_val + 1 if max_val > 0 else 2
    
    fig = go.Figure(data=go.Scatterpolar(
        r=counts + [counts[0]], theta=intents + [intents[0]], fill='toself',
        fillcolor='rgba(99, 102, 241, 0.4)', line=dict(color='#6366f1', width=2),
        marker=dict(symbol='circle', size=6, color='#4338ca'), mode='lines+markers'
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
    """Hàm khởi tạo kết nối Google Sheet và lưu vào Cache để dùng lại, giúp giảm thời gian chờ từ 2s xuống 0.001s"""
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
    """Hàm ghi log vào Sheet với Giờ Việt Nam chuẩn xác"""
    try:
        # Gọi lại client đã được lưu trong Cache
        client = get_gsheet_client()
        if not client:
            return False
            
        sheet = client.open_by_url(SPREADSHEET_URL).sheet1
        
        # Cấu hình chuẩn Múi giờ Việt Nam (UTC+7)
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        timestamp = datetime.now(vietnam_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        # Đóng gói dữ liệu bot trả về dạng JSON để không vỡ bảng Sheet
        bot_response_json = json.dumps({"response": bot_response}, ensure_ascii=False)
        
        # Tạo mảng dữ liệu 7 cột
        row_data = [timestamp, user_query, intent, bot_response_json, round(response_time, 2), tokens, model_name]
        
        # Đẩy dữ liệu lên hàng mới
        sheet.append_row(row_data)
        return True
    except Exception as e:
        print(f"Lỗi bắn log GSheet: {e}")
        return False

def get_safe_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets: return st.secrets["GROQ_API_KEY"]
    except Exception: pass 
    return os.getenv("GROQ_API_KEY")

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

def generate_suggestions(llm, prompt):
    res = llm.invoke([("system", "Bạn là Kỹ sư Sales VPRINT."), ("user", prompt)])
    if hasattr(res, 'usage_metadata') and res.usage_metadata:
        st.session_state.api_tokens += res.usage_metadata.get('total_tokens', 0)
    return res.content

def deduplicate_docs(docs):
    seen, unique_docs = set(), []
    for d in docs:
        url = d.metadata.get("product_url", d.page_content[:20])
        if url not in seen:
            seen.add(url); unique_docs.append(d)
    return unique_docs

def parse_and_clean_suggestions(full_text):
    for marker in ["💡 **Có thể bạn quan tâm:**", "💡 Có thể bạn quan tâm:", "💡 **Có thể bạn quan tâm**"]:
        if marker in full_text:
            main_text, sug_text = full_text.split(marker, 1)
            sugs = [line.strip().lstrip("-*•1234567890. ") for line in sug_text.split('\n') if line.strip().lstrip("-*•1234567890. ")]
            return main_text.strip(), sugs
    return full_text, []

def format_specs_to_table(spec_text):
    if not spec_text: return ""
    try:
        data = json.loads(spec_text)
        if not isinstance(data, dict): return spec_text.replace(r'\n', '\n')
        models = list(data.keys())
        features = list({k for md in data.values() if isinstance(md, dict) for k in md.keys()})
        if not features: return spec_text.replace(r'\n', '\n')
        header = "| **Thông số** | " + " | ".join([f"**{m}**" for m in models]) + " |"
        separator = "|---|" + "|".join(["---"] * len(models)) + "|"
        rows = [f"| {feat} | " + " | ".join([str(data[m].get(feat, "-")) for m in models]) + " |" for feat in features]
        return "\n".join([header, separator] + rows)
    except: return spec_text.replace(r'\n', '\n')

# ==========================================
# 5. LOAD SYSTEM & ROUTER
# ==========================================
@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    machine_embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_MACHINES, model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": True})
    docs = load_csv_docs(CSV_PATH)
    machine_store = Chroma.from_documents(documents=docs, embedding=machine_embedder, collection_name=COLLECTION_MACHINES)
    vector_retriever = machine_store.as_retriever(search_kwargs={"k": K_VECTOR})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = K_BM25
    machine_ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

    book_embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_BOOK, model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": True})
    book_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=book_embedder, collection_name=COLLECTION_BOOK)
    book_retriever = book_store.as_retriever(search_kwargs={"k": 4})

    return machine_ensemble, book_retriever, machine_embedder, device

@st.cache_resource
def init_vector_router(_embedder, device):
    intent_labels, corpus_texts = [], []
    for intent, questions in ROUTING_SAMPLES.items():
        for q in questions:
            corpus_texts.append(q)
            intent_labels.append(intent)
    corpus_tensor = torch.tensor(_embedder.embed_documents(corpus_texts), dtype=torch.float32).to(device)
    return corpus_tensor, intent_labels

class RouterDecision:
    def __init__(self):
        self.intent = "normal_rag"
        self.use_rag = True
        self.reset_focus = False

def fast_vector_route_query(user_query, embedder, corpus_tensor, intent_labels, device, threshold=0.45):
    query_tensor = torch.tensor([embedder.embed_query(user_query)], dtype=torch.float32).to(device)
    cos_scores = torch.mm(query_tensor, corpus_tensor.transpose(0, 1))[0]
    best_score, best_idx = torch.max(cos_scores, dim=0)
    best_intent = intent_labels[best_idx.item()]
    
    decision = RouterDecision()
    if best_score.item() < threshold:
        decision.intent, decision.use_rag = "normal_rag", True
    else:
        decision.intent, decision.use_rag = best_intent, best_intent not in ["direct_chat", "out_of_domain"]
        if best_intent == "find_machine": decision.reset_focus = True
    return decision

machine_retriever, book_retriever, sys_embedder, sys_device = load_system()
router_tensor, router_labels = init_vector_router(sys_embedder, sys_device)

# ==========================================
# 6. UI RENDER (SIDEBAR & MAIN)
# ==========================================
with st.sidebar:
    st.image("img/logo.png", width=200)
    st.header("⚙️ Model Settings")
    selected_model = st.selectbox("Chọn Groq model", ["openai/gpt-oss-20b", "qwen/qwen3-32b", "llama-3.1-8b-instant"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    
    st.subheader("📊 Phân tích Hành vi (Intent)")
    st.plotly_chart(plot_intent_radar(), use_container_width=True)
    
    st.metric(label="🪙 Tổng Token đã dùng", value=f"{st.session_state.total_session_tokens:,}")
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.history, st.session_state.last_docs, st.session_state.current_suggestions = [], [], []
            st.session_state.intent_counts = {k: 0 for k in st.session_state.intent_counts}
            st.session_state.api_tokens = 0
            st.session_state.total_session_tokens = 0
            st.rerun()
    with col2:
        chat_data = generate_chat_export()
        st.download_button(label="📥 Tải Lịch sử", data=chat_data, file_name=f"VPRINT_Chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain", use_container_width=True)

st.title("🤖 VPRINT Sales AI")

for role, msg in st.session_state.history:
    avatar = "👤" if role == "user" else "img/logo.png" 
    with st.chat_message(role, avatar=avatar): 
        st.markdown(msg, unsafe_allow_html=True)

if len(st.session_state.history) == 0: st.info(WELCOME)

if len(st.session_state.history) > 0 and st.session_state.history[-1][0] == "assistant" and st.session_state.current_suggestions:
    st.markdown("<br>💡 <i>Gợi ý hành động:</i>", unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.current_suggestions))
    for idx, sug in enumerate(st.session_state.current_suggestions):
        with cols[idx]:
            if st.button(sug, key=f"btn_sug_{idx}", use_container_width=True):
                st.session_state.suggestion_clicked = sug
                st.rerun()

# ==========================================
# 7. LOGIC XỬ LÝ CHÍNH
# ==========================================
user_query = st.chat_input("Nhập câu hỏi hoặc chọn gợi ý...")
if st.session_state.suggestion_clicked:
    user_query = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None 

if user_query:
    with st.chat_message("user", avatar="👤"): st.markdown(user_query)
    st.session_state.history.append(("user", user_query))
    st.session_state.current_suggestions = []
    st.session_state.api_tokens = 0 

    api_key = get_safe_api_key()
    if not api_key:
        st.error("⚠️ Không tìm thấy API Key.")
        st.stop()
        
    llm = ChatGroq(model_name=selected_model, temperature=temperature, groq_api_key=api_key)
    start = time.perf_counter()

    decision = fast_vector_route_query(user_query, sys_embedder, router_tensor, router_labels, sys_device)
    decision = apply_router_guards(user_query, decision)
    
    if decision.reset_focus: st.session_state.last_docs = []
    if decision.intent in st.session_state.intent_counts: st.session_state.intent_counts[decision.intent] += 1

    try:
        raw_answer = ""
        if decision.use_rag:
            if decision.intent == "book_knowledge":
                with st.spinner("📚 Đang tra cứu cẩm nang ngành in..."): 
                    docs = book_retriever.invoke(user_query)
                context = format_book_context(docs)
                messages = build_book_rag_messages(user_query, context, st.session_state.history, MAX_HISTORY)
                with st.chat_message("assistant", avatar="img/logo.png"):
                    raw_answer = st.write_stream(stream_response(messages, llm))
            
            else:
                with st.spinner("🔍 Đang lục tìm kho máy móc VPRINT..."):
                    fused_docs = machine_retriever.invoke(user_query)
                    # SỬA LỖI: Chỉ dùng deduplicate_docs, KHÔNG gọi hàm filter nữa
                    filtered_docs = deduplicate_docs(fused_docs)[:K_FINAL]
                context = format_context(filtered_docs)

                if decision.intent == "find_machine":
                    suggested_docs = filtered_docs[:3]
                    if not suggested_docs: raw_answer = "Xin lỗi, tôi không tìm thấy máy phù hợp."
                    else:
                        raw_answer = "👋 Hệ thống đã tìm thấy các dòng máy phù hợp:\n\n"
                        m_names = []
                        for i, doc in enumerate(suggested_docs):
                            name = doc.metadata.get("name", f"Sản phẩm {i+1}")
                            m_names.append(name)
                            raw_answer += f"### 🏆 Top {i+1}: **{name}**\n"
                            
                            desc_text = extract_labeled_value(doc.page_content, "Description") or extract_labeled_value(doc.page_content, "Summary")
                            if desc_text:
                                desc_lines = [f"- {d.strip().lstrip('-*• ').capitalize()}" for d in desc_text.replace(r'\n', '\n').split('\n') if d.strip().lstrip('-*• ')][:3]
                                raw_answer += f"**Mô tả:**\n{chr(10).join(desc_lines)}\n\n"

                            url = doc.metadata.get("product_url", "")
                            if url: raw_answer += f"🔗 **Xem chi tiết:** [Nhấn vào đây]({url})\n\n"

                            imgs = parse_images(doc.metadata.get("images", ""))
                            if imgs: 
                                img_html = "".join([f'<img src="{img}" style="height:140px;margin-right:8px;border-radius:8px;border:1px solid #ddd;object-fit:contain;">' for img in imgs])
                                raw_answer += f"<div>{img_html}</div><br>\n\n"
                            raw_answer += "---\n\n"
                        
                        sug_prompt = f"Khách vừa hỏi: '{user_query}'. Bạn đề xuất: {m_names}. Hãy viết 3 câu gợi ý ngắn gọn (dưới 10 chữ/câu) để khách bấm vào hỏi tiếp. Bắt buộc Format:\n💡 **Có thể bạn quan tâm:**\n- [Gợi ý 1]\n- [Gợi ý 2]\n- [Gợi ý 3]"
                        llm_sug = generate_suggestions(llm, sug_prompt)
                        raw_answer += f"{llm_sug}"
                        st.session_state.last_docs = suggested_docs
                    
                    with st.chat_message("assistant", avatar="img/logo.png"): st.markdown(raw_answer, unsafe_allow_html=True)

                elif decision.intent == "spec_query":
                    best_doc = st.session_state.last_docs[0] if st.session_state.last_docs else (filtered_docs[0] if filtered_docs else None)
                    
                    if best_doc:
                        verify_prompt = f"Khách hỏi: '{user_query}'. Máy đang xét: '{best_doc.metadata.get('name')}'. Máy này có đáp ứng nhu cầu không? Trả lời YES hoặc NO."
                        verify_res = llm.invoke([("system", "Bạn là bộ kiểm duyệt."), ("user", verify_prompt)])
                        if hasattr(verify_res, 'usage_metadata') and verify_res.usage_metadata:
                            st.session_state.api_tokens += verify_res.usage_metadata.get('total_tokens', 0)
                            
                        if "NO" in verify_res.content.strip().upper():
                            raw_answer = f"Xin lỗi, hiện tại kho dữ liệu VPRINT không có dòng máy đáp ứng chính xác yêu cầu '{user_query}'.\n\nTuy nhiên, dòng máy gần nhất là **{best_doc.metadata.get('name')}**. Bạn có muốn tư vấn dòng này không?"
                        else:
                            raw_answer = build_specs_answer(user_query, best_doc)
                            table = format_specs_to_table(extract_labeled_value(best_doc.page_content, "Specifications"))
                            if table: raw_answer += f"\n\n---\n**🔧 Bảng Thông Số:**\n\n{table}"
                            
                        sug_prompt = f"Khách xem máy {best_doc.metadata.get('name')}. Đưa ra 3 gợi ý ngắn (VD: Báo giá, bảo hành). Format:\n💡 **Có thể bạn quan tâm:**\n- [Gợi ý 1]\n- [Gợi ý 2]\n- [Gợi ý 3]"
                        llm_sug = generate_suggestions(llm, sug_prompt)
                        raw_answer += f"\n\n{llm_sug}"
                        
                    with st.chat_message("assistant", avatar="img/logo.png"): st.markdown(raw_answer, unsafe_allow_html=True)

                else:
                    messages = build_rag_messages(user_query, context, st.session_state.history, MAX_HISTORY)
                    with st.chat_message("assistant", avatar="img/logo.png"): 
                        raw_answer = st.write_stream(stream_response(messages, llm))

        else:
            messages = build_direct_messages(user_query, st.session_state.history, MAX_HISTORY)
            with st.chat_message("assistant", avatar="img/logo.png"): 
                raw_answer = st.write_stream(stream_response(messages, llm))

        if raw_answer:
            clean_answer, suggestions = parse_and_clean_suggestions(raw_answer)
            st.session_state.history.append(("assistant", clean_answer))
            st.session_state.current_suggestions = suggestions
            
            tokens_used = st.session_state.api_tokens
            if tokens_used == 0: tokens_used = (len(str(user_query)) + len(str(raw_answer))) // 3
            
            st.session_state.total_session_tokens += tokens_used
            process_time = time.perf_counter() - start
            
            is_logged = log_chat_to_gsheet(user_query, raw_answer, decision.intent, process_time, tokens_used, selected_model)
            if is_logged:
                st.toast('Đã đồng bộ dữ liệu chat!', icon='✅')
            
            if suggestions: st.rerun()

    except Exception as e:
        error_msg = f"⚠️ Hệ thống AI đang bận hoặc gián đoạn kết nối. Vui lòng thử lại! (Chi tiết: {str(e)})"
        with st.chat_message("assistant", avatar="img/logo.png"): st.markdown(error_msg)
        st.session_state.history.append(("assistant", error_msg))
        log_chat_to_gsheet(user_query, error_msg, decision.intent, 0, 0, selected_model)

    st.caption(f"⏱ Phản hồi: **{round(time.perf_counter() - start, 2)}s** | 🎯 Phân tích: `{decision.intent}` | 🪙 Token lượt này: **{st.session_state.api_tokens}**")
