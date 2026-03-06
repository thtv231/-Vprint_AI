


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
#from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI # Import thêm OpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Import các hàm từ file xử lý local
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
    extract_labeled_value,
    get_optimized_history
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
        "normal_rag": 0, "direct_chat": 0, "out_of_domain": 0,
        "solution_consulting": 0
    }
    st.session_state.api_tokens = 0 
    st.session_state.total_session_tokens = 0

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
EMBED_MODEL_MACHINES = "bkai-foundation-models/vietnamese-bi-encoder"

COLLECTION_BOOK = "vprint_knowledge_base"
EMBED_MODEL_BOOK = "intfloat/multilingual-e5-large" 

K_VECTOR, K_BM25, K_FINAL, MAX_HISTORY = 10, 10, 5, 5

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

def get_safe_api_key(key_name="GROQ_API_KEY"):
    try:
        if key_name in st.secrets: return st.secrets[key_name]
    except Exception: pass 
    return os.getenv(key_name)

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

def generate_suggestions(llm_router, prompt):
    res = llm_router.invoke([("system", "Bạn là Kỹ sư Sales VPRINT tinh tế. Hãy tạo câu chuyển tiếp mượt mà."), ("user", prompt)])
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
# 5. LOAD SYSTEM & LLM ROUTER MỚI
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

class RouterDecision:
    def __init__(self, intent="normal_rag", use_rag=True, reset_focus=False):
        self.intent = intent
        self.use_rag = use_rag
        self.reset_focus = reset_focus

def llm_classify_intent(user_query, llm_router):
    system_prompt = """Bạn là Chuyên gia phân loại ý định cho VPRINT AI.
    Gán câu hỏi của khách vào 1 trong các nhãn sau. CHỈ trả về tên nhãn:
    - find_machine: Tìm mua, danh sách máy, chủng loại máy.
    - spec_query: Chi tiết kỹ thuật, công suất, kích thước, điện áp.
    - book_knowledge: Kỹ thuật in, lỗi in, quy trình (chế bản, sau in).
    - solution_consulting: Tư vấn trọn gói, dây chuyền, setup xưởng, tính toán sản lượng, làm sao để sản xuất...
    - normal_rag: Thông tin chung (bảo hành, địa chỉ, giao hàng).
    - direct_chat: Chào hỏi, khen ngợi, xã giao.
    - out_of_domain: Thời tiết, nấu ăn, chủ đề không liên quan.

    Câu hỏi: "{query}"
    """
    try:
        response = llm_router.invoke([("system", system_prompt.format(query=user_query))])
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            st.session_state.api_tokens += response.usage_metadata.get('total_tokens', 0)
        intent = response.content.strip().lower()
        valid_intents = ["find_machine", "spec_query", "book_knowledge", "solution_consulting", "normal_rag", "direct_chat", "out_of_domain"]
        for valid in valid_intents:
            if valid in intent: return valid
        return "normal_rag"
    except Exception: return "normal_rag"

machine_retriever, book_retriever, sys_embedder, sys_device = load_system()

# ==========================================
# 6. UI RENDER (SIDEBAR & MAIN)
# ==========================================
with st.sidebar:
    st.image("img/logo.png", width=200)
    st.header("⚙️ Model Settings")
    
    # Đã thêm GPT-4o vào danh sách lựa chọn
    selected_model = st.selectbox("Chọn Model", [
        "gpt-4o", 
        "openai/gpt-oss-20b", 
        "qwen/qwen3-32b", 
        "llama-3.1-8b-instant"
    ])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    
    st.subheader("📊 Phân tích (Intent)")
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
    with st.chat_message(role, avatar=avatar): st.markdown(msg, unsafe_allow_html=True)

if len(st.session_state.history) == 0: 
    st.markdown(WELCOME, unsafe_allow_html=True)

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

    # Lấy API Key động dựa trên Model
    groq_api_key = get_safe_api_key("GROQ_API_KEY")
    openai_api_key = get_safe_api_key("OPENAI_API_KEY")
    
    if not groq_api_key:
        st.error("⚠️ Không tìm thấy GROQ_API_KEY để chạy Router.")
        st.stop()
        
    # Setup LLM Router (Luôn dùng Groq Llama cho nhẹ, rẻ và cực nhanh)
    llm_router = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=groq_api_key)

    # Setup LLM Main (Chuyển đổi linh hoạt giữa OpenAI và Groq)
    if selected_model == "gpt-4o":
        if not openai_api_key:
            st.error("⚠️ Không tìm thấy OPENAI_API_KEY. Vui lòng thêm vào file .env!")
            st.stop()
        # Tham số stream_usage=True giúp LangChain đếm token của OpenAI chính xác
        llm_main = ChatOpenAI(model_name="gpt-4o", temperature=temperature, openai_api_key=openai_api_key, stream_usage=True)
    else:
        llm_main = ChatGroq(model_name=selected_model, temperature=temperature, groq_api_key=groq_api_key)
    
    start = time.perf_counter()

    intent_label = llm_classify_intent(user_query, llm_router)
    decision = RouterDecision(
        intent=intent_label,
        use_rag=(intent_label not in ["direct_chat", "out_of_domain"]),
        reset_focus=(intent_label == "find_machine")
    )
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
                    raw_answer = st.write_stream(stream_response(messages, llm_main))
            
            else:
                with st.spinner("🔍 Đang lục tìm kho máy móc VPRINT..."):
                    fused_docs = machine_retriever.invoke(user_query)
                    filtered_docs = deduplicate_docs(fused_docs)[:K_FINAL + 3] 
                context = format_context(filtered_docs)

                if decision.intent == "find_machine":
                    suggested_docs = filtered_docs[:3]
                    if not suggested_docs: 
                        raw_answer = "Xin lỗi, tôi không tìm thấy máy phù hợp."
                        with st.chat_message("assistant", avatar="img/logo.png"): 
                            st.markdown(raw_answer, unsafe_allow_html=True)
                    else:
                        raw_answer = "👋 Dựa trên yêu cầu, đây là các dòng máy có thông số phù hợp nhất:\n\n"
                        expert_context = ""
                        for i, doc in enumerate(suggested_docs):
                            raw_content = doc.page_content[:800] 
                            expert_context += f"MÁY SỐ {i+1}:\n{raw_content}\n\n"
                            
                        sys_prompt = f"""Bạn là Kỹ sư Trưởng VPRINT. Khách hàng là chuyên gia kỹ thuật đang tìm máy.
                        Nhiệm vụ: Đọc thông số thô của {len(suggested_docs)} máy dưới đây và tóm tắt thành 4 ý. 
                        TUYỆT ĐỐI KHÔNG BỊA ĐẶT. Ngắn gọn, súc tích.
                        
                        FORMAT BẮT BUỘC BẰNG MARKDOWN CỦA TỪNG MÁY:
                        - **Mô tả:** [1 câu tóm tắt chức năng]
                        - **Tốc độ/Hiệu suất:** [Thông số tốc độ]
                        - **Công nghệ:** [Các công nghệ đáng chú ý]
                        - **Điểm ưu việt:** [Đặc điểm nổi bật]
                        
                        LƯU Ý: Giữa mỗi máy BẮT BUỘC phải có ký hiệu `|||` để phân cách. Không lặp lại tên máy.
                        DỮ LIỆU THÔ:
                        {expert_context}
                        """
                        
                        with st.spinner("🧠 Đang phân tích thông số kỹ thuật chuyên sâu..."):
                            usp_response = llm_main.invoke([("system", sys_prompt), ("user", user_query)])
                            usp_text = usp_response.content
                            if hasattr(usp_response, 'usage_metadata') and usp_response.usage_metadata:
                                st.session_state.api_tokens += usp_response.usage_metadata.get('total_tokens', 0)
                                
                        usp_blocks = [block.strip() for block in usp_text.split('|||') if block.strip()]
                        
                        with st.chat_message("assistant", avatar="img/logo.png"):
                            display_container = st.empty()
                            
                            for i, doc in enumerate(suggested_docs):
                                name = doc.metadata.get("name", f"Sản phẩm {i+1}")
                                url = doc.metadata.get("product_url", "")
                                imgs = parse_images(doc.metadata.get("images", ""))
                                
                                raw_answer += f"### 🏆 Top {i+1}: **[{name}]({url})**\n"
                                if imgs: 
                                    img_html = "".join([f'<img src="{img}" style="height:140px;margin-right:8px;border-radius:8px;border:1px solid #ddd;object-fit:contain;">' for img in imgs[:3]])
                                    raw_answer += f"<div>{img_html}</div><br>\n\n"
                                
                                try: raw_answer += f"{usp_blocks[i]}\n\n"
                                except IndexError: raw_answer += "- Đang cập nhật tóm tắt thông số...\n\n"
                                    
                                raw_answer += "---\n\n"
                            
                            # CÂU DẪN DẮT + GỢI Ý ĐÃ ĐƯỢC TỐI ƯU
                            sug_prompt = f"""Khách vừa hỏi: '{user_query}'.
                            Nhiệm vụ: Viết 1 câu dẫn dắt thật tự nhiên tóm tắt 1 điểm đáng chú ý nhất của các máy vừa tìm được để nối tiếp câu chuyện, sau đó đưa ra 3 câu hỏi ngắn để khách đi sâu hơn.
                            TUYỆT ĐỐI KHÔNG dùng tiêu đề (như "Gợi ý:", "Có thể bạn quan tâm:"), KHÔNG dùng dấu ngoặc vuông.
                            
                            Format chuẩn BẮT BUỘC:
                            [1 câu dẫn dắt tự nhiên]
                            - [Câu hỏi 1]?
                            - [Câu hỏi 2]?
                            - [Câu hỏi 3]?
                            """
                            llm_sug = generate_suggestions(llm_router, sug_prompt) 
                            raw_answer += f"{llm_sug}"
                            
                            display_container.markdown(raw_answer, unsafe_allow_html=True)
                            
                        st.session_state.last_docs = suggested_docs

                elif decision.intent == "spec_query":
                    best_doc = st.session_state.last_docs[0] if st.session_state.last_docs else (filtered_docs[0] if filtered_docs else None)
                    
                    if best_doc:
                        sys_prompt = f"""Bạn là Kỹ sư Tư vấn Máy móc VPRINT.
                        Khách hàng hỏi: '{user_query}'
                        Hệ thống tìm thấy máy phù hợp nhất: **{best_doc.metadata.get('name')}**
                        Chi tiết thông số: {best_doc.page_content}
                        
                        Nhiệm vụ:
                        1. Trả lời TRỰC TIẾP câu hỏi. KHÔNG BỊA ĐẶT.
                        2. NẾU khách hỏi theo tên gọi khác mà trong thông số có đề cập đến chi tiết đó, hãy GIẢI THÍCH RÕ để khách hiểu.
                        3. Trình bày thông số kỹ thuật dưới dạng Bảng (Markdown Table) đẹp mắt.
                        """
                        messages = [("system", sys_prompt), ("user", user_query)]
                        
                        with st.chat_message("assistant", avatar="img/logo.png"):
                            raw_answer = st.write_stream(stream_response(messages, llm_main))
                            
                        # CÂU DẪN DẮT + GỢI Ý ĐÃ ĐƯỢC TỐI ƯU
                        sug_prompt = f"""Khách vừa xem thông số máy {best_doc.metadata.get('name')}. 
                        Nhiệm vụ: Viết 1 câu dẫn dắt thật tự nhiên tóm tắt 1 điểm đáng chú ý nhất của máy này để nối tiếp, sau đó đưa ra 3 câu hỏi ngắn (như hỏi giá, bảo hành, so sánh).
                        TUYỆT ĐỐI KHÔNG dùng tiêu đề, KHÔNG dùng dấu ngoặc vuông.
                        
                        Format chuẩn BẮT BUỘC:
                        [1 câu dẫn dắt tự nhiên]
                        - [Câu hỏi 1]?
                        - [Câu hỏi 2]?
                        - [Câu hỏi 3]?
                        """
                        llm_sug = generate_suggestions(llm_router, sug_prompt) 
                        raw_answer += f"\n\n{llm_sug}"
                        
                    else:
                        error_msg = f"Xin lỗi, hiện tại kho dữ liệu VPRINT không có dòng máy đáp ứng yêu cầu '{user_query}'."
                        with st.chat_message("assistant", avatar="img/logo.png"):
                            st.markdown(error_msg)
                            

                elif decision.intent == "solution_consulting":
                    suggested_docs = filtered_docs[:8] 
                    context = format_context(suggested_docs)
                    
                    # SYSTEM PROMPT ĐÃ CẬP NHẬT HOÀN CHỈNH CHO CÂU DẪN DẮT
                    sys_prompt = f"""Bạn là Kỹ sư Trưởng Tư vấn Giải pháp của VPRINT - chuyên gia cấp cao về hệ thống in ấn và sản xuất bao bì.

                    QUYỀN HẠN VÀ RÀNG BUỘC (ĐỌC KỸ BẮT BUỘC TUÂN THỦ):
                    1. **Về Tư vấn Kỹ thuật & Quy trình (ĐƯỢC TỰ DO):** Bạn ĐƯỢC PHÉP sử dụng toàn bộ tri thức chuyên sâu sẵn có về ngành in (thuật ngữ, kỹ thuật Offset/Flexo/Digital, công thức tính toán...) để vạch ra giải pháp tối ưu nhất.
                    2. **Về Đề xuất Máy móc (RÀNG BUỘC NGHIÊM NGẶT):** Khi ráp thiết bị vào quy trình, CHỈ ĐƯỢC PHÉP sử dụng các dòng máy có trong [DỮ LIỆU KHO MÁY VPRINT] bên dưới. TUYỆT ĐỐI KHÔNG tự bịa tên máy. NẾU THIẾU máy cho công đoạn nào, hãy ghi: "Hiện VPRINT chưa có sẵn dữ liệu máy cho công đoạn này...".

                    HÃY SUY LUẬN TỪNG BƯỚC:
                    1. **Phân tích Bài toán**: Dùng tri thức phân tích yêu cầu. Tự tính toán tốc độ/công suất yêu cầu.
                    2. **Thiết kế Quy trình & Công nghệ**: Vẽ ra các công đoạn. Giải thích ngắn gọn tại sao chọn công nghệ đó.
                    3. **Khớp nối Thiết bị VPRINT**: Nhặt máy móc phù hợp từ kho VPRINT để điền vào các công đoạn ở bước 2.
                    4. **Trình bày**: Lập 1 Bảng Tóm tắt Dây chuyền rõ ràng.
                    5. **Gợi ý Câu hỏi tiếp nối (QUAN TRỌNG)**: Ở phần dưới cùng, BẮT BUỘC viết 1 câu dẫn dắt thật tự nhiên tóm tắt 1 điểm mấu chốt của dây chuyền bạn vừa thiết kế, sau đó liệt kê 3 câu hỏi ngắn để khách đi sâu hơn. 
                    TUYỆT ĐỐI KHÔNG ghi tiêu đề (như "Gợi ý:", "Có thể bạn quan tâm:"), KHÔNG dùng dấu ngoặc vuông.
                    
                    Format chuẩn BẮT BUỘC:
                    [1 câu tóm tắt nối tiếp tự nhiên, VD: 'Với tổng công suất 15Kw và tốc độ 100 cái/phút của dây chuyền trên, bạn có muốn:']
                    - Xem báo giá chi tiết dây chuyền này?
                    - Tìm hiểu cách bố trí mặt bằng xưởng?
                    - So sánh với phương án khác?

                    [DỮ LIỆU KHO MÁY VPRINT]:
                    {context}
                    """
                    
                    opt_history = get_optimized_history(st.session_state.history, MAX_HISTORY)
                    messages = [("system", sys_prompt)] + opt_history + [("user", f"Yêu cầu khách hàng: '{user_query}'")]
                    
                    with st.chat_message("assistant", avatar="img/logo.png"):
                        raw_answer = st.write_stream(stream_response(messages, llm_main))
                
                else: # normal_rag
                    messages = build_rag_messages(user_query, context, st.session_state.history, MAX_HISTORY)
                    with st.chat_message("assistant", avatar="img/logo.png"): 
                        raw_answer = st.write_stream(stream_response(messages, llm_main))

        elif decision.intent == "out_of_domain":
            # FALLBACK PROMPT ĐÃ CẬP NHẬT CHO CÂU DẪN DẮT
            fallback_prompt = f"""Bạn là VPRINT AI - Chuyên gia tư vấn máy móc ngành in và bao bì.
            Khách hỏi: '{user_query}' (KHÔNG LIÊN QUAN chuyên môn).
            
            Nhiệm vụ:
            1. Từ chối lịch sự, vui vẻ (dưới 40 chữ).
            2. Ở dưới cùng, BẮT BUỘC viết 1 câu dẫn dắt chuyển hướng mượt mà quay lại việc tìm máy móc, kèm 3 câu hỏi tùy chọn.
            TUYỆT ĐỐI KHÔNG dùng tiêu đề, KHÔNG dùng ngoặc vuông.
            
            Format chuẩn:
            [1 câu dẫn dắt tự nhiên, VD: 'Thay vì chủ đề này, tôi có thể tư vấn cho bạn về các dòng máy công nghiệp. Bạn đang quan tâm đến:']
            - Máy in Offset hay Flexo?
            - Dây chuyền làm ly giấy?
            - Thiết bị bế hộp tự động?
            """
            messages = [("system", fallback_prompt)]
            with st.chat_message("assistant", avatar="img/logo.png"): 
                raw_answer = st.write_stream(stream_response(messages, llm_main))

        else: # direct_chat
            messages = build_direct_messages(user_query, st.session_state.history, MAX_HISTORY)
            with st.chat_message("assistant", avatar="img/logo.png"): 
                raw_answer = st.write_stream(stream_response(messages, llm_main))

        if raw_answer:
            st.session_state.history.append(("assistant", raw_answer))
            
            tokens_used = st.session_state.api_tokens
            if tokens_used == 0: tokens_used = (len(str(user_query)) + len(str(raw_answer))) // 3
            
            st.session_state.total_session_tokens += tokens_used
            process_time = time.perf_counter() - start
            
            is_logged = log_chat_to_gsheet(user_query, raw_answer, decision.intent, process_time, tokens_used, selected_model)

    except Exception as e:
        error_msg = f"⚠️ Hệ thống AI đang bận. Vui lòng thử lại! (Chi tiết: {str(e)})"
        with st.chat_message("assistant", avatar="img/logo.png"): st.markdown(error_msg)
        st.session_state.history.append(("assistant", error_msg))
        log_chat_to_gsheet(user_query, error_msg, decision.intent, 0, 0, selected_model)

    st.caption(f"⏱ Phản hồi: **{round(time.perf_counter() - start, 2)}s** | 🎯 Phân tích: `{decision.intent}` | 🪙 Token: **{st.session_state.api_tokens}**")