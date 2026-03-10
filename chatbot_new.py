import streamlit as st
import time
import re
import torch
import json
import os
import gspread
import plotly.graph_objects as go
import uuid
from google.oauth2.service_account import Credentials
from datetime import datetime
from dotenv import load_dotenv
import pytz

# --- LANGCHAIN & LLM IMPORTS ---
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate







# --- LANGGRAPH IMPORTS ---
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Import các hàm từ file xử lý local của bạn
from chatbot_vprint_hybrid_local import (
    load_csv_docs,
    format_context,
    format_book_context,
    parse_images
)

# ==========================================
# 1. CẤU HÌNH TRANG VÀ SESSION STATE
# ==========================================
st.set_page_config(page_title="VPRINT AI Expert", page_icon="img/logo.png", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver() 

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.total_session_tokens = 0

# ==========================================
# 2. CONFIG & LOAD SYSTEM (SỬ DỤNG RETRIEVER CŨ CHO TỐC ĐỘ)
# ==========================================
load_dotenv()
CSV_PATH = "vprint_products_clean.csv"
PERSIST_DIR = "vprint_agentic_db_local"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1GvR6uMGIT0J1MHJplPsepbWDALntCgLAT9ENrhqidEc/edit"

COLLECTION_MACHINES = "vprint_products_local"
EMBED_MODEL_MACHINES = "bkai-foundation-models/vietnamese-bi-encoder"
COLLECTION_BOOK = "vprint_knowledge_base"
EMBED_MODEL_BOOK = "intfloat/multilingual-e5-large" 

@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # KHO MÁY MÓC (Sử dụng Bi-encoder cũ cho tốc độ phản hồi ấn tượng)
    machine_embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_MACHINES, model_kwargs={"device": device})
    docs = load_csv_docs(CSV_PATH)
    machine_store = Chroma.from_documents(documents=docs, embedding=machine_embedder, collection_name=COLLECTION_MACHINES)
    
    # Tăng k=15 để quét rộng hơn, không bỏ sót máy
    vector_retriever = machine_store.as_retriever(search_kwargs={"k": 15})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 15
    machine_ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

    # CẨM NANG NGÀNH IN
    book_embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_BOOK, model_kwargs={"device": device})
    book_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=book_embedder, collection_name=COLLECTION_BOOK)
    book_retriever = book_store.as_retriever(search_kwargs={"k": 4})

    return machine_ensemble, book_retriever

machine_retriever, book_retriever = load_system()

# ==========================================
# 3. TOOLS VỚI BỘ LỌC PYTHON (GUARDRAILS)
# ==========================================
@tool
def search_vprint_machines(query: str) -> str:
    """
    Sử dụng công cụ này để tìm máy móc, thông số, và tư vấn thiết bị VPRINT.
    MẸO TÌM KIẾM QUAN TRỌNG CHO LLM: 
    - Nếu khách hỏi kích thước quá cụ thể (VD: 72*102), hãy truyền vào tham số query các từ khóa RỘNG HƠN về chủng loại (VD: "máy ghi bản kẽm CTP offset khổ lớn") để quét được nhiều máy, sau đó bạn tự đọc thông số để lọc lại.
    """
    raw_docs = machine_retriever.invoke(query)
    
    # --- BỘ LỌC THÉP (PYTHON GUARDRAILS) ---
    final_docs = []
    q_low = query.lower()
    for doc in raw_docs:
        text = (doc.page_content + " " + doc.metadata.get('name', '')).lower()
        
        # Chặn nhầm lẫn Offset và Flexo
        if "offset" in q_low and "flexo" in text and "offset" not in text: continue
        if "flexo" in q_low and "offset" in text and "flexo" not in text: continue
        
        # Chặn nhầm lẫn Hộp và Nhãn
        if ("hộp" in q_low or "carton" in q_low) and "nhãn" in text and "hộp" not in text: continue
        
        final_docs.append(doc)

    # Lọc trùng lặp và lấy TOP 6 để Agent có nhiều không gian lựa chọn hơn
    seen, unique_docs = set(), []
    for d in final_docs:
        url = d.metadata.get("product_url", d.page_content[:20])
        if url not in seen:
            seen.add(url); unique_docs.append(d)
            if len(unique_docs) == 6: break

    if not unique_docs:
        return "THÔNG BÁO TỪ HỆ THỐNG: Kho dữ liệu VPRINT hiện không có máy khớp chính xác với yêu cầu này. Yêu cầu Agent từ chối khéo léo và không tự bịa ra máy."

    return format_context(unique_docs)

@tool
def consult_printing_handbook(query: str) -> str:
    """Tra cứu kiến thức kỹ thuật in và giải pháp khắc phục lỗi sản xuất."""
    docs = book_retriever.invoke(query)
    return format_book_context(docs)

tools = [search_vprint_machines, consult_printing_handbook]

# ==========================================
# 4. KIẾN TRÚC LANGGRAPH (RE-ACT AGENT)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def route_logic(state: AgentState):
    last_msg = state["messages"][-1]
    return "tools" if last_msg.tool_calls else END

def create_agent_graph(llm):
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: AgentState):
        sys_prompt = """Bạn là Kỹ sư Trưởng VPRINT. Bạn tư vấn dựa trên sự thật khách quan.

        LUẬT BẮT BUỘC (GUARDRAILS KHẮT KHE):
        1. KHÔNG tự báo giá tiền. Nếu khách hỏi giá, hãy bảo họ liên hệ Hotline VPRINT.
        2. KHÔNG bịa máy cho dây chuyền. Chỉ tư vấn máy có thật trong dữ liệu Tool trả về.
        3. KIỂM TRA KHỔ MÁY: Nếu khách tìm khổ lớn (VD: 72x102) mà máy Tool trả về chỉ có khổ nhỏ, phải nói rõ là không đáp ứng được (hoặc tìm tiếp).
        4. CÔNG NGHỆ: Khách hỏi Offset thì không đưa Flexo. Khách làm Hộp thì không đưa máy làm Nhãn.
        5. TOÀN VẸN DỮ LIỆU SỐ (QUAN TRỌNG): TUYỆT ĐỐI giữ nguyên các dấu câu, dấu gạch ngang (-), dấu ngã (~) biểu thị khoảng cách thông số. Không được nối liền số làm sai lệch thông số (VD: sai '1.02.4', đúng '1.0 - 2.4').
        6. TRÌNH BÀY THÔNG SỐ: Khi khách yêu cầu xem thông số kỹ thuật, BẮT BUỘC trình bày các chỉ số dưới dạng Bảng (Markdown Table) cho dễ đọc. Dưới bảng phải có mục "💡 Tại sao phù hợp:".
        7. KẾT THÚC: PHẢI kết thúc bằng 1 câu hỏi gợi mở để khai thác thêm thông số kỹ thuật khách chưa nêu.
        """
        messages = [SystemMessage(content=sys_prompt)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}
        
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", route_logic, {"tools": "tools", END: END}) # Có Map để vẽ mũi tên
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=st.session_state.memory)

# ==========================================
# 5. UI & THỰC THI (OPTIMIZED)
# ==========================================
@st.cache_resource
def get_gsheet_worksheet():
    try:
        creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), 
                                                     scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        return gspread.authorize(creds).open_by_url(SPREADSHEET_URL).sheet1
    except: return None

with st.sidebar:
    st.image("img/logo.png", width=200)
    # Khuyến nghị dùng GPT-4o cho sự ổn định tối đa
    selected_model = st.selectbox("Chọn Model", ["gpt-4o", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    
    with st.expander("📊 Sơ đồ hoạt động"):
        try:
            temp_llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
            st.image(create_agent_graph(temp_llm).get_graph().draw_mermaid_png())
        except Exception: pass

    if st.button("🔄 Reset Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

st.title("🤖 VPRINT Agentic Sales AI")

for role, msg in st.session_state.history:
    with st.chat_message(role, avatar="👤" if role == "user" else "img/logo.png"): st.markdown(msg)

user_query = st.chat_input("Nhập yêu cầu...")
if user_query:
    st.chat_message("user", avatar="👤").markdown(user_query)
    st.session_state.history.append(("user", user_query))
    
    # Khởi tạo LLM với Temperature = 0.0 để chống ảo giác tối đa
    if selected_model == "gpt-4o":
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        llm = ChatGroq(model_name=selected_model, temperature=0.0, groq_api_key=os.getenv("GROQ_API_KEY"))

    vprint_agent = create_agent_graph(llm)
    
    with st.chat_message("assistant", avatar="img/logo.png"):
        status = st.status("🧠 Agent đang phân tích yêu cầu...", expanded=True)
        final_answer = ""
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        for event in vprint_agent.stream({"messages": [("user", user_query)]}, config=config, stream_mode="updates"):
            if "agent" in event:
                msg = event["agent"]["messages"][-1]
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        status.write(f"🛠️ Đang gọi: `{tc['name']}` với từ khóa: `{tc['args']}`")
                else:
                    status.update(label="✅ Đã có câu trả lời!", state="complete", expanded=False)
                    final_answer = msg.content
            elif "tools" in event:
                status.write("📥 Đã tìm thấy dữ liệu từ kho VPRINT, đang tổng hợp...")

        st.markdown(final_answer)
        st.session_state.history.append(("assistant", final_answer))
        
        # Log nhanh với Cached Worksheet
        sheet = get_gsheet_worksheet()
        if sheet: sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_query, final_answer])