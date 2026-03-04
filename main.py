from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import time
import re
import torch
import json
import ast
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq

# Import các hàm logic từ file của bạn (Giữ nguyên như cũ)
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
# 1. CONFIG & GLOBALS
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

# Bộ nhớ tạm lưu ngữ cảnh theo từng user/phiên chat
SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# Khai báo các biến toàn cục cho hệ thống AI
ensemble_retriever = None
all_docs = None
embedder = None
system_device = None
router_tensor = None
router_labels = None

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
# 2. UTILS FUNCTIONS (Giữ nguyên logic của Streamlit)
# ==========================================
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
    if not spec_text: return ""
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
    markers = ["💡 **Có thể bạn quan tâm:**", "💡 Có thể bạn quan tâm:"]
    for marker in markers:
        if marker in full_text:
            parts = full_text.split(marker)
            main_text = parts[0].strip()
            sug_text = parts[1].strip()
            
            sugs = []
            for line in sug_text.split('\n'):
                clean_line = line.strip().lstrip("-*•1234567890. ")
                if clean_line: sugs.append(clean_line)
            return main_text, sugs
    return full_text, []

# ==========================================
# 3. AI SYSTEM LOADING (Lifespan thay thế cho @st.cache_resource)
# ==========================================
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
    ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
    return ensemble, docs, embeddings, device

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo mô hình khi Server chạy
    print("🚀 Đang khởi động hệ thống VPRINT AI...")
    global ensemble_retriever, all_docs, embedder, system_device, router_tensor, router_labels
    ensemble_retriever, all_docs, embedder, system_device = load_system()
    router_tensor, router_labels = init_vector_router(embedder, system_device)
    print("✅ Hệ thống AI đã sẵn sàng!")
    yield
    # Dọn dẹp tài nguyên khi tắt server (nếu cần)
    SESSION_STORE.clear()

app = FastAPI(title="VPRINT Sales AI API", lifespan=lifespan)

# ==========================================
# 4. API MODELS & ENDPOINTS
# ==========================================
class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2

class ChatResponse(BaseModel):
    answer: str
    suggestions: List[str]
    intent: str
    latency: float

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    start_time = time.perf_counter()

    # 1. Quản lý trạng thái ngữ cảnh (Thay thế st.session_state)
    if req.session_id not in SESSION_STORE:
        SESSION_STORE[req.session_id] = {"history": [], "last_docs": []}
    
    current_session = SESSION_STORE[req.session_id]
    user_query = req.message
    
    # 2. Định tuyến (Routing)
    decision = fast_vector_route_query(user_query, embedder, router_tensor, router_labels, system_device, threshold=0.5)
    decision = apply_router_guards(user_query, decision)

    if decision.reset_focus:
        current_session["last_docs"] = []

    # Khởi tạo LLM
    llm = ChatGroq(model_name=req.model, temperature=req.temperature, groq_api_key=GROQ_API_KEY)
    raw_answer = ""

    try:
        # ----------------------------------------------------
        # LUỒNG 1 & 2: RAG & XỬ LÝ DỮ LIỆU
        # ----------------------------------------------------
        if decision.use_rag:
            fused_docs = ensemble_retriever.invoke(user_query)
            filtered_docs = filter_docs_by_need_profile(deduplicate_docs(fused_docs)[:K_FINAL], user_query)
            context = format_context(filtered_docs)

            # LUỒNG TÌM MÁY
            if decision.intent == "find_machine":
                suggested_docs = filtered_docs[:3]
                if not suggested_docs:
                    raw_answer = "Xin lỗi, tôi không tìm thấy máy phù hợp."
                else:
                    raw_answer = "👋 Hệ thống đã tìm thấy các dòng máy phù hợp:\n\n"
                    for i, doc in enumerate(suggested_docs):
                        machine_name = doc.metadata.get("name", f"Sản phẩm {i+1}")
                        raw_answer += f"### 🏆 Top {i+1}: **{machine_name}**\n"
                        
                        content = doc.page_content
                        
                        # Mô tả
                        desc_text = get_field(content, "Description") or get_field(content, "Summary")
                        if desc_text:
                            raw_desc = desc_text.replace(r'\n', '\n')
                            desc_list = [d.strip() for d in raw_desc.split('\n') if d.strip()] if '\n' in raw_desc else [d.strip() + '.' for d in raw_desc.split('.') if len(d.strip()) > 1]
                            formatted_desc_lines = [f"- {d.lstrip('-*• ')[0].upper() + d.lstrip('-*• ')[1:]}" for d in desc_list[:3] if d.lstrip('-*• ')]
                            formatted_desc = "\n".join(formatted_desc_lines)
                            if len(desc_list) > 3: formatted_desc += "\n- *... (Xem thêm tại link chi tiết)*"
                            raw_answer += f"**Mô tả:**\n{formatted_desc}\n\n"

                        # Đặc điểm
                        features = get_field(content, "Features")
                        if features:
                            raw_features = features.replace(r'\n', '\n')
                            feature_list = [f.strip() for f in raw_features.split('\n') if f.strip()] if '\n' in raw_features else [f.strip() for f in raw_features.split(',') if f.strip()]
                            formatted_feat_lines = [f"- {f.lstrip('-*• ')[0].upper() + f.lstrip('-*• ')[1:]}" for f in feature_list[:3] if f.lstrip('-*• ')]
                            formatted_features = "\n".join(formatted_feat_lines)
                            if len(feature_list) > 3: formatted_features += "\n- *... (Xem chi tiết thông số bên dưới)*"
                            raw_answer += f"**Đặc điểm nổi bật:**\n{formatted_features}\n\n"
                        
                        # Link & Hình ảnh
                        product_url = doc.metadata.get("product_url", "")
                        if product_url: raw_answer += f"🔗 **Xem chi tiết:** [Nhấn vào đây]({product_url})\n\n"

                        images = parse_images(doc.metadata.get("images", ""))
                        if images:
                            img_html = "".join([f'<img src="{img}" style="height:140px;margin-right:8px;border-radius:8px;border:1px solid #ddd;object-fit:contain;">' for img in images])
                            raw_answer += f"<div>{img_html}</div><br>\n\n"
                        
                        raw_answer += "---\n\n"
                    
                    top1_name = suggested_docs[0].metadata.get("name", "dòng máy này")
                    raw_answer += f"💡 **Có thể bạn quan tâm:**\n- Thông số kỹ thuật của {top1_name}\n- Yêu cầu báo giá {top1_name}\n- Tư vấn các dòng máy khác"
                    current_session["last_docs"] = suggested_docs

            # LUỒNG THÔNG SỐ
            elif decision.intent == "spec_query":
                best_doc = None
                if current_session["last_docs"]:
                    best_doc = pick_best_doc_for_query(user_query, current_session["last_docs"])
                if not best_doc:
                    best_doc = pick_best_doc_for_query(user_query, filtered_docs)
                if not best_doc and current_session["last_docs"]:
                    best_doc = current_session["last_docs"][0]

                raw_answer = build_specs_answer(user_query, best_doc)
                if best_doc:
                    table_specs = format_specs_to_table(get_field(best_doc.page_content, "Specifications"))
                    if table_specs: raw_answer += f"\n\n---\n**🔧 Bảng Thông Số Kỹ Thuật:**\n\n{table_specs}"
                    
                    machine_name = best_doc.metadata.get("name", "máy này")
                    raw_answer += f"\n\n💡 **Có thể bạn quan tâm:**\n- Chế độ bảo hành của {machine_name}\n- Báo giá {machine_name}\n- Xem các dòng máy tương tự"

            # LUỒNG RAG BÌNH THƯỜNG
            else:
                if not filtered_docs:
                    raw_answer = "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
                else:
                    messages = build_rag_messages(user_query, context, current_session["history"], MAX_HISTORY)
                    # Khác với Streamlit dùng stream(), ở API ta gọi invoke() để lấy string cuối cùng
                    response = llm.invoke(messages)
                    raw_answer = response.content

        # ----------------------------------------------------
        # LUỒNG 3: CHAT NGOÀI LỀ / TRỰC TIẾP
        # ----------------------------------------------------
        else:
            messages = build_direct_messages(user_query, current_session["history"], MAX_HISTORY)
            response = llm.invoke(messages)
            raw_answer = response.content

        # ==========================================
        # 5. POST-PROCESSING & TRẢ VỀ API
        # ==========================================
        clean_answer, suggestions = parse_and_clean_suggestions(raw_answer)
        
        # Cập nhật lịch sử
        current_session["history"].append(("user", user_query))
        current_session["history"].append(("assistant", clean_answer))

        end_time = time.perf_counter()
        
        return ChatResponse(
            answer=clean_answer,
            suggestions=suggestions,
            intent=decision.intent,
            latency=round(end_time - start_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")