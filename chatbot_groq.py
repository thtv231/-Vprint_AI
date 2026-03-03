


# NÂNG CẤP BỘ ĐỊNH TUYẾN


import streamlit as st
import subprocess
import time
import re
import torch
import json
import ast
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
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

# ----------------------------
# CONFIG & DATA
# ----------------------------

# Load biến môi trường
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CSV_PATH = "vprint_products_clean.csv"
PERSIST_DIR = "vprint_agentic_db_local"
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

# Dữ liệu mẫu để huấn luyện Vector Router siêu tốc
ROUTING_SAMPLES = {
    "find_machine": [
        "Giới thiệu cho tôi các dòng máy bế hộp tự động.",
        "Xưởng mình mới mở, cần tìm máy in offset 4 màu loại nhỏ.",
        "Bên VPRINT có bán máy dán thùng carton sóng E không?",
        "Tư vấn giúp tôi hệ thống máy in flexo cuộn.",
        "Mình đang tìm mua máy cán màng nhiệt tốc độ cao.",
        "Có dòng máy nào in được trực tiếp lên màng nhựa PET không bạn?",
        "Báo giá cho tôi các loại máy bế tròn.",
        "Cần tìm máy chia cuộn decal giấy.",
        "Máy ép nhũ viền tự động loại nào tốt hiện nay?",
        "Gợi ý giải pháp máy móc sản xuất hộp cứng cao cấp."
    ],
    "spec_query": [
        "Tốc độ tối đa của máy này là bao nhiêu tờ/giờ?",
        "Khổ giấy lớn nhất mà máy số 1 có thể chạy là bao nhiêu?",
        "Máy đó tiêu thụ điện năng như thế nào? Dùng điện 3 pha à?",
        "Kích thước tổng thể nặng nhẹ ra sao, tốn bao nhiêu diện tích?",
        "Dòng này dùng cụm bế nam châm hay bế phẳng?",
        "Độ chính xác chồng màu của máy in này là bao nhiêu milimet?",
        "Hệ thống lô mực của dòng thứ 2 cấu tạo gồm mấy lô?",
        "Máy này có tích hợp sẵn cụm sấy UV không hay phải lắp thêm?",
        "Thời gian bảo hành và chế độ bảo trì thế nào?",
        "Option nâng cấp của nó gồm những gì?",
        "Thông số kỹ thuật chi tiết của máy này?"
    ],
    "normal_rag": [
        "Sự khác biệt lớn nhất giữa in offset và in flexo là gì?",
        "Làm thế nào để khắc phục lỗi lem mực khi in màng mạ kim loại?",
        "Quy trình tiêu chuẩn sản xuất hộp giấy mỹ phẩm gồm những bước nào?",
        "Nên dùng màng BOPP hay màng PET cán bóng bao bì thực phẩm?",
        "Tại sao khi bế hộp carton sóng hay bị nứt nếp gấp?",
        "Giải thích công nghệ in dữ liệu biến đổi VDP.",
        "Xu hướng bao bì thân thiện môi trường hiện nay là gì?",
        "Độ phân giải DPI ảnh hưởng thế nào đến bài in?",
        "Sự khác nhau giữa mực UV và mực gốc nước?",
        "Tư vấn các tiêu chuẩn an toàn thực phẩm bao bì giấy."
    ],
    "direct_chat": [
        "Chào VPRINT, chúc một ngày tốt lành.",
        "Alo, có tư vấn viên ở đó không?",
        "Cảm ơn bạn đã hỗ trợ nhé, thông tin rất hữu ích.",
        "Tuyệt vời, để mình xem xét thêm rồi báo lại.",
        "Bạn là bot tự động hay người thật vậy?",
        "Hi ad.",
        "Bot trả lời chán quá, không hiểu ý mình gì cả.",
        "Cho mình hỏi thời tiết hôm nay thế nào?",
        "Ok, mình hiểu rồi.",
        "Tạm biệt nhé."
    ]
}

# ----------------------------
# STREAMLIT PAGE
# ----------------------------

st.set_page_config(page_title="VPRINT AI", page_icon="🤖", layout="wide")

if "initialized" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_docs = []
    st.session_state.initialized = True

st.title("🤖 VPRINT Sales AI")

# ----------------------------
# UTIL FUNCTIONS
# ----------------------------

def get_ollama_models():
    """
    Trả về danh sách model Groq.
    Giữ nguyên tên hàm để không ảnh hưởng code cũ.
    """
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
    """Parse chuỗi JSON/Dict thành bảng Markdown"""
    if not spec_text:
        return ""
    
    try:
        try:
            data = json.loads(spec_text)
        except json.JSONDecodeError:
            data = ast.literal_eval(spec_text)

        if not isinstance(data, dict):
            return spec_text.replace(r'\n', '\n')

        models = list(data.keys())
        features = []
        for model_data in data.values():
            if isinstance(model_data, dict):
                for k in model_data.keys():
                    if k not in features:
                        features.append(k)
        
        if not features:
            return spec_text.replace(r'\n', '\n')

        header = "| **Thông số / Model** | " + " | ".join([f"**{m}**" for m in models]) + " |"
        separator = "|---|" + "|".join(["---"] * len(models)) + "|"
        
        rows = []
        for feat in features:
            row_vals = []
            for m in models:
                val = data[m].get(feat, "-")
                row_vals.append(str(val))
            row_str = f"| {feat} | " + " | ".join(row_vals) + " |"
            rows.append(row_str)

        return "\n".join([header, separator] + rows)
        
    except Exception:
        return spec_text.replace(r'\n', '\n')

# ----------------------------
# LOAD SYSTEM & ROUTER
# ----------------------------

@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": K_VECTOR})
    docs = load_csv_docs(CSV_PATH)
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = K_BM25

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble_retriever, docs, embeddings, device


@st.cache_resource
def init_vector_router(_embedder, device):
    """Nhúng tập dữ liệu định tuyến 1 lần duy nhất"""
    intent_labels = []
    corpus_texts = []
    
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
    """So sánh Cosine Similarity để quyết định rẽ nhánh"""
    query_emb = embedder.embed_query(user_query)
    query_tensor = torch.tensor([query_emb], dtype=torch.float32).to(device)
    
    cos_scores = torch.mm(query_tensor, corpus_tensor.transpose(0, 1))[0]
    best_score, best_idx = torch.max(cos_scores, dim=0)
    best_intent = intent_labels[best_idx.item()]
    
    decision = RouterDecision()
    decision.score = round(best_score.item() * 100, 2)
    
    if best_score.item() < threshold:
        decision.intent = "normal_rag"
        decision.use_rag = True
    else:
        decision.intent = best_intent
        decision.use_rag = True if best_intent != "direct_chat" else False
        
        if best_intent == "find_machine":
            decision.reset_focus = True
            
    return decision


# Tải hệ thống
ensemble_retriever, all_docs, embedder, system_device = load_system()
router_tensor, router_labels = init_vector_router(embedder, system_device)

# ----------------------------
# SIDEBAR
# ----------------------------

with st.sidebar:
    st.header("⚙️ Model Settings")
    models = get_ollama_models()
    selected_model = st.selectbox("Chọn Ollama model", models)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)

    st.divider()

    st.write("**System info**")
    st.write(f"Vector DB: {PERSIST_DIR}")
    st.write(f"Products loaded: {len(all_docs)}")

    st.divider()

    if st.button("🔄 Reset Chat"):
        st.session_state.history = []
        st.session_state.last_docs = []
        st.rerun()

# ----------------------------
# RENDER HISTORY
# ----------------------------

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg, unsafe_allow_html=True)

if len(st.session_state.history) == 0:
    st.info(WELCOME)

# ----------------------------
# USER INPUT
# ----------------------------

user_query = st.chat_input("Nhập câu hỏi...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.history.append(("user", user_query))

    llm = ChatGroq(
    model_name=selected_model,
    temperature=temperature,
    groq_api_key=GROQ_API_KEY
)
    start = time.perf_counter()

    # Định tuyến bằng Vector thay vì LLM
    decision = fast_vector_route_query(
        user_query, embedder, router_tensor, router_labels, system_device, threshold=0.5
    )
    decision = apply_router_guards(user_query, decision)

    if decision.reset_focus:
        st.session_state.last_docs = []

    # ----------------------------
    # RAG
    # ----------------------------

    if decision.use_rag:
        with st.spinner("🔍 Đang tìm kiếm thông tin..."):
            fused_docs = ensemble_retriever.invoke(user_query)
            fused_docs = deduplicate_docs(fused_docs)
            fused_docs = fused_docs[:K_FINAL]
            filtered_docs = filter_docs_by_need_profile(fused_docs, user_query)

        context = format_context(filtered_docs)

        # ----------------------------
        # FIND MACHINE
        # ----------------------------

        if decision.intent == "find_machine":
            suggested_docs = filtered_docs[:3]

            if not suggested_docs:
                fallback_msg = "Xin lỗi, tôi không tìm thấy máy phù hợp."
                with st.chat_message("assistant"):
                    st.markdown(fallback_msg)
                st.session_state.history.append(("assistant", fallback_msg))
            else:
                full_response = "👋 Hệ thống đã tìm thấy các dòng máy phù hợp:\n\n"

                for i, doc in enumerate(suggested_docs):
                    machine_name = doc.metadata.get("name", f"Sản phẩm {i+1}")
                    full_response += f"### 🏆 Top {i+1}: **{machine_name}**\n"

                    content = doc.page_content
                    summary = get_field(content, "Summary")
                    description = get_field(content, "Description")
                    features = get_field(content, "Features")

                    desc_text = description if description else summary

                    # MÔ TẢ TỐI ƯU
                    if desc_text:
                        raw_desc = desc_text.replace(r'\n', '\n')
                        desc_list = [d.strip() for d in raw_desc.split('\n') if d.strip()] if '\n' in raw_desc else [d.strip() + '.' for d in raw_desc.split('.') if len(d.strip()) > 1]
                            
                        MAX_DESC_LINES = 3
                        top_desc = desc_list[:MAX_DESC_LINES]
                        
                        formatted_desc_lines = []
                        for d in top_desc:
                            clean_line = d.lstrip('-*• ')
                            if clean_line:
                                clean_line = clean_line[0].upper() + clean_line[1:]
                                formatted_desc_lines.append(f"- {clean_line}")
                                
                        formatted_desc = "\n".join(formatted_desc_lines)
                        if len(desc_list) > MAX_DESC_LINES:
                            formatted_desc += "\n- *... (Xem thêm tại link chi tiết)*"

                        full_response += f"**Mô tả:**\n{formatted_desc}\n\n"

                    # ĐẶC ĐIỂM TỐI ƯU
                    if features:
                        raw_features = features.replace(r'\n', '\n')
                        feature_list = [f.strip() for f in raw_features.split('\n') if f.strip()] if '\n' in raw_features else [f.strip() for f in raw_features.split(',') if f.strip()]
                            
                        MAX_FEAT_LINES = 3
                        top_features = feature_list[:MAX_FEAT_LINES]
                        
                        formatted_feat_lines = []
                        for f in top_features:
                            clean_feat = f.lstrip('-*• ')
                            if clean_feat:
                                clean_feat = clean_feat[0].upper() + clean_feat[1:]
                                formatted_feat_lines.append(f"- {clean_feat}")
                                
                        formatted_features = "\n".join(formatted_feat_lines)
                        if len(feature_list) > MAX_FEAT_LINES:
                            formatted_features += "\n- *... (Xem chi tiết thông số bên dưới)*"

                        full_response += f"**Đặc điểm nổi bật:**\n{formatted_features}\n\n"

                    # LINK & HÌNH ẢNH
                    product_url = doc.metadata.get("product_url", "")
                    if product_url:
                        full_response += f"🔗 **Xem chi tiết:** [Nhấn vào đây]({product_url})\n\n"

                    images = parse_images(doc.metadata.get("images", ""))
                    if images:
                        img_html = "".join([f'<img src="{img}" style="height:160px;margin-right:10px;border-radius:8px;border:1px solid #ddd;object-fit:contain;max-width:200px;">' for img in images])
                        full_response += f"<div>{img_html}</div><br>\n\n"

                    full_response += "---\n\n"

                full_response += "**Bạn muốn xem chi tiết thông số của máy nào?**"

                with st.chat_message("assistant"):
                    st.markdown(full_response, unsafe_allow_html=True)

                st.session_state.history.append(("assistant", full_response))
                st.session_state.last_docs = suggested_docs

        # ----------------------------
        # SPEC QUERY
        # ----------------------------

        elif decision.intent == "spec_query":
            candidate_docs = st.session_state.last_docs + filtered_docs
            best_doc = pick_best_doc_for_query(user_query, candidate_docs)

            answer = build_specs_answer(user_query, best_doc)

            if best_doc:
                content = best_doc.page_content
                
                # TRÍCH XUẤT VÀ FORMAT THÔNG SỐ THÀNH BẢNG MARKDOWN
                raw_specs = get_field(content, "Specifications") 
                if raw_specs:
                    table_specs = format_specs_to_table(raw_specs)
                    answer += f"\n\n---\n**🔧 Bảng Thông Số Kỹ Thuật Chi Tiết:**\n\n{table_specs}"

                images = parse_images(best_doc.metadata.get("images", ""))
                if images:
                    img_html = "".join([f'<img src="{img}" style="height:160px;margin-right:10px;border-radius:8px;border:1px solid #ddd;object-fit:contain;max-width:200px;">' for img in images])
                    answer += f"\n\n<div>{img_html}</div>"

            with st.chat_message("assistant"):
                st.markdown(answer, unsafe_allow_html=True)

            st.session_state.history.append(("assistant", answer))

        # ----------------------------
        # NORMAL RAG
        # ----------------------------

        else:
            if not filtered_docs:
                answer = "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.history.append(("assistant", answer))
            else:
                messages = build_rag_messages(user_query, context, st.session_state.history, MAX_HISTORY)
                with st.chat_message("assistant"):
                    stream = llm.stream(messages)
                    def token_stream():
                        for chunk in stream:
                            if hasattr(chunk, "content"):
                                yield chunk.content
                    answer = st.write_stream(token_stream)
                st.session_state.history.append(("assistant", answer))

    # ----------------------------
    # DIRECT CHAT
    # ----------------------------

    else:
        messages = build_direct_messages(user_query, st.session_state.history, MAX_HISTORY)
        with st.chat_message("assistant"):
            stream = llm.stream(messages)
            def token_stream():
                for chunk in stream:
                    if hasattr(chunk, "content"):
                        yield chunk.content
            answer = st.write_stream(token_stream)
        st.session_state.history.append(("assistant", answer))

    # ----------------------------
    # LATENCY & DEBUG LOG
    # ----------------------------

    end = time.perf_counter()
    latency = round(end - start, 2)

    st.caption(f"⏱ Thời gian trả lời: **{latency}s** | Model: `{selected_model}` | 🎯 Router: `{decision.intent}` (Độ tự tin: `{decision.score}%`)")
