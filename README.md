# 🖨️ VPRINT Sales AI - Agentic RAG Consultation Chatbot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-⚡-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![LLM](https://img.shields.io/badge/LLM-GPT--4o%20%7C%20Llama--3.1-black)

> **VPRINT Sales AI** là trợ lý ảo thông minh chuyên sâu cho ngành in ấn và bao bì công nghiệp. Vượt xa kiến trúc RAG thông thường, hệ thống sử dụng **Agentic Workflow** và **Hybrid Search (Vector + BM25 kết hợp RRF)** để tư vấn cấu hình dây chuyền máy móc, giải thích kiến thức kỹ thuật và tự động thu thập thông tin khách hàng tiềm năng (Lead Generation).

---

## ✨ Tính năng Nổi bật (Core Features)

* 🧠 **Semantic Routing & Intent Detection:** Tự động phân loại luồng ý định của người dùng (Tìm máy, Hỏi kiến thức ngành, Tư vấn giải pháp, Trò chuyện) để kích hoạt Pipeline xử lý phù hợp.
* 🔗 **Sequential Agentic RAG:** Khả năng suy luận đa bước (Multi-step reasoning). Bóc tách yêu cầu khách hàng (sản phẩm, vật liệu, quy mô) -> Phân rã quy trình -> Truy hồi thiết bị chính xác cho từng công đoạn -> Tổng hợp giải pháp dây chuyền.
* 🎯 **Hybrid Search với Heuristic Boosting:** Kết hợp tìm kiếm ngữ nghĩa (OpenAI Embeddings) và tìm kiếm từ khóa (BM25). Can thiệp điểm số thuật toán RRF bằng Hard-rules (VD: Ép hệ thống không đề xuất máy cuộn cho vật liệu tờ rời).
* 📚 **Expert Persona (Chuyên gia ngành in):** Trả lời các câu hỏi kỹ thuật sâu (Offset vs Flexo, CTP Thermal vs Violet) dựa trên Cẩm nang nội bộ (Handbook of Print Media).
* 📩 **Automated Lead Capture:** Tự động nhận diện ý định đặt lịch, trích xuất thông tin liên hệ (Tên, SĐT, Email) và gửi thông báo qua hệ thống SMTP nội bộ.
* 📊 **Dashboard & Analytics:** Lưu trữ lịch sử hội thoại lên Google Sheets, thống kê lượng Token sử dụng, trực quan hóa luồng Intent bằng biểu đồ Radar.

---

## 🏗️ Kiến trúc Hệ thống (Architecture)

1.  **Giao diện:** Xây dựng trên `Streamlit` với UI/UX được tinh chỉnh (Custom CSS, Typing effect, Thinking indicator).
2.  **Cơ sở dữ liệu:** * `vprint_knowledge_base`: Kho vector chứa cẩm nang ngành in.
    * `vprint_products_local`: Kho vector + metadata chứa thông số kỹ thuật máy móc.
3.  **LLM Engine:** Hỗ trợ chuyển đổi linh hoạt giữa các mô hình mạnh nhất hiện nay:
    * OpenAI (GPT-4o, GPT-3.5-turbo) cho tư duy logic phức tạp.
    * Groq (Llama-3.1-8b) cho tốc độ phản hồi siêu tốc (Ultra-fast inference).

---

## 🚀 Cài đặt & Khởi chạy (Installation & Setup)

### 1. Yêu cầu hệ thống
* Python 3.9 trở lên
* Git

### 2. Cài đặt môi trường
```bash
# Clone repository
git clone https://github.com/thtv231/-Vprint_AI.git
cd vprint-sales-ai

# Tạo và kích hoạt virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
