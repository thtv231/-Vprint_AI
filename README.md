# 🤖 VPRINT Sales AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Groq](https://img.shields.io/badge/LLM-Groq-black)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange)

**VPRINT Sales AI** là một hệ thống Chatbot RAG (Retrieval-Augmented Generation) thông minh, được thiết kế chuyên biệt để hỗ trợ tư vấn bán hàng, tra cứu thông số kỹ thuật và giải đáp thắc mắc về các dòng máy in, máy bế, và thiết bị ngành bao bì.

---

## ✨ Tính năng nổi bật

* 🚀 **Bộ định tuyến siêu tốc (Fast Vector Router):** Sử dụng Pytorch và Cosine Similarity để phân loại ý định người dùng (Intent Routing) chỉ trong vài mili-giây, giúp điều hướng câu hỏi về đúng luồng xử lý (Tìm máy, Hỏi thông số, Chat kỹ thuật, hoặc Giao tiếp ngoài lề) mà không tốn token LLM.
* 🔍 **Tìm kiếm lai (Hybrid Search):** Kết hợp sức mạnh của tìm kiếm từ khóa chính xác (`BM25`) và tìm kiếm ngữ nghĩa (`Vector Search`) thông qua LangChain `EnsembleRetriever`, đảm bảo độ chính xác tuyệt đối khi tra cứu mã máy và thông số kỹ thuật.
* 🧠 **Quản lý ngữ cảnh thông minh:** Tự động ghi nhớ các thiết bị vừa tư vấn, cho phép người dùng hỏi tiếp các câu hỏi dạng rút gọn (VD: *"Thông số máy này thế nào?"*, *"Bảo hành bao lâu?"*) một cách tự nhiên.
* 💡 **Nút bấm gợi ý tương tác (Smart UI Pills):** Tự động sinh ra các câu hỏi gợi ý tiếp theo và hiển thị dưới dạng nút bấm UI bo góc (chuẩn phong cách ChatGPT/Claude), giúp tăng trải nghiệm người dùng (UX) và dẫn dắt luồng Sales hiệu quả.
* 🛡️ **Kiểm soát Out-of-Domain:** Tự động nhận diện và từ chối khéo léo các câu hỏi không thuộc lĩnh vực in ấn/bao bì, đồng thời chủ động xin thông tin (SĐT/Zalo) khi câu hỏi vượt quá khả năng xử lý hoặc liên quan đến báo giá phức tạp.

---

## 🛠️ Công nghệ sử dụng

* **Giao diện (UI):** [Streamlit](https://streamlit.io/)
* **Mô hình ngôn ngữ (LLM):** Llama 3 / Mixtral / Gemma 2 (thông qua [Groq API](https://groq.com/) để tối ưu tốc độ sinh text).
* **Mô hình nhúng (Embeddings):** `bkai-foundation-models/vietnamese-bi-encoder` (tối ưu cho tiếng Việt).
* **Cơ sở dữ liệu Vector:** [Chroma](https://www.trychroma.com/) (In-memory).
* **Khung ứng dụng AI:** [LangChain](https://www.langchain.com/).

---

## 📂 Cấu trúc dự án

```text
📁 vprint-sales-ai/
│
├── chatbot_groq.py                # File main chạy giao diện Streamlit và luồng chính
├── chatbot_vprint_hybrid_local.py # Chứa các hàm logic, xử lý dữ liệu và System Prompts
├── vprint_products_clean.csv      # Database chứa thông tin sản phẩm (Đầu vào cho RAG)
├── requirements.txt               # Danh sách các thư viện Python cần thiết
├── .env                           # File chứa biến môi trường (API Keys)
└── README.md                      # Tài liệu hướng dẫn dự án