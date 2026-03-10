import pandas as pd
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI # Đã đổi sang OpenAI

load_dotenv()

# Kiểm tra xem đã có OPENAI_API_KEY chưa
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ LỖI: Không tìm thấy OPENAI_API_KEY. Hãy thêm vào file .env nhé!")
    exit()

print("✅ Đã nhận diện API Key. Đang khởi tạo GPT-4o...")

# 1. Khởi tạo LLM (Dùng GPT-4o)
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.1, 
    max_tokens=150
)

system_prompt = """Bạn là một Chuyên gia Kỹ thuật ngành in ấn của VPRINT.
Nhiệm vụ: Đọc thông tin máy và trích xuất danh sách "Từ khóa ứng dụng" (Tags).
Hãy suy luận các từ khóa dựa trên:
1. Phân loại máy (vd: máy in phun, máy ghi kẽm, máy in offset, máy bế...)
2. Khổ in / Khổ giấy / Khổ kẽm (vd: 72x102, khổ B1, 8-up, 580mm, khổ lớn...)
3. Vật liệu in (vd: carton sóng, túi giấy, kẽm nhiệt, decal...)
4. Ứng dụng thành phẩm (vd: in bao bì, in thương mại, hộp carton, in nhãn...)

Quy định: TUYỆT ĐỐI CHỈ TRẢ VỀ CÁC TỪ KHÓA CÁCH NHAU BẰNG DẤU PHẨY. Không giải thích, không viết thành câu.
Ví dụ mẫu đầu ra: máy in phun, in bao bì, carton sóng, hộp giấy, túi giấy, in singlepass"""

def enrich_all_data():
    input_file = "vprint_products_clean.csv"
    output_file = "vprint_products_enriched.csv"
    
    print(f"📦 Đang đọc dữ liệu từ: {input_file}")
    df = pd.read_csv(input_file).fillna("")
    
    enriched_tags = []
    
    print(f"🚀 Bắt đầu làm giàu dữ liệu cho {len(df)} máy móc...\n")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Tiến trình GPT-4o"):
        name = str(row.get('name', ''))
        short_desc = str(row.get('short_desc', ''))
        features = str(row.get('features', ''))[:500]
        specs = str(row.get('specs', ''))[:800]
        
        user_content = f"Tên máy: {name}\nMô tả: {short_desc}\nĐặc tính: {features}\nThông số kỹ thuật: {specs}\n\nTừ khóa ứng dụng:"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]
            
            # Gọi API OpenAI
            response = llm.invoke(messages)
            tags = response.content.strip().replace('"', '').replace('\n', ' ')
            enriched_tags.append(tags)
            
            # OpenAI tier thấp có thể có rate limit, tạm nghỉ một chút để an toàn
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"\n⚠️ Lỗi ở máy '{name}': {e}")
            enriched_tags.append("") 
            time.sleep(2) 
            
    df['tags_ung_dung'] = enriched_tags
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 HOÀN THÀNH! Đã lưu file dữ liệu được GPT-4o phân tích tại: {output_file}")

if __name__ == "__main__":
    enrich_all_data()