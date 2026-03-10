import pandas as pd
import json
import re

# 1. ĐỌC DỮ LIỆU THÔ
input_file = 'vprint_products_clean.csv'
output_file = 'vprint_products_final.csv'

print(f"📥 Đang đọc dữ liệu thô từ: {input_file}...")
try:
    # BẮT BUỘC phải có encoding='utf-8' để giữ nguyên vẹn Tiếng Việt
    df = pd.read_csv(input_file, encoding='utf-8')
except UnicodeDecodeError:
    # Dự phòng nếu file gốc được lưu bằng Excel Windows
    df = pd.read_csv(input_file, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {input_file}. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# 2. HÀM DỌN DẸP VĂN BẢN (Text Cleaning)
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).strip()
    
    # Xóa tiền tố thừa
    text = re.sub(r'^Mô tả chung\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Đặc tính\s*', '', text, flags=re.IGNORECASE)
    
    # Xóa câu quảng cáo lặp lại
    text = re.sub(r'VPRINT MACHINERY\s*-\s*Giải pháp mới cho ngành in\s*(?:và|&)\s*bao bì Việt Nam', '', text, flags=re.IGNORECASE)
    
    # Xóa khoảng trắng kép
    text = re.sub(r'\s+', ' ', text).strip()
    
    if text.lower() in ["nan", "none", "null", ""]: return ""
    return text

# 3. HÀM DỌN DẸP JSON (Recursive JSON)
def clean_specs_recursive(specs_str):
    if pd.isna(specs_str): return ""
    try:
        specs_dict = json.loads(specs_str)
        if not isinstance(specs_dict, dict): return ""
        
        def process_dict(d):
            clean_d = {}
            for k, v in d.items():
                k_clean = str(k).strip()
                if not k_clean: continue
                
                if isinstance(v, dict):
                    cleaned_v = process_dict(v)
                    if cleaned_v: clean_d[k_clean] = cleaned_v
                elif str(v).strip() and str(v).lower() not in ["nan", "none", "null", ""]:
                    clean_d[k_clean] = str(v).strip()
            return clean_d
            
        final_dict = process_dict(specs_dict)
        return json.dumps(final_dict, ensure_ascii=False) if final_dict else ""
    except Exception:
        return ""

# 4. ÁP DỤNG LÀM SẠCH LÊN CÁC CỘT
print("🧹 Đang dọn dẹp các cột văn bản...")
df['short_desc'] = df['short_desc'].apply(clean_text)
df['description'] = df['description'].apply(clean_text)
df['features'] = df['features'].apply(clean_text)

print("🛠️ Đang dọn dẹp cột JSON (Thông số kỹ thuật)...")
df['specs_json'] = df['specs'].apply(clean_specs_recursive)

df = df.fillna("")

# 5. CHUẨN HÓA CỘT CHO CHROMADB
# Mapping từ file của bạn sang cấu trúc chuẩn
rename_map = {
    'url': 'product_url',
    'short_desc': 'summary',
    'images': 'image_urls'
}
df = df.rename(columns=rename_map)

# Khai báo cấu trúc chuẩn mực cần thiết cho Ingest Pipeline
required_cols = [
    'name', 'source_url', 'category_id', 'product_url', 'sku', 
    'price', 'view_count', 'summary', 'description', 'features', 
    'specs_json', 'unused', 'image_urls'
]

# Thêm các cột còn thiếu bằng chuỗi rỗng để không bị lỗi Key
for col in required_cols:
    if col not in df.columns:
        df[col] = ""

df_final = df[required_cols]

# 6. LƯU FILE (BẮT BUỘC DÙNG utf-8-sig CHO WINDOWS)
# Dùng utf-8-sig để khi bạn mở file bằng Excel trên Windows, Tiếng Việt không bị lỗi
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ Hoàn tất! File dữ liệu SIÊU SẠCH đã được lưu tại: {output_file}")