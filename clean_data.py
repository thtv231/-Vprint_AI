import json
import re

def normalize_text(text):
    """
    Chuẩn hóa chuỗi: chuyển thành chữ thường và xóa các khoảng trắng thừa 
    để so sánh chính xác hơn.
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_duplicate_questions(input_file, output_file):
    seen_questions = set()
    unique_records = []
    duplicate_count = 0

    # Đọc file dữ liệu đầu vào
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                question_content = ""
                
                # Tìm nội dung câu hỏi của User trong mảng messages
                for msg in record.get("messages", []):
                    if msg.get("role") == "user":
                        question_content = msg.get("content", "")
                        break
                
                # Chuẩn hóa câu hỏi để so sánh
                normalized_q = normalize_text(question_content)
                
                # Kiểm tra trùng lặp
                if normalized_q in seen_questions:
                    duplicate_count += 1
                else:
                    seen_questions.add(normalized_q)
                    unique_records.append(record)
                    
            except json.JSONDecodeError:
                print("Lỗi cú pháp JSON ở dòng:", line)

    # Ghi dữ liệu đã lọc ra file mới
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record in unique_records:
            # ensure_ascii=False để giữ nguyên tiếng Việt có dấu
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("=== KẾT QUẢ XỬ LÝ ===")
    print(f"- Số dòng bị trùng lặp đã loại bỏ: {duplicate_count}")
    print(f"- Số dòng duy nhất (unique) giữ lại: {len(unique_records)}")
    print(f"- Đã lưu file sạch tại: {output_file}")

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    # Thay đổi tên file cho phù hợp với máy của bạn
    FILE_DAU_VAO = 'vprint_openai_finetune.jsonl'
    FILE_DAU_RA = 'vprint_openai_finetune_clean.jsonl'
    
    remove_duplicate_questions(FILE_DAU_VAO, FILE_DAU_RA)