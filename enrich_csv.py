# =============================================================
# ENRICH CSV — SINH USER ALIASES CHO TOÀN BỘ SẢN PHẨM
# =============================================================
# Cách chạy:
#   python enrich_csv.py                  → Resume (bỏ qua sản phẩm đã có alias)
#   python enrich_csv.py --force          → Chạy lại TẤT CẢ, ghi đè alias cũ
#   python enrich_csv.py --force --limit 10  → Test với 10 sản phẩm đầu
#
# Yêu cầu: pip install pandas python-dotenv
# =============================================================

import os
import sys

# Fix Unicode/emoji trên terminal Windows (CP1252) — PHẢI đặt trước mọi import khác
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import argparse
import pandas as pd
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ==========================================
# CONFIG
# ==========================================
INPUT_CSV    = "vprint_products_clean.csv"
OUTPUT_CSV   = "vprint_products_enriched.csv"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ENRICH_MODEL = "llama-3.3-70b-versatile"   # model ổn định, hỗ trợ json_object
FALLBACK_MODEL = "llama-3.1-8b-instant"    # fallback nếu rate limit
MAX_RETRIES  = 3
DELAY_SEC    = 0.6   # tránh rate limit Groq

# Phải khớp với DEFAULT_COLUMNS trong chatbot_groq.py
DEFAULT_COLUMNS = [
    "name", "source_url", "category_id", "product_url", "sku",
    "price", "view_count", "summary", "description", "features",
    "specs_json", "unused", "image_urls",
]

# ==========================================
# PROMPTS
# ==========================================
SYSTEM_PROMPT = """Bạn là chuyên gia ngành in bao bì Việt Nam với 20 năm kinh nghiệm tư vấn bán hàng.
Nhiệm vụ: Với mỗi thiết bị được cung cấp, sinh ra alias và use case thực tế.

QUY TẮC:
- user_aliases: Cách người mua THỰC TẾ gọi sản phẩm — tiếng lóng, mô tả vòng vo, viết tắt, tiếng Anh lẫn tiếng Việt.
  Đây là những từ khách hàng thực sự gõ khi tìm kiếm, KHÔNG phải tên kỹ thuật chính thức.
- use_cases: Bài toán/nhu cầu cụ thể từ góc nhìn người mua.
- KHÔNG lặp lại tên máy chính thức trong alias.
- Ưu tiên cách nói đời thường, thực tế.

Chỉ trả về JSON, không thêm bất kỳ text nào khác:
{"user_aliases": ["alias 1", "alias 2", ...], "use_cases": ["use case 1", ...]}"""

USER_PROMPT_TEMPLATE = """Thiết bị:
- Tên: {name}
- Mô tả: {summary}
- Tính năng: {features}
- Danh mục: {category_id}

Hãy sinh ra:
1. user_aliases (15-20 cách người Việt thực tế gọi thiết bị này)
   Ví dụ với "Máy làm ly giấy":
   → ["tô giấy", "bát giấy", "cốc giấy dùng 1 lần", "ly nhựa thay thế", "máy dập cốc",
      "paper cup machine", "coffee cup maker", "máy làm cốc cafe", "hộp giấy đựng nước nóng",
      "máy làm bát giấy", "cup forming machine", "máy sản xuất ly giấy", ...]

2. use_cases (5-8 bài toán cụ thể máy giải quyết)
   Ví dụ: ["sản xuất ly giấy cho quán cà phê takeaway", "làm cốc giấy chịu nhiệt đựng trà sữa", ...]"""


# ==========================================
# GROQ API CALLER
# ==========================================
_groq_client: Groq | None = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

def call_groq(prompt_user: str, model: str = ENRICH_MODEL) -> str:
    """Gọi Groq API qua SDK, trả về raw content string."""
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_user},
        ],
        temperature=0.3,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def parse_llm_json(raw: str) -> dict:
    """Parse JSON từ LLM, xử lý markdown fences nếu có."""
    clean = raw.strip()
    # Strip ```json ... ``` nếu model vẫn sinh ra dù đã dặn không
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean
    return json.loads(clean)


def enrich_one(row: pd.Series) -> dict:
    """
    Sinh alias cho 1 sản phẩm với retry.
    Trả về {"user_aliases": "str", "use_cases": "str"}
    hoặc {"user_aliases": "", "use_cases": ""} nếu thất bại.
    """
    prompt = USER_PROMPT_TEMPLATE.format(
        name        = str(row.get("name", "")).strip(),
        summary     = str(row.get("summary", "")).strip()[:400],
        features    = str(row.get("features", "")).strip()[:400],
        category_id = str(row.get("category_id", "")).strip(),
    )

    for attempt in range(MAX_RETRIES):
        # Dùng fallback model ở lần thử cuối nếu model chính bị lỗi
        model = FALLBACK_MODEL if attempt == MAX_RETRIES - 1 else ENRICH_MODEL
        try:
            raw    = call_groq(prompt, model=model)
            result = parse_llm_json(raw)

            aliases   = result.get("user_aliases", [])
            use_cases = result.get("use_cases", [])

            if not isinstance(aliases, list) or not aliases:
                raise ValueError(f"user_aliases rỗng hoặc sai kiểu: {type(aliases)}")

            return {
                "user_aliases": ", ".join(str(a).strip() for a in aliases if a),
                "use_cases":    ", ".join(str(u).strip() for u in use_cases if u),
            }

        except Exception as e:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            print(f"   ⚠️  Attempt {attempt+1}/{MAX_RETRIES} [{model}]: {e} — cho {wait}s")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    return {"user_aliases": "", "use_cases": ""}


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Enrich VPRINT product CSV with user aliases")
    parser.add_argument("--force", action="store_true",
                        help="Chạy lại TẤT CẢ sản phẩm, ghi đè alias cũ")
    parser.add_argument("--limit", type=int, default=0,
                        help="Chỉ xử lý N sản phẩm đầu (để test). 0 = xử lý hết")
    args = parser.parse_args()

    # --- Validate API key ---
    if not GROQ_API_KEY:
        print("❌ Không tìm thấy GROQ_API_KEY.")
        print("   Tạo file .env với nội dung: GROQ_API_KEY=your_key")
        sys.exit(1)

    print("=" * 60)
    print("🚀 ENRICH CSV — SINH USER ALIASES")
    if args.force:
        print("   Mode: FORCE — ghi đè toàn bộ alias cũ")
    else:
        print("   Mode: RESUME — bỏ qua sản phẩm đã có alias")
    print("=" * 60)

    # --- Load INPUT CSV ---
    if not Path(INPUT_CSV).exists():
        print(f"❌ Không tìm thấy file: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, header=0).fillna("")

    # Rename columns theo DEFAULT_COLUMNS (giữ nguyên logic chatbot_groq.py)
    rename_map = {
        col: DEFAULT_COLUMNS[idx] if idx < len(DEFAULT_COLUMNS) else f"extra_{idx}"
        for idx, col in enumerate(df.columns)
    }
    df = df.rename(columns=rename_map)

    # Thêm cột alias nếu chưa có
    if "user_aliases" not in df.columns:
        df["user_aliases"] = ""
    if "use_cases" not in df.columns:
        df["use_cases"] = ""

    # --- Resume: load alias đã có từ OUTPUT CSV ---
    if not args.force and Path(OUTPUT_CSV).exists():
        try:
            df_prev = pd.read_csv(OUTPUT_CSV).fillna("")
            if "user_aliases" in df_prev.columns and "name" in df_prev.columns:
                # Merge alias từ file cũ vào df hiện tại theo tên máy
                alias_map = {
                    str(r["name"]).strip(): {
                        "user_aliases": str(r.get("user_aliases", "")).strip(),
                        "use_cases":    str(r.get("use_cases", "")).strip(),
                    }
                    for _, r in df_prev.iterrows()
                    if str(r.get("user_aliases", "")).strip()
                }
                for idx, row in df.iterrows():
                    name = str(row.get("name", "")).strip()
                    if name in alias_map:
                        df.at[idx, "user_aliases"] = alias_map[name]["user_aliases"]
                        df.at[idx, "use_cases"]    = alias_map[name]["use_cases"]

                already_done = df[df["user_aliases"].str.strip() != ""].shape[0]
                print(f"♻️  Resume: đã load {already_done} alias từ {OUTPUT_CSV}")
        except Exception as e:
            print(f"⚠️  Không đọc được file cũ ({e}), bắt đầu từ đầu.")

    # --- Nếu force: xóa toàn bộ alias để chạy lại ---
    if args.force:
        df["user_aliases"] = ""
        df["use_cases"]    = ""
        print("🔄 Đã xóa alias cũ, bắt đầu enrich lại từ đầu.")

    # --- Xác định danh sách cần xử lý ---
    todo_mask = (
        df["name"].str.strip().astype(bool) &          # có tên
        (df["user_aliases"].str.strip() == "")         # chưa có alias
    )
    todo_indices = df[todo_mask].index.tolist()

    if args.limit > 0:
        todo_indices = todo_indices[:args.limit]
        print(f"🔬 Test mode: chỉ xử lý {len(todo_indices)} sản phẩm")

    total_todo = len(todo_indices)
    total_all  = len(df)
    already    = total_all - len(df[todo_mask])

    print(f"📦 Tổng sản phẩm  : {total_all}")
    print(f"✅ Đã có alias    : {already}")
    print(f"⏳ Cần xử lý     : {total_todo}")
    print("-" * 60)

    if total_todo == 0:
        print("✅ Tất cả sản phẩm đã có alias. Dùng --force để chạy lại.")
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"📄 Output: {OUTPUT_CSV}")
        return

    # --- Enrich từng sản phẩm ---
    enriched = 0
    errors   = 0

    for order, idx in enumerate(todo_indices, start=1):
        row  = df.loc[idx]
        name = str(row.get("name", "")).strip()

        print(f"\n[{order}/{total_todo}] 🔧 {name[:65]}")

        result = enrich_one(row)

        if result["user_aliases"]:
            df.at[idx, "user_aliases"] = result["user_aliases"]
            df.at[idx, "use_cases"]    = result["use_cases"]
            preview = result["user_aliases"][:90]
            print(f"   ✅ {preview}...")
            enriched += 1
        else:
            print(f"   ❌ Thất bại sau {MAX_RETRIES} lần thử")
            errors += 1

        # Checkpoint sau mỗi sản phẩm
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        if order < total_todo:
            time.sleep(DELAY_SEC)

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH!")
    print(f"   ✅ Đã enrich : {enriched}")
    print(f"   ❌ Lỗi       : {errors}")
    print(f"   ⏭️  Bỏ qua   : {already}")
    print(f"   📄 Output    : {OUTPUT_CSV}")
    print("=" * 60)

    if errors > 0:
        print(f"\n⚠️  {errors} sản phẩm lỗi. Chạy lại để retry (resume tự động).")

    print("\n📌 Bước tiếp theo:")
    print("   1. Mở OUTPUT_CSV kiểm tra alias có hợp lý không")
    print("   2. Upload lên Google Drive")
    print("   3. Chạy build_rag_db_fixed.py trên Colab để rebuild VectorDB")
    print("   4. Copy DB mới lên Streamlit Cloud / Git repo")


if __name__ == "__main__":
    main()