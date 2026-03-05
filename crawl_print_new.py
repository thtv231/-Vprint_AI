



import asyncio
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig


# =====================================================
# CONFIG
# =====================================================
BASE_URL = "https://vprintmachinery.com"
LIST_URL = "https://vprintmachinery.com/san-pham.html&p={}"

MAX_PAGE = 20
CONCURRENT = 8
OUTPUT = "vprint_products_clean.csv"


# =====================================================
# UTILS
# =====================================================
def clean_text(x):
    if not x:
        return ""
    return re.sub(r"\s+", " ", x.replace("\xa0", " ")).strip()


# =====================================================
# LIST PAGE
# =====================================================
def parse_list(html, page):

    soup = BeautifulSoup(html, "lxml")

    items = []

    for it in soup.select(".wap_item .item"):

        a = it.select_one(".sp_name a")

        if not a:
            continue

        items.append({
            "name": clean_text(a.get_text()),
            "link": urljoin(BASE_URL, a["href"]),
            "page": page
        })

    return items


# =====================================================
# ⭐ PRODUCT IMAGE PARSER
# chỉ lấy ảnh trong .wap_pro .zoom_slick
# =====================================================
def parse_product_images(soup):

    images = set()

    container = soup.select_one(".wap_pro .zoom_slick")

    if not container:
        return []

    for img in container.select("img"):

        src = img.get("src")

        if not src:
            continue

        src_lower = src.lower()

        # bỏ thumbnail
        if "thumb" in src_lower or "100x80" in src_lower:
            continue

        # chỉ lấy ảnh sản phẩm
        if (
            "upload/sanpham" in src_lower
            or "upload/hinhthem" in src_lower
            or "upload/images" in src_lower
        ):

            images.add(urljoin(BASE_URL, src))

    return sorted(images)


# =====================================================
# SPECS PARSER
# =====================================================
def parse_specs(soup):

    container = soup.select_one(".tab_2") or soup

    specs = {}

    for table in container.select("table"):

        rows = table.select("tr")

        if not rows:
            continue

        headers = [
            clean_text(c.get_text(" ", strip=True))
            for c in rows[0].find_all(["td", "th"])
        ]

        # ===== 2 columns =====
        if len(headers) == 2:

            for r in rows:

                td = r.find_all("td")

                if len(td) != 2:
                    continue

                k = clean_text(td[0].get_text())
                v = clean_text(td[1].get_text())

                if k and v:
                    specs[k] = v

        # ===== multi column =====
        elif len(headers) > 2:

            models = headers[1:]
            model_specs = {m: {} for m in models}

            for r in rows[1:]:

                cols = r.find_all("td")

                if len(cols) != len(headers):
                    continue

                key = clean_text(cols[0].get_text())

                for i, m in enumerate(models):

                    val = clean_text(cols[i+1].get_text())

                    if val:
                        model_specs[m][key] = val

            specs.update(model_specs)

    # UL specs
    for li in container.select("li"):

        t = clean_text(li.get_text())

        if ":" in t:
            k, v = t.split(":", 1)
            specs[clean_text(k)] = clean_text(v)

    return specs


# =====================================================
# DETAIL PAGE
# =====================================================
def parse_detail(html, url, base):

    soup = BeautifulSoup(html, "lxml")

    info = soup.select_one(".product_info")

    if not info:
        return None

    def t(css):

        el = soup.select_one(css)

        return clean_text(el.get_text(" ", strip=True)) if el else ""

    name = t(".ten")
    price = t(".gia")

    # SKU
    sku = ""

    for li in info.select("li"):

        txt = clean_text(li.get_text())

        if "Mã sản phẩm" in txt:
            sku = txt.split(":")[-1]

    # views
    views = ""

    for li in info.select("li"):

        if "Lượt xem" in li.get_text():
            views = clean_text(li.get_text().split(":")[-1])

    # short description
    short_desc = ""

    lis = info.select("li")

    if len(lis) >= 2:
        short_desc = clean_text(lis[-2].get_text())

    specs = parse_specs(soup)

    docs = [
        urljoin(BASE_URL, a["href"])
        for a in soup.select(".tab_3 a[href]")
    ]

    # ⭐ chỉ lấy ảnh máy
    images = parse_product_images(soup)

    return {
        **base,
        "url": url,
        "name": name,
        "sku": sku,
        "price": price,
        "views": views,
        "short_desc": short_desc,
        "description": t(".tab_0"),
        "features": t(".tab_1"),
        "specs": json.dumps(specs, ensure_ascii=False),
        "documents": ", ".join(docs),
        "images": ", ".join(images)
    }


# =====================================================
# FETCH DETAIL
# =====================================================
async def fetch_detail(crawler, product, cfg, sem):

    async with sem:

        res = await crawler.arun(url=product["link"], config=cfg)

        if res.success:
            return parse_detail(res.html, product["link"], product)

        return None


# =====================================================
# MAIN
# =====================================================
async def main():

    browser_cfg = BrowserConfig(headless=True)

    list_cfg = CrawlerRunConfig(wait_for="css:.wap_item")
    detail_cfg = CrawlerRunConfig(wait_for="css:.product_info")

    async with AsyncWebCrawler(config=browser_cfg) as crawler:

        products = []

        # -------- LIST --------
        for page in range(1, MAX_PAGE + 1):

            url = BASE_URL + "/san-pham.html" if page == 1 else LIST_URL.format(page)

            res = await crawler.arun(url=url, config=list_cfg)

            if not res.success:
                break

            products.extend(parse_list(res.html, page))

        print("Total products:", len(products))

        # -------- DETAIL --------
        sem = asyncio.Semaphore(CONCURRENT)

        tasks = [
            fetch_detail(crawler, p, detail_cfg, sem)
            for p in products
        ]

        rows = []

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):

            r = await coro

            if r:
                rows.append(r)

        df = pd.DataFrame(rows)

        df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

        print("\n✅ DONE →", OUTPUT)


# =====================================================
if __name__ == "__main__":

    asyncio.run(main())