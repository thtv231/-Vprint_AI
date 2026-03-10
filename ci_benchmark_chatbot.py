import argparse
import json
import os
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List


@dataclass
class Case:
    case_id: str
    question: str
    expect_any: List[str]
    expect_all: List[str]
    forbid_any: List[str]


CASES: List[Case] = [
    Case("Q1", "Máy nào dùng để cán màng BOPP cho bao bì hộp giấy?", ["bopp", "can mang", "laminat", "lm360"], [], ["khoan giay", "bo day"]),
    Case("Q2", "Máy nào dùng để bế hộp carton sau khi in offset?", ["be", "die cut"], [], ["khoan giay"]),
    Case("Q3", "Máy nào dùng để dán hộp carton tự động?", ["dan hop", "folder gluer"], [], ["khoan giay"]),
    Case("Q4", "Máy nào dùng để ghi bản kẽm offset?", ["ctp", "computer-to-plate", "ghi ban"], [], ["inkjet"]),
    Case("Q5", "Máy nào dùng để in bao bì nhựa PE/PP?", ["flexo", "gravure", "pe", "pp"], [], ["khoan giay"]),
    Case("Q6", "Sự khác nhau giữa flexo và gravure printing là gì?", ["flexo", "gravure"], ["chi phi", "san luong"], ["top 1"]),
    Case("Q7", "Tại sao gravure printing phù hợp cho bao bì snack?", ["gravure", "toc do", "chat luong"], [], ["top 1"]),
    Case("Q8", "Trong offset printing, vai trò của dampening solution là gì?", ["dampening", "offset", "vung khong in"], [], ["top 1"]),
    Case("Q9", "Dot gain là gì trong in offset?", ["dot gain", "tram", "offset"], [], ["top 1"]),
    Case("Q10", "Sheet-fed offset và web offset khác nhau thế nào?", ["sheet-fed", "web", "offset"], [], ["top 1"]),
    Case("Q11", "Tôi muốn sản xuất bao bì snack số lượng lớn. Nên chọn flexo hay gravure?", ["gravure"], [], []),
    Case("Q12", "Tôi có máy in offset 72×102, nên đầu tư CTP loại nào?", ["ctp", "72", "102"], [], ["khoan giay"]),
    Case("Q13", "Sau khi in hộp giấy offset, cần các công đoạn nào?", ["lamination", "die", "glu"], [], []),
    Case("Q14", "Nhà máy in bao bì mềm cần những máy gì?", ["gravure", "lamination", "slitter", "bag"], [], []),
    Case("Q15", "Nếu cần in nhãn chai nước giải khát, nên dùng công nghệ gì?", ["flexo"], [], ["top 1"]),
    Case("Q16", "Flexographic printing có dùng dampening solution như offset không?", ["khong", "không"], [], ["co", "có"]),
    Case("Q17", "Inkjet printing có cần printing plate không?", ["khong", "không"], [], ["co", "có"]),
    Case("Q18", "Gravure printing có dùng laser toner không?", ["khong", "không"], [], ["co", "có"]),
    Case("Q19", "Offset printing có thể in trực tiếp lên kim loại nóng chảy không?", ["khong", "không"], [], ["co", "có"]),
    Case("Q20", "Screen printing có thể in 1000 m/phút không?", ["khong", "không"], [], ["co", "có"]),
]


def norm(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(s.split())


def contains_any(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    return any(norm(k) in text for k in keywords)


def contains_all(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    return all(norm(k) in text for k in keywords)


def contains_forbid(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    return any(norm(k) in text for k in keywords)


def post_chat(base_url: str, question: str, case_id: str, model: str, temperature: float, timeout: float) -> dict:
    payload = {
        "session_id": f"bench-{case_id}-{int(time.time() * 1000)}",
        "message": question,
        "model": model,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/docs"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except Exception:
            time.sleep(1.0)
    return False


def write_junit(path: str, results: List[dict], duration: float):
    testsuite = ET.Element(
        "testsuite",
        name="chatbot-benchmark",
        tests=str(len(results)),
        failures=str(sum(1 for r in results if not r["passed"])),
        time=f"{duration:.3f}",
    )
    for r in results:
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            classname="chatbot.benchmark",
            name=r["case_id"],
            time=f"{r['latency']:.3f}",
        )
        if not r["passed"]:
            failure = ET.SubElement(testcase, "failure", message="keyword assertion failed")
            failure.text = json.dumps(
                {
                    "question": r["question"],
                    "answer_preview": r["answer"][:700],
                    "expect_any": r["expect_any"],
                    "expect_all": r["expect_all"],
                    "forbid_any": r["forbid_any"],
                    "errors": r["errors"],
                },
                ensure_ascii=False,
                indent=2,
            )
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark chatbot answers with 20 fixed test questions.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    parser.add_argument("--output-json", default="benchmark_report.json")
    parser.add_argument("--junit-xml", default="benchmark_report.xml")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-cmd", default="python -m uvicorn main:app --host 127.0.0.1 --port 8000")
    parser.add_argument("--skip-if-no-key", action="store_true")
    args = parser.parse_args()

    if args.skip_if_no_key and not os.getenv("GROQ_API_KEY"):
        print("SKIP: GROQ_API_KEY is missing.")
        return 0

    server_proc = None
    if args.start_server:
        server_proc = subprocess.Popen(
            args.server_cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    started = time.time()
    try:
        if not wait_for_server(args.base_url, timeout=90.0):
            print("ERROR: Chatbot API server is not ready.")
            return 2

        results = []
        pass_count = 0
        for c in CASES:
            t0 = time.perf_counter()
            errors = []
            answer = ""
            try:
                data = post_chat(
                    base_url=args.base_url,
                    question=c.question,
                    case_id=c.case_id,
                    model=args.model,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                answer = str(data.get("answer", ""))
            except urllib.error.HTTPError as e:
                errors.append(f"http_error={e.code}")
            except Exception as e:
                errors.append(f"exception={e}")

            normalized_answer = norm(answer)
            if not contains_any(normalized_answer, c.expect_any):
                errors.append("missing_expect_any")
            if not contains_all(normalized_answer, c.expect_all):
                errors.append("missing_expect_all")
            if contains_forbid(normalized_answer, c.forbid_any):
                errors.append("contains_forbid_any")

            passed = len(errors) == 0
            if passed:
                pass_count += 1

            results.append(
                {
                    "case_id": c.case_id,
                    "question": c.question,
                    "answer": answer,
                    "expect_any": c.expect_any,
                    "expect_all": c.expect_all,
                    "forbid_any": c.forbid_any,
                    "passed": passed,
                    "errors": errors,
                    "latency": round(time.perf_counter() - t0, 3),
                }
            )
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {c.case_id} - {c.question}")
            if not passed:
                print(f"  errors={errors}")
                print(f"  answer={answer[:260]}")

        total = len(CASES)
        pass_rate = pass_count / total if total else 0.0
        duration = time.time() - started
        summary = {"total": total, "passed": pass_count, "failed": total - pass_count, "pass_rate": round(pass_rate, 4), "duration_sec": round(duration, 2)}

        report = {"summary": summary, "results": results}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        write_junit(args.junit_xml, results, duration)

        print(json.dumps(summary, ensure_ascii=False))
        return 0 if pass_rate >= args.min_pass_rate else 1
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except Exception:
                server_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())

