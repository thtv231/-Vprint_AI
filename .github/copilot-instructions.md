## Purpose

This repository implements VPRINT Sales AI — a Retrieval-Augmented Generation (RAG) chatbot for Vietnamese-language sales and technical support around printing and packaging machines. The instructions below help AI coding agents be immediately productive: where core logic lives, how data flows, important guardrails, and concrete commands to run and test locally.

## Quick map (where to look)
- Entry points:
  - `chatbot_groq.py` — primary Streamlit UI (modern UI + router). Good for UI changes and interactive debugging.
  - `main.py` — FastAPI server exposing `/api/chat`. Use this for headless API and automated tests.
- Core logic and helpers:
  - `chatbot_vprint_hybrid_local.py` — most reusable functions: `load_csv_docs`, `format_context`, `build_rag_messages`, `build_direct_messages`, `build_specs_answer`, `parse_images`, intent detectors and router guards. ALWAYS check here before changing retrieval or prompt logic.
- Ingestion & DB
  - `ingest_chroma_local.py`, `ingest_book_local.py`, `ingest_book_local.py` (colab variant) — scripts to create/refresh Chroma DB in `vprint_agentic_db_local`.
  - Source CSV: `vprint_products_clean.csv` (canonical product catalog used to build vectors).

## Architecture & data flow (short)
1. CSV rows are converted to Document chunks (`chatbot_vprint_hybrid_local.load_csv_docs`) and stored into Chroma (persist dir `vprint_agentic_db_local`).
2. At runtime the system builds an EnsembleRetriever (BM25 + Vector) and an optional fast vector router. Retriever -> `format_context` -> prompt builder -> LLM call.
3. Routing: `ROUTING_SAMPLES` + `fast_vector_route_query` (or intent detectors in `chatbot_vprint_hybrid_local.py`) decide whether to: find_machine / spec_query / normal_rag / direct_chat / out_of_domain.

## Important conventions and guardrails (project-specific)
- Language: Vietnamese is primary. Prompts & UI copy are Vietnamese; preserve punctuation and formatting when editing.
- Prompt invariants:
  - All RAG/direct prompts append the same suggestion block marker exactly as: `💡 **Có thể bạn quan tâm:**` followed by three `-` bullet items. Many post-processors rely on this marker (`parse_and_clean_suggestions`). Do not change this phrasing without updating those parsers.
  - System prompts in `chatbot_vprint_hybrid_local.py` explicitly instruct: “TUYỆT ĐỐI KHÔNG BỊA ĐẶT.” Agents must follow that and not fabricate machines or specs.
  - Specs must be presented as a Markdown table where code expects it (see `format_specs_to_table` / `build_specs_answer`). If you change specs formatting, update all callers.
- CSV → Document mapping relies on specific column positions (DEFAULT_COLUMNS). Avoid renaming CSV header columns unless you modify `DEFAULT_COLUMNS` and `load_csv_docs` accordingly.
- Images: `parse_images` accepts JSON arrays, comma/semicolon/pipe-separated lists. Metadata `images` often stored as JSON string.
- Retrieval: EnsembleRetriever weights are 0.5/0.5 by default; tests rely on hybrid search behavior. Adjust weights consciously and run `test_semantic_search_local.py` after changes.

## Environment & keys
- Put API keys in `.env` at repo root. Common keys used:
  - `GROQ_API_KEY` — used by Groq LLM bindings (ChatGroq).
  - `OPENAI_API_KEY` — used by scripts that call OpenAI (enrichment, tests).

## Run / debug commands (Windows / PowerShell)
- Streamlit UI (local interactive):
  - streamlit run chatbot_groq.py
- FastAPI server (headless API):
  - uvicorn main:app --reload --port 8000
- Rebuild book knowledge DB (PDF ingestion):
  - python ingest_book_local.py --pdf-path "<path-to-pdf>" --rebuild
  - Note: `ingest_book_local.py` defaults to small `--batch-size` to work on low-VRAM GPUs; pass `--batch-size` to tune.
- Rebuild product vector DB (CSV ingestion):
  - see `ingest_chroma_local.py` / commented ingestion in `ingest_chroma_local.py` — run with appropriate args; backup existing `vprint_agentic_db_local` if you need to preserve it.
- Quick hybrid-search smoke test (local):
  - python test_semantic_search_local.py --run-tests

## Tests & checks to run after edits
- After changing retriever, routing, or prompt text, run:
  - `python test_semantic_search_local.py --run-tests` (sanity-check hybrid ranking)
  - Start `main.py` and exercise `/api/chat` with representative queries for `find_machine`, `spec_query`, and `out_of_domain`.

## Patterns & code snippets agents should reuse
- Build RAG messages: see `build_rag_messages(user_query, context, history)` — it injects the dataset block and enforces the suggestion footer.
- Router: `fast_vector_route_query(...)` + `apply_router_guards(...)` — change `ROUTING_SAMPLES` or thresholds to tune routing.
- Presentation: `format_specs_to_table` + `build_specs_answer` produce Markdown tables for specs — follow this contract when formatting output.

## Non-obvious gotchas
- Do not remove or rename the suggestion footer marker `💡 **Có thể bạn quan tâm:**` — UI and parsing depend on it.
- Many modules detect device with torch (cuda/mps/cpu) and set fp16 for embeddings when available; embedding failures often stem from mismatched model kwargs—test embedding loads in a small script before bulk ingestion.
- Guardrails for safety and correctness live in multiple places (`chatbot_vprint_hybrid_local.apply_router_guards`, LangGraph tools `search_vprint_machines`), so changes that affect what documents are considered “matching” must be synced across these files.

## If you edit retrieval/ingestion
- Update `DEFAULT_COLUMNS` in `chatbot_vprint_hybrid_local.py` AND document any new column mapping in this file.
- After ingestion, validate DB exists at `vprint_agentic_db_local` and run `test_semantic_search_local.py`.

---
If anything in these instructions is unclear or you'd like me to include additional examples (e.g., exact prompt text for the router, a checklist for safe schema changes, or sample test queries), tell me which sections to expand and I'll iterate.
