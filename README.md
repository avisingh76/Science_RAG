# 🔬 Science RAG — Agentic RAG Assistant

An agentic Retrieval-Augmented Generation (RAG) assistant for science textbooks. Upload PDFs, ask questions, and get answers with page citations, powered by LangGraph, FAISS + BM25, FastAPI and OpenRouter.

**Live demo:** [add link]

## ✨ Features

- **Multi-PDF support** with automatic index rebuild and hot reload after each upload
- **Hybrid search:** FAISS (semantic) + BM25 (keyword) fused with Reciprocal Rank Fusion
- **Optional reranking** with a CrossEncoder (`ENABLE_RERANKING=false` saves ~800 MB RAM)
- **Agentic pipeline:** LangGraph agent chooses between textbook search, read-only SQL and Wikipedia
- **User accounts** (JWT + Argon2), saved session history, and an **admin dashboard**
- **Evaluation suite** with an LLM-generated test set, retrieval benchmark and LLM-as-judge

## 🏗️ Project Structure

```
science_rag/
├── agent/        graph.py (LangGraph agent), tools.py (RAG, SQL, Wikipedia tools)
├── api/          main.py (FastAPI backend)
├── ingestion/    ingest.py (PDF -> chunks -> FAISS + BM25 indexes)
├── ui/           app.py (chat UI), admin.py (admin dashboard)
├── evaluation/   generate_testset.py, eval_retrieval.py, eval_answers.py
├── data/         pdfs/, faiss_index.bin, bm25_index.pkl, chunks_meta.pkl, metadata.db
├── Dockerfile, Dockerfile.streamlit, requirements.txt, .env
```

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + SQLAlchemy (async), SQLite (local) / PostgreSQL (production) |
| Agent / LLM | LangGraph + LangChain, OpenRouter (any model) |
| Retrieval | `all-MiniLM-L6-v2` + FAISS, BM25, CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| Frontend | Streamlit (chat + admin) |
| Auth / Security | JWT, Argon2, rate limiting, SELECT-only SQL tool (sqlparse whitelist) |
| Deploy | Docker, Railway |

## 🚀 Local Setup

```bash
git clone https://github.com/your-username/science-rag.git && cd science-rag
pip install -r requirements.txt
pip install "pydantic[email]" "python-jose[cryptography]" "passlib[bcrypt]" argon2-cffi rank-bm25 slowapi tenacity sqlparse aiosqlite asyncpg python-multipart
```

Create `.env` (generate secrets with `python -c "import secrets; print(secrets.token_hex(32))"`):

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini
APP_API_KEY=your_secret_admin_key
JWT_SECRET=your_jwt_secret
DATABASE_URL=sqlite+aiosqlite:///./data/metadata.db
ALLOWED_ORIGINS=http://localhost:8501
API_URL=http://localhost:8000
ENABLE_RERANKING=true
EVAL_JUDGE_MODEL=nvidia/nemotron-3.5-lightning:free
```

Put PDFs in `data/pdfs/`, then:

```bash
python -m ingestion.ingest                         # build indexes
uvicorn api.main:app --reload --port 8000          # backend
streamlit run ui/app.py                            # chat UI
streamlit run ui/admin.py --server.port 8502       # admin dashboard
```

## 🔌 API Endpoints

| Access | Endpoints |
|---|---|
| Public | `GET /health`, `POST /register`, `POST /login` |
| User (JWT) | `POST /chat`, `GET /sessions`, `GET /history/{id}`, `DELETE /sessions/{id}`, `GET /sources` |
| Admin (`X-API-Key`) | `POST /ingest`, `GET /admin/stats`, `GET /admin/users`, `GET /admin/sessions`, `DELETE /admin/books/{filename}` |

## 🤖 Agent Pipeline

```
Question -> LangGraph agent -> tool selection:
   rag_search: FAISS + BM25 -> RRF fusion -> (optional) CrossEncoder rerank -> top 3 passages
   sql_query: read-only metadata queries   |   wikipedia_search: real-world context
-> Final answer with source PDF and page citations
```

## 📊 Evaluation

```bash
python -m evaluation.generate_testset --n 60   # LLM-generated questions from your books
python -m evaluation.eval_retrieval            # FAISS vs BM25 vs hybrid vs rerank
python -m evaluation.eval_answers --n 30       # agent + LLM-as-judge (set EVAL_JUDGE_MODEL in .env)
```

**Retrieval** (60 LLM-generated questions, one textbook, page-level):

| Method | Hit@1 | Hit@3 | MRR | ms/query |
|---|---|---|---|---|
| FAISS only | 63.3% | 83.3% | 0.761 | 101 |
| BM25 only | 48.3% | 68.3% | 0.596 | 7 |
| Hybrid (RRF) | 70.0% | 85.0% | 0.785 | 108 |
| Hybrid + rerank | 75.0% | 88.3% | 0.817 | 409 |

**Answers** (30 questions, judge: [model]): correctness [x.x]/5, faithfulness [x.x]/5, hallucination rate [x]%.

Evaluation exposed duplicated chunks in the index; removing them (2,462 to 1,231) raised FAISS Hit@10 from 90% to 100%. Results come from a small LLM-generated test set, so treat them as indicative.

## 🚢 Deploy on Railway

1. Push the repo to GitHub, create a Railway project and add PostgreSQL
2. Deploy the FastAPI service (`Dockerfile`) and set the env vars below
3. Deploy the Streamlit service (`Dockerfile.streamlit`) and set `API_URL`
4. Upload PDFs through the admin dashboard or `/ingest`

Env vars: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `APP_API_KEY`, `JWT_SECRET`, `DATABASE_URL`, `ALLOWED_ORIGINS` (backend); `API_URL`, `APP_API_KEY` (frontend); `ENABLE_RERANKING=false` on the free tier.

## 📄 License

MIT License — free to use and modify.
