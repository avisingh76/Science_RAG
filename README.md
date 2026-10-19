# 🔬 Science RAG — Agentic RAG Assistant

An agentic Retrieval-Augmented Generation (RAG) assistant for science textbooks. Upload PDFs, ask questions, and get answers with source-page references, powered by LangGraph, FAISS + BM25, FastAPI and OpenRouter. **Live demo:** https://pacific-celebration-production-09c6.up.railway.app/

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
├── data/         pdfs/ and generated indexes / metadata.db (not in the repo)
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
git clone https://github.com/avisingh76/Science_RAG.git && cd science-rag
pip install -r requirements.txt
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
EVAL_GEN_MODEL=openai/gpt-4o-mini
EVAL_JUDGE_MODEL=anthropic/claude-haiku-5.5
```

Add your own PDFs to `data/pdfs/` (textbooks are not included in the repo), then:

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

## 📊 Evaluation
```bash
python -m evaluation.generate_testset --n 60   # LLM-generated questions from your books
python -m evaluation.eval_retrieval            # FAISS vs BM25 vs hybrid vs rerank
python -m evaluation.eval_answers --n 60       # agent + LLM-as-judge (set EVAL_JUDGE_MODEL in .env)
```

**Retrieval** (60 LLM-generated questions, one textbook, page-level):

| Method | Hit@1 | Hit@3 | MRR | ms/query |
|---|---|---|---|---|
| FAISS only | 63.3% | 83.3% | 0.761 | 105 |
| BM25 only | 48.3% | 68.3% | 0.596 | 4 |
| Hybrid (RRF) | 70.0% | 85.0% | 0.785 | 109 |
| Hybrid + rerank | 75.0% | 88.3% | 0.817 | 323 |

**Answers** (60 questions, judge: Claude Haiku 5.5): correctness 4.38/5, faithfulness 4.25/5, relevance 4.70/5, hallucination 3.3%, correct page in context 78%, mean latency 3.5 s.

**Findings:** removing duplicate chunks (2,462 to 1,231) raised FAISS Hit@10 from 90% to 100%; a prompt fix for invalid agent tool calls raised correctness from 3.6 to 4.7/5 on a 15-question pilot. Questions are LLM-generated from one textbook, so treat results as indicative.

## 🚢 Deploy on Railway
1. Push the repo to GitHub, create a Railway project, add PostgreSQL and deploy the FastAPI service (`Dockerfile`) with the env vars below
2. Deploy the Streamlit service (`Dockerfile.streamlit`) and set `API_URL`
3. Upload PDFs through the admin dashboard or `/ingest`

Env vars: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `APP_API_KEY`, `JWT_SECRET`, `DATABASE_URL`, `ALLOWED_ORIGINS` (backend); `API_URL`, `APP_API_KEY` (frontend); `ENABLE_RERANKING=false` on the free tier.

## 📄 License
MIT — free to use and modify.
