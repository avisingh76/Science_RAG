"""
main.py  —  FastAPI backend (Admin Dashboard endpoints added).

New admin endpoints (all require X-API-Key header):
  GET    /admin/stats              → Overview counts
  GET    /admin/users              → All registered users
  GET    /admin/sessions           → All sessions across all users
  DELETE /admin/books/{filename}   → Remove a book from vector store + DB

All existing endpoints retained.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
try:
    import python_multipart  # noqa
except ImportError:
    pass
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from agent.graph import run_agent, generate_session_title

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("science_rag.api")


# ── Env validation ─────────────────────────────────────────────────────────
def _validate_env():
    required = ["OPENROUTER_API_KEY", "OPENROUTER_MODEL", "APP_API_KEY",
                "DATABASE_URL", "JWT_SECRET"]
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

_validate_env()

APP_API_KEY        = os.getenv("APP_API_KEY")
DATABASE_URL       = os.getenv("DATABASE_URL")
JWT_SECRET         = os.getenv("JWT_SECRET")
JWT_ALGORITHM      = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

# ── Password hashing ───────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# ── CORS ───────────────────────────────────────────────────────────────────
_raw_origins    = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PDF_DIR  = BASE_DIR / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True, echo=False,
)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    SERIAL PRIMARY KEY,
                email      TEXT UNIQUE NOT NULL,
                username   TEXT NOT NULL,
                password   TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id    INTEGER,
                created_at TEXT
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id         SERIAL PRIMARY KEY,
                session_id TEXT,
                role       TEXT,
                content    TEXT,
                sources    TEXT,
                created_at TEXT
            )
        """))
    logger.info("Database tables verified / created.")


# ── Rate limiter ────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await engine.dispose()


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Science RAG API",
    description="Agentic RAG — multi-PDF + user accounts + admin",
    version="5.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ── Auth helpers ───────────────────────────────────────────────────────────
def _create_token(user_id: int, email: str) -> str:
    expire  = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "email": email, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": int(payload["sub"]), "email": payload["email"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def _check_admin(request: Request):
    key = request.headers.get("X-API-Key", "")
    if not key or key != APP_API_KEY:
        logger.warning("Unauthorized admin request from %s", request.client.host)
        raise HTTPException(status_code=401, detail="Admin API key required.")


# ── Pydantic models ────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email:    EmailStr
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
    email:        str


class ChatRequest(BaseModel):
    message:    str        = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, max_length=200)


class ChatResponse(BaseModel):
    session_id: str
    answer:     str
    sources:    list[str]
    created_at: str


class MessageRecord(BaseModel):
    role:       str
    content:    str
    sources:    list[str]
    created_at: str


# ── Public endpoints ───────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Science RAG API", "version": "5.0.0"}


@app.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest):
    hashed = pwd_context.hash(body.password)
    now    = datetime.utcnow().isoformat()
    async with get_db() as db:
        existing = (await db.execute(
            text("SELECT user_id FROM users WHERE email = :email"),
            {"email": body.email},
        )).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered.")
        result = await db.execute(
    text("INSERT INTO users (email, username, password, created_at) "
         "VALUES (:email, :username, :password, :ts) RETURNING user_id"),
    {"email": body.email, "username": body.username,
     "password": hashed, "ts": now},
)
user_id = result.scalar()
    logger.info("New user registered: %s (id=%s)", body.email, user_id)
    return TokenResponse(
        access_token=_create_token(user_id, body.email),
        username=body.username, email=body.email,
    )


@app.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    async with get_db() as db:
        row = (await db.execute(
            text("SELECT user_id, username, password FROM users WHERE email = :email"),
            {"email": body.email},
        )).fetchone()
    if not row or not pwd_context.verify(body.password, row.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    logger.info("User logged in: %s", body.email)
    return TokenResponse(
        access_token=_create_token(row.user_id, body.email),
        username=row.username, email=body.email,
    )


# ── User endpoints (JWT) ───────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    user       = _get_current_user(request)
    session_id = body.session_id or generate_session_title(body.message)
    now        = datetime.utcnow().isoformat()

    async with get_db() as db:
        await db.execute(
            text("INSERT INTO sessions (session_id, user_id, created_at) "
                 "VALUES (:sid, :uid, :ts) ON CONFLICT DO NOTHING"),
            {"sid": session_id, "uid": user["user_id"], "ts": now},
        )
        rows = (await db.execute(
            text("SELECT role, content FROM messages WHERE session_id = :sid ORDER BY id"),
            {"sid": session_id},
        )).fetchall()
        history = [{"role": r.role, "content": r.content} for r in rows]

    try:
        result = run_agent(body.message, history=history)
    except Exception:
        logger.exception("Agent failed | session=%s", session_id)
        raise HTTPException(status_code=500, detail="Agent encountered an internal error.")

    answer, sources = result["answer"], result["sources"]

    async with get_db() as db:
        for role, content, src in [
            ("user",      body.message, "[]"),
            ("assistant", answer,       json.dumps(sources)),
        ]:
            await db.execute(
                text("INSERT INTO messages (session_id, role, content, sources, created_at) "
                     "VALUES (:sid, :role, :content, :sources, :ts)"),
                {"sid": session_id, "role": role, "content": content,
                 "sources": src, "ts": now},
            )

    return ChatResponse(
        session_id=session_id, answer=answer, sources=sources, created_at=now
    )


@app.get("/history/{session_id}", response_model=list[MessageRecord])
async def get_history(session_id: str, request: Request):
    user = _get_current_user(request)
    async with get_db() as db:
        sess = (await db.execute(
            text("SELECT user_id FROM sessions WHERE session_id = :sid"),
            {"sid": session_id},
        )).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found.")
        if sess.user_id != user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied.")
        rows = (await db.execute(
            text("SELECT role, content, sources, created_at FROM messages "
                 "WHERE session_id = :sid ORDER BY id"),
            {"sid": session_id},
        )).fetchall()
    return [
        MessageRecord(
            role=r.role, content=r.content,
            sources=json.loads(r.sources or "[]"), created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/sessions")
async def list_sessions(request: Request):
    user = _get_current_user(request)
    async with get_db() as db:
        rows = (await db.execute(
            text("SELECT session_id, created_at FROM sessions "
                 "WHERE user_id = :uid ORDER BY created_at DESC"),
            {"uid": user["user_id"]},
        )).fetchall()
    return [{"session_id": r.session_id, "created_at": r.created_at} for r in rows]


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    user = _get_current_user(request)
    async with get_db() as db:
        sess = (await db.execute(
            text("SELECT user_id FROM sessions WHERE session_id = :sid"),
            {"sid": session_id},
        )).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found.")
        if sess.user_id != user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied.")
        await db.execute(text("DELETE FROM messages WHERE session_id = :sid"), {"sid": session_id})
        await db.execute(text("DELETE FROM sessions WHERE session_id = :sid"), {"sid": session_id})
    return {"status": "deleted", "session_id": session_id}


@app.get("/sources")
async def list_sources(request: Request):
    # Accept either JWT (regular users) or X-API-Key (admin dashboard)
    api_key = request.headers.get("X-API-Key", "")
    if not (api_key and api_key == APP_API_KEY):
        _get_current_user(request)   # JWT check for regular users
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(BASE_DIR / "data" / "metadata.db"))
    rows = conn.execute(
        "SELECT filename, num_chunks, num_pages, ingested_at FROM sources "
        "ORDER BY ingested_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"filename": r[0], "num_chunks": r[1], "num_pages": r[2], "ingested_at": r[3]}
        for r in rows
    ]


# ── Admin: ingest PDF ──────────────────────────────────────────────────────
@app.post("/ingest")
async def ingest_pdf(request: Request, file: UploadFile = File(...)):
    _check_admin(request)
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    dest = PDF_DIR / file.filename
    try:
        contents = await file.read()
        with open(dest, "wb") as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
    try:
        import sqlite3 as _sqlite3
        from sentence_transformers import SentenceTransformer
        from pipeline.ingest import init_db as _init_db, ingest_pdf as _ingest_pdf, rebuild_faiss
        from agent.tools import reload_index
        model = SentenceTransformer("all-MiniLM-L6-v2")
        conn  = _sqlite3.connect(str(BASE_DIR / "data" / "metadata.db"))
        _init_db(conn)
        _ingest_pdf(dest, conn, model)
        rebuild_faiss(conn, model)
        conn.close()
        reload_index()
        logger.info("Ingest complete: %s", file.filename)
        return {"status": "ingested", "filename": file.filename}
    except Exception:
        logger.exception("Ingest failed: %s", file.filename)
        raise HTTPException(status_code=500, detail="Ingestion failed.")


# ── Admin: stats ───────────────────────────────────────────────────────────
@app.get("/admin/stats")
async def admin_stats(request: Request):
    _check_admin(request)
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(BASE_DIR / "data" / "metadata.db"))
    stats = {
        "total_users":    conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        "total_sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        "total_messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "total_books":    conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
        "total_chunks":   conn.execute("SELECT COALESCE(SUM(num_chunks),0) FROM sources").fetchone()[0],
    }
    conn.close()
    return stats


# ── Admin: all users ───────────────────────────────────────────────────────
@app.get("/admin/users")
async def admin_users(request: Request):
    _check_admin(request)
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(BASE_DIR / "data" / "metadata.db"))
    rows = conn.execute(
        "SELECT u.user_id, u.email, u.username, u.created_at, "
        "COUNT(s.session_id) as session_count "
        "FROM users u LEFT JOIN sessions s ON u.user_id = s.user_id "
        "GROUP BY u.user_id ORDER BY u.created_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"user_id": r[0], "email": r[1], "username": r[2],
         "created_at": r[3], "session_count": r[4]}
        for r in rows
    ]


# ── Admin: all sessions ────────────────────────────────────────────────────
@app.get("/admin/sessions")
async def admin_sessions(request: Request):
    _check_admin(request)
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(BASE_DIR / "data" / "metadata.db"))
    rows = conn.execute(
        "SELECT s.session_id, s.created_at, u.username, u.email, "
        "COUNT(m.id) as message_count "
        "FROM sessions s "
        "LEFT JOIN users u ON s.user_id = u.user_id "
        "LEFT JOIN messages m ON s.session_id = m.session_id "
        "GROUP BY s.session_id ORDER BY s.created_at DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return [
        {"session_id": r[0], "created_at": r[1], "username": r[2],
         "email": r[3], "message_count": r[4]}
        for r in rows
    ]


# ── Admin: delete book ─────────────────────────────────────────────────────
@app.delete("/admin/books/{filename}")
async def admin_delete_book(filename: str, request: Request):
    _check_admin(request)
    import sqlite3 as _sqlite3
    from agent.tools import reload_index

    # Remove from DB
    conn = _sqlite3.connect(str(BASE_DIR / "data" / "metadata.db"))
    conn.execute("DELETE FROM chunks  WHERE source   = ?", (filename,))
    conn.execute("DELETE FROM pages   WHERE source   = ?", (filename,))
    conn.execute("DELETE FROM sources WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()

    # Remove PDF file
    pdf_path = PDF_DIR / filename
    if pdf_path.exists():
        pdf_path.unlink()

    # Rebuild FAISS + BM25 without deleted book
    try:
        import sqlite3 as _sq
        from sentence_transformers import SentenceTransformer
        from pipeline.ingest import rebuild_faiss
        model = SentenceTransformer("all-MiniLM-L6-v2")
        conn2 = _sq.connect(str(BASE_DIR / "data" / "metadata.db"))
        rebuild_faiss(conn2, model)
        conn2.close()
        reload_index()
    except Exception:
        logger.exception("FAISS rebuild failed after book deletion.")

    logger.info("Book deleted: %s", filename)
    return {"status": "deleted", "filename": filename}
