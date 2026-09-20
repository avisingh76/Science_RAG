"""
ingest.py  —  Ingest one or ALL PDFs from the data/pdfs/ folder.

Usage:
  python -m pipeline.ingest                          # all PDFs
  python -m pipeline.ingest --pdf data/pdfs/book.pdf # single PDF

Changes from v2:
  - BM25 index saved to data/bm25_index.pkl after every ingest
  - BM25 built from all chunks in DB (same as FAISS rebuild logic)
"""

import argparse
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
PDF_DIR    = BASE_DIR / "data" / "pdfs"
DATA_DIR   = BASE_DIR / "data"
FAISS_PATH = DATA_DIR / "faiss_index.bin"
META_PATH  = DATA_DIR / "chunks_meta.pkl"
BM25_PATH  = DATA_DIR / "bm25_index.pkl"
DB_PATH    = DATA_DIR / "metadata.db"

PDF_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
ST_MODEL      = "all-MiniLM-L6-v2"


# ── Text helpers ───────────────────────────────────────────────────────────
def is_clean_text(text: str) -> bool:
    if not text or len(text) < 50:
        return False
    printable = sum(1 for c in text if 32 <= ord(c) < 127)
    return (printable / len(text)) > 0.80


def extract_text_by_page(pdf_path: Path) -> list[dict]:
    reader  = PdfReader(str(pdf_path))
    pages   = []
    skipped = 0
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if is_clean_text(text):
            pages.append({"page": i + 1, "text": text})
        else:
            skipped += 1
    print(f"    Pages extracted: {len(pages)}  (skipped {skipped} garbled/blank)")
    return pages


def chunk_text(pages: list[dict], source: str) -> list[dict]:
    chunks = []
    for page_data in pages:
        text     = page_data["text"]
        page_num = page_data["page"]
        start    = 0
        while start < len(text):
            end         = start + CHUNK_SIZE
            chunk       = text[start:end]
            last_period = chunk.rfind(". ")
            if last_period > CHUNK_SIZE // 2:
                chunk = chunk[: last_period + 1]
            chunk = chunk.strip()
            if len(chunk) > 50 and is_clean_text(chunk):
                chunks.append({
                    "page":    page_num,
                    "source":  source,
                    "text":    chunk,
                    "preview": chunk[:120] + "...",
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    print(f"    Chunks created: {len(chunks)}")
    return chunks


# ── Database helpers ───────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            source    TEXT,
            page      INTEGER,
            preview   TEXT,
            text      TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            source  TEXT,
            page    INTEGER,
            preview TEXT,
            UNIQUE(source, page)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            filename    TEXT PRIMARY KEY,
            num_chunks  INTEGER,
            num_pages   INTEGER,
            ingested_at TEXT
        )
    """)
    conn.commit()


def delete_source_chunks(conn: sqlite3.Connection, source: str):
    conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
    conn.execute("DELETE FROM pages  WHERE source = ?", (source,))
    conn.execute("DELETE FROM sources WHERE filename = ?", (source,))
    conn.commit()


def save_chunks_to_db(conn: sqlite3.Connection, chunks: list[dict]):
    rows = [(c["source"], c["page"], c["preview"], c["text"]) for c in chunks]
    conn.executemany(
        "INSERT INTO chunks (source, page, preview, text) VALUES (?,?,?,?)", rows
    )
    page_rows = list({
        (c["source"], c["page"], c["preview"])
        for i, c in enumerate(chunks) if i % 3 == 0
    })
    conn.executemany(
        "INSERT OR IGNORE INTO pages (source, page, preview) VALUES (?,?,?)", page_rows
    )
    conn.execute(
        "INSERT OR REPLACE INTO sources (filename, num_chunks, num_pages, ingested_at) "
        "VALUES (?,?,?,?)",
        (
            chunks[0]["source"],
            len(chunks),
            len(set(c["page"] for c in chunks)),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


# ── FAISS + BM25 rebuild ───────────────────────────────────────────────────
def rebuild_faiss(conn: sqlite3.Connection, model: SentenceTransformer):
    """Re-embed ALL chunks and rebuild FAISS + BM25 indexes from scratch."""
    rows = conn.execute(
        "SELECT chunk_id, source, page, preview, text FROM chunks ORDER BY chunk_id"
    ).fetchall()

    if not rows:
        print("  No chunks in DB — skipping index rebuild.")
        return []

    all_chunks = [
        {"chunk_id": r[0], "source": r[1], "page": r[2], "preview": r[3], "text": r[4]}
        for r in rows
    ]
    texts = [c["text"] for c in all_chunks]

    # ── FAISS (semantic) ───────────────────────────────────────────────────
    print(f"  Embedding {len(texts)} chunks for FAISS...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"  FAISS index saved  ({index.ntotal} vectors, dim={embeddings.shape[1]})")

    # ── BM25 (keyword) ─────────────────────────────────────────────────────
    print("  Building BM25 index...")
    tokenized = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in all_chunks]}, f)
    print(f"  BM25 index saved → {BM25_PATH}")

    # ── Chunk metadata ─────────────────────────────────────────────────────
    with open(META_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"  Chunk metadata saved → {META_PATH}  ({len(all_chunks)} chunks)")

    return all_chunks


# ── Main ingest ────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: Path, conn: sqlite3.Connection, model: SentenceTransformer):
    source = pdf_path.name
    print(f"\n  📄 Ingesting: {source}")
    pages = extract_text_by_page(pdf_path)
    if not pages:
        print(f"  ⚠️  No readable text in {source} — skipping.")
        return
    chunks = chunk_text(pages, source)
    delete_source_chunks(conn, source)
    save_chunks_to_db(conn, chunks)
    print(f"  ✅ {source} saved to DB  ({len(chunks)} chunks, {len(pages)} pages)")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Science RAG")
    parser.add_argument("--pdf", type=str, default=None)
    args = parser.parse_args()

    print("\n=== Science RAG — Multi-PDF Ingestion Pipeline ===\n")

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(ST_MODEL)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"ERROR: File not found: {pdf_path}")
            conn.close()
            return
        ingest_pdf(pdf_path, conn, model)
    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {PDF_DIR}")
            conn.close()
            return
        print(f"Found {len(pdfs)} PDF(s): {[p.name for p in pdfs]}\n")
        for pdf_path in pdfs:
            ingest_pdf(pdf_path, conn, model)

    print("\nRebuilding FAISS + BM25 indexes...")
    rebuild_faiss(conn, model)
    conn.close()

    print("\n✅ Ingestion complete!")
    print("   Next: uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()