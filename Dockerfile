# ── FastAPI Backend Dockerfile ─────────────────────────────────────────────
# Optimized for Railway free tier (512MB RAM)
# CrossEncoder disabled to save RAM — only FAISS + BM25 used

FROM python:3.12-slim

# System dependencies for faiss-cpu and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install dependencies
# torch CPU-only to save space and RAM
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Pre-download sentence-transformers model at build time
# so it doesn't download on first request (slow cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
