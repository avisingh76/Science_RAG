FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV CACHE_BUST=4

COPY requirements.txt .

# Install torch CPU-only first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install all requirements
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install packages that may be missed
RUN pip install --no-cache-dir \
    "python-jose[cryptography]==3.3.0" \
    "passlib==1.7.4" \
    "argon2-cffi==23.1.0" \
    "pydantic[email]==2.9.2" \
    "rank-bm25==0.2.2" \
    "python-multipart==0.0.9"

COPY . .

# Pre-download sentence-transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
