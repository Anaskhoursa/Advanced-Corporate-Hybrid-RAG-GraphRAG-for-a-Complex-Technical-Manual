FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/app/.cache/huggingface FASTEMBED_CACHE_PATH=/app/.cache/fastembed
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"
COPY scripts/ ./scripts/
COPY data/ ./data/
CMD ["python", "scripts/run_assistant.py"]
