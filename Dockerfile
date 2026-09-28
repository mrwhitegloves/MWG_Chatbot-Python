FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.cloudrun.txt .
RUN pip install -r requirements.cloudrun.txt

COPY main.py knowledge_data.py seed_knowledge.py ./

# Cloud Run provides PORT (default 8080)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
