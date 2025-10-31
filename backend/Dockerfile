# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
