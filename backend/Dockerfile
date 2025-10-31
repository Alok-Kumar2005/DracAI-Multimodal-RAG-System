FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements first
COPY backend/requirements.txt /code/backend/requirements.txt

# Install uv
RUN pip install --no-cache-dir uv

# Install all packages
RUN uv pip install --system --no-cache -r /code/backend/requirements.txt

# Copy the backend directory
COPY backend/ /code/backend/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/code

# Run uvicorn
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]