FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install uv
RUN pip install --no-cache-dir uv

# Use uv to install all packages (this is the fast part)
RUN uv pip install --system --no-cache -r /app/requirements.txt

# Copy your application code
COPY . /app

ENV PYTHONUNBUFFERED=1

# This is the corrected line:
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]