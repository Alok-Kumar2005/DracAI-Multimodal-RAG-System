FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install uv
RUN pip install --no-cache-dir uv

# Use uv to install all packages
RUN uv pip install --system --no-cache -r /app/requirements.txt

# Copy your application code
COPY . /app

ENV PYTHONUNBUFFERED=1

# Use "python -m" to avoid "command not found" errors
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]