FROM python:3.10-slim

# Change APT sources to use Fastly's CDN, which is much faster globally
RUN echo "deb https://cdn-fastly.debian.org/debian/ bullseye main" > /etc/apt/sources.list && \
    echo "deb https://cdn-fastly.debian.org/debian/ bullseye-updates main" >> /etc/apt/sources.list && \
    echo "deb https://security.debian.org/debian-security bullseye-security main" >> /etc/apt/sources.list

# Original apt-get step (will now be much faster)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install uv, a high-speed package installer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# Use uv to install the requirements
RUN uv pip install --no-cache --system -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]