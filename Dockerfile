FROM python:3.11-slim

WORKDIR /app

# Disable .pyc generation and Python output buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies

# ffmpeg   — GIF conversion
# libmagic — used by Pillow for format detection
# ca-certs — required for HTTPS calls
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    \
    # Install Colab CLI
    && pip install --no-cache-dir google-colab-cli \
    && colab version

# Python application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Python application source code
COPY . .

CMD ["python", "main.py"]
