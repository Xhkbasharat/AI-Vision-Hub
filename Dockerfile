
# ==========================================
# AI VISION HUB - DOCKER CONFIGURATION
# ==========================================

FROM python:3.11-slim

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=5000

# ==========================================
# WORKING DIRECTORY
# ==========================================

WORKDIR /app

# ==========================================
# SYSTEM DEPENDENCIES
# ==========================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# PYTHON DEPENDENCIES
# ==========================================

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# ==========================================
# APPLICATION FILES
# ==========================================

COPY . .

# ==========================================
# CREATE APPLICATION DIRECTORIES
# ==========================================

RUN mkdir -p \
    data/uploads \
    data/known_faces \
    models \
    logs

# ==========================================
# NON-ROOT USER
# ==========================================

RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app

USER appuser

# ==========================================
# NETWORK CONFIGURATION
# ==========================================

EXPOSE 5000

# ==========================================
# HEALTH CHECK
# ==========================================

HEALTHCHECK --interval=30s --timeout=10s \
    --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5)" || exit 1

# ==========================================
# START APPLICATION
# ==========================================

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 180 app:app"]
