FROM python:3.10-slim

# ─── System deps ──────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ─── Python env tweaks ────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1           \
    # המיץ ↓ הlegacy resolver חוסך 20‑40 דקות, אבל אפשר להסירו אם תרצה
    PIP_USE_DEPRECATED=legacy-resolver

WORKDIR /app

# ─── Dependencies ────────────────────────────────────────────────────────
COPY constraints.txt requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt   # משתמש אוטומטית ב‑constraints וב‑legacy

# ─── App code ────────────────────────────────────────────────────────────
COPY . .

CMD ["gunicorn", "-k", "gevent", "-b", ":$PORT", "--timeout=900", "--log-level", "debug", "app:app"]
