# =============================================================================
# Dockerfile
# Multi-stage build for GCP Billing Cost Forecaster
#
# Stage 1 (builder) — installs all Python dependencies
# Stage 2 (runtime) — copies only what's needed, keeps image lean
#
# Build:   docker build -t gcp-billing-forecaster .
# Run:     docker run -p 5000:5000 gcp-billing-forecaster
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# System deps needed to compile some packages (LightGBM, Prophet)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first — leverages Docker layer cache
# If requirements.txt hasn't changed, this layer is reused on rebuild
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Non-root user for security
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY app.py          ./app.py
COPY setup.py        ./setup.py
COPY requirements.txt ./requirements.txt
COPY src/            ./src/
COPY configs/        ./configs/
COPY templates/      ./templates/
COPY static/         ./static/

# artifacts/ is mounted at runtime via docker-compose volume —
# NOT baked into the image (model.pkl changes with each retrain)
RUN mkdir -p artifacts logs

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_DEBUG=false

EXPOSE 5000

# Health check — hits /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run with gunicorn in production
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]