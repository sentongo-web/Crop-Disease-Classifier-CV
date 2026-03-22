# ─── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch (smaller image for deployment)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        numpy pandas scikit-learn matplotlib seaborn Pillow \
        albumentations opencv-python-headless tqdm \
        fastapi "uvicorn[standard]" python-multipart aiofiles jinja2 \
        pyyaml python-dotenv requests

# ─── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/     ./src/
COPY app/     ./app/
COPY config.yaml .

# Create models directory
RUN mkdir -p models

# Copy trained model if it exists
COPY models/best_model.pth ./models/best_model.pth 2>/dev/null || true

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

ENV MODEL_PATH="/app/models/best_model.pth" \
    HOST="0.0.0.0" \
    PORT="8000" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE="1" \
    PYTHONUNBUFFERED="1"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
