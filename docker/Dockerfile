# ==============================================================================
# ProToPhen — CPU Inference Container
# ==============================================================================
# Build:
#   docker build -t protophen:latest -f docker/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 \
#       -v ./checkpoints:/app/checkpoints \
#       -v ./model_registry:/app/model_registry \
#       protophen:latest \
#       --checkpoint /app/checkpoints/best.pt

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- dependency layer (cached unless pyproject.toml changes) ----------
COPY pyproject.toml setup.py ./
# Minimal __init__.py so pip can resolve the package name
COPY protophen/__init__.py protophen/__init__.py

# Install CPU-only PyTorch first to avoid the large CUDA bundle,
# then install protophen with the serving optional extras.
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        fastapi "uvicorn[standard]" prometheus-client && \
    pip install --no-cache-dir -e ".[jumpcp]"

# ---------- source layer ----------------------------------------------------
COPY . .

# Re-install in editable mode now that full source is present
RUN pip install --no-cache-dir -e .

# ---------- runtime ---------------------------------------------------------
# Default config (can be overridden with -v bind mount)
COPY configs/deployment.yaml /app/configs/deployment.yaml

# Expose the default API port
EXPOSE 8000

# Health check — hits the /health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint: the serve script; users pass CLI args via docker run
ENTRYPOINT ["python", "scripts/serve.py", "--config", "configs/deployment.yaml"]

# Default command arguments (override at run-time)
CMD ["--host", "0.0.0.0", "--port", "8000"]