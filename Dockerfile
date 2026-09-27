# Dockerfile for MediaPipe Bouldering Pose Analysis Streamlit App
# Supports both CPU and GPU (with NVIDIA Container Toolkit)

# ============================================================
# BASE IMAGE
# ============================================================
FROM python:3.11-slim as base

# Install system dependencies
# Note: Package names updated for Debian 13 (trixie) / Python 3.11-slim
# Added libegl1 and libgles2 for MediaPipe GPU/EGL support
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# APPLICATION CODE
# ============================================================
COPY . .

# Create necessary directories
RUN mkdir -p /app/.streamlit /app/temp_uploads /app/temp_outputs

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================
# Copy streamlit config
COPY .streamlit/config.toml /app/.streamlit/config.toml

# ============================================================
# NGROK TUNNEL (OPTIONAL)
# ============================================================
# Install ngrok for tunneling (optional, for external access)
RUN wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -O /tmp/ngrok.tgz && \
    tar -xzf /tmp/ngrok.tgz -C /usr/local/bin && \
    rm /tmp/ngrok.tgz

# ============================================================
# ENTRYPOINT
# ============================================================
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# ============================================================
# GPU VARIANT (uncomment to use)
# ============================================================
# FROM nvidia/cuda:12.2-runtime-ubuntu22.04 as gpu-base
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.11 python3.11-venv python3-pip \
#     ffmpeg \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     libgomp1 \
#     wget \
#     curl \
#     && rm -rf /var/lib/apt/lists/*
# 
# WORKDIR /app
# 
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt
# 
# COPY . .
# RUN mkdir -p /app/.streamlit /app/temp_uploads /app/temp_outputs
# COPY .streamlit/config.toml /app/.streamlit/config.toml
# 
# RUN wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -O /tmp/ngrok.tgz && \
#     tar -xzf /tmp/ngrok.tgz -C /usr/local/bin && \
#     rm /tmp/ngrok.tgz
# 
# EXPOSE 8501
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8501/_stcore/health || exit 1
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# ============================================================
# DEVELOPMENT VARIANT
# ============================================================
# FROM base as dev
# 
# # Install development tools
# RUN pip install --no-cache-dir \
#     pytest \
#     black \
#     flake8 \
#     mypy \
#     jupyter
# 
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]