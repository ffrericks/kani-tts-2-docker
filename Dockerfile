# CUDA 12.1 — compatibel met driver 535.x (max CUDA 12.2)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 \
    git gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install kani-tts-2 dependencies
RUN pip install --no-cache-dir kani-tts-2

# Upgrade naar torch 2.6.0 cu121 — vereist door NeMo (CVE-2025-32434), compatibel met driver 535
RUN pip install --no-cache-dir --force-reinstall \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu121

# nemo-toolkit installeert een oude transformers — force-reinstall naar 4.56.0+
# Lfm2HybridConvCache bestaat pas vanaf transformers 4.54.0
RUN pip install --no-cache-dir --force-reinstall \
    "transformers==4.56.0"

# Server dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    scipy \
    python-multipart

WORKDIR /app

# Directory for saved voice embeddings
RUN mkdir -p /app/voices

# Copy custom server + UI + assets
COPY server.py .
COPY index.html .
COPY public/ ./public/

EXPOSE 8001

ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "server.py"]
