# CUDA 12.6 base — install Python + torch 2.8.0 manually
FROM nvidia/cuda:12.6.3-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg libsndfile1 \
    git gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3 available as python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install torch 2.8.0 with CUDA 12.6 support
RUN pip install --no-cache-dir \
    "torch>=2.8.0" \
    "torchaudio>=2.8.0" \
    --index-url https://download.pytorch.org/whl/cu126

# Install kani-tts-2 (includes nemo-toolkit, transformers, etc.)
RUN pip install --no-cache-dir kani-tts-2

# nemo-toolkit can overwrite torchvision/torchaudio — force correct versions
RUN pip install --no-cache-dir --force-reinstall \
    "torchaudio>=2.8.0" \
    --index-url https://download.pytorch.org/whl/cu126

# Server dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    scipy \
    python-multipart

WORKDIR /app

# Directory for saved voice embeddings
RUN mkdir -p /app/voices

# Copy custom server + UI
COPY server.py .
COPY index.html .

EXPOSE 8000

ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "server.py"]
