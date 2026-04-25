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

# Upgrade naar torch>=2.6 — vereist door NeMo CVE; cu118 heeft 2.6+ en werkt met driver 535
RUN pip install --no-cache-dir --force-reinstall \
    "torch>=2.6.0" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

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
