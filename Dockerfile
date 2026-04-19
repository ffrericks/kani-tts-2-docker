# Zelfde bewezen base image als kani-tts v1
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 \
    git gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Upgrade torch naar 2.8.0+ inclusief torchvision (kani-tts-2 vereist dit)
RUN pip install --no-cache-dir \
    "torch>=2.8.0" \
    "torchvision>=0.23.0" \
    "torchaudio>=2.8.0" \
    --index-url https://download.pytorch.org/whl/cu126

# Install kani-tts-2 (nemo-toolkit kan torchvision overschrijven)
RUN pip install --no-cache-dir kani-tts-2

# Force-reinstall torchvision + torchaudio zodat ze matchen met torch 2.8.0
RUN pip install --no-cache-dir --force-reinstall \
    "torchvision>=0.23.0" \
    "torchaudio>=2.8.0" \
    --index-url https://download.pytorch.org/whl/cu126

# Lfm2HybridConvCache zit alleen in de git versie van transformers, niet op PyPI
# ARG bust_cache voorkomt dat GitHub Actions een oude gecachede versie gebruikt
ARG TRANSFORMERS_BUILD_DATE=20260419
RUN pip install --no-cache-dir --force-reinstall \
    "transformers @ git+https://github.com/huggingface/transformers.git"

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
