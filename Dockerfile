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

# Install kani-tts-2 from PyPI to pull all dependencies (nemo-toolkit etc.)
RUN pip install --no-cache-dir kani-tts-2

# Patch alleen core.py (NeMo codec op CPU, low_cpu_mem_usage)
# Andere bestanden (incl. SpeakerEmbedder) van PyPI versie bewaren
COPY .temp-wheel/kani_tts/core.py /tmp/kani_core_patch.py
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    cp /tmp/kani_core_patch.py "$SITE/kani_tts/core.py"

# Force-reinstall torchvision + torchaudio zodat ze matchen met torch 2.8.0
RUN pip install --no-cache-dir --force-reinstall \
    "torchvision>=0.23.0" \
    "torchaudio>=2.8.0" \
    --index-url https://download.pytorch.org/whl/cu126

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
