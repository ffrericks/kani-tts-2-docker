# Use official PyTorch image with CUDA 12.4 - torch is pre-installed!
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install kani-tts - torch is already pre-installed in this base image
RUN pip install --no-cache-dir kani-tts

# nemo-toolkit installs an incompatible torchvision - force the correct version
# torchvision 0.22.0 is compatible with torch 2.7.0 + CUDA 12.6
RUN pip install --no-cache-dir --force-reinstall \
    "torchvision==0.22.0" \
    "torchaudio==2.7.0" \
    --index-url https://download.pytorch.org/whl/cu126

# Install server dependencies
RUN pip install --no-cache-dir fastapi uvicorn scipy "transformers @ git+https://github.com/huggingface/transformers.git"

# Clone the kani-tts repository to get the server source files
RUN git clone https://github.com/nineninesix-ai/kani-tts.git /kani-tts-repo

# Set working directory
WORKDIR /app

# Copy server files from cloned repo
RUN cp -r /kani-tts-repo/examples/basic/* .

# Copy our custom server+UI files (overwrite defaults)
COPY server.py .
COPY config.py .
COPY index.html .

# Expose the port used by the server
EXPOSE 8000

# Set environment variables for model caching
ENV HF_HOME=/root/.cache/huggingface

CMD ["python3", "server.py"]
