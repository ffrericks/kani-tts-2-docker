# Use NVIDIA CUDA runtime for GPU support
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.8+ first (kani-tts requires >=2.8.0)
# Using the default index which has the latest versions
RUN pip3 install --no-cache-dir "torch>=2.8.0" torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install kani-tts - this will automatically install nemo-toolkit==2.4.0
# and compatible transformers as its own dependencies
RUN pip3 install --no-cache-dir kani-tts

# Install server dependencies
RUN pip3 install --no-cache-dir fastapi uvicorn scipy

# Clone the kani-tts repository to get the server source files
RUN git clone https://github.com/nineninesix-ai/kani-tts.git /kani-tts-repo

# Set working directory to the basic example
WORKDIR /app

# Copy server files from cloned repo and our custom overrides
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
