# Use NVIDIA CUDA base for GPU support, fallback to CPU works automatically
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    g++ \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/nineninesix-ai/kani-tts.git .

# Upgrade pip and install build dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir Cython

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    "nemo-toolkit[tts]==2.4.0" \
    "transformers==4.47.1" \
    scipy \
    kani-tts

# Expose the port used by the server
EXPOSE 8000

# Set environment variables for model caching
ENV HF_HOME=/root/.cache/huggingface

# Command to run the server
# We run from the examples/basic directory to ensure local modules are found
WORKDIR /app/examples/basic

# Copy our custom files
COPY server.py .
COPY config.py .
COPY index.html .

CMD ["python3", "server.py"]
