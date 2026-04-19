# Kani-TTS Docker for CasaOS

A dockerized version of [Kani-TTS](https://github.com/nineninesix-ai/kani-tts), ready to run on **CasaOS** and Proxmox environments with NVIDIA GPU passthrough.

Includes a premium Web UI for easy speech generation directly in your browser.

---

## 🚀 Quick Install (CasaOS)

1.  Open **CasaOS** → **App Store** → **Custom Install** (top right)
2.  Paste the following Docker Compose:

```yaml
version: '3.8'
services:
  kani-tts:
    image: ghcr.io/ffrericks/kani-tts-docker:latest
    container_name: kani-tts
    ports:
      - "8000:8000"
    volumes:
      - / hf_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    environment:
      - TZ=Europe/Amsterdam
volumes:
  hf_cache:
```

3.  Click **Install**
4.  Access the Web UI at `http://<your-server-ip>:8000`

> **Note:** The first start takes a few minutes to download the AI model (~1GB). This only happens once thanks to the persistent volume.

---

## 🛠 Features

- **Pre-built image** — No local compilation needed, pulled directly from GitHub Container Registry
- **GPU Acceleration** — Pre-configured for NVIDIA GPUs via Docker runtime (4GB+ VRAM recommended)
- **Persistent Model Cache** — Models are stored in a Docker volume, no re-downloads on restart
- **Modern Web UI** — Built-in studio interface for speech generation in your browser
- **FastAPI backend** — `/tts` and `/stream-tts` endpoints for integration with other tools

---

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB |

The multilingual `kani-tts-370m` model is used by default — it fits comfortably within 4GB VRAM.

---

## 📦 Supported Models

By default this image uses `nineninesix/kani-tts-370m` (English, Spanish, Chinese, German, Korean, Arabic).

To change the model, edit `config.py` and rebuild, or override the `MODEL_NAME` environment variable.

---

## 📜 License & Attribution

This project is licensed under the **Apache License 2.0**.

This is a dockerized wrapper around the excellent [Kani-TTS](https://github.com/nineninesix-ai/kani-tts) by **nineninesix-ai**. All credits for the underlying TTS architecture and models go to the original authors.

Please adhere to the [Ethical Usage Guidelines](https://github.com/nineninesix-ai/kani-tts?tab=readme-ov-file) of the original project — no impersonation, hate speech, or harmful content.
