# Kani-TTS Docker for CasaOS

This repository provides a dockerized version of [Kani-TTS](https://github.com/nineninesix-ai/kani-tts), optimized for **CasaOS** and Proxmox environments with NVIDIA GPU passthrough.

Features a premium Web UI and easy setup for home servers.

## 🚀 Quick Start (CasaOS)

1.  **Clone this repo** on your server:
    ```bash
    git clone https://github.com/ffrericks/kani-tts-docker.git
    cd kani-tts-docker
    ```
2.  **Import to CasaOS**:
    - Open CasaOS -> App Store -> Custom Install.
    - Click **Import** and select the `docker-compose.yml`.
3.  **Install** and open the app!

## 🛠 Features
- **GPU Acceleration**: Pre-configured for NVIDIA GPUs (4GB VRAM+).
- **Persistent Caching**: AI models are stored in a Docker volume to avoid re-downloading.
- **Modern Web UI**: Built-in studio interface for easy speech generation.
- **OpenAI Compatible API**: Use it as a backend for other AI tools.

## 📜 License
This project is licensed under the **Apache License 2.0**.

### Attribution
This project is a dockerized wrapper around the excellent [Kani-TTS](https://github.com/nineninesix-ai/kani-tts) by **nineninesix-ai**. All credits for the underlying TTS architecture and models go to the original authors.

Please adhere to the [Ethical Usage Guidelines](https://github.com/nineninesix-ai/kani-tts#content-requirements--usage-policy) of the original project.
