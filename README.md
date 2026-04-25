# Kani-TTS 2 Docker for CasaOS

NOTE: I don't get this working on my machine, Use Kani-TTS instead.
A dockerized version of [Kani-TTS 2](https://github.com/nineninesix-ai/kani-tts-2), ready to run on **CasaOS** and Proxmox environments with NVIDIA GPU passthrough.

Includes a Web UI with built-in **Voice Cloning** — upload a short audio clip and clone any voice.

---

## 🚀 Quick Install (CasaOS)

1. Open **CasaOS** → **App Store** → **Custom Install** (top right)
2. Paste the following Docker Compose:

```yaml
version: '3.8'

services:
  kani-tts-2:
    image: ghcr.io/ffrericks/kani-tts-2-docker:latest
    container_name: kani-tts-2
    ports:
      - "8001:8001"
    volumes:
      - /DATA/AppData/kani-tts-2/huggingface:/root/.cache/huggingface
      - /DATA/AppData/kani-tts-2/voices:/app/voices
    runtime: nvidia
    environment:
      - TZ=Europe/Amsterdam
      - NVIDIA_VISIBLE_DEVICES=all
    restart: unless-stopped
```

3. Under **"Web UI"**, set the port to **`8001`** and the path to **`/`**
4. Click **Install**
5. Access the Web UI at `http://<your-server-ip>:8001`

> **Geen GPU?** Verwijder de `runtime: nvidia` en `NVIDIA_VISIBLE_DEVICES` regels — de container draait dan op CPU.

> **Note:** The first start takes several minutes to download the AI models. This only happens once thanks to the persistent volume.

> **Running alongside v1?** v1 uses port `8000`, v2 uses `8001` — they can run simultaneously without conflict.

---

## ✨ Features

- **Voice Cloning** — Upload any 3–20 second audio clip and clone that voice (zero-shot, no training needed)
- **Voice Gallery** — Save and manage multiple voice profiles, reuse them anytime
- **Language Tags** — Hint the model which language to use (`nl_NL`, `en_US`, `de_DE`, etc.)
- **Pre-built image** — No local compilation needed, pulled directly from GitHub Container Registry
- **GPU Acceleration** — Pre-configured for NVIDIA GPUs via Docker runtime
- **Persistent Storage** — Models and voice profiles are stored in Docker volumes
- **FastAPI backend** — `/tts`, `/voices` endpoints for integration with n8n and other tools

---

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB | 20 GB |

> **4GB VRAM:** Kani-TTS 2 requires 4–8GB VRAM. With 4GB it may work, but is tight. If the container crashes on startup, there is not enough VRAM available.

---

## 🎙 Voice Cloning

1. Open the Web UI at `http://<your-server-ip>:8001`
2. Scroll to **Voice Gallery**
3. Upload a WAV or MP3 file (3–20 seconds of clean speech)
4. Give the voice a name and click **Opslaan**
5. Select the saved voice in the **Stem** dropdown and generate

---

## 🔌 API — n8n Integration

### Generate speech (POST /tts)

```json
{
  "text": "{{ $json.textPlain }}",
  "voice_name": "mijn_stem",
  "language_tag": "nl_NL",
  "temperature": 0.7
}
```

Set **Response Format** to `File` in the n8n HTTP Request node to receive the WAV file directly.

### List saved voices (GET /voices)

Returns a JSON array of saved voice names.

### Upload a voice (POST /voices)

Send as `multipart/form-data` with fields `file` (audio) and `name` (string).

---

## 📦 Model

This image uses `nineninesix/kani-tts-2-pt` — the pretrained multilingual Kani-TTS 2 model.

---

## 📜 License & Attribution

This project is licensed under the **Apache License 2.0**.

This is a dockerized wrapper around [Kani-TTS 2](https://github.com/nineninesix-ai/kani-tts-2) by **nineninesix-ai**. All credits for the TTS architecture and models go to the original authors.

Please adhere to the [Ethical Usage Guidelines](https://github.com/nineninesix-ai/kani-tts-2) of the original project — no impersonation, hate speech, or harmful content.
