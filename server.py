"""FastAPI server voor Kani TTS 2 met voice cloning, device switching en Web UI"""

import io
import os
import sys
import shutil
import time
import threading
import torch
import numpy as np
from scipy.io.wavfile import write as wav_write
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from typing import Optional

from kani_tts import KaniTTS, SpeakerEmbedder

# Constants
SAMPLE_RATE  = 22050
VOICES_DIR   = "/app/voices"
DEVICE_FILE  = "/app/device.cfg"
MODEL_NAME   = "nineninesix/kani-tts-2-pt"
TEMPERATURE  = 0.7
TOP_P        = 0.9
REPETITION_PENALTY = 1.2

# ── Device voorkeur ────────────────────────────────────────────────────────────
def get_device_pref() -> str:
    """Lees device voorkeur uit bestand of env var."""
    try:
        with open(DEVICE_FILE) as f:
            val = f.read().strip()
            if val in ("cpu", "cuda", "auto"):
                return val
    except FileNotFoundError:
        pass
    return os.environ.get("DEVICE", "auto")

def save_device_pref(device: str):
    with open(DEVICE_FILE, "w") as f:
        f.write(device)

# Zet CUDA_VISIBLE_DEVICES VOOR torch CUDA initialisatie
_pref = get_device_pref()
if _pref == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Kani TTS 2 API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(VOICES_DIR, exist_ok=True)

tts            = None
embedder       = None
current_device = "unknown"
is_ready       = False


# ── Model initialisatie ────────────────────────────────────────────────────────
async def _init_models():
    global tts, embedder, current_device, is_ready
    is_ready = False
    pref = get_device_pref()
    print(f"🔧 Device voorkeur: {pref}")

    try:
        if pref == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tts = KaniTTS(MODEL_NAME)
            current_device = "cpu"
        else:
            # Probeer GPU, val terug op CPU bij fout
            try:
                tts = KaniTTS(MODEL_NAME)
                current_device = "cuda" if torch.cuda.is_available() and \
                    torch.cuda.device_count() > 0 else "cpu"
            except Exception as gpu_err:
                print(f"⚠️  GPU mislukt: {gpu_err}")
                print("↩️  Terugvallen op CPU...")
                torch.cuda.empty_cache()
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                tts = KaniTTS(MODEL_NAME)
                current_device = "cpu"

        try:
            embedder = SpeakerEmbedder()
        except Exception as emb_err:
            print(f"⚠️  SpeakerEmbedder niet geladen: {emb_err}")
            embedder = None

        is_ready = True
        print(f"✅ Kani TTS 2 gereed op {current_device.upper()}")

    except Exception as e:
        print(f"❌ Initialisatie mislukt: {e}")


@app.on_event("startup")
async def startup_event():
    await _init_models()


# ── Health & Device ────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status":           "ready" if is_ready else "initializing",
        "tts_initialized":  tts is not None,
        "cloning_available": embedder is not None,
        "device":           current_device,
        "device_pref":      get_device_pref(),
    }


@app.post("/set-device")
async def set_device(device: str):
    """Sla device voorkeur op en herstart de server."""
    if device not in ("cpu", "cuda", "auto"):
        raise HTTPException(status_code=400, detail="Gebruik: cpu, cuda of auto")

    save_device_pref(device)
    print(f"🔄 Device gewijzigd naar '{device}' — herstart...")

    def _restart():
        time.sleep(0.6)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_restart, daemon=True).start()
    return {"status": "restarting", "device": device}


# ── Voice Gallery ──────────────────────────────────────────────────────────────
@app.get("/voices")
async def list_voices():
    return sorted(f[:-3] for f in os.listdir(VOICES_DIR) if f.endswith(".pt"))


@app.post("/voices")
async def upload_voice(file: UploadFile = File(...), name: str = Form(...)):
    if not embedder:
        raise HTTPException(status_code=501, detail="Speaker Embedder niet beschikbaar")
    safe = name.replace("-", "").replace("_", "")
    if not safe.isalnum():
        raise HTTPException(status_code=400,
            detail="Ongeldige naam. Gebruik letters, cijfers, - of _")

    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        embedding = embedder.embed_audio_file(temp_path)
        torch.save(embedding, os.path.join(VOICES_DIR, f"{name}.pt"))
        return {"status": "success", "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/voices/{name}")
async def delete_voice(name: str):
    fp = os.path.join(VOICES_DIR, f"{name}.pt")
    if os.path.exists(fp):
        os.remove(fp)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Voice niet gevonden")


def _load_speaker_emb(voice_name: Optional[str]):
    if not voice_name:
        return None
    fp = os.path.join(VOICES_DIR, f"{voice_name}.pt")
    return torch.load(fp) if os.path.exists(fp) else None


# ── TTS ────────────────────────────────────────────────────────────────────────
def _to_wav_bytes(audio) -> bytes:
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    wav_buf = io.BytesIO()
    wav_write(wav_buf, SAMPLE_RATE, audio.astype(np.float32))
    wav_buf.seek(0)
    return wav_buf.read()


class TTSRequest(BaseModel):
    text:               str
    temperature:        Optional[float] = TEMPERATURE
    top_p:              Optional[float] = TOP_P
    repetition_penalty: Optional[float] = REPETITION_PENALTY
    voice_name:         Optional[str]   = None
    language_tag:       Optional[str]   = None


@app.post("/tts")
async def generate_speech(request: TTSRequest):
    if not tts:
        raise HTTPException(status_code=503, detail="TTS model niet geïnitialiseerd")

    try:
        speaker_emb = _load_speaker_emb(request.voice_name)
        kwargs = dict(
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        if speaker_emb is not None:
            kwargs["speaker_emb"] = speaker_emb
        if request.language_tag:
            kwargs["language_tag"] = request.language_tag

        t_start = time.time()
        audio, _ = tts(request.text, **kwargs)
        t_total  = round(time.time() - t_start, 2)

        audio_np  = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
        audio_dur = round(len(audio_np) / SAMPLE_RATE, 2)

        return Response(
            content=_to_wav_bytes(audio),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech_{request.voice_name or 'default'}.wav",
                "X-Generation-Time":   str(t_total),
                "X-Audio-Duration":    str(audio_dur),
                "X-Device":            current_device,
                "Access-Control-Expose-Headers":
                    "X-Generation-Time, X-Audio-Duration, X-Device",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/public/{filename}")
async def serve_public(filename: str):
    fp = os.path.join("public", filename)
    if os.path.exists(fp):
        return FileResponse(fp)
    raise HTTPException(status_code=404, detail="Bestand niet gevonden")

@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Kani TTS 2 Server is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
