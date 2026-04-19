"""FastAPI server for Kani TTS 2 with voice cloning support and Web UI"""

import io
import os
import shutil
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
SAMPLE_RATE = 22050
VOICES_DIR = "/app/voices"
MODEL_NAME = "nineninesix/kani-tts-2-pt"
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2

app = FastAPI(title="Kani TTS 2 API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(VOICES_DIR, exist_ok=True)

tts = None
embedder = None


class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = TEMPERATURE
    top_p: Optional[float] = TOP_P
    repetition_penalty: Optional[float] = REPETITION_PENALTY
    voice_name: Optional[str] = None      # naam van opgeslagen voice embedding
    language_tag: Optional[str] = None    # bijv. "en_US", "nl_NL", "de_DE"


@app.on_event("startup")
async def startup_event():
    global tts, embedder
    print("Initializing Kani TTS 2 models...")
    try:
        tts = KaniTTS(MODEL_NAME)
        embedder = SpeakerEmbedder()
        print("Kani TTS 2 initialized successfully!")
    except Exception as e:
        print(f"Initialization failed: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if tts is not None else "initializing",
        "tts_initialized": tts is not None,
        "cloning_available": embedder is not None,
    }


# --- Voice Gallery ---

@app.get("/voices")
async def list_voices():
    """Geef lijst van opgeslagen voice profielen"""
    voices = [f[:-3] for f in os.listdir(VOICES_DIR) if f.endswith(".pt")]
    return sorted(voices)


@app.post("/voices")
async def upload_voice(file: UploadFile = File(...), name: str = Form(...)):
    """Upload een audio bestand en sla de speaker embedding op"""
    if not embedder:
        raise HTTPException(status_code=501, detail="Speaker Embedder niet beschikbaar")

    # Naam validatie — letters, cijfers, koppeltekens en underscores
    safe = name.replace("-", "").replace("_", "")
    if not safe.isalnum():
        raise HTTPException(
            status_code=400,
            detail="Ongeldige naam. Gebruik alleen letters, cijfers, - of _"
        )

    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Computing speaker embedding for: {name}")
        embedding = embedder.embed_audio_file(temp_path)

        save_path = os.path.join(VOICES_DIR, f"{name}.pt")
        torch.save(embedding, save_path)

        return {"status": "success", "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/voices/{name}")
async def delete_voice(name: str):
    """Verwijder een opgeslagen voice profiel"""
    file_path = os.path.join(VOICES_DIR, f"{name}.pt")
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Voice niet gevonden")


# --- TTS ---

def _load_speaker_emb(voice_name: Optional[str]):
    if not voice_name:
        return None
    file_path = os.path.join(VOICES_DIR, f"{voice_name}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    return None


def _audio_to_wav_bytes(audio) -> bytes:
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    audio = audio.astype(np.float32)
    wav_buffer = io.BytesIO()
    wav_write(wav_buffer, SAMPLE_RATE, audio)
    wav_buffer.seek(0)
    return wav_buffer.read()


@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """Genereer audio (non-streaming)"""
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

        audio, _ = tts(request.text, **kwargs)

        filename = f"speech_{request.voice_name or 'default'}.wav"
        return Response(
            content=_audio_to_wav_bytes(audio),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Kani TTS 2 Server is running. index.html not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
