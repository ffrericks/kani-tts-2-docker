"""FastAPI server for Kani TTS with streaming support and Web UI"""

import io
import time
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, FileResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
from scipy.io.wavfile import write as wav_write

from audio import LLMAudioPlayer, StreamingAudioWriter
from generation import TTSGenerator
from config import CHUNK_SIZE, LOOKBACK_FRAMES, TEMPERATURE, TOP_P, MAX_TOKENS

from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

app = FastAPI(title="Kani TTS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = None
player = None

class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    top_p: Optional[float] = TOP_P
    chunk_size: Optional[int] = CHUNK_SIZE
    lookback_frames: Optional[int] = LOOKBACK_FRAMES

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global generator, player
    print("🚀 Initializing TTS models...")
    try:
        generator = TTSGenerator()
        player = LLMAudioPlayer(generator.tokenizer)
        print("✅ TTS models initialized successfully!")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

@app.get("/health")
async def health_check():
    """Check if server is ready"""
    return {
        "status": "healthy" if generator is not None else "initializing",
        "tts_initialized": generator is not None and player is not None
    }

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """Generate complete audio file (non-streaming)"""
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    try:
        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames
        )
        audio_writer.start()

        generator.generate(
            request.text,
            audio_writer,
            max_tokens=request.max_tokens
        )

        audio_writer.finalize()

        if not audio_writer.audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        full_audio = np.concatenate(audio_writer.audio_chunks)
        wav_buffer = io.BytesIO()
        wav_write(wav_buffer, 22050, full_audio)
        wav_buffer.seek(0)

        return Response(
            content=wav_buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-tts")
async def stream_speech(request: TTSRequest):
    """Stream audio chunks as they're generated"""
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS models not initialized")

    import queue
    import threading
    import struct

    async def audio_chunk_generator():
        chunk_queue = queue.Queue()
        class ChunkList(list):
            def append(self, chunk):
                super().append(chunk)
                chunk_queue.put(("chunk", chunk))

        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames
        )
        audio_writer.audio_chunks = ChunkList()

        def generate():
            try:
                audio_writer.start()
                generator.generate(request.text, audio_writer, max_tokens=request.max_tokens)
                audio_writer.finalize()
                chunk_queue.put(("done", None))
            except Exception as e:
                chunk_queue.put(("error", str(e)))

        gen_thread = threading.Thread(target=generate)
        gen_thread.start()

        try:
            while True:
                msg_type, data = chunk_queue.get(timeout=60)
                if msg_type == "chunk":
                    pcm_data = (data * 32767).astype(np.int16)
                    chunk_bytes = pcm_data.tobytes()
                    length_prefix = struct.pack('<I', len(chunk_bytes))
                    yield length_prefix + chunk_bytes
                elif msg_type == "done":
                    yield struct.pack('<I', 0)
                    break
                elif msg_type == "error":
                    yield struct.pack('<I', 0xFFFFFFFF)
                    break
        finally:
            gen_thread.join()

    return StreamingResponse(
        audio_chunk_generator(),
        media_type="application/octet-stream"
    )

@app.get("/")
async def root():
    """Serve the Web UI"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Kani TTS Server is running. index.html not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
