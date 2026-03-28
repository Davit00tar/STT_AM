"""
Medical Audio to Clinical Notes — FastAPI REST API
Exposes the audio-processing pipeline from logic.py as HTTP endpoints.

Usage:
    uvicorn api:app --reload --port 8000
"""

import os
import tempfile
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from logic import init_clients, process_audio_pipeline

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HF_SPACE_NAME = os.getenv("HF_SPACE_NAME", "davtar10/whisper-am-server")

ALLOWED_EXTENSIONS = {".mp3", ".m4a", ".wav"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MIN_FILE_SIZE_BYTES = 1_000             # 1 KB


# ── Pydantic response models ───────────────────────────────────────────────

class TimingStats(BaseModel):
    noise_reduction: float
    chunking: float
    transcription: float
    gemini: float


class TranscriptionResponse(BaseModel):
    clinical_note: str
    raw_transcript: str | None = None
    processing_time_seconds: float
    timing_stats: TimingStats
    source_filename: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


# ── Application lifespan (init clients once) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise API clients at startup; nothing to tear down."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not HF_API_KEY:
        missing.append("HF_API_KEY")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in a .env file or your shell."
        )

    groq_client, hf_client = init_clients(
        groq_key=GROQ_API_KEY,
        hf_key=HF_API_KEY,
        google_key=GOOGLE_API_KEY,
    )
    # Store on app.state so endpoints can access if needed later
    app.state.groq_client = groq_client
    app.state.hf_client = hf_client

    yield  # App is running


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Armenian Medical STT API",
    description=(
        "Upload an audio file of a medical consultation and receive "
        "a structured Armenian clinical note."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from any origin (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple readiness / health probe."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accept an audio file and return a structured clinical note.

    - **file**: Audio file (mp3, m4a, or wav), max 50 MB.
    """
    # ── Validate extension ──
    filename = file.filename or "upload.mp3"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # ── Read and validate size ──
    contents = await file.read()
    if len(contents) < MIN_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="File is too small (< 1 KB) — likely empty or corrupted.",
        )
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large ({len(contents) / 1024 / 1024:.1f} MB). Max is 50 MB.",
        )

    # ── Save to temp file ──
    unique_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(tempfile.gettempdir(), unique_name)

    try:
        with open(temp_path, "wb") as f:
            f.write(contents)

        # ── Run the pipeline in a thread pool ──
        # process_audio_pipeline is synchronous and blocking (CPU + network I/O).
        # Awaiting it via run_in_threadpool keeps the event loop free so that
        # concurrent requests are handled in parallel instead of queuing.
        clinical_note, proc_time, timing_stats, debug_data, _ = await run_in_threadpool(
            process_audio_pipeline,
            audio_file_path=temp_path,
            space_name=HF_SPACE_NAME,
            hf_token=HF_API_KEY,
            debug=True,  # Always collect transcript for the API response
        )

        # Extract raw transcript from debug_data
        raw_transcript = None
        if debug_data and "full_transcript" in debug_data:
            raw_transcript = debug_data["full_transcript"]

        return TranscriptionResponse(
            clinical_note=clinical_note,
            raw_transcript=raw_transcript,
            processing_time_seconds=round(proc_time, 2),
            timing_stats=TimingStats(**timing_stats),
            source_filename=filename,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        error_msg = str(e)

        # Map known pipeline errors to appropriate HTTP status codes
        if "EMPTY_AUDIO" in error_msg or "EMPTY_TRANSCRIPT" in error_msg:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in the audio. Please try with a longer or clearer recording.",
            )
        if "EMPTY_RESPONSE" in error_msg:
            raise HTTPException(
                status_code=422,
                detail="The audio was unclear. Please record again with clearer speech.",
            )
        if "ReadTimeout" in error_msg or "timed out" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail=(
                    "Transcription server timed out (likely a cold start). "
                    "Wait ~30 seconds and try again."
                ),
            )
        if "ConnectionError" in error_msg or "connection" in error_msg.lower():
            raise HTTPException(
                status_code=502,
                detail="Could not reach the transcription server. Check your internet connection.",
            )

        raise HTTPException(status_code=500, detail=f"Processing error: {error_msg}")

    finally:
        # Always clean up the temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass
