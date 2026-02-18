"""
Audio Chunker & Parallel Inference Module
==========================================
Smart chunking with silence detection and parallel async transcription
for the Armenian Medical STT pipeline.
"""
from dataclasses import dataclass

import os
import glob
import shutil
import time
import threading
import tempfile
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.silence import detect_silence
import numpy as np
import noisereduce as nr


logger = logging.getLogger(__name__)

# ── Auto-discover ffmpeg for pydub ──────────────────────
# static_ffmpeg bundles both ffmpeg AND ffprobe as properly named
# executables and adds them to PATH so pydub can find them.
if not shutil.which("ffmpeg"):
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        logger.info("Added static-ffmpeg binaries to PATH.")
    except ImportError:
        logger.warning(
            "ffmpeg not found on PATH and static-ffmpeg not installed. "
            "pydub will fail to load audio files."
        )

# ──────────────────────────────────────────────────────────
#  0. Noise Reduction
# ──────────────────────────────────────────────────────────

def remove_background_noise(filepath: str) -> str:
    """
    Apply non-stationary spectral-gating noise reduction to an audio file.

    Uses dynamic tracking (adapts to changing background noise every 2s)
    with moderate reduction (0.6) to preserve voice clarity.
    Normalizes volume back to 0 dBFS after cleaning.
    Returns the path to the cleaned audio file (mp3 in temp dir).
    """
    logger.info("Cleaning audio: %s", filepath)

    # Load with pydub
    audio = AudioSegment.from_file(filepath)

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())

    # Handle stereo vs mono
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).T

    # Apply non-stationary noise reduction
    reduced_noise = nr.reduce_noise(
        y=samples,
        sr=audio.frame_rate,
        stationary=False,       # Dynamically tracks changing background noise
        time_constant_s=2.0,    # Adapts to new background sounds every 2 seconds
        prop_decrease=0.6,      # 60% reduction to prevent voice muffling
    )

    # Convert back to pydub format
    if audio.channels > 1:
        reduced_noise = reduced_noise.T.flatten()

    clean_audio = audio._spawn(reduced_noise.tobytes())

    # Volume restoration — boost back to maximum safe level (0 dBFS)
    clean_audio = clean_audio.apply_gain(-clean_audio.max_dBFS)

    # Export cleaned audio to temp file
    cleaned_path = os.path.join(tempfile.gettempdir(), "cleaned_audio.mp3")
    clean_audio.export(cleaned_path, format="mp3")
    logger.info("Noise reduction complete → %s", cleaned_path)
    return cleaned_path


# ──────────────────────────────────────────────────────────
#  1. Smart Chunking
# ──────────────────────────────────────────────────────────

# Chunking parameters (milliseconds)
TARGET_SEGMENT_MS = 30_000        # Target chunk length: 30 s
SAFETY_WINDOW_START_MS = 25_000   # Start of the safety window: 25 s
SAFETY_WINDOW_END_MS = 35_000     # End of the safety window: 35 s
MIN_CHUNK_MS = 10_000             # Minimum chunk length: 10 s (stride is 5 s)
MIN_SILENCE_LEN_MS = 300          # Minimum silence duration to detect
SILENCE_THRESH_DB = -40           # dBFS threshold for silence



def smart_chunk_audio(audio_file_path: str) -> list[str]:
    """
    Split an audio file into ~30-second chunks using silence-aware cutting.

    Algorithm for each segment:
      1. Look at the 25 s – 35 s "safety window" relative to the current offset.
      2. If silence is found in that window → cut at the midpoint of the silence.
      3. If no silence → hard-cut at exactly 30 s.

    Args:
        audio_file_path: Path to the source audio file.

    Returns:
        Ordered list of chunk file paths (chunk_000.mp3, chunk_001.mp3, …).
    """
    audio = AudioSegment.from_file(audio_file_path)
    total_ms = len(audio)

    # Short audio → return as a single chunk (no splitting needed)
    if total_ms <= SAFETY_WINDOW_END_MS:
        chunk_path = _export_chunk(audio, 0)
        logger.info("Audio ≤ 35 s — single chunk, no splitting.")
        return [chunk_path]

    chunks: list[str] = []
    offset = 0
    idx = 0

    while offset < total_ms:
        remaining = total_ms - offset

        # If what's left fits inside the safety window, take it all
        if remaining <= SAFETY_WINDOW_END_MS:
            chunk = audio[offset:]
            chunks.append(_export_chunk(chunk, idx))
            break

        # Determine cut point using the safety window
        window_start = offset + SAFETY_WINDOW_START_MS
        window_end = min(offset + SAFETY_WINDOW_END_MS, total_ms)
        window = audio[window_start:window_end]

        silences = detect_silence(
            window,
            min_silence_len=MIN_SILENCE_LEN_MS,
            silence_thresh=SILENCE_THRESH_DB,
        )

        if silences:
            # Cut at the midpoint of the first detected silence gap
            sil_start, sil_end = silences[0]
            # Translate back to absolute position
            cut_point = window_start + (sil_start + sil_end) // 2
            logger.debug(
                "Chunk %d: silence found at %d–%d ms in window → cutting at %d ms",
                idx, window_start + sil_start, window_start + sil_end, cut_point,
            )
        else:
            # No silence → hard-cut at exactly 30 s
            cut_point = offset + TARGET_SEGMENT_MS
            logger.debug("Chunk %d: no silence → hard-cut at %d ms", idx, cut_point)

        chunk = audio[offset:cut_point]
        chunks.append(_export_chunk(chunk, idx))

        offset = cut_point
        idx += 1

    logger.info("Created %d chunks from audio (%d ms total).", len(chunks), total_ms)

    # Merge very short final chunk into the previous one to avoid
    # stride_length_s issues on the Space side.
    if len(chunks) > 1:
        last_audio = AudioSegment.from_file(chunks[-1])
        if len(last_audio) < MIN_CHUNK_MS:
            prev_audio = AudioSegment.from_file(chunks[-2])
            merged = prev_audio + last_audio
            # Re-export merged chunk
            merged.export(chunks[-2], format="mp3")
            # Remove the tiny tail chunk
            os.remove(chunks[-1])
            chunks.pop()
            logger.info(
                "Merged short final chunk (%d ms) into chunk %d.",
                len(last_audio), len(chunks) - 1,
            )

    return chunks


def _export_chunk(segment: AudioSegment, index: int) -> str:
    """Export an AudioSegment as chunk_NNN.mp3 in the temp directory."""
    chunk_name = f"chunk_{index:03d}.mp3"
    chunk_path = os.path.join(tempfile.gettempdir(), chunk_name)
    segment.export(chunk_path, format="mp3")
    return chunk_path


# ──────────────────────────────────────────────────────────
#  2. Parallel Inference (ThreadPool + Gradio Client)
# ──────────────────────────────────────────────────────────

MAX_CONCURRENCY = 5               # Thread pool size / semaphore limit
MAX_RETRIES = 3                   # Retry attempts per chunk
RETRY_DELAY_BASE = 2.0            # Exponential back-off base (seconds)


def _transcribe_one_chunk(
    chunk_path: str,
    index: int,
    space_name: str,
    hf_token: str,
    semaphore: threading.Semaphore,
) -> tuple[int, str]:
    """
    Transcribe a single chunk via the Gradio Client.
    Uses exponential back-off on failure.
    Returns (index, transcription_text).
    """
    from gradio_client import Client as GradioClient, handle_file

    with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = GradioClient(space_name, token=hf_token)
                result = client.predict(
                    handle_file(chunk_path),
                    fn_index=0,
                )
                text = result if isinstance(result, str) else str(result)
                logger.info("Chunk %d transcribed (%d chars).", index, len(text))
                return (index, text)

            except Exception as e:
                tb = traceback.format_exc()
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_BASE ** attempt
                    logger.warning(
                        "Chunk %d attempt %d failed (%s) — retrying in %.1f s\n%s",
                        index, attempt, e, delay, tb,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Chunk %d failed after %d attempts:\n%s",
                        index, MAX_RETRIES, tb,
                    )
                    raise RuntimeError(
                        f"Transcription failed for chunk {index} after {MAX_RETRIES} retries: {type(e).__name__}: {e}"
                    ) from e

    return (index, "")  # unreachable, satisfies type checker


@dataclass
class TranscriptionResult:
    """Holds the full transcript and per-chunk details."""
    full_transcript: str
    chunk_details: list  # list of dict: {index, path, text, duration_s}


def transcribe_chunks_parallel(
    chunk_paths: list[str],
    space_name: str,
    hf_token: str,
) -> TranscriptionResult:
    """
    Transcribe all chunks in parallel using a thread pool.

    Args:
        chunk_paths: Ordered list of chunk file paths.
        space_name: HF Space name (e.g. "davtar10/whisper-am-server").
        hf_token: Hugging Face API token.

    Returns:
        TranscriptionResult with full_transcript and per-chunk details.
    """
    semaphore = threading.Semaphore(MAX_CONCURRENCY)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        futures = [
            executor.submit(
                _transcribe_one_chunk, path, idx, space_name, hf_token, semaphore
            )
            for idx, path in enumerate(chunk_paths)
        ]
        results = [f.result() for f in futures]

    # Sort by index and stitch
    results.sort(key=lambda r: r[0])
    full_transcript = " ".join(text.strip() for _, text in results if text.strip())

    # Build per-chunk details
    chunk_details = []
    for idx, text in results:
        path = chunk_paths[idx]
        try:
            audio = AudioSegment.from_file(path)
            duration_s = len(audio) / 1000.0
        except Exception:
            duration_s = 0.0
        chunk_details.append({
            "index": idx,
            "path": path,
            "text": text,
            "duration_s": duration_s,
        })

    return TranscriptionResult(
        full_transcript=full_transcript,
        chunk_details=chunk_details,
    )


# ──────────────────────────────────────────────────────────
#  3. Cleanup
# ──────────────────────────────────────────────────────────

def cleanup_chunks(chunk_paths: list[str]) -> None:
    """Delete all chunk files. Silently ignores missing/locked files."""
    for path in chunk_paths:
        try:
            os.remove(path)
            logger.debug("Deleted %s", path)
        except OSError:
            pass
