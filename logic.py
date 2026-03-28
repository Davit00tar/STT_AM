"""
Medical Audio to Clinical Notes - AI Processing Logic
All AI, transcription, preprocessing, and Gemini formatting logic lives here.
No Streamlit imports — this module is independently testable.
"""

import os
import time
import tempfile
from groq import Groq
from huggingface_hub import InferenceClient
from gradio_client import Client as GradioClient, handle_file
import google.generativeai as genai
from audio_chunker import (
    smart_chunk_audio,
    transcribe_chunks_parallel,
    cleanup_chunks,
)


# ── API Client Initialization ──────────────────────────────────────────────

def init_clients(groq_key: str, hf_key: str, google_key: str):
    """
    Create and return API clients.
    Returns:
        tuple: (groq_client, hf_client)
    Side-effect: configures google.generativeai with the provided key.
    """
    groq_client = Groq(api_key=groq_key)
    hf_client = InferenceClient(token=hf_key)
    genai.configure(api_key=google_key)
    return groq_client, hf_client


# ── Transcription Functions ─────────────────────────────────────────────────

def transcribe_audio_groq(groq_client, audio_file_path: str) -> str:
    """
    Transcribe audio using Groq's Whisper API (fallback).
    Args:
        groq_client: Initialised Groq client.
        audio_file_path: Path to audio file.
    Returns:
        str: Transcribed text in Armenian.
    """
    try:
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            raise Exception("EMPTY_AUDIO")

        with open(audio_file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3-turbo",
                language="hy",
                temperature=0,
                response_format="verbose_json",
            )

            if not transcription.text or len(transcription.text.strip()) < 3:
                raise Exception("EMPTY_TRANSCRIPT")

            return transcription.text
    except Exception as e:
        if "EMPTY" in str(e):
            raise Exception("EMPTY_AUDIO")
        raise Exception(f"Transcription error: {str(e)}")


def transcribe_audio_hf(hf_client, audio_file_path: str) -> str:
    """
    Transcribe audio using Hugging Face Inference API with fine-tuned
    Armenian Whisper model (Chillarmo/whisper-large-v3-turbo-armenian).
    Args:
        hf_client: Initialised InferenceClient.
        audio_file_path: Path to audio file.
    Returns:
        str: Transcribed text in Armenian.
    """
    try:
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            raise Exception("EMPTY_AUDIO")

        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        result = hf_client.automatic_speech_recognition(
            audio=audio_bytes,
            model="Chillarmo/whisper-large-v3-turbo-armenian",
        )

        transcription_text = result.text if hasattr(result, "text") else str(result)

        if not transcription_text or len(transcription_text.strip()) < 3:
            raise Exception("EMPTY_TRANSCRIPT")

        return transcription_text
    except Exception as e:
        print(f"DEBUG HF ERROR: {type(e).__name__}: {str(e)}")
        if "EMPTY" in str(e):
            raise Exception("EMPTY_AUDIO")
        raise Exception(f"Transcription error: {str(e)}")


def transcribe_audio_hf_space(audio_file_path: str, space_name: str, hf_token: str) -> str:
    """
    Transcribe audio using the Hugging Face Space via Gradio Client.
    Args:
        audio_file_path: Path to audio file.
        space_name: HF Space identifier (e.g. 'davtar10/whisper-am-server').
        hf_token: Hugging Face API token.
    Returns:
        str: Transcribed text in Armenian.
    """
    try:
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            raise Exception("EMPTY_AUDIO")

        client = GradioClient(space_name, token=hf_token)

        result = client.predict(
            handle_file(audio_file_path),
            fn_index=0,
        )

        transcription_text = result if isinstance(result, str) else str(result)

        if not transcription_text or len(transcription_text.strip()) < 3:
            raise Exception("EMPTY_TRANSCRIPT")

        return transcription_text
    except Exception as e:
        print(f"DEBUG HF SPACE ERROR: {type(e).__name__}: {str(e)}")
        if "EMPTY" in str(e):
            raise Exception("EMPTY_AUDIO")
        raise Exception(f"Transcription error: {str(e)}")


# ── Gemini Medical Formatting ───────────────────────────────────────────────

def process_with_gemini(chunk_texts: list[str]) -> str:
    """
    Process transcript chunks with Google Gemini for medical formatting.
    Each chunk is sent as a labeled segment so Gemini can see boundaries.
    Args:
        chunk_texts: List of per-chunk transcription strings (ordered).
    Returns:
        str: Cleaned and formatted clinical note.
    """
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            system_instruction="""
**Role:** You are a strict Medical Transcription Editor. Your ONLY job is to take raw speech-to-text chunks and merge them into one clean, continuous clinical note.

### **Input Format:**
You will receive the transcript split into sequential chunks labeled `[Chunk 1]`, `[Chunk 2]`, etc.
These are adjacent segments of one continuous recording. Merge them into a single seamless narrative.
Do NOT mention chunks, labels, or the splitting process in your output.

### **ABSOLUTE RULES — READ CAREFULLY:**

1. **ASR ERROR CORRECTION (Phonetic Reasoning):**
   * The input comes from an automatic speech recognition (ASR) system. It WILL contain misheard words.
   * If a word **has no meaning in Armenian** but **sounds like** a real Armenian word that makes sense in context → replace it with that meaningful word.
   * Example: if ASR wrote " delays" but in context it clearly sounds like " delays" → use "adekvat".
   * If a word IS a real, meaningful Armenian word → keep it EXACTLY, even if a synonym might sound "better". Do NOT rephrase.
   * Do NOT replace meaningful words with synonyms. Only fix words that are clearly ASR errors (nonsense/garbled text).
   * Remove filler sounds like "umm", "aaaa", "hmm" and greetings at the very start.

2. **NUMBERS, DATES, DOSAGES — NEVER CHANGE:**
   * All numbers, measurements, dosages, dates, lab values, and quantities MUST remain EXACTLY as spoken.
   * Do NOT round, convert, or "correct" any numerical value.
   * Examples: "150/90", "37.2", "5 mg", "3 times" — keep them exactly as-is.

3. **MEDICAL TERMS & SPECIAL TERMS — PRESERVE CAREFULLY:**
   * Medication names, diagnosis names, procedure names, anatomical terms — keep them as the doctor said them.
   * If a medical term looks garbled by ASR (nonsense syllables), try to recover the correct term based on how it sounds.
   * If the term is already recognizable and meaningful, do NOT change it.
   * If the doctor said a term in Russian, Armenian, or Latin — keep it in that language.

4. **NO ADDITIONS — ZERO TOLERANCE:**
   * Do NOT add any word, phrase, sentence, or piece of information not in the original.
   * Do NOT add introductory phrases, conclusions, summaries, or section headers.
   * Do NOT add medical advice or logical next steps.
   * Do NOT rephrase or restructure sentences. Keep the doctor's sentence structure.

5. **FORMATTING:**
   * Merge the chunks into one continuous, readable clinical narrative.
   * Fix punctuation and paragraph breaks for readability.
   * Remove duplicate text at chunk boundaries (where the same phrase may appear in two adjacent chunks).

### **Output Format:**
* **Language:** Same as input (Armenian, with medical terms as spoken).
* **Structure:** Clean narrative text, no section headers unless the doctor explicitly dictated them.
""",
        )

        labeled_input = "\n\n".join(
            f"[Chunk {i + 1}]\n{text.strip()}"
            for i, text in enumerate(chunk_texts)
            if text.strip()
        )

        response = model.generate_content(labeled_input)

        if not response.candidates or not response.candidates[0].content.parts:
            raise Exception("EMPTY_RESPONSE")

        return response.text
    except Exception as e:
        if "EMPTY_RESPONSE" in str(e) or "finish_reason" in str(e):
            raise Exception("EMPTY_RESPONSE")
        raise Exception(f"Processing error: {str(e)}")


# ── Utility ─────────────────────────────────────────────────────────────────

def save_uploaded_file(file_name: str, file_buffer: bytes) -> str | None:
    """
    Save uploaded file bytes to a temporary directory and return the path.
    Args:
        file_name: Original file name.
        file_buffer: Raw bytes of the file.
    Returns:
        str | None: Path to saved file, or None on failure.
    """
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(file_buffer)
        return file_path
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return None


# ── Full Processing Pipeline ────────────────────────────────────────────────

def process_audio_pipeline(
    audio_file_path: str,
    space_name: str,
    hf_token: str,
    debug: bool = False,
) -> tuple[str, float, dict, dict | None, bytes | None]:
    """
    Complete processing pipeline from audio to clinical note.
    Steps: Noise Reduction → Smart Chunk → Parallel Transcribe → Gemini Format → Cleanup.

    Args:
        audio_file_path: Path to the input audio file.
        space_name: HF Space identifier for transcription.
        hf_token: Hugging Face API token.
        debug: Whether to collect per-chunk debug data.

    Returns:
        tuple of:
            clinical_note (str),
            processing_time (float),
            timing_stats (dict),
            debug_data (dict | None) – chunk details if debug=True,
            debug_cleaned_audio (bytes | None) – cleaned audio bytes if debug=True.
    """
    start_time = time.time()
    cleaned_path = None
    chunk_paths = []
    timing_stats = {
        "noise_reduction": 0.0,
        "chunking": 0.0,
        "transcription": 0.0,
        "gemini": 0.0,
    }
    debug_data = None
    debug_cleaned_audio = None

    try:
        # ── Step 0: Noise Reduction ──
        # nr_start = time.time()
        # cleaned_path = remove_background_noise(audio_file_path)
        # timing_stats["noise_reduction"] = time.time() - nr_start

        # Save cleaned audio bytes for debug
        if debug:
            try:
                # Use original audio since noise reduction is disabled
                with open(audio_file_path, "rb") as f:
                    debug_cleaned_audio = f.read()
            except Exception:
                debug_cleaned_audio = None

        # ── Step 1: Smart Chunking ──
        pk_start = time.time()
        # Pass audio_file_path directly instead of cleaned_path
        chunk_paths = smart_chunk_audio(audio_file_path)
        timing_stats["chunking"] = time.time() - pk_start

        # ── Step 2: Parallel Transcription ──
        tx_start = time.time()
        tx_result = transcribe_chunks_parallel(chunk_paths, space_name, hf_token)
        timing_stats["transcription"] = time.time() - tx_start

        raw_transcript = tx_result.full_transcript

        # Validate transcript
        if not raw_transcript or len(raw_transcript.strip()) < 3:
            raise Exception("EMPTY_TRANSCRIPT")

        # Collect debug data (read audio bytes before cleanup)
        if debug:
            debug_chunks = []
            for chunk in tx_result.chunk_details:
                audio_bytes = None
                try:
                    with open(chunk["path"], "rb") as f:
                        audio_bytes = f.read()
                except FileNotFoundError:
                    pass
                debug_chunks.append(
                    {
                        "index": chunk["index"],
                        "duration_s": chunk["duration_s"],
                        "text": chunk["text"],
                        "audio_bytes": audio_bytes,
                    }
                )
            debug_data = {
                "chunks": debug_chunks,
                "full_transcript": raw_transcript,
            }

        # ── Step 3: Gemini Formatting ──
        gem_start = time.time()
        chunk_texts = [chunk["text"] for chunk in tx_result.chunk_details]
        clinical_note = process_with_gemini(chunk_texts)
        timing_stats["gemini"] = time.time() - gem_start

    finally:
        # ── Cleanup chunks + cleaned audio ──
        cleanup_chunks(chunk_paths)
        if cleaned_path:
            try:
                os.remove(cleaned_path)
            except OSError:
                pass

    processing_time = time.time() - start_time
    return clinical_note, processing_time, timing_stats, debug_data, debug_cleaned_audio
