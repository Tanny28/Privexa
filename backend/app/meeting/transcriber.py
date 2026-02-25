# ──────────────────────────────────────────────
# Audio Transcriber — faster-whisper integration
# Ported from teammate's notebook with async support
# ──────────────────────────────────────────────
import time
import logging
import asyncio
from typing import Optional, Tuple
from functools import lru_cache

import torch
from faster_whisper import WhisperModel

from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
    WHISPER_VAD_MIN_SILENCE_MS,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
    """
    Load and cache the Whisper model (singleton).
    Auto-detects GPU vs CPU and selects optimal compute type.

    Returns:
        WhisperModel instance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        f"Loading Whisper model '{WHISPER_MODEL_SIZE}' "
        f"on {device} with {compute_type}"
    )

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        num_workers=4,
    )

    logger.info("Whisper model loaded successfully")
    return model


def transcribe_audio_sync(file_path: str) -> Tuple[str, dict]:
    """
    Transcribe an audio file to text (synchronous).
    Uses teammate's optimized settings for maximum speed.

    Args:
        file_path: Path to the audio file (.mp3, .wav, etc.)

    Returns:
        Tuple of (transcript_text, metadata_dict).
        metadata includes: duration, elapsed, speed_ratio.
    """
    model = _get_whisper_model()
    start = time.time()

    segments, info = model.transcribe(
        file_path,
        beam_size=WHISPER_BEAM_SIZE,
        language=WHISPER_LANGUAGE,
        vad_filter=WHISPER_VAD_FILTER,
        vad_parameters=dict(
            min_silence_duration_ms=WHISPER_VAD_MIN_SILENCE_MS,
        ),
        condition_on_previous_text=False,
        temperature=0.0,
    )

    transcript_parts = []
    for segment in segments:
        transcript_parts.append(segment.text)

    transcript = " ".join(transcript_parts).strip()

    elapsed = time.time() - start
    speed_ratio = info.duration / elapsed if elapsed > 0 else 0

    metadata = {
        "audio_duration_sec": round(info.duration, 1),
        "transcription_time_sec": round(elapsed, 1),
        "speed_ratio": round(speed_ratio, 1),
        "language": info.language,
        "language_probability": round(info.language_probability, 2),
    }

    logger.info(
        f"Transcription done in {elapsed:.1f}s | "
        f"Audio: {info.duration:.1f}s | "
        f"Speed: {speed_ratio:.1f}x realtime"
    )

    return transcript, metadata


async def transcribe_audio(file_path: str) -> Tuple[str, dict]:
    """
    Async wrapper for audio transcription.
    Runs the CPU/GPU-bound transcription in a thread pool
    so it doesn't block the FastAPI event loop.

    Args:
        file_path: Path to the audio file.

    Returns:
        Tuple of (transcript_text, metadata_dict).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        transcribe_audio_sync,
        file_path,
    )


def is_whisper_available() -> bool:
    """Check if Whisper can be loaded (for health checks)."""
    try:
        _get_whisper_model()
        return True
    except Exception as e:
        logger.error(f"Whisper not available: {e}")
        return False
