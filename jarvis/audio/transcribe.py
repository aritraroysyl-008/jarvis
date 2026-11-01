"""
Jarvis Voice Assistant - Audio: Transcription

Whisper-based speech-to-text with hallucination filtering and echo detection.
"""

import re

import numpy as np
import whisper

from jarvis.config.settings import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    MIN_UTTERANCE_SAMPLES,
    HALLUCINATIONS,
    USE_FP16,
    DEVICE_STR,
)
from jarvis.core.turn import Turn


# Load Whisper model
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE_STR)


def transcribe(audio_i16: np.ndarray) -> str:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_i16: int16 audio array at 16 kHz

    Returns:
        Transcribed text, or empty string if too short / hallucination
    """
    if len(audio_i16) < MIN_UTTERANCE_SAMPLES:
        return ""

    audio = audio_i16.astype(np.float32) / 32768.0

    try:
        result = whisper_model.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            fp16=USE_FP16,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
    except Exception as e:
        print("\nWhisper error:", e)
        return ""

    # Filter low-confidence segments
    parts = [
        s["text"]
        for s in result.get("segments", [])
        if not (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0)
    ]
    text = " ".join(p.strip() for p in parts).strip()

    # Reject non-speech
    if not re.search(r"[A-Za-z0-9]", text):
        return ""

    # Filter common hallucinations
    cleaned = re.sub(r"[^a-z ]", "", text.lower()).strip()
    if cleaned in HALLUCINATIONS:
        return ""

    return text


def _words(text: str) -> list[str]:
    """Extract words from text for comparison."""
    return re.findall(r"[a-z']+", text.lower())


def looks_like_echo(text: str, turn: Turn | None) -> bool:
    """
    Check if user speech is mostly the assistant's own words (speaker bleed).

    Args:
        text: Transcribed user speech
        turn: Current turn (for assistant's spoken text)

    Returns:
        True if likely echo
    """
    if turn is None:
        return False

    words = _words(text)
    if len(words) < 3:
        return False

    said = set(_words(" ".join(turn.said)))
    overlap = sum(w in said for w in words)

    return overlap / len(words) >= 0.8