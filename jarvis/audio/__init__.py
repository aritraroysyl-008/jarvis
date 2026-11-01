"""Jarvis Voice Assistant - Audio Package"""

from jarvis.audio.vad import VADWorker, create_vad_model
from jarvis.audio.microphone import mic_queue, mic_callback, create_input_stream
from jarvis.audio.transcribe import transcribe, looks_like_echo, whisper_model

__all__ = [
    "VADWorker",
    "create_vad_model",
    "mic_queue",
    "mic_callback",
    "create_input_stream",
    "transcribe",
    "looks_like_echo",
    "whisper_model",
]