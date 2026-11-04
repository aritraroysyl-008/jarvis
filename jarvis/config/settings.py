"""
Jarvis Voice Assistant - Configuration

Centralized configuration for all modules.
Modify these values to customize behavior without touching code.
"""

from typing import Optional, List

# Torch is imported lazily on first access to avoid a slow / hard-failing
# module-level import for consumers that don't need it.
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# NETWORK / SERVICE ENDPOINTS

# Text-to-Speech server (PocketTTS compatible)
TTS_URL = "http://127.0.0.1:8080/v1/audio/speech"
TTS_VOICE = "alba"

# LLM Server (OpenAI-compatible /v1/chat/completions)
LLM_BASE_URL = "http://127.0.0.1:1337/v1"
LLM_CHAT_URL = f"{LLM_BASE_URL}/chat/completions"
LLM_MODELS_URL = f"{LLM_BASE_URL}/models"
LLM_MODEL = "no_func.gguf"  # fallback if /models lookup fails


TTS_SAMPLE_RATE = 24000
MIC_SAMPLE_RATE = 16000

OUTPUT_BLOCK_SIZE = 480               # 20 ms @ 24 kHz
MIC_QUEUE_SIZE = 200

# VAD / TURN TAKING

VAD_FRAME_SIZE = 512                  # Silero needs exactly 512 samples @ 16 kHz
VAD_THRESHOLD = 0.60                  # Speech threshold while assistant is silent
VAD_SPEECH_FRAMES = 3                 # Consecutive speech frames to start recording
VAD_SILENCE_FRAMES = 20               # ~640 ms of silence ends utterance
BARGE_IN_THRESHOLD = 0.80             # Stricter while assistant is talking
BARGE_IN_FRAMES = 6                   # ~190 ms of speech to trigger barge-in
PRE_ROLL_FRAMES = 10                  # Frames to keep before speech starts
MAX_UTTERANCE_FRAMES = 30 * MIC_SAMPLE_RATE // VAD_FRAME_SIZE
MIN_UTTERANCE_SAMPLES = MIC_SAMPLE_RATE // 4  # 250 ms minimum

# ECHO CANCELLATION (optional, needs `pip install speexdsp`)

USE_AEC = True
AEC_FILTER_LEN = 4096                 # ~256 ms tail

# LLM SETTINGS

MAX_TOOL_ROUNDS = 4
MAX_HISTORY = 40
MAX_CHUNK_CHARS = 300                 # Split very long sentences at comma
CHUNK_SENTENCES = 5                   # Sentences per TTS chunk
FIRST_CHUNK_SENTENCES = 5             # First chunk size (lower = starts sooner)

# Jitter buffer: playback starts once this much audio is queued (or TTS finished)
PREBUFFER_SECONDS = 0.15
PREBUFFER_BYTES = int(TTS_SAMPLE_RATE * 2 * PREBUFFER_SECONDS)

PAUSE_TIMEOUT = 45.0                  # Safety: auto-resume if pause never resolved

# WHISPER SETTINGS

WHISPER_MODEL = "tiny"
WHISPER_LANGUAGE = "en"               # None = auto-detect (tiny is weak at that)

# SYSTEM PROMPT

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep replies short, conversational, and easy to speak aloud. "
    "Do not use markdown, bullet points, code blocks, or emojis. "
    "When a tool would give a better answer, call it, then reply in one or two spoken sentences. "
    "Use full curse words naturally (fucking, shit, damn, etc.) - never censor with asterisks like f*cking or sh*t."
)

# HALLUCINATION FILTER

HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "bye", "uh", "um", "hmm"
}

# DEVICE SELECTION (set via CLI flag or settings)
MIC_DEVICE: Optional[int] = None      # Microphone input device index
OUTPUT_DEVICE: Optional[int] = None   # Audio output device index


# CONFIG VALIDATION

import urllib.request
import urllib.error


def validate_config() -> List[str]:
    """Check that TTS and LLM servers are reachable.

    Returns a list of error messages (empty if all OK).
    """
    errors = []

    # Check TTS server
    try:
        req = urllib.request.Request(
            TTS_URL,
            method="OPTIONS",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status not in (200, 204, 405):
                errors.append(f"TTS server at {TTS_URL} returned status {resp.status}")
    except urllib.error.URLError as e:
        errors.append(f"Cannot reach TTS server at {TTS_URL}: {e}")
    except Exception as e:
        errors.append(f"TTS server check failed: {e}")

    # Check LLM server
    try:
        req = urllib.request.Request(LLM_MODELS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status not in (200,):
                errors.append(f"LLM server at {LLM_MODELS_URL} returned status {resp.status}")
    except urllib.error.URLError as e:
        errors.append(f"Cannot reach LLM server at {LLM_MODELS_URL}: {e}")
    except Exception as e:
        errors.append(f"LLM server check failed: {e}")

    return errors

def _init_device():
    torch = _get_torch()
    torch.set_num_threads(1)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return str(device), device == torch.device("cuda")


DEVICE_STR, USE_FP16 = _init_device()


