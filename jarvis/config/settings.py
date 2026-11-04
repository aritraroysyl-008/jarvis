"""
Jarvis Voice Assistant - Configuration

Centralized configuration for all modules.
Modify these values to customize behavior without touching code.
"""

import torch

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

# DEVICE DETECTION

torch.set_num_threads(1)

if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DEVICE_STR = str(DEVICE)
USE_FP16 = DEVICE_STR == "cuda"


