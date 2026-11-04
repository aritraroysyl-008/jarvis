# Jarvis

**Jarvis** — a modular, full-duplex Python voice assistant with tool calling, barge-in interruption, echo cancellation, and streaming LLM + TTS.

Built as an installable Python package (`jarvis/`), it runs entirely on your machine and streams speech as the LLM generates it — no waiting for full sentences before audio starts.

## Features

- **Full-duplex audio** — listen and speak at the same time; no turn-taking pauses
- **Barge-in / interruption** — talk over the assistant to cut it off instantly
- **Tool calling (function calling)** — LLM can call registered tools and speak the results
- **Streaming LLM + TTS** — audio starts playing while the LLM is still streaming tokens
- **Echo cancellation** — optional SpeexDSP AEC so the mic doesn't hear its own output
- **PocketTTS.cpp TTS server** — GGML-based, zero-shot voice-cloning TTS via OpenAI-compatible HTTP API ([ServeurpersoCom/pockettts.cpp](https://github.com/ServeurpersoCom/pockettts.cpp))
- **Silero VAD** — voice activity detection for utterance segmentation
- **Whisper transcription** — local speech-to-text
- **Modular, extensible** — clean separation of config, core, pipeline, audio, and tools

## Architecture

```
jarvis/
├── __init__.py          # Package entry, version
├── main.py              # Main loop: VAD → transcribe → LLM → TTS → playback
├── config/
│   ├── __init__.py
│   └── settings.py      # All endpoints, sample rates, VAD, system prompt
├── core/
│   ├── __init__.py
│   ├── turn.py          # Turn lifecycle (cancel, commit, spoken tracking)
│   ├── player.py        # Always-on output stream with jitter buffer + barge-in
│   ├── session.py       # Conversation history + turn management
│   └── echo_canceller.py
├── pipeline/
│   ├── __init__.py
│   ├── llm.py           # Streaming OpenAI-compatible chat + tool calls
│   ├── tts.py           # Streaming TTS client (PocketTTS.cpp /v1/audio/speech)
│   └── run_turn.py      # Turn orchestration: LLM + tools + TTS + playback
├── audio/
│   ├── __init__.py
│   ├── microphone.py    # Input stream + mic queue
│   ├── vad.py           # Silero VAD worker
│   └── transcribe.py    # Whisper transcription
└── tools/
    ├── __init__.py
    └── ...              # Built-in tool registry
```

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/aritraroysyl-008/jarvis.git
cd jarvis
```

### 2. Create a virtual environment and install dependencies

Jarvis is a pure-Python package. Use `uv` or `pip`:

```bash
# With uv (recommended)
uv venv .venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .

# Or with pip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Set up the TTS server (PocketTTS.cpp)

Jarvis speaks through a PocketTTS.cpp `tts-server` at `http://127.0.0.1:8080/v1/audio/speech` — an OpenAI-compatible HTTP server with a voice directory and runtime voice registry.

Build and run PocketTTS.cpp ([source](https://github.com/ServeurpersoCom/pockettts.cpp)):

```bash
# Clone PocketTTS.cpp (with submodules)
git clone --recurse-submodules https://github.com/ServeurpersoCom/pockettts.cpp.git
cd pockettts.cpp

# Build (choose one backend)
./buildcpu.sh                  # CPU only
# ./buildcuda.sh               # NVIDIA GPU
# ./buildvulkan.sh             # AMD/Intel GPU (Vulkan)
# ./buildall.sh                # all backends, runtime DL loading

# Download pre-converted GGUF models (English + French, Q8_0, with voices)
./models.sh

# Start the OpenAI-compatible TTS server (port 8080)
./tts-server --port 8080
```

The server exposes:
- `POST /v1/audio/speech` — OpenAI-compatible TTS (what Jarvis calls)
- Runtime voice registry — register predefined or cloned voices at startup

Jarvis's `TTS_VOICE` in `jarvis/config/settings.py` should match a registered voice name (default: `"alba"`).

### 4. Register a custom voice (optional)

Use `register_voice.bat` (Windows) or `register_voice.sh` (Linux/macOS) to convert an MP3 to 24 kHz mono WAV and register it with the PocketTTS `tts-server` voice registry:

```bash
# Windows
register_voice.bat

# Linux / macOS
chmod +x register_voice.sh
./register_voice.sh
```

It will prompt for the MP3 path and a voice name, then register it with the `tts-server` runtime voice registry. Set `TTS_VOICE` in `jarvis/config/settings.py` to the name you chose.

### 5. Set up the LLM server

Jarvis talks to any OpenAI-compatible LLM server at `http://127.0.0.1:1337/v1` (e.g. Ollama, LM Studio, llama.cpp, vLLM). Configure `LLM_MODEL` in `jarvis/config/settings.py`.

### 5. Run Jarvis

```bash
python -m jarvis.main
# or
python jarvis/main.py
```

You'll see:
```
Jarvis Voice Assistant - Modular Edition
Using LLM model: no_func.gguf
Tools: ...
AEC: enabled

Listening (full duplex). Talk any time, Ctrl+C to quit.
```

Speak anytime — Jarvis transcribes, streams the LLM response, and plays TTS audio as it arrives. Say "stop" to interrupt. `Ctrl+C` to quit.

## Configuration

All settings live in `jarvis/config/settings.py`:

| Setting | Default | Description |
|---|---|---|
| `TTS_URL` | `http://127.0.0.1:8080/v1/audio/speech` | PocketTTS.cpp OpenAI-compatible endpoint |
| `TTS_VOICE` | `"alba"` | Registered voice name in PocketTTS `tts-server` voice registry |
| `LLM_BASE_URL` | `http://127.0.0.1:1337/v1` | OpenAI-compatible LLM server |
| `LLM_MODEL` | `"no_func.gguf"` | Fallback model name |
| `TTS_SAMPLE_RATE` | `24000` | TTS audio sample rate (Hz) |
| `MIC_SAMPLE_RATE` | `16000` | Microphone sample rate (Hz) |
| `USE_AEC` | `True` | Enable echo cancellation (`pip install speexdsp`) |
| `WHISPER_MODEL` | `"tiny"` | Whisper STT model size |
| `SYSTEM_PROMPT` | — | System prompt sent to the LLM |

## Install as a package

```bash
pip install -e .
```

Then import and use Jarvis programmatically:

```python
from jarvis.main import main
main()
```

Or run the entry point:

```bash
python -m jarvis.main
```

## Requirements

- Python 3.10+
- `uv` (recommended) or `pip`
- PocketTTS.cpp server running locally (for TTS)
- OpenAI-compatible LLM server (for chat)
- `sounddevice`, `requests`, `torch`, `whisper`, `silero-vad`, `numpy`

Install runtime deps:

```bash
uv pip install sounddevice requests torch whisper numpy
# optional: uv pip install speexdsp   # for echo cancellation
```

## License

MIT

---

TTS powered by [PocketTTS.cpp](https://github.com/ServeurpersoCom/pockettts.cpp) — GGML-based local AI TTS with voice cloning, CPU/CUDA/Metal/Vulkan support.
