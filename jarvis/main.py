#!/usr/bin/env python3
"""
Jarvis - Modular Full-Duplex Voice Assistant with Tool Calling

Main entry point. Initializes all components and runs the main loop.

Architecture:
    config/     - Settings and configuration
    tools/      - Tool registry and built-in tools
    core/       - Turn, Player, Session, EchoCanceller
    pipeline/   - LLM streaming, TTS, turn orchestration
    audio/      - VAD, microphone, transcription
"""

import queue
import re
import threading

from jarvis.config.settings import LLM_MODELS_URL, LLM_MODEL, USE_AEC, AEC_FILTER_LEN
from jarvis.tools import list_tools
from jarvis.core import Turn, Player, EchoCanceller, Session
from jarvis.audio import (
    mic_queue,
    create_input_stream,
    create_vad_model,
    VADWorker,
    transcribe,
    looks_like_echo,
)
from jarvis.pipeline import run_turn


# Stop words that immediately interrupt the assistant
STOP_WORDS = [
    r"\bstop\b",
    r"\bshut\s+(up|the\s+fuck\s+up)\b",
    r"\bbe\s+quiet\b",
    r"\bquiet\b",
    r"\bshush\b",
    r"\bhalt\b",
    r"\bcancel\b",
    r"\babort\b",
    r"\bshut\s+it\b",
]
STOP_REGEX = re.compile("|".join(STOP_WORDS), re.IGNORECASE)


def is_stop_command(text: str) -> bool:
    """Check if text contains a stop command."""
    return bool(STOP_REGEX.search(text))


def resolve_llm_model() -> str:
    """Query LLM server for available model, fallback to config."""
    import requests
    try:
        r = requests.get(LLM_MODELS_URL, timeout=5)
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            return data[0]["id"]
    except Exception as e:
        print("Warning: could not query LLM /models:", e)
    return LLM_MODEL


def main():
    print("=" * 50)
    print("Jarvis Voice Assistant - Modular Edition")
    print("=" * 50)

    # Resolve LLM model
    model = resolve_llm_model()
    print(f"Using LLM model: {model}")
    print(f"Tools: {', '.join(list_tools()) or 'none'}")
    print(f"AEC: {'enabled' if USE_AEC else 'disabled'}")

    # Initialize core components
    echo = EchoCanceller(
        frame_size=512,  # VAD_FRAME_SIZE
        filter_len=AEC_FILTER_LEN,
        mic_rate=16000,  # MIC_SAMPLE_RATE
        enabled=USE_AEC,
    )

    player = Player(echo_canceller=echo)
    session = Session()

    # Load VAD model
    vad_model = create_vad_model()

    # Start VAD worker
    stop_event = threading.Event()
    utterance_queue = queue.Queue()

    vad_worker = VADWorker(
        mic_queue=mic_queue,
        utterance_queue=utterance_queue,
        session=session,
        player=player,
        echo=echo,
        stop_event=stop_event,
        vad_model=vad_model,
    )
    vad_thread = threading.Thread(target=vad_worker.run, daemon=True)

    # Main audio context
    with player, create_input_stream(echo):
        vad_thread.start()
        print("\nListening (full duplex). Talk any time, Ctrl+C to quit.\n")

        try:
            while True:
                # Commit finished turns
                session.commit_if_finished()

                # Get next utterance
                try:
                    audio, barge = utterance_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Transcribe
                text = transcribe(audio)

                # Filter echo / hallucinations
                if not text or (barge and looks_like_echo(text, session.turn)):
                    if barge:
                        player.resume()  # False alarm, resume playback
                    continue

                # Check for stop commands - immediate interrupt
                if is_stop_command(text):
                    print(f"\n\nYou: {text}")
                    print("[Stop command detected - interrupting]")
                    session.cancel_current(player)
                    player.resume()
                    continue

                # Ensure player is resumed after any barge-in (in case it wasn't echo)
                if barge:
                    player.resume()

                print(f"\n\nYou: {text}")
                session.start_turn(text, player)

                # Start turn in background
                turn = session.turn
                turn.thread = threading.Thread(
                    target=run_turn,
                    args=(turn, list(session.history), player),
                    daemon=True,
                )
                print("Assistant: ", end="", flush=True)
                turn.thread.start()

        except KeyboardInterrupt:
            print("\n\nBye.")
        finally:
            stop_event.set()
            session.cancel_current()


if __name__ == "__main__":
    main()