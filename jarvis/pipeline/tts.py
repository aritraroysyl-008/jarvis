"""
Jarvis Voice Assistant - Pipeline: Text-to-Speech

Handles streaming TTS requests and audio playback queuing.
"""

import queue
import threading

import requests

from jarvis.config.settings import TTS_URL, TTS_VOICE
from jarvis.core.turn import Turn
from jarvis.core.player import Player


def speak(turn: Turn, text: str, player: Player):
    """
    Stream TTS for a single sentence and queue audio for playback.

    Args:
        turn: Current turn (for cancellation)
        text: Sentence to speak
        player: Player instance for audio queuing
    """
    resp = None
    try:
        resp = requests.post(
            TTS_URL,
            json={"input": text, "voice": TTS_VOICE, "response_format": "pcm"},
            stream=True,
            timeout=(10, 120),
        )
        turn.tts_resp = resp
        resp.raise_for_status()

        leftover = b""
        for chunk in resp.iter_content(chunk_size=1024):
            if turn.cancel.is_set():
                return
            if not chunk:
                continue
            data = leftover + chunk
            n = len(data) & ~1  # Keep int16 alignment
            leftover = data[n:]
            if n and not player.push(turn, text, data[:n], False):
                return
    except Exception as e:
        if not turn.cancel.is_set():
            print("\nTTS error:", e)
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        turn.tts_resp = None
        if not turn.cancel.is_set():
            player.push(turn, text, b"", True)  # Marks sentence as fully played


def tts_worker(turn: Turn, tts_q: queue.Queue, player: Player):
    """
    Background worker: pulls sentences from queue and streams TTS.

    Runs until turn cancelled or None sentinel received.
    """
    try:
        _tts_loop(turn, tts_q, player)
    finally:
        turn.tts_done.set()


def _tts_loop(turn: Turn, tts_q: queue.Queue, player: Player):
    """Main TTS loop."""
    while True:
        try:
            text = tts_q.get(timeout=0.1)
        except queue.Empty:
            if turn.cancel.is_set():
                return
            continue
        if text is None or turn.cancel.is_set():
            return
        speak(turn, text, player)