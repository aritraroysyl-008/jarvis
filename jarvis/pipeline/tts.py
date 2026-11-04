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
        print(f"[TTS DEBUG] Response status: {resp.status_code}, headers: {dict(resp.headers)}")
        resp.raise_for_status()

        leftover = b""
        audio_sent = False
        total_bytes = 0
        for chunk in resp.iter_content(chunk_size=1024):
            if turn.cancel.is_set():
                return
            if not chunk:
                continue
            data = leftover + chunk
            n = len(data) & ~1  # Keep int16 alignment
            leftover = data[n:]
            if n:
                total_bytes += n
                pushed = player.push(turn, text, data[:n], False)
                if not pushed:
                    print(f"[TTS DEBUG] Player rejected push (cancelled?)")
                    return
                audio_sent = True
        print(f"[TTS DEBUG] Sent {total_bytes} bytes for: {text[:50]}...")
        if not audio_sent:
            print(f"\n[TTS WARNING] No audio data received for: {text[:60]}...")
    except requests.exceptions.ConnectionError:
        print(f"\n[TTS ERROR] Cannot connect to TTS server at {TTS_URL}")
    except requests.exceptions.Timeout:
        print(f"\n[TTS ERROR] TTS request timed out")
    except Exception as e:
        if not turn.cancel.is_set():
            print(f"\n[TTS ERROR] {e}")
            import traceback
            traceback.print_exc()
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        turn.tts_resp = None
        # Only mark as played if audio was actually sent or turn cancelled
        if not turn.cancel.is_set():
            player.push(turn, text, b"", True)  # Marks sentence as fully played


def tts_worker(turn: Turn, tts_q: queue.Queue, player: Player, tts_pending: threading.Event):
    """
    Background worker: pulls sentences from queue and streams TTS.

    Runs until turn cancelled, queue empty, and no more sentences pending.
    """
    try:
        _tts_loop(turn, tts_q, player, tts_pending)
    except Exception as e:
        print(f"\n[TTS WORKER CRASHED] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Only set tts_done when worker actually finishes all work
        turn.tts_done.set()


def _tts_loop(turn: Turn, tts_q: queue.Queue, player: Player, tts_pending: threading.Event):
    """Main TTS loop."""
    while True:
        try:
            text = tts_q.get(timeout=0.1)
        except queue.Empty:
            # Exit only if cancelled AND no more sentences coming AND queue empty
            if turn.cancel.is_set() and tts_pending.is_set():
                return
            continue
        if turn.cancel.is_set():
            return
        print(f"[TTS speak] {text[:60]}...", flush=True)
        speak(turn, text, player)