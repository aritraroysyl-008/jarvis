"""
Jarvis Voice Assistant - Pipeline: Turn Orchestration

Coordinates LLM streaming, tool execution, TTS, and playback for a single turn.
"""

import queue
import threading
import time

from typing import Callable, Optional

from jarvis.config.settings import MAX_TOOL_ROUNDS
from jarvis.core.turn import Turn
from jarvis.core.player import Player
from jarvis.pipeline.llm import run_llm_with_tools
from jarvis.pipeline.tts import tts_worker


def run_turn(turn: Turn, messages: list, player: Player, speak_fn: Optional[Callable[[str], None]] = None):
    """
    Execute a complete turn: LLM + tools + TTS + playback.

    This runs in a background thread. The main thread monitors
    turn.finished and turn.cancel for lifecycle management.

    Args:
        turn: Turn object to execute
        messages: Conversation history
        player: Player instance for audio output
    """
    tts_q = queue.Queue()
    tts_pending = threading.Event()  # Set when all TTS sentences are queued
    tts_thread = threading.Thread(target=tts_worker, args=(turn, tts_q, player, tts_pending), daemon=True)
    tts_thread.start()

    try:
        run_llm_with_tools(turn, messages, tts_q, MAX_TOOL_ROUNDS, speak_fn=speak_fn)
    except Exception as e:
        if not turn.cancel.is_set():
            print("\nTurn error:", e)
    finally:
        # Signal no more sentences will be queued
        tts_pending.set()
        # Wait for TTS worker to finish all sentences
        tts_thread.join()
        # Wait for playback to finish
        while not turn.cancel.is_set() and player.pending(turn):
            time.sleep(0.02)
        turn.finished.set()