"""
Jarvis Voice Assistant - Core: Turn Management

A Turn represents one assistant response cycle: LLM streaming + tool calls + TTS + playback.
Supports cancellation, barge-in, and history commit.
"""

import threading
import uuid


class Turn:
    """
    Represents a single assistant turn (response cycle).

    Lifecycle:
        1. Created by Session.start_turn()
        2. LLM streams response (may include tool calls)
        3. Tools execute in background
        4. TTS generates audio chunks
        5. Player plays audio
        6. Turn finishes or is cancelled (barge-in)
        7. Session.commit() folds results into history
    """

    def __init__(self):
        self.cancel = threading.Event()       # Set to abort the turn
        self.finished = threading.Event()     # Set when turn completes naturally
        self.tts_done = threading.Event()     # All TTS audio for this turn queued
        self.new_messages = []                # Completed assistant(tool_calls)/tool messages
        self.final_text = ""                  # Full text of the final assistant reply
        self.said = []                        # Every sentence sent to TTS (for echo check)
        self.spoken = []                      # Sentences that were fully played
        self.llm_resp = None                  # Active LLM HTTP response (for abort)
        self.tts_resp = None                  # Active TTS HTTP response (for abort)
        self.thread = None                    # Worker thread running run_turn()
        self.committed = False                # Whether Session.commit() has run

    def abort(self):
        """Abort the turn: cancel events, close HTTP connections, flush player."""
        self.cancel.set()
        for resp in (self.llm_resp, self.tts_resp):
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass

    def generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:12]}"