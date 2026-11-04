"""
Jarvis application class.

Encapsulates the full-duplex voice assistant: VAD, transcription,
streaming LLM, tool calling, TTS, and playback.

Example:
    >>> from jarvis import JARVIS
    >>> jarvis = JARVIS()
    >>> jarvis.run()
"""

import queue
import re
import threading

from jarvis.config.settings import (
    LLM_MODELS_URL,
    LLM_MODEL,
    USE_AEC,
    AEC_FILTER_LEN,
)
from jarvis.core import Turn, Player, EchoCanceller, Session
from jarvis.pipeline import run_turn
from jarvis.triggers import TriggerManager


class JARVIS:
    """Modular full-duplex voice assistant with tool calling.

    Attributes:
        session: Conversation session manager.
        player: Audio output player with barge-in support.
        echo: Echo canceller instance, or None if disabled.
        running: Whether the assistant main loop is active.
    """

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

    def __init__(self):
        self.session = Session()
        self.player = None
        self.echo = None
        self.vad_model = None
        self.vad_thread = None
        self.stop_event = threading.Event()
        self.utterance_queue = queue.Queue()
        self.running = False
        self.trigger_manager = TriggerManager()

    # Lifecycle

    def start(self):
        """Initialize components and run the main assistant loop."""
        from jarvis.audio import (
            mic_queue,
            create_input_stream,
            create_vad_model,
            VADWorker,
            transcribe,
            looks_like_echo,
        )

        from jarvis.config.settings import (
            TTS_SAMPLE_RATE,
            MIC_SAMPLE_RATE,
            VAD_FRAME_SIZE,
        )

        # Create echo canceller before player
        self.echo = EchoCanceller(
            frame_size=512,
            filter_len=AEC_FILTER_LEN,
            mic_rate=16000,
            enabled=USE_AEC,
        )

        self.player = Player(echo_canceller=self.echo)
        self.vad_model = create_vad_model()
        self.stop_event = threading.Event()
        self.utterance_queue = queue.Queue()

        vad_worker = VADWorker(
            mic_queue=mic_queue,
            utterance_queue=self.utterance_queue,
            session=self.session,
            player=self.player,
            echo=self.echo,
            stop_event=self.stop_event,
            vad_model=self.vad_model,
        )
        self.vad_thread = threading.Thread(target=vad_worker.run, daemon=True)

        self.running = True
        self.trigger_manager.start()
        print("Jarvis")
        print("Listening (full duplex). Talk any time. Ctrl+C to quit.\n")

        with self.player, create_input_stream(self.echo):
            self.vad_thread.start()
            try:
                while self.running:
                    self.session.commit_if_finished()

                    try:
                        audio, barge = self.utterance_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    text = transcribe(audio)
                    # Check keyword triggers (auto-fire .jext triggers)
                    trigger_result = self.trigger_manager.check_keyword(text)
                    if trigger_result:
                        print(f"\n[Trigger auto-fired] {trigger_result}")
                    if not text or (barge and looks_like_echo(text, self.session.turn)):
                        if barge:
                            self.player.resume()
                        continue

                    if self.is_stop_command(text):
                        print(f"\n\nYou: {text}")
                        print("[Stop command detected - interrupting]")
                        self.session.cancel_current(self.player)
                        self.player.resume()
                        continue

                    if barge:
                        self.player.resume()

                    print(f"\n\nYou: {text}")
                    self.session.start_turn(text, self.player)

                    turn = self.session.turn
                    turn.thread = threading.Thread(
                        target=run_turn,
                        args=(turn, list(self.session.history), self.player),
                        daemon=True,
                    )
                    print("Assistant: ", end="", flush=True)
                    turn.thread.start()

            except KeyboardInterrupt:
                print("\n\nBye.")
            finally:
                self.stop()

    def run(self):
        """Run the assistant. Alias for :meth:`start`."""
        self.start()

    def stop(self):
        """Stop the assistant and release resources."""
        self.running = False
        self.trigger_manager.stop()
        self.stop_event.set()
        if self.session and self.session.turn:
            self.session.cancel_current()
        if self.vad_thread and self.vad_thread.is_alive():
            self.vad_thread.join(timeout=2)

    # Helpers

    def is_stop_command(self, text: str) -> bool:
        """Return True if *text* contains a stop/interrupt command."""
        return bool(self.STOP_REGEX.search(text))


def main():
    """Console entry point for ``python -m jarvis.main``."""
    JARVIS().run()


if __name__ == "__main__":
    main()
