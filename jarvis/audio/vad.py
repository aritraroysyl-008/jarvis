"""
Jarvis Voice Assistant - Audio: Voice Activity Detection

Silero VAD-based voice activity detection with barge-in support.
"""

import queue
import threading
from collections import deque

import numpy as np
import torch

from jarvis.config.settings import (
    MIC_SAMPLE_RATE,
    VAD_FRAME_SIZE,
    VAD_THRESHOLD,
    VAD_SPEECH_FRAMES,
    VAD_SILENCE_FRAMES,
    BARGE_IN_THRESHOLD,
    BARGE_IN_FRAMES,
    PRE_ROLL_FRAMES,
    MAX_UTTERANCE_FRAMES,
)
from jarvis.core.session import Session
from jarvis.core.player import Player
from jarvis.core.echo_canceller import EchoCanceller


class VADWorker:
    """
    Voice Activity Detection worker.

    Runs in a background thread, consumes microphone frames,
    detects speech/silence, and emits utterances.
    """

    def __init__(
        self,
        mic_queue: queue.Queue,
        utterance_queue: queue.Queue,
        session: Session,
        player: Player,
        echo: EchoCanceller,
        stop_event: threading.Event,
        vad_model,
    ):
        self.mic_queue = mic_queue
        self.utterance_queue = utterance_queue
        self.session = session
        self.player = player
        self.echo = echo
        self.stop_event = stop_event
        self.vad_model = vad_model

    def run(self):
        """Main VAD loop - runs in background thread."""
        pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        captured = []
        recording = False
        barge = False
        speech_count = 0
        silence_count = 0

        self.vad_model.reset_states()

        def finish_utterance():
            nonlocal captured, recording, barge, speech_count, silence_count
            if captured:
                self.utterance_queue.put((np.concatenate(captured), barge))
            captured = []
            recording = False
            barge = False
            speech_count = 0
            silence_count = 0
            pre_roll.clear()
            self.vad_model.reset_states()

        while not self.stop_event.is_set():
            try:
                mic, ref = self.mic_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                # Apply echo cancellation
                frame = self.echo.process(mic, ref)
                if len(frame) != VAD_FRAME_SIZE:
                    continue

                # Run VAD
                tensor = torch.from_numpy(frame.astype(np.float32) / 32768.0)
                with torch.inference_mode():
                    prob = self.vad_model(tensor, MIC_SAMPLE_RATE).item()
            except Exception as e:
                print("\nVAD error:", e)
                continue

            active = self.session.assistant_active()

            if not recording:
                # Pre-roll buffer
                pre_roll.append(frame)

                # Threshold depends on whether assistant is speaking
                threshold = BARGE_IN_THRESHOLD if active else VAD_THRESHOLD
                needed = BARGE_IN_FRAMES if active else VAD_SPEECH_FRAMES

                speech_count = speech_count + 1 if prob >= threshold else 0

                if speech_count >= needed:
                    recording = True
                    silence_count = 0
                    captured = list(pre_roll)
                    if active:
                        barge = True
                        self.player.pause()  # Instant pause, Whisper decides if real
            else:
                captured.append(frame)
                silence_count = 0 if prob >= VAD_THRESHOLD else silence_count + 1

                if (
                    silence_count >= VAD_SILENCE_FRAMES
                    or len(captured) >= MAX_UTTERANCE_FRAMES
                ):
                    finish_utterance()


def create_vad_model():
    """Load and return Silero VAD model (stays on CPU)."""
    model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True, force_reload=False
    )
    model.eval()
    return model