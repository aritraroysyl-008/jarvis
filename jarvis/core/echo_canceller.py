"""
Jarvis Voice Assistant - Core: Echo Cancellation

Optional acoustic echo cancellation using speexdsp.
Falls back gracefully if speexdsp is not installed.
"""

import threading
from collections import deque

import numpy as np


class EchoCanceller:
    """
    SpeexDSP-based echo canceller.

    Maintains a reference buffer of playback audio and subtracts it
    from the microphone signal to reduce speaker-to-mic bleed.
    """

    def __init__(self, frame_size: int, filter_len: int, mic_rate: int, enabled: bool = True):
        self.ec = None
        self.ref = deque()
        self.ref_len = 0
        self.lock = threading.Lock()
        self.frame_size = frame_size
        self.mic_rate = mic_rate

        if enabled:
            try:
                from speexdsp import EchoCanceller as SpeexEC
                self.ec = SpeexEC.create(frame_size, filter_len, mic_rate)
                print("AEC: speexdsp enabled")
            except Exception as e:
                print(f"AEC: disabled ({e}). Use headphones for best results.")

    @property
    def enabled(self) -> bool:
        return self.ec is not None

    def push_ref(self, pcm_out: bytes):
        """
        Called from Player callback with exactly what is sent to the speaker.

        Args:
            pcm_out: PCM16 bytes at TTS sample rate (24 kHz)
        """
        if not self.enabled:
            return

        # Convert to float for resampling
        x = np.frombuffer(pcm_out, dtype=np.int16).astype(np.float32)

        # Resample from TTS rate to MIC rate
        n_out = len(x) * self.mic_rate // 24000  # TTS_RATE = 24000
        if n_out == 0:
            return

        xs = np.interp(
            np.linspace(0, len(x), n_out, endpoint=False),
            np.arange(len(x)),
            x
        ).astype(np.int16)

        with self.lock:
            self.ref.append(xs)
            self.ref_len += len(xs)
            # Keep reference tightly in sync (~400 ms max)
            while self.ref_len > self.mic_rate * 0.4:
                self.ref_len -= len(self.ref.popleft())

    def pop_ref(self, n: int) -> np.ndarray:
        """
        Pop N samples from reference buffer for echo cancellation.

        Args:
            n: Number of samples needed (at MIC rate)

        Returns:
            int16 array of length n
        """
        out = np.zeros(n, dtype=np.int16)
        filled = 0

        with self.lock:
            while filled < n and self.ref:
                a = self.ref[0]
                take = min(n - filled, len(a))
                out[filled:filled + take] = a[:take]
                filled += take

                if take == len(a):
                    self.ref.popleft()
                else:
                    self.ref[0] = a[take:]

                self.ref_len -= take

        return out

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Apply echo cancellation to microphone signal.

        Args:
            mic: Microphone signal (int16, length = frame_size)
            ref: Reference signal from pop_ref (int16, length = frame_size)

        Returns:
            Echo-cancelled microphone signal (int16)
        """
        if not self.enabled or ref is None:
            return mic

        out = self.ec.process(mic.tobytes(), ref.tobytes())
        return np.frombuffer(out, dtype=np.int16)