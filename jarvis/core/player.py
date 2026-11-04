"""
Jarvis Voice Assistant - Core: Audio Player

Always-on output stream with instant pause/flush for barge-in support.
Uses a jitter buffer to prevent gaps at chunk boundaries.
"""

import threading
import time
from collections import deque

import sounddevice as sd

from jarvis.config.settings import (
    TTS_SAMPLE_RATE,
    OUTPUT_BLOCK_SIZE,
    PREBUFFER_BYTES,
    PAUSE_TIMEOUT,
)


class Player:
    """
    Always-on audio output stream.

    Features:
        - Instant pause/resume for barge-in
        - Jitter buffer prevents gaps between TTS chunks
        - Thread-safe queue with turn tracking
        - Echo canceller reference signal feed
    """

    def __init__(self, echo_canceller=None):
        self.echo = echo_canceller
        self.q = deque()              # Items: [turn, sentence_text, pcm_bytes, is_last_chunk]
        self.lock = threading.Lock()
        self.paused = False
        self.paused_at = 0.0
        self.buffered = 0             # Bytes of audio currently queued
        self.started = False          # False = still filling jitter buffer

        self.stream = sd.RawOutputStream(
            samplerate=TTS_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=OUTPUT_BLOCK_SIZE,
            latency="low",
            callback=self._callback,
        )

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def push(self, turn, text: str, data: bytes, last: bool) -> bool:
        """
        Queue audio for playback.

        Args:
            turn: The Turn this audio belongs to (for cancellation tracking)
            text: Sentence text (for echo detection)
            data: PCM16 audio bytes
            last: True if this is the final chunk for the sentence

        Returns:
            False if turn was cancelled, True if queued.
        """
        with self.lock:
            if turn.cancel.is_set():
                print(f"[PLAYER DEBUG] Push rejected: turn cancelled")
                return False
            print(f"[PLAYER DEBUG] Push: {len(data)} bytes, last={last}, qsize={len(self.q)}, buffered={self.buffered}, started={self.started}, paused={self.paused}")
            self.q.append([turn, text, data, last])
            self.buffered += len(data)
        return True

    def flush(self):
        """Clear all queued audio (used on barge-in)."""
        with self.lock:
            self.q.clear()
            self.buffered = 0
            self.started = False
            self.paused = False

    def pause(self):
        """Instant pause playback (barge-in)."""
        with self.lock:
            if not self.paused:
                self.paused = True
                self.paused_at = time.monotonic()

    def resume(self):
        """Resume playback after pause."""
        with self.lock:
            self.paused = False

    def pending(self, turn) -> bool:
        """Check if a turn still has audio queued or playing."""
        with self.lock:
            return any(item[0] is turn for item in self.q)

    def _callback(self, outdata, frames, time_info, status):
        """Sounddevice output callback - runs on audio thread."""
        need = frames * 2  # bytes needed (int16 = 2 bytes/sample)
        buf = bytearray()

        with self.lock:
            # Safety: auto-resume if pause timeout exceeded
            if self.paused and time.monotonic() - self.paused_at > PAUSE_TIMEOUT:
                self.paused = False
                print(f"[PLAYER DEBUG] Auto-resume after pause timeout")

            # Jitter buffer: wait for enough audio before starting
            if not self.started and self.q:
                turn = self.q[0][0]
                tts_done = getattr(turn, 'tts_done', None)
                # Start if: buffer full, OR tts_done, OR first item is a complete sentence (last=True)
                first_item_last = self.q[0][3] if self.q else False
                if self.buffered >= PREBUFFER_BYTES or (tts_done is not None and tts_done.is_set()) or first_item_last:
                    self.started = True
                    print(f"[PLAYER DEBUG] Started playback: buffered={self.buffered}, prebuffer={PREBUFFER_BYTES}, tts_done={tts_done.is_set() if tts_done else 'N/A'}, first_last={first_item_last}")

            # Safety: force resume if paused but we have audio queued (barge-in false alarm)
            if self.paused and self.q and self.started:
                print(f"[PLAYER DEBUG] Force resume: was paused but audio queued")
                self.paused = False

            # Pull audio from queue
            if self.started and not self.paused:
                while len(buf) < need and self.q:
                    item = self.q[0]
                    data = item[2]
                    take = need - len(buf)

                    if len(data) <= take:
                        # Consume entire chunk
                        buf += data
                        self.buffered -= len(data)
                        self.q.popleft()
                        if item[3]:  # sentence fully played
                            item[0].spoken.append(item[1])
                    else:
                        # Partial consume
                        buf += data[:take]
                        self.buffered -= take
                        item[2] = data[take:]

                if not self.q:
                    self.started = False  # Underrun: rebuffer before resuming
                    print(f"[PLAYER DEBUG] Underrun, stopping playback")
            elif self.started and self.paused:
                print(f"[PLAYER DEBUG] Paused, not pulling audio")

        # Pad with silence if needed
        if len(buf) < need:
            buf += bytes(need - len(buf))

        out = bytes(buf)
        outdata[:] = out

        # Feed echo canceller with what we actually played
        if self.echo:
            self.echo.push_ref(out)