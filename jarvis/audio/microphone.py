"""
Jarvis Voice Assistant - Audio: Microphone Input

Handles microphone callback and audio queue management.
"""

import queue

import numpy as np
import sounddevice as sd

from typing import Optional

from jarvis.config.settings import MIC_SAMPLE_RATE, VAD_FRAME_SIZE, MIC_QUEUE_SIZE
from jarvis.core.echo_canceller import EchoCanceller


# Global microphone queue
mic_queue = queue.Queue(maxsize=MIC_QUEUE_SIZE)


def mic_callback(indata, frames, time_info, status):
    """
    Sounddevice input callback - runs on audio thread.

    Captures microphone audio and reference signal for echo cancellation.
    """
    mic = indata[:, 0].copy()
    ref = None
    # Note: echo_canceller.push_ref is called from Player callback
    # We pop reference here for the VAD frame
    # The echo canceller instance is passed from main
    try:
        mic_queue.put_nowait((mic, ref))
    except queue.Full:
        pass  # Drop frame if queue full


def create_input_stream(echo: EchoCanceller, callback=None):
    """
    Create and return a sounddevice InputStream.

    Args:
        echo: EchoCanceller instance
        callback: Optional custom callback (uses default if None)

    Returns:
        sounddevice.InputStream (not started)
    """
    if callback is None:
        def default_callback(indata, frames, time_info, status):
            mic = indata[:, 0].copy()
            ref = echo.pop_ref(frames) if echo.enabled else None
            try:
                mic_queue.put_nowait((mic, ref))
            except queue.Full:
                pass
        callback = default_callback

    return sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=VAD_FRAME_SIZE,
        latency="low",
        callback=callback,
    )