"""Phase 10 — audio format constants for the in-house loop.

Single source of truth for the rates, so the mic/SIP ingress, Silero, Deepgram, Cartesia, and
the playback device all agree and nothing hidden resamples on the audio path (the Phase-1
choppy-playback diagnosis).

``INPUT_SAMPLE_RATE`` is 16 kHz because Silero's ONNX model accepts only 8 kHz or 16 kHz, and
16 kHz is also what Deepgram's streaming endpoint wants — so one rate serves both and the
capture path needs no conversion at all.
"""

from __future__ import annotations

INPUT_SAMPLE_RATE = 16000    # mic / SIP ingress -> VAD + STT
OUTPUT_SAMPLE_RATE = 24000   # Cartesia TTS -> playback / SIP egress
NUM_CHANNELS = 1
FRAME_MS = 20
BYTES_PER_SAMPLE = 2

INPUT_FRAME_BYTES = int(INPUT_SAMPLE_RATE * FRAME_MS / 1000) * BYTES_PER_SAMPLE * NUM_CHANNELS
OUTPUT_FRAME_BYTES = int(OUTPUT_SAMPLE_RATE * FRAME_MS / 1000) * BYTES_PER_SAMPLE * NUM_CHANNELS


def duration_ms(pcm: bytes, sample_rate: int = INPUT_SAMPLE_RATE, channels: int = NUM_CHANNELS) -> float:
    """Wall-clock duration of a 16-bit PCM buffer, in milliseconds."""
    if sample_rate <= 0 or channels <= 0:
        return 0.0
    samples = (len(pcm) // BYTES_PER_SAMPLE) // channels
    return samples / sample_rate * 1000.0
