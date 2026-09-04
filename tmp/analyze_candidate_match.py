from __future__ import annotations

import sys
import wave

import numpy as np


RATE = 8000


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return np.diff(audio, prepend=audio[0])


def best_match(source: np.ndarray, query: np.ndarray) -> tuple[float, int]:
    query = query - query.mean()
    query_energy = float(np.sum(query * query))
    needed = len(source) + len(query) - 1
    fft_size = 1 << (needed - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(source, fft_size) * np.fft.rfft(query[::-1], fft_size),
        fft_size,
    )
    numerator = correlation[len(query) - 1 : len(source)]
    squares = np.concatenate(([0.0], np.cumsum(source * source, dtype=np.float64)))
    source_energy = squares[len(query) :] - squares[: -len(query)]
    scores = numerator / np.sqrt(np.maximum(source_energy * query_energy, 1e-12))
    index = int(np.argmax(scores))
    return float(scores[index]), index


source = read_wav(sys.argv[1])
candidate = read_wav(sys.argv[2])
window_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
step_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
window = int(window_seconds * RATE)
step = int(step_seconds * RATE)

for offset in range(0, len(candidate) - window + 1, step):
    score, source_index = best_match(source, candidate[offset : offset + window])
    inferred_start = (source_index - offset) / RATE
    if score >= 0.22:
        print(
            f"candidate={offset / RATE:6.3f}-{(offset + window) / RATE:6.3f} "
            f"intro={source_index / RATE:6.3f}-{(source_index + window) / RATE:6.3f} "
            f"start={inferred_start:7.3f} score={score:.3f}"
        )
