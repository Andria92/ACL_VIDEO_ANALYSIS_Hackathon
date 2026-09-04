from __future__ import annotations

import glob
import os
import sys
import wave

import numpy as np


RATE = 8000


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    # First difference deemphasizes changing background beds and DC offsets.
    return np.diff(audio, prepend=audio[0])


def best_match(source: np.ndarray, query: np.ndarray) -> tuple[float, int]:
    query = query - query.mean()
    query_energy = float(np.sum(query * query))
    if query_energy <= 1e-12 or len(query) > len(source):
        return 0.0, 0

    needed = len(source) + len(query) - 1
    fft_size = 1 << (needed - 1).bit_length()
    source_fft = np.fft.rfft(source, fft_size)
    query_fft = np.fft.rfft(query[::-1], fft_size)
    correlation = np.fft.irfft(source_fft * query_fft, fft_size)
    numerator = correlation[len(query) - 1 : len(source)]

    squares = np.concatenate(([0.0], np.cumsum(source * source, dtype=np.float64)))
    source_energy = squares[len(query) :] - squares[: -len(query)]
    denominator = np.sqrt(np.maximum(source_energy * query_energy, 1e-12))
    scores = numerator / denominator
    index = int(np.argmax(scores))
    return float(scores[index]), index


source_path = sys.argv[1] if len(sys.argv) > 1 else "tmp/audio_match/original.wav"
original = read_wav(source_path)

for candidate_path in sorted(glob.glob("tmp/audio_match/candidates/*.wav")):
    candidate = read_wav(candidate_path)
    name = os.path.splitext(os.path.basename(candidate_path))[0]

    requested_window = float(os.environ.get("MATCH_WINDOW", "3.0"))
    window_seconds = min(requested_window, len(candidate) / RATE)
    window_samples = max(1, int(window_seconds * RATE))
    step_samples = max(1, int(1.0 * RATE))
    offsets = list(range(0, max(1, len(candidate) - window_samples + 1), step_samples))
    final_offset = max(0, len(candidate) - window_samples)
    if final_offset not in offsets:
        offsets.append(final_offset)

    matches = []
    for offset in offsets:
        query = candidate[offset : offset + window_samples]
        score, original_index = best_match(original, query)
        inferred_start = (original_index - offset) / RATE
        matches.append((score, original_index / RATE, offset / RATE, inferred_start))

    matches.sort(reverse=True)
    best = matches[0]
    supporting = sum(1 for item in matches if item[0] >= 0.35 and abs(item[3] - best[3]) <= 0.35)
    print(
        f"{name:12s} score={best[0]:.3f} "
        f"original={best[1]:7.3f}s candidate_offset={best[2]:6.3f}s "
        f"inferred_start={best[3]:7.3f}s supporting={supporting}"
    )
