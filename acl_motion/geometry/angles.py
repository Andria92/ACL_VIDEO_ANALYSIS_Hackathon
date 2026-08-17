"""Projected 2D angle primitives."""

from __future__ import annotations

import numpy as np

from acl_motion.geometry._points import as_point


def angle_2d(a: object, vertex: object, b: object) -> float:
    """Return the unsigned angle A-V-B in degrees.

    The calculation uses image-plane vectors ``A - V`` and ``B - V``:

    ``acos(dot(A-V, B-V) / (|A-V| |B-V|))``.

    Missing points, non-finite coordinates, or zero-length vectors return
    ``NaN``. The cosine is clamped to ``[-1, 1]`` before ``arccos``.
    """

    a_point = as_point(a)
    vertex_point = as_point(vertex)
    b_point = as_point(b)
    if a_point is None or vertex_point is None or b_point is None:
        return float("nan")

    vector_a = np.array(a_point, dtype=float) - np.array(vertex_point, dtype=float)
    vector_b = np.array(b_point, dtype=float) - np.array(vertex_point, dtype=float)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 0 or norm_b <= 0:
        return float("nan")

    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def wrapped_angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """Return ``angle_a - angle_b`` wrapped to ``[-180, 180)`` degrees."""

    try:
        a_value = float(angle_a)
        b_value = float(angle_b)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(a_value) or not np.isfinite(b_value):
        return float("nan")
    return float((a_value - b_value + 180.0) % 360.0 - 180.0)
