"""Projected 2D distance primitives."""

from __future__ import annotations

import numpy as np

from acl_motion.geometry._points import as_math_point, as_point


def distance_2d(a: object, b: object) -> float:
    """Return Euclidean image-plane distance in pixels, or ``NaN`` if unavailable."""

    a_point = as_point(a)
    b_point = as_point(b)
    if a_point is None or b_point is None:
        return float("nan")
    return float(np.hypot(a_point[0] - b_point[0], a_point[1] - b_point[1]))


def signed_point_line_distance(point: object, line_start: object, line_end: object) -> float:
    """Return signed perpendicular distance from a point to a directed line.

    Convention: points are interpreted in image pixels, then converted to
    mathematical coordinates by negating y. Positive distance means the point
    lies to the counter-clockwise/left side of the directed line from
    ``line_start`` to ``line_end``. A zero-length line returns ``NaN``.
    """

    point_math = as_math_point(point)
    start_math = as_math_point(line_start)
    end_math = as_math_point(line_end)
    if point_math is None or start_math is None or end_math is None:
        return float("nan")

    line = np.array(end_math, dtype=float) - np.array(start_math, dtype=float)
    relative = np.array(point_math, dtype=float) - np.array(start_math, dtype=float)
    length = float(np.linalg.norm(line))
    if length <= 0:
        return float("nan")
    cross = line[0] * relative[1] - line[1] * relative[0]
    return float(cross / length)


def midpoint(a: object, b: object) -> tuple[float, float] | None:
    """Return midpoint of two finite image-plane points, or ``None`` if unavailable."""

    a_point = as_point(a)
    b_point = as_point(b)
    if a_point is None or b_point is None:
        return None
    return (a_point[0] + b_point[0]) / 2.0, (a_point[1] + b_point[1]) / 2.0


def normalized_distance(a: object, b: object, scale: float) -> float:
    """Return point distance divided by a positive body-scale reference."""

    try:
        scale_value = float(scale)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(scale_value) or scale_value <= 0:
        return float("nan")
    distance = distance_2d(a, b)
    if not np.isfinite(distance):
        return float("nan")
    return float(distance / scale_value)
