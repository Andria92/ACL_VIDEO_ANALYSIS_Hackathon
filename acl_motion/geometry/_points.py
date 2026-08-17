"""Internal point parsing helpers for geometry primitives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Point = tuple[float, float]


def as_point(value: object) -> Point | None:
    """Return a finite ``(x, y)`` point, or ``None`` for missing/invalid input."""

    if value is None:
        return None
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        return None
    try:
        x_value = float(value[0])
        y_value = float(value[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x_value) or not np.isfinite(y_value):
        return None
    return x_value, y_value


def as_math_point(value: object) -> Point | None:
    """Return image point converted to mathematical coordinates.

    Image coordinates use positive y downward. The geometry convention for
    orientations and signed distances uses positive y upward, so this helper
    negates y.
    """

    point = as_point(value)
    if point is None:
        return None
    return point[0], -point[1]
