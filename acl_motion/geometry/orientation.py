"""Projected 2D orientation primitives."""

from __future__ import annotations

import numpy as np

from acl_motion.geometry._points import as_point


def segment_orientation_2d(a: object, b: object) -> float:
    """Return directed segment orientation in degrees.

    Convention:

    - ``0°`` is the positive image x-axis.
    - Positive rotation is counter-clockwise in mathematical coordinates.
    - Image y is inverted internally because image coordinates increase
      downward.

    The result is wrapped to ``[-180, 180)``. Missing points or a zero-length
    segment return ``NaN``.
    """

    a_point = as_point(a)
    b_point = as_point(b)
    if a_point is None or b_point is None:
        return float("nan")
    dx = b_point[0] - a_point[0]
    dy_math = -(b_point[1] - a_point[1])
    if dx == 0 and dy_math == 0:
        return float("nan")
    angle = float(np.degrees(np.arctan2(dy_math, dx)))
    return float((angle + 180.0) % 360.0 - 180.0)
