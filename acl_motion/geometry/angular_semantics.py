"""Semantic types and canonical changes for projected angular measurements."""

from __future__ import annotations

from enum import StrEnum

import numpy as np


class AngleType(StrEnum):
    """Mathematical type of an existing projected angular measurement."""

    INTERNAL = "internal"
    DIRECTED = "directed"
    AXIS = "axis"


# This registry describes existing values; it does not create new measurements.
ANGLE_TYPES: dict[str, AngleType] = {
    # Connected-segment configurations and differences derived from them.
    "left_hka_angle_2d_deg": AngleType.INTERNAL,
    "right_hka_angle_2d_deg": AngleType.INTERNAL,
    "injured_hka_angle_2d_deg": AngleType.INTERNAL,
    "contralateral_hka_angle_2d_deg": AngleType.INTERNAL,
    "hka_projected_bilateral_difference_deg": AngleType.INTERNAL,
    "hka_projected_bilateral_absolute_difference_deg": AngleType.INTERNAL,
    "left_elbow_angle_2d_deg": AngleType.INTERNAL,
    "right_elbow_angle_2d_deg": AngleType.INTERNAL,
    "injured_elbow_angle_2d_deg": AngleType.INTERNAL,
    "contralateral_elbow_angle_2d_deg": AngleType.INTERNAL,
    "elbow_projected_bilateral_difference_deg": AngleType.INTERNAL,
    "elbow_projected_bilateral_absolute_difference_deg": AngleType.INTERNAL,
    # Endpoint-ordered vectors in the current image-coordinate implementation.
    "projected_trunk_axis_angle_deg": AngleType.DIRECTED,
    "left_upper_arm_orientation_2d_deg": AngleType.DIRECTED,
    "right_upper_arm_orientation_2d_deg": AngleType.DIRECTED,
    "path:projected_heading_deg": AngleType.DIRECTED,
    # Existing signed shoulder-line minus hip-line result. Both source segments
    # are ordered left-to-right, so this preserves its current directed meaning.
    "projected_shoulder_pelvis_orientation_difference_deg": AngleType.DIRECTED,
    # Body lines whose reversed endpoints describe the same visible axis.
    "projected_hip_line_angle_deg": AngleType.AXIS,
    "projected_shoulder_line_angle_deg": AngleType.AXIS,
}


def angle_type_for_metric(metric_name: str) -> AngleType | None:
    """Return the audited angular type for an existing metric, if registered."""

    return ANGLE_TYPES.get(metric_name)


def angular_difference(
    start_angle: float,
    end_angle: float,
    angle_type: AngleType | str,
) -> float:
    """Return the canonical signed change from ``start_angle`` to ``end_angle``.

    Internal/configuration values use a direct difference. Directed orientations
    use the shortest circular change in ``[-180, 180)``. Undirected body axes use
    the shortest axial change in ``[-90, 90)`` because orientations separated by
    180 degrees describe the same line.

    Non-numeric, non-finite, or unknown angle types return ``NaN``.
    """

    try:
        start_value = float(start_angle)
        end_value = float(end_angle)
        semantic_type = AngleType(angle_type)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(start_value) or not np.isfinite(end_value):
        return float("nan")
    if semantic_type is AngleType.INTERNAL:
        return end_value - start_value
    if semantic_type is AngleType.DIRECTED:
        return float((end_value - start_value + 180.0) % 360.0 - 180.0)
    return float((end_value - start_value + 90.0) % 180.0 - 90.0)


def angular_difference_for_metric(
    metric_name: str,
    start_angle: float,
    end_angle: float,
) -> float:
    """Return a registered metric's canonical signed angular change."""

    angle_type = angle_type_for_metric(metric_name)
    if angle_type is None:
        return float("nan")
    return angular_difference(start_angle, end_angle, angle_type)
