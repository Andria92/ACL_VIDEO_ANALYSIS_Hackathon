"""Semantic types and canonical changes for projected angular measurements."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

import numpy as np

ANGULAR_STATISTICS_VERSION = "angular_statistics_v1_shortest_arc"


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


def angular_range(
    values: Iterable[float],
    angle_type: AngleType | str,
) -> float:
    """Return the smallest semantically valid arc containing ``values``.

    Internal/configuration angles retain their ordinary linear range. Directed
    orientations use a 360-degree period, while undirected axes use a 180-degree
    period. This prevents a boundary crossing such as ``179, -179`` from being
    reported as a 358-degree movement range.

    Non-numeric, non-finite-only, or unknown inputs return ``NaN``.
    """

    try:
        semantic_type = AngleType(angle_type)
        numeric = np.asarray(list(values), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float("nan")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return float("nan")
    if numeric.size == 1:
        return 0.0
    if semantic_type is AngleType.INTERNAL:
        return float(np.ptp(numeric))

    period = 360.0 if semantic_type is AngleType.DIRECTED else 180.0
    ordered = np.sort(np.mod(numeric, period))
    gaps = np.diff(np.concatenate((ordered, [ordered[0] + period])))
    result = float(period - np.max(gaps))
    return 0.0 if np.isclose(result, 0.0, atol=1e-12) else result


def angular_mean(
    values: Iterable[float],
    angle_type: AngleType | str,
) -> float:
    """Return the circular or axial mean for an angular measurement.

    Directed orientations use a 360-degree period. Undirected axes use a
    180-degree period, so values separated by 180 degrees contribute to the
    same visible axis. Internal/configuration angles retain their ordinary
    arithmetic mean.
    """

    semantic_type, numeric = _finite_angular_values(values, angle_type)
    if semantic_type is None or numeric.size == 0:
        return float("nan")
    if semantic_type is AngleType.INTERNAL:
        return float(np.mean(numeric))

    period = 360.0 if semantic_type is AngleType.DIRECTED else 180.0
    radians = numeric * (2.0 * np.pi / period)
    mean_sine = float(np.mean(np.sin(radians)))
    mean_cosine = float(np.mean(np.cos(radians)))
    if np.hypot(mean_sine, mean_cosine) <= np.finfo(float).eps:
        return float("nan")
    mean = float(np.arctan2(mean_sine, mean_cosine) * period / (2.0 * np.pi))
    return float((mean + period / 2.0) % period - period / 2.0)


def angular_standard_deviation(
    values: Iterable[float],
    angle_type: AngleType | str,
) -> float:
    """Return circular/axial standard deviation in degrees.

    A single observation has no sample uncertainty and therefore returns
    ``NaN``. Internal/configuration angles retain sample standard deviation.
    """

    semantic_type, numeric = _finite_angular_values(values, angle_type)
    if semantic_type is None or numeric.size < 2:
        return float("nan")
    if semantic_type is AngleType.INTERNAL:
        return float(np.std(numeric, ddof=1))

    period = 360.0 if semantic_type is AngleType.DIRECTED else 180.0
    radians = numeric * (2.0 * np.pi / period)
    resultant = float(
        np.hypot(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    )
    if resultant <= np.finfo(float).eps:
        return float("nan")
    resultant = min(resultant, 1.0)
    return float(np.sqrt(-2.0 * np.log(resultant)) * period / (2.0 * np.pi))


def _finite_angular_values(
    values: Iterable[float],
    angle_type: AngleType | str,
) -> tuple[AngleType | None, np.ndarray]:
    try:
        semantic_type = AngleType(angle_type)
        numeric = np.asarray(list(values), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None, np.asarray([], dtype=float)
    return semantic_type, numeric[np.isfinite(numeric)]


def measurement_range_for_metric(
    metric_name: str,
    values: Iterable[float],
) -> float:
    """Return a registered angular range or an ordinary linear range."""

    numeric_values = list(values)
    angle_type = angle_type_for_metric(metric_name)
    if angle_type is not None:
        return angular_range(numeric_values, angle_type)
    try:
        numeric = np.asarray(numeric_values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float("nan")
    numeric = numeric[np.isfinite(numeric)]
    return float(np.ptp(numeric)) if numeric.size else float("nan")


def range_semantics_for_metric(metric_name: str) -> str:
    """Return the auditable range rule used for ``metric_name``."""

    angle_type = angle_type_for_metric(metric_name)
    if angle_type is None or angle_type is AngleType.INTERNAL:
        return "linear"
    if angle_type is AngleType.DIRECTED:
        return "shortest_directed_arc"
    return "shortest_axial_arc"
