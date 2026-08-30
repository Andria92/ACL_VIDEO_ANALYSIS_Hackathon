import math

import numpy as np
import pytest

from acl_motion.geometry.angles import angle_2d
from acl_motion.geometry.angular_semantics import (
    AngleType,
    angle_type_for_metric,
    angular_difference,
    angular_range,
    measurement_range_for_metric,
    range_semantics_for_metric,
)
from acl_motion.geometry.distances import (
    distance_2d,
    midpoint,
    normalized_distance,
    signed_point_line_distance,
)
from acl_motion.geometry.orientation import segment_orientation_2d


def test_angle_2d_known_angles():
    assert angle_2d((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)
    assert angle_2d((-1, 0), (0, 0), (1, 0)) == pytest.approx(180.0)
    assert angle_2d((1, 0), (0, 0), (2, 0)) == pytest.approx(0.0)


def test_angle_2d_handles_zero_length_nan_and_missing():
    assert math.isnan(angle_2d((0, 0), (0, 0), (1, 0)))
    assert math.isnan(angle_2d((np.nan, 0), (0, 0), (1, 0)))
    assert math.isnan(angle_2d(None, (0, 0), (1, 0)))


def test_signed_point_line_distance_convention():
    assert signed_point_line_distance((0.5, 0), (0, 0), (1, 0)) == pytest.approx(0.0)
    assert signed_point_line_distance((0.5, -1), (0, 0), (1, 0)) == pytest.approx(1.0)
    assert signed_point_line_distance((0.5, 1), (0, 0), (1, 0)) == pytest.approx(-1.0)
    assert math.isnan(signed_point_line_distance((1, 1), (0, 0), (0, 0)))


def test_segment_orientation_2d_known_directions():
    assert segment_orientation_2d((0, 0), (1, 0)) == pytest.approx(0.0)
    assert segment_orientation_2d((0, 0), (0, -1)) == pytest.approx(90.0)
    assert segment_orientation_2d((0, 0), (0, 1)) == pytest.approx(-90.0)
    assert segment_orientation_2d((0, 0), (1, -1)) == pytest.approx(45.0)
    assert math.isnan(segment_orientation_2d((0, 0), (0, 0)))


def test_midpoint_distance_and_normalized_distance():
    assert midpoint((0, 0), (2, 4)) == pytest.approx((1, 2))
    assert midpoint(None, (2, 4)) is None
    assert distance_2d((0, 0), (3, 4)) == pytest.approx(5.0)
    assert normalized_distance((0, 0), (3, 4), 10) == pytest.approx(0.5)
    assert math.isnan(normalized_distance((0, 0), (3, 4), 0))
    assert math.isnan(normalized_distance((0, 0), (3, 4), np.nan))


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (158.6, 85.5, -73.1),
        (101.4, 123.5, 22.1),
    ],
)
def test_internal_angular_difference(start, end, expected):
    assert angular_difference(start, end, AngleType.INTERNAL) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (-175.0, 178.0, -7.0),
        (178.0, -175.0, 7.0),
        (10.0, 40.0, 30.0),
        (40.0, 10.0, -30.0),
    ],
)
def test_directed_angular_difference_uses_shortest_circular_change(start, end, expected):
    assert angular_difference(start, end, AngleType.DIRECTED) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (-0.1, 173.9, -6.0),
        (0.0, 180.0, 0.0),
        (10.0, 170.0, -20.0),
        (170.0, 10.0, 20.0),
    ],
)
def test_axis_angular_difference_uses_shortest_axial_change(start, end, expected):
    assert angular_difference(start, end, AngleType.AXIS) == pytest.approx(expected)


def test_angular_difference_handles_invalid_values_and_registry_types():
    assert math.isnan(angular_difference(np.nan, 10.0, AngleType.INTERNAL))
    assert math.isnan(angular_difference(None, 10.0, AngleType.DIRECTED))
    assert math.isnan(angular_difference(10.0, 20.0, "unknown"))
    assert angle_type_for_metric("injured_hka_angle_2d_deg") is AngleType.INTERNAL
    assert angle_type_for_metric("projected_trunk_axis_angle_deg") is AngleType.DIRECTED
    assert angle_type_for_metric("projected_hip_line_angle_deg") is AngleType.AXIS
    assert angle_type_for_metric("not_an_angle") is None


@pytest.mark.parametrize(
    ("values", "angle_type", "expected"),
    [
        ([130.0, 170.0], AngleType.INTERNAL, 40.0),
        ([179.0, -179.0], AngleType.DIRECTED, 2.0),
        ([170.0, -170.0], AngleType.DIRECTED, 20.0),
        ([89.0, -89.0], AngleType.AXIS, 2.0),
        ([10.0], AngleType.AXIS, 0.0),
    ],
)
def test_angular_range_respects_metric_period(values, angle_type, expected):
    assert angular_range(values, angle_type) == pytest.approx(expected)


def test_measurement_range_uses_registry_and_linear_fallback():
    assert measurement_range_for_metric(
        "projected_hip_line_angle_deg",
        [89.0, -89.0],
    ) == pytest.approx(2.0)
    assert measurement_range_for_metric("synthetic_metric", [1.0, 7.0]) == pytest.approx(6.0)
    assert math.isnan(measurement_range_for_metric("synthetic_metric", [np.nan]))
    assert range_semantics_for_metric("projected_trunk_axis_angle_deg") == (
        "shortest_directed_arc"
    )
    assert range_semantics_for_metric("projected_hip_line_angle_deg") == "shortest_axial_arc"
    assert range_semantics_for_metric("injured_hka_angle_2d_deg") == "linear"
