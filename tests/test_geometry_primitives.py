import math

import numpy as np
import pytest

from acl_motion.geometry.angles import angle_2d
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
