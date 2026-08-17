"""Reusable image-plane geometry primitives and M3 feature extraction."""

from acl_motion.geometry.angles import angle_2d, wrapped_angle_difference_deg
from acl_motion.geometry.distances import (
    distance_2d,
    midpoint,
    normalized_distance,
    signed_point_line_distance,
)
from acl_motion.geometry.orientation import segment_orientation_2d

__all__ = [
    "angle_2d",
    "distance_2d",
    "midpoint",
    "normalized_distance",
    "segment_orientation_2d",
    "signed_point_line_distance",
    "wrapped_angle_difference_deg",
]
