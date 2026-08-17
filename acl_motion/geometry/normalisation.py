"""Body-scale normalisation helpers for M3 projected geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from acl_motion.geometry.distances import distance_2d, midpoint

NORMALISATION_METHOD = "median_torso_length_valid_processed_frames"


@dataclass(frozen=True, slots=True)
class NormalisationReference:
    """Per-view body-scale reference used for normalized geometry features."""

    method: str
    reference_value_px: float
    frames_used_for_reference: tuple[int, ...]

    @property
    def is_available(self) -> bool:
        return bool(np.isfinite(self.reference_value_px) and self.reference_value_px > 0)

    def to_metadata(self) -> dict:
        return {
            "normalisation_method": self.method,
            "reference_value_px": self.reference_value_px,
            "frames_used_for_reference": list(self.frames_used_for_reference),
            "frames_used_count": len(self.frames_used_for_reference),
        }


def compute_body_scale_reference(processed_pose: pd.DataFrame) -> NormalisationReference:
    """Compute median torso length across valid processed frames.

    Torso length is ``distance(shoulder_mid, pelvis_mid)``. Per-frame noisy
    scale is not used for feature normalization; this function returns a single
    robust reference for the view.
    """

    lengths: list[float] = []
    frame_indices: list[int] = []
    for frame_index, frame in processed_pose.groupby("frame_index", sort=True):
        points = {
            landmark: _feature_point(frame, landmark)
            for landmark in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        }
        shoulder_mid = midpoint(points["left_shoulder"], points["right_shoulder"])
        pelvis_mid = midpoint(points["left_hip"], points["right_hip"])
        torso_length = distance_2d(shoulder_mid, pelvis_mid)
        if np.isfinite(torso_length) and torso_length > 0:
            lengths.append(torso_length)
            frame_indices.append(int(frame_index))
    reference = float(np.nanmedian(lengths)) if lengths else float("nan")
    return NormalisationReference(
        method=NORMALISATION_METHOD,
        reference_value_px=reference,
        frames_used_for_reference=tuple(frame_indices),
    )


def _feature_point(frame: pd.DataFrame, landmark_name: str) -> tuple[float, float] | None:
    rows = frame[frame["landmark_name"].eq(landmark_name)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    for prefix in ("smoothed", "clean"):
        x_value = row.get(f"{prefix}_x")
        y_value = row.get(f"{prefix}_y")
        if pd.notna(x_value) and pd.notna(y_value):
            return float(x_value), float(y_value)
    return None
