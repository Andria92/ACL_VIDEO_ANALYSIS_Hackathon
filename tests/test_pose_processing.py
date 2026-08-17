import numpy as np
import pandas as pd

from acl_motion.processing.pose_processing import ProcessingConfig, process_pose_coordinates
from acl_motion.quality.models import LandmarkQualityStatus


def test_processing_does_not_interpolate_across_invalid_segment():
    raw = pd.DataFrame(
        [
            _row(0, 0.0, True),
            _row(1, 1.0, True),
            _row(2, np.nan, False),
            _row(3, 3.0, True),
        ]
    )
    quality = pd.DataFrame(
        [
            _quality(0, "VALID_TARGET", 1, LandmarkQualityStatus.OBSERVED_VALID),
            _quality(1, "VALID_TARGET", 1, LandmarkQualityStatus.OBSERVED_VALID),
            _quality(2, "TARGET_NOT_FOUND", None, LandmarkQualityStatus.MISSING),
            _quality(3, "VALID_TARGET", 2, LandmarkQualityStatus.OBSERVED_VALID),
        ]
    )

    processed = process_pose_coordinates(
        raw,
        quality,
        ProcessingConfig(max_interpolation_gap_frames=2, smoothing_window_frames=1),
    )

    missing = processed.loc[processed.frame_index.eq(2)].iloc[0]
    assert missing["clean_x"] != missing["clean_x"]
    assert bool(missing["interpolated"]) is False


def test_processing_interpolates_short_gap_within_valid_segment():
    raw = pd.DataFrame(
        [
            _row(0, 0.0, True),
            _row(1, np.nan, False),
            _row(2, 2.0, True),
        ]
    )
    quality = pd.DataFrame(
        [
            _quality(0, "VALID_TARGET", 1, LandmarkQualityStatus.OBSERVED_VALID),
            _quality(1, "VALID_TARGET", 1, LandmarkQualityStatus.MISSING),
            _quality(2, "VALID_TARGET", 1, LandmarkQualityStatus.OBSERVED_VALID),
        ]
    )

    processed = process_pose_coordinates(
        raw,
        quality,
        ProcessingConfig(max_interpolation_gap_frames=2, smoothing_window_frames=1),
    )

    filled = processed.loc[processed.frame_index.eq(1)].iloc[0]
    assert filled["clean_x"] == 1.0
    assert bool(filled["interpolated"]) is True


def _row(frame_index: int, x_value: float, observed: bool) -> dict:
    return {
        "case_id": "case",
        "source_id": "source",
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 33.3,
        "landmark_name": "left_knee",
        "x_px": x_value,
        "y_px": x_value,
        "x_norm": x_value,
        "y_norm": x_value,
        "confidence": 0.9 if observed else np.nan,
        "observed": observed,
        "backend": "test",
        "quality_flags": [],
        "target_bbox_x": 0,
        "target_bbox_y": 0,
        "target_bbox_width": 100,
        "target_bbox_height": 100,
    }


def _quality(
    frame_index: int,
    frame_status: str,
    segment_id: int | None,
    landmark_status: LandmarkQualityStatus,
) -> dict:
    return {
        "frame_index": frame_index,
        "landmark_name": "left_knee",
        "frame_status": frame_status,
        "valid_target_frame": frame_status == "VALID_TARGET",
        "valid_segment_id": segment_id,
        "landmark_status": landmark_status.value,
        "landmark_rejection_reason": "",
        "landmark_jump_px": np.nan,
    }
