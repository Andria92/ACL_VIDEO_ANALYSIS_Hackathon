import numpy as np
import pandas as pd

from acl_motion.quality.models import FrameQualityStatus
from acl_motion.quality.pose_quality import QualityConfig, classify_frame_quality

LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _pose_rows(
    frame_index: int,
    *,
    x_offset: float = 0.0,
    roi_x_offset: float | None = None,
    observed: bool = True,
    confidence: float = 0.9,
    pose_count: int = 1,
    selected_pose_index: int = 0,
):
    if roi_x_offset is None:
        roi_x_offset = x_offset
    base = {
        "left_shoulder": (0, 0),
        "right_shoulder": (40, 0),
        "left_hip": (0, 100),
        "right_hip": (40, 100),
        "left_knee": (0, 170),
        "right_knee": (40, 170),
        "left_ankle": (0, 240),
        "right_ankle": (40, 240),
    }
    rows = []
    for landmark, (x_value, y_value) in base.items():
        rows.append(
            {
                "case_id": "case",
                "source_id": "source",
                "frame_index": frame_index,
                "timestamp_ms": frame_index * 33.3,
                "landmark_name": landmark,
                "x_px": x_value + x_offset if observed else np.nan,
                "y_px": y_value if observed else np.nan,
                "confidence": confidence if observed else np.nan,
                "observed": observed,
                "backend_metadata": {
                    "pose_count": pose_count,
                    "selected_pose_index": selected_pose_index,
                },
                "target_bbox_x": -30.0 + roi_x_offset,
                "target_bbox_y": -30.0,
                "target_bbox_width": 120.0,
                "target_bbox_height": 320.0,
            }
        )
    return rows


def test_frame_quality_marks_short_reacquisition_invalid():
    rows = []
    for frame in range(6):
        rows.extend(_pose_rows(frame, x_offset=frame))
    for frame in range(6, 10):
        rows.extend(_pose_rows(frame, observed=False))
    for frame in range(10, 13):
        rows.extend(_pose_rows(frame, x_offset=200))
    raw = pd.DataFrame(rows)

    quality = classify_frame_quality(raw, QualityConfig(min_valid_segment_frames=5))

    assert quality.loc[quality.frame_index.eq(6), "frame_status"].iloc[0] == FrameQualityStatus.TARGET_NOT_FOUND
    assert (
        quality.loc[quality.frame_index.eq(11), "frame_status"].iloc[0]
        == FrameQualityStatus.INVALID_TRACK_SEGMENT
    )


def test_frame_quality_allows_target_reacquisition_inside_human_roi():
    rows = []
    for frame in range(6):
        rows.extend(_pose_rows(frame, x_offset=frame))
    for frame in range(6, 11):
        rows.extend(_pose_rows(frame, observed=False))
    for frame in range(11, 16):
        rows.extend(_pose_rows(frame, x_offset=200 + frame, roi_x_offset=200 + frame))
    raw = pd.DataFrame(rows)

    quality = classify_frame_quality(raw, QualityConfig(min_valid_segment_frames=5))

    first_segment = quality.loc[quality.frame_index.eq(5), "valid_segment_id"].iloc[0]
    reacquired_segment = quality.loc[quality.frame_index.eq(11), "valid_segment_id"].iloc[0]
    assert quality.loc[quality.frame_index.eq(8), "frame_status"].iloc[0] == FrameQualityStatus.TARGET_NOT_FOUND
    assert quality.loc[quality.frame_index.between(11, 15), "frame_status"].eq(
        FrameQualityStatus.VALID_TARGET
    ).all()
    assert reacquired_segment != first_segment


def test_frame_quality_marks_no_yolo_pose_target_not_found():
    rows = _pose_rows(0, observed=False)
    raw = pd.DataFrame(rows)

    quality = classify_frame_quality(raw)

    assert quality.loc[quality.frame_index.eq(0), "frame_status"].iloc[0] == FrameQualityStatus.TARGET_NOT_FOUND


def test_frame_quality_marks_large_jump_identity_uncertain():
    rows = []
    for frame in range(5):
        rows.extend(_pose_rows(frame, x_offset=frame))
    rows.extend(_pose_rows(5, x_offset=300))
    for frame in range(6, 11):
        rows.extend(_pose_rows(frame, x_offset=300 + frame))
    raw = pd.DataFrame(rows)

    quality = classify_frame_quality(
        raw,
        QualityConfig(
            min_valid_segment_frames=1,
            temporal_jump_min_px=50,
            suspicious_padding_frames=0,
        ),
    )

    assert (
        quality.loc[quality.frame_index.eq(5), "frame_status"].iloc[0]
        == FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN
    )


def test_frame_quality_preserves_source_and_analysis_frame_indices():
    raw = pd.DataFrame(
        [
            {
                **row,
                "source_frame_index": row["frame_index"] + 200,
                "analysis_frame_index": row["frame_index"],
            }
            for frame in range(2)
            for row in _pose_rows(frame)
        ]
    )

    quality = classify_frame_quality(raw)

    assert quality.loc[quality.frame_index.eq(0), "source_frame_index"].iloc[0] == 200
    assert quality.loc[quality.frame_index.eq(1), "analysis_frame_index"].iloc[0] == 1


def test_target_pose_overlap_ambiguity_does_not_silently_switch_player():
    rows = []
    for frame in range(4):
        rows.extend(_pose_rows(frame, x_offset=frame))
    rows.extend(_pose_rows(4, x_offset=95, pose_count=2))
    raw = pd.DataFrame(rows)

    quality = classify_frame_quality(
        raw,
        QualityConfig(
            min_valid_segment_frames=1,
            temporal_jump_min_px=500,
            suspicious_padding_frames=0,
        ),
    )

    row = quality.loc[quality.frame_index.eq(4)].iloc[0]
    assert row["frame_status"] == FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN
    assert row["target_overlap_ambiguous_flag"]
    assert "TARGET_OVERLAP_AMBIGUOUS" in row["frame_rejection_reason"]


def test_human_roi_rejects_wrong_nearby_player_candidate():
    rows = _pose_rows(0, x_offset=300, roi_x_offset=0, confidence=0.95)
    raw = pd.DataFrame(rows)

    quality = classify_frame_quality(raw, QualityConfig(min_valid_segment_frames=1))

    row = quality.loc[quality.frame_index.eq(0)].iloc[0]
    assert row["frame_status"] == FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN
    assert bool(row["pose_centroid_consistent_with_roi"]) is False
    assert "human target ROI" in row["frame_rejection_reason"]


def test_target_identity_uncertainty_starts_new_valid_segment_after_resume():
    rows = []
    for frame in range(5):
        rows.extend(_pose_rows(frame, x_offset=frame))
    rows.extend(_pose_rows(5, x_offset=100, pose_count=2))
    for frame in range(6, 11):
        rows.extend(_pose_rows(frame, x_offset=100 + frame))
    raw = pd.DataFrame(rows)

    quality = classify_frame_quality(
        raw,
        QualityConfig(
            min_valid_segment_frames=5,
            temporal_jump_min_px=500,
            suspicious_padding_frames=0,
        ),
    )

    assert quality.loc[quality.frame_index.eq(5), "frame_status"].iloc[0] != FrameQualityStatus.VALID_TARGET
    first_segment = quality.loc[quality.frame_index.eq(4), "valid_segment_id"].iloc[0]
    resumed_segment = quality.loc[quality.frame_index.eq(6), "valid_segment_id"].iloc[0]
    assert quality.loc[quality.frame_index.between(6, 10), "frame_status"].eq(
        FrameQualityStatus.VALID_TARGET
    ).all()
    assert resumed_segment != first_segment
