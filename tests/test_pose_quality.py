import numpy as np
import pandas as pd

from acl_motion.geometry.features import FeatureStatus, compute_geometry_features
from acl_motion.processing.pose_processing import process_pose_coordinates
from acl_motion.quality.models import FrameQualityStatus, LandmarkQualityStatus
from acl_motion.quality.pose_quality import (
    QualityConfig,
    classify_frame_quality,
    classify_landmark_quality,
)

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
    competing_pose_max_iou: float | None = None,
    competing_pose_min_center_distance_fraction: float | None = None,
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
                    "competing_pose_count": max(pose_count - 1, 0),
                    "competing_pose_max_iou": competing_pose_max_iou,
                    "competing_pose_min_center_distance_fraction": (
                        competing_pose_min_center_distance_fraction
                    ),
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


def test_human_target_unavailable_frame_is_explicitly_withheld():
    rows = _pose_rows(0, observed=False)
    for row in rows:
        row["human_target_unavailable"] = True
        row["human_target_unavailable_reason"] = "PLAYER_OVERLAP"
        row["human_target_unavailable_note"] = "Operator could not defend target identity."
        row["human_target_unavailable_start_frame"] = 0
        row["human_target_unavailable_end_frame"] = 0
    raw = pd.DataFrame(rows)

    quality = classify_frame_quality(raw)

    frame = quality.iloc[0]
    assert frame["frame_status"] == FrameQualityStatus.TARGET_NOT_FOUND
    assert bool(frame["human_target_unavailable"])
    assert frame["human_target_unavailable_reason"] == "PLAYER_OVERLAP"
    assert "HUMAN_TARGET_UNAVAILABLE" in frame["frame_rejection_reason"]


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


def test_close_competing_pose_is_not_called_valid_despite_stable_confident_joints():
    raw = pd.DataFrame(
        _pose_rows(
            0,
            pose_count=2,
            competing_pose_max_iou=0.18,
            competing_pose_min_center_distance_fraction=0.30,
        )
    )

    quality = classify_frame_quality(
        raw,
        QualityConfig(min_valid_segment_frames=1),
    )

    row = quality.iloc[0]
    assert row["frame_status"] == FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN
    assert bool(row["competing_pose_proximity_flag"])
    assert "another detected person overlaps" in row["frame_rejection_reason"]


def test_human_acceptance_resolves_identity_ambiguity_without_erasing_auto_qc():
    rows = _pose_rows(
        0,
        pose_count=2,
        competing_pose_max_iou=0.18,
        competing_pose_min_center_distance_fraction=0.30,
    )
    for row in rows:
        row["human_target_accepted"] = True
        row["human_target_accepted_note"] = "Reviewer confirmed the white-shirted athlete."
        row["human_target_accepted_start_frame"] = 0
        row["human_target_accepted_end_frame"] = 0

    quality = classify_frame_quality(
        pd.DataFrame(rows),
        QualityConfig(min_valid_segment_frames=1),
    )

    frame = quality.iloc[0]
    assert frame["automatic_frame_status"] == FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN
    assert frame["frame_status"] == FrameQualityStatus.VALID_TARGET
    assert frame["manual_review_decision"] == "ACCEPTED"
    assert bool(frame["manual_override_applied"])
    assert frame["human_target_accepted_note"].startswith("Reviewer confirmed")


def test_human_acceptance_cannot_manufacture_a_missing_pose():
    rows = _pose_rows(0, observed=False)
    for row in rows:
        row["human_target_accepted"] = True

    quality = classify_frame_quality(
        pd.DataFrame(rows),
        QualityConfig(min_valid_segment_frames=1),
    )

    frame = quality.iloc[0]
    assert frame["automatic_frame_status"] == FrameQualityStatus.TARGET_NOT_FOUND
    assert frame["frame_status"] == FrameQualityStatus.TARGET_NOT_FOUND
    assert frame["manual_review_decision"] == "ACCEPTED"
    assert not bool(frame["manual_override_applied"])


def test_human_acceptance_keeps_measurable_joints_from_a_partial_pose():
    rows = _pose_rows(0)
    for row in rows:
        row["human_target_accepted"] = True
        if row["landmark_name"] in {"left_knee", "left_ankle", "right_ankle"}:
            row["observed"] = False
            row["x_px"] = np.nan
            row["y_px"] = np.nan
            row["confidence"] = np.nan
    raw = pd.DataFrame(rows)

    frame_quality = classify_frame_quality(
        raw,
        QualityConfig(min_valid_segment_frames=1),
    )
    landmark_quality = classify_landmark_quality(raw, frame_quality)

    assert frame_quality.iloc[0]["frame_status"] == FrameQualityStatus.PARTIAL_POSE
    assert (
        landmark_quality.loc[
            landmark_quality["landmark_name"].eq("left_shoulder"),
            "landmark_status",
        ].iloc[0]
        == LandmarkQualityStatus.OBSERVED_VALID
    )
    assert (
        landmark_quality.loc[
            landmark_quality["landmark_name"].eq("left_ankle"),
            "landmark_status",
        ].iloc[0]
        == LandmarkQualityStatus.MISSING
    )
    processed = process_pose_coordinates(raw, landmark_quality)
    features, _ = compute_geometry_features(processed)
    trunk = features.loc[
        features["feature_name"].eq("projected_trunk_axis_angle_deg")
    ].iloc[0]
    left_hka = features.loc[
        features["feature_name"].eq("left_hka_angle_2d_deg")
    ].iloc[0]

    assert trunk["status"] == FeatureStatus.SUPPORTED
    assert left_hka["status"] == FeatureStatus.INSUFFICIENT_LANDMARKS


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


def test_landmark_quality_rejects_confirmed_isolated_spike():
    raw, frame_quality = _landmark_sequence([0.0, 2.0, 100.0, 4.0, 6.0])

    quality = classify_landmark_quality(
        raw,
        frame_quality,
        QualityConfig(temporal_jump_min_px=50.0),
    )

    spike = quality.loc[quality.frame_index.eq(2)].iloc[0]
    assert spike["landmark_status"] == LandmarkQualityStatus.TEMPORAL_OUTLIER
    assert bool(spike["landmark_temporal_outlier"])
    assert spike["landmark_jump_px"] > 50.0
    assert spike["landmark_next_jump_px"] > 50.0
    assert spike["landmark_bridge_px"] < 10.0


def test_landmark_quality_preserves_sustained_large_transition():
    raw, frame_quality = _landmark_sequence([0.0, 2.0, 100.0, 104.0, 108.0])

    quality = classify_landmark_quality(
        raw,
        frame_quality,
        QualityConfig(temporal_jump_min_px=50.0),
    )

    transition = quality.loc[quality.frame_index.eq(2)].iloc[0]
    assert transition["landmark_status"] == LandmarkQualityStatus.OBSERVED_VALID
    assert bool(transition["landmark_temporal_outlier"]) is False
    assert transition["landmark_jump_px"] > 50.0
    assert transition["landmark_next_jump_px"] < 50.0


def _landmark_sequence(x_values: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(
        [
            {
                "case_id": "case",
                "source_id": "source",
                "frame_index": frame_index,
                "source_frame_index": frame_index,
                "analysis_frame_index": frame_index,
                "timestamp_ms": frame_index * 33.3,
                "landmark_name": "right_ankle",
                "x_px": x_value,
                "y_px": 10.0,
                "confidence": 0.9,
                "observed": True,
            }
            for frame_index, x_value in enumerate(x_values)
        ]
    )
    frame_quality = pd.DataFrame(
        [
            {
                "frame_index": frame_index,
                "frame_status": FrameQualityStatus.VALID_TARGET.value,
                "valid_target_frame": True,
                "identity_uncertain_flag": False,
                "valid_segment_id": 1,
                "body_scale_reference_px": 100.0,
            }
            for frame_index in range(len(x_values))
        ]
    )
    return raw, frame_quality
