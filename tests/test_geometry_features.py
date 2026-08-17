import numpy as np
import pandas as pd
import pytest

from acl_motion.cases.models import InjurySide
from acl_motion.geometry.features import (
    FORBIDDEN_CLINICAL_LABELS,
    FeatureStatus,
    _declared_feature_names,
    compute_geometry_features,
)

BASE_POINTS = {
    "left_shoulder": (0.0, 0.0),
    "right_shoulder": (40.0, 0.0),
    "left_hip": (0.0, 100.0),
    "right_hip": (40.0, 100.0),
    "left_knee": (0.0, 170.0),
    "right_knee": (80.0, 140.0),
    "left_ankle": (0.0, 240.0),
    "right_ankle": (40.0, 100.0),
    "left_elbow": (-20.0, 50.0),
    "right_elbow": (60.0, 50.0),
    "left_wrist": (-30.0, 110.0),
    "right_wrist": (70.0, 110.0),
}


def test_valid_landmarks_support_core_features():
    features, normalisation = compute_geometry_features(_processed_pose())

    assert normalisation.is_available
    left_hka = _feature(features, "left_hka_angle_2d_deg", 0)
    trunk = _feature(features, "projected_trunk_axis_angle_deg", 0)

    assert left_hka.status == FeatureStatus.SUPPORTED.value
    assert left_hka.feature_value == pytest.approx(180.0)
    assert trunk.status == FeatureStatus.SUPPORTED.value


def test_missing_ankle_blocks_hka_but_not_trunk():
    pose = _processed_pose(missing={"left_ankle"})

    features, _ = compute_geometry_features(pose)

    left_hka = _feature(features, "left_hka_angle_2d_deg", 0)
    trunk = _feature(features, "projected_trunk_axis_angle_deg", 0)

    assert left_hka.status == FeatureStatus.INSUFFICIENT_LANDMARKS.value
    assert trunk.status == FeatureStatus.SUPPORTED.value


def test_invalid_target_frame_rejects_features():
    pose = _processed_pose(frame_status="TARGET_IDENTITY_UNCERTAIN")

    features, _ = compute_geometry_features(pose)

    assert _feature(features, "projected_trunk_axis_angle_deg", 0).status == (
        FeatureStatus.INVALID_TARGET_FRAME.value
    )


def test_interpolated_landmark_metadata_propagates():
    pose = _processed_pose(interpolated={"right_knee"})

    features, _ = compute_geometry_features(pose)

    right_hka = _feature(features, "right_hka_angle_2d_deg", 0)

    assert right_hka.status == FeatureStatus.SUPPORTED.value
    assert bool(right_hka.input_interpolated) is True
    assert bool(right_hka.input_smoothed) is True


def test_partially_usable_frame_does_not_globally_disappear():
    pose = _processed_pose(missing={"left_ankle", "right_ankle"})

    features, _ = compute_geometry_features(pose)

    assert _feature(features, "left_hka_angle_2d_deg", 0).status != FeatureStatus.SUPPORTED.value
    assert _feature(features, "projected_hip_line_angle_deg", 0).status == (
        FeatureStatus.SUPPORTED.value
    )
    assert _feature(features, "left_elbow_angle_2d_deg", 0).status == FeatureStatus.SUPPORTED.value


def test_injured_contralateral_mapping_left_and_right():
    pose = _processed_pose()

    left_features, _ = compute_geometry_features(pose, injured_side=InjurySide.LEFT)
    right_features, _ = compute_geometry_features(pose, injured_side=InjurySide.RIGHT)

    left_diff = _feature(left_features, "hka_projected_bilateral_difference_deg", 0)
    right_diff = _feature(right_features, "hka_projected_bilateral_difference_deg", 0)
    left_hka = _feature(left_features, "left_hka_angle_2d_deg", 0).feature_value
    right_hka = _feature(left_features, "right_hka_angle_2d_deg", 0).feature_value

    assert left_diff.feature_value == pytest.approx(left_hka - right_hka)
    assert right_diff.feature_value == pytest.approx(right_hka - left_hka)
    assert _feature(left_features, "hka_projected_bilateral_absolute_difference_deg", 0).feature_value == (
        pytest.approx(abs(left_hka - right_hka))
    )


def test_unknown_injury_side_does_not_invent_laterality_features():
    features, _ = compute_geometry_features(_processed_pose(), injured_side=InjurySide.UNKNOWN)

    assert "injured_hka_angle_2d_deg" not in set(features["feature_name"])
    assert "hka_projected_bilateral_difference_deg" not in set(features["feature_name"])


def test_m3_feature_names_remain_generic_not_clinical():
    names = _declared_feature_names()

    assert not [
        forbidden for forbidden in FORBIDDEN_CLINICAL_LABELS for name in names if forbidden in name
    ]


def _feature(features: pd.DataFrame, feature_name: str, frame_index: int):
    return features[
        features["feature_name"].eq(feature_name) & features["frame_index"].eq(frame_index)
    ].iloc[0]


def _processed_pose(
    *,
    frame_status: str = "VALID_TARGET",
    missing: set[str] | None = None,
    interpolated: set[str] | None = None,
) -> pd.DataFrame:
    missing = missing or set()
    interpolated = interpolated or set()
    rows = []
    for landmark_name, point in BASE_POINTS.items():
        is_missing = landmark_name in missing
        is_interpolated = landmark_name in interpolated
        rows.append(
            {
                "case_id": "synthetic_case",
                "source_id": "synthetic_view",
                "frame_index": 0,
                "source_frame_index": 100,
                "analysis_frame_index": 0,
                "timestamp_ms": 0.0,
                "landmark_name": landmark_name,
                "frame_status": frame_status,
                "valid_target_frame": frame_status == "VALID_TARGET",
                "valid_segment_id": 1 if frame_status == "VALID_TARGET" else pd.NA,
                "landmark_status": "MISSING" if is_missing else "OBSERVED_VALID",
                "processing_status": "MISSING" if is_missing else "SMOOTHED",
                "clean_x": np.nan if is_missing else point[0],
                "clean_y": np.nan if is_missing else point[1],
                "smoothed_x": np.nan if is_missing else point[0],
                "smoothed_y": np.nan if is_missing else point[1],
                "interpolated": is_interpolated,
                "smoothed": not is_missing,
                "rejected": is_missing,
            }
        )
    return pd.DataFrame(rows)
