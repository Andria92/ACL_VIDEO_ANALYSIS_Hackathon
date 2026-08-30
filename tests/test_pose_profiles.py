import pytest

from acl_motion.pose.profiles import (
    DEFAULT_POSE_ANALYSIS_PROFILE_ID,
    POSE_ANALYSIS_PROFILES,
    pose_analysis_profile,
    pose_analysis_profiles_payload,
)


def test_yolov8n_is_the_only_pose_profile():
    assert len(POSE_ANALYSIS_PROFILES) == 1
    profile = pose_analysis_profile(DEFAULT_POSE_ANALYSIS_PROFILE_ID)
    assert profile.profile_id == "yolov8n_legacy_tight"
    assert profile.model_filename == "yolov8n-pose.pt"
    assert profile.selection_strategy == "largest"
    assert profile.image_size == 640
    assert profile.roi_padding_fraction == 0.0


def test_other_pose_profiles_are_rejected():
    with pytest.raises(ValueError, match="Only the YOLOv8n"):
        pose_analysis_profile("unsupported-model")


def test_fixed_profile_payload_keeps_plain_and_scientific_context():
    payload = pose_analysis_profiles_payload()

    assert len(payload) == 1
    assert payload[0]["plain_language_description"]
    assert payload[0]["scientific_description"]
    assert payload[0]["limitations"]
