import pandas as pd
import pytest

from acl_motion.validation.pose_reference import validate_pose_against_reference


def test_pose_reference_validation_reports_detection_and_normalized_accuracy():
    predicted = pd.DataFrame(
        [
            {
                "source_frame_index": 1,
                "landmark_name": "left_knee",
                "x_px": 13.0,
                "y_px": 14.0,
                "observed": True,
                "target_bbox_width": 60.0,
                "target_bbox_height": 80.0,
            },
            {
                "source_frame_index": 1,
                "landmark_name": "right_knee",
                "x_px": None,
                "y_px": None,
                "observed": False,
                "target_bbox_width": 60.0,
                "target_bbox_height": 80.0,
            },
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "source_frame_index": 1,
                "landmark_name": "left_knee",
                "x_px": 10.0,
                "y_px": 10.0,
                "visible": True,
            },
            {
                "source_frame_index": 1,
                "landmark_name": "right_knee",
                "x_px": 30.0,
                "y_px": 30.0,
                "visible": True,
            },
        ]
    )

    report = validate_pose_against_reference(predicted, reference)

    assert report["reference_point_count"] == 2
    assert report["detected_reference_point_count"] == 1
    assert report["detection_rate"] == 0.5
    assert report["median_pixel_error"] == 5.0
    assert report["pck_0_05"] == 1.0
    assert report["per_landmark"][0]["landmark_name"] == "left_knee"


def test_pose_reference_validation_requires_explicit_reference_coordinates():
    with pytest.raises(ValueError, match="Reference pose is missing"):
        validate_pose_against_reference(
            pd.DataFrame(),
            pd.DataFrame({"source_frame_index": [1], "landmark_name": ["left_knee"]}),
        )
