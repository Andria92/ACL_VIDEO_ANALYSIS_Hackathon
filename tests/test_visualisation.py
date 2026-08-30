import numpy as np
import pandas as pd

from acl_motion.video.roi import BBox
from acl_motion.visualisation.overlay import draw_pose_overlay
from acl_motion.visualisation.trajectories import plot_joint_coordinate_diagnostics


def test_draw_pose_overlay_preserves_frame_shape():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    landmarks = {
        "left_shoulder": {
            "x_px": 20.0,
            "y_px": 20.0,
            "confidence": 0.9,
            "observed": True,
        },
        "right_shoulder": {
            "x_px": 60.0,
            "y_px": 20.0,
            "confidence": 0.9,
            "observed": True,
        },
    }

    output = draw_pose_overlay(frame, landmarks, bbox=BBox(10, 10, 70, 50))

    assert output.shape == frame.shape
    assert output.sum() > 0


def test_draw_pose_overlay_hides_very_low_confidence_landmarks_by_default():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    landmarks = {
        "left_shoulder": {
            "x_px": 20.0,
            "y_px": 20.0,
            "confidence": 0.1,
            "observed": True,
        },
        "right_shoulder": {
            "x_px": 60.0,
            "y_px": 20.0,
            "confidence": 0.1,
            "observed": True,
        },
    }

    output = draw_pose_overlay(frame, landmarks)

    assert output.sum() == 0


def test_draw_pose_overlay_renders_provisional_pose_less_strongly():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    supported = {
        name: {
            "x_px": x,
            "y_px": 20.0,
            "confidence": 0.9,
            "observed": True,
        }
        for name, x in (("left_shoulder", 20.0), ("right_shoulder", 80.0))
    }
    provisional = {
        name: {**landmark, "confidence": 0.3}
        for name, landmark in supported.items()
    }

    supported_output = draw_pose_overlay(frame, supported)
    provisional_output = draw_pose_overlay(frame, provisional)

    assert np.count_nonzero(provisional_output) < np.count_nonzero(supported_output)


def test_plot_joint_coordinate_diagnostics_writes_file(tmp_path):
    pose_df = pd.DataFrame(
        [
            {
                "timestamp_ms": 0.0,
                "landmark_name": "left_shoulder",
                "x_px": 10.0,
                "y_px": 20.0,
                "observed": True,
            },
            {
                "timestamp_ms": 33.3,
                "landmark_name": "left_shoulder",
                "x_px": 12.0,
                "y_px": 22.0,
                "observed": True,
            },
        ]
    )

    output = plot_joint_coordinate_diagnostics(
        pose_df,
        tmp_path / "diagnostics.png",
        landmark_names=("left_shoulder",),
    )

    assert output.exists()
    assert output.stat().st_size > 0
