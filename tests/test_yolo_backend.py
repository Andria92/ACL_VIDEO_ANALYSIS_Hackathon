from types import SimpleNamespace

import numpy as np

from acl_motion.pose.models import QualityFlag, QualityFlagCode
from acl_motion.pose.yolo_backend import COCO_POSE_LANDMARKS, YoloPoseBackend


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values

    def __len__(self):
        return len(self._values)


def _pose_result(boxes, confidences):
    candidate_count = len(boxes)
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=_FakeTensor(boxes)),
        keypoints=SimpleNamespace(
            xy=_FakeTensor(np.zeros((candidate_count, len(COCO_POSE_LANDMARKS), 2))),
            conf=_FakeTensor(confidences),
        ),
    )


def test_yolo_schema_contains_milestone_one_core_joints():
    required = {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }

    assert required.issubset(set(COCO_POSE_LANDMARKS))


def test_yolo_empty_landmarks_preserve_unavailable_rows():
    backend = object.__new__(YoloPoseBackend)
    flag = QualityFlag(QualityFlagCode.TARGET_NOT_FOUND, frame_index=4)

    landmarks = backend._empty_landmarks(flag)

    assert set(landmarks) == set(COCO_POSE_LANDMARKS)
    assert landmarks["left_knee"].observed is False
    assert landmarks["left_knee"].quality_flags[0].code == QualityFlagCode.TARGET_NOT_FOUND


def test_temporal_selection_keeps_previous_target_when_another_player_is_centered():
    backend = object.__new__(YoloPoseBackend)
    backend._selection_strategy = "temporal"
    backend._temporal_max_gap_frames = 12
    backend.reset_tracking()
    confidence = np.full((1, len(COCO_POSE_LANDMARKS)), 0.8)

    first_index = backend._select_pose_index(
        _pose_result([[10, 20, 30, 80]], confidence),
        crop_width=100,
        crop_height=100,
        frame_index=1,
    )
    second_confidence = np.full((2, len(COCO_POSE_LANDMARKS)), 0.8)
    second_index = backend._select_pose_index(
        _pose_result(
            [
                [15, 20, 35, 80],
                [40, 20, 60, 80],
            ],
            second_confidence,
        ),
        crop_width=100,
        crop_height=100,
        frame_index=2,
    )

    assert first_index == 0
    assert second_index == 0
    assert backend._last_selection_metadata["target_continuity_used"] is True
    assert backend._last_selection_metadata["target_continuity_iou"] > 0


def test_temporal_selection_expires_stale_target_history():
    backend = object.__new__(YoloPoseBackend)
    backend._selection_strategy = "temporal"
    backend._temporal_max_gap_frames = 2
    backend.reset_tracking()
    confidence = np.full((1, len(COCO_POSE_LANDMARKS)), 0.8)
    backend._select_pose_index(
        _pose_result([[10, 20, 30, 80]], confidence),
        crop_width=100,
        crop_height=100,
        frame_index=1,
    )

    selected_index = backend._select_pose_index(
        _pose_result(
            [[15, 20, 35, 80], [40, 20, 60, 80]],
            np.full((2, len(COCO_POSE_LANDMARKS)), 0.8),
        ),
        crop_width=100,
        crop_height=100,
        frame_index=10,
    )

    assert selected_index == 1
    assert backend._last_selection_metadata["target_continuity_used"] is False


def test_selection_preserves_competing_person_overlap_for_identity_qc():
    backend = object.__new__(YoloPoseBackend)
    backend._selection_strategy = "largest"
    backend._last_selection_metadata = {}

    selected_index = backend._select_pose_index(
        _pose_result(
            [[10, 10, 70, 90], [45, 15, 95, 90]],
            np.full((2, len(COCO_POSE_LANDMARKS)), 0.9),
        ),
        crop_width=100,
        crop_height=100,
        frame_index=3,
    )

    assert selected_index == 0
    assert backend._last_selection_metadata["competing_pose_count"] == 1
    assert backend._last_selection_metadata["competing_pose_max_iou"] > 0
    assert (
        backend._last_selection_metadata[
            "competing_pose_min_center_distance_fraction"
        ]
        < 1
    )
