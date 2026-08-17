from acl_motion.pose.models import QualityFlag, QualityFlagCode
from acl_motion.pose.yolo_backend import COCO_POSE_LANDMARKS, YoloPoseBackend


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
