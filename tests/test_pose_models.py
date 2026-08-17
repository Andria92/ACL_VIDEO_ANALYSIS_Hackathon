from acl_motion.pose.models import Landmark, PoseFrame, PoseSequence, QualityFlag, QualityFlagCode
from acl_motion.video.roi import BBox


def test_landmark_row_preserves_traceability_fields():
    flag = QualityFlag(
        code=QualityFlagCode.LOW_LANDMARK_CONFIDENCE,
        message="Synthetic test flag.",
        frame_index=12,
        landmark_name="left_knee",
    )
    landmark = Landmark(
        name="left_knee",
        x_px=100.0,
        y_px=200.0,
        x_norm=0.25,
        y_norm=0.5,
        confidence=0.8,
        visibility=0.8,
        observed=True,
        quality_flags=(flag,),
    )

    row = landmark.to_row(
        case_id="ACL_TEST",
        source_id="VIEW_01",
        frame_index=12,
        timestamp_ms=400.0,
        backend="dummy",
        target_bbox=BBox(10, 20, 200, 300),
    )

    assert row["case_id"] == "ACL_TEST"
    assert row["source_id"] == "VIEW_01"
    assert row["landmark_name"] == "left_knee"
    assert row["observed"] is True
    assert row["backend"] == "dummy"
    assert row["target_bbox_x"] == 10
    assert row["quality_flags"][0]["quality_code"] == "LOW_LANDMARK_CONFIDENCE"


def test_pose_sequence_yields_long_landmark_rows():
    frame = PoseFrame(
        frame_index=0,
        timestamp_ms=0.0,
        source_id="VIEW_01",
        backend="dummy",
        landmarks={
            "left_hip": Landmark(
                name="left_hip",
                x_px=1.0,
                y_px=2.0,
                x_norm=0.1,
                y_norm=0.2,
                observed=True,
            )
        },
    )
    sequence = PoseSequence(
        case_id="ACL_TEST",
        source_id="VIEW_01",
        backend="dummy",
        frames=(frame,),
    )

    rows = list(sequence.iter_landmark_rows())

    assert len(rows) == 1
    assert rows[0]["frame_index"] == 0
    assert rows[0]["landmark_name"] == "left_hip"
