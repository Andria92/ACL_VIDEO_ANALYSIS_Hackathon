import numpy as np

from acl_motion.pose.base import PoseBackend
from acl_motion.pose.models import Landmark, PoseFrame


class DummyBackend(PoseBackend):
    name = "dummy"
    model_name = "dummy-model"

    @property
    def landmark_names(self):
        return ("left_hip",)

    def extract_frame(self, image, roi=None, *, frame_index, timestamp_ms, source_id):
        return PoseFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            source_id=source_id,
            backend=self.name,
            target_bbox=roi,
            landmarks={
                "left_hip": Landmark(
                    name="left_hip",
                    x_px=10.0,
                    y_px=20.0,
                    x_norm=0.1,
                    y_norm=0.2,
                    observed=True,
                )
            },
        )


def test_backend_contract_returns_pose_frame():
    backend = DummyBackend()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    frame = backend.extract_frame(
        image,
        frame_index=3,
        timestamp_ms=100.0,
        source_id="VIEW_01",
    )

    assert frame.backend == "dummy"
    assert frame.frame_index == 3
    assert "left_hip" in frame.landmarks
