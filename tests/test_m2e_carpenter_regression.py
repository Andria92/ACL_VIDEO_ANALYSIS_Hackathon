from pathlib import Path

import pandas as pd
import pytest


def test_dense_carpenter_known_overlap_frames_have_auditable_states():
    frame_quality_path = Path("data/quality/ellie_carpenter_dense_frame_quality.csv")
    raw_pose_path = Path("data/pose/ellie_carpenter_raw_manual_roi_dense.parquet")
    if not frame_quality_path.exists() or not raw_pose_path.exists():
        pytest.skip("Dense Carpenter M2 artifacts have not been generated locally.")

    frame_quality = pd.read_csv(frame_quality_path).set_index("frame_index")
    raw_pose = pd.read_parquet(raw_pose_path)

    assert frame_quality.loc[245, "frame_status"] == "VALID_TARGET"
    assert frame_quality.loc[252, "frame_status"] != "VALID_TARGET"

    keyframe_rows = raw_pose[raw_pose["frame_index"].isin([245, 252])]
    assert keyframe_rows.groupby("frame_index")["manual_roi_is_keyframe"].first().to_dict() == {
        245: True,
        252: True,
    }
    assert (
        keyframe_rows.groupby("frame_index")["manual_roi_keyframes_path"].first()
        == "data/annotations/ellie_carpenter_roi_keyframes_dense.csv"
    ).all()
