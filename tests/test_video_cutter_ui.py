from __future__ import annotations

import cv2
import numpy as np

from acl_motion.ui.video_cutter import cut_video_segment, smoke_test
from acl_motion.video.io import read_video_metadata


def test_video_cutter_ui_smoke_has_review_controls() -> None:
    result = smoke_test()

    assert result["html_has_video_player"] is True
    assert result["html_has_mark_in"] is True
    assert result["html_has_mark_out"] is True
    assert result["html_has_five_frame_controls"] is True
    assert result["html_has_reload_player"] is True
    assert result["html_has_player_error_recovery"] is True
    assert result["html_has_cut"] is True
    assert result["writes_files"] is False


def test_cut_video_segment_with_opencv_fallback(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    try:
        assert writer.isOpened()
        for index in range(20):
            frame = np.full((48, 64, 3), index * 8, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    result = cut_video_segment(
        video_path=source_path,
        output_dir=tmp_path / "cuts",
        start_seconds=0.2,
        end_seconds=1.0,
        output_name="review_clip",
        mode="opencv",
    )
    output_path = tmp_path / "cuts" / "review_clip.mp4"
    metadata = read_video_metadata(output_path)

    assert result["saved"] is True
    assert result["method"] == "opencv"
    assert result["width"] == 64
    assert result["height"] == 48
    assert output_path.exists()
    assert metadata.frame_count > 0
    assert 0.2 <= metadata.duration_seconds <= 1.2
