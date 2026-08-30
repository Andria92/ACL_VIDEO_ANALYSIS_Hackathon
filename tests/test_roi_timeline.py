from pathlib import Path

from acl_motion.video.roi import BBox, RoiKeyframe, RoiTimeline
from scripts.extract_pose import pose_extraction_roi


def test_roi_timeline_interpolates_between_keyframes():
    timeline = RoiTimeline(
        (
            RoiKeyframe(0, BBox(0, 10, 100, 200)),
            RoiKeyframe(10, BBox(20, 30, 120, 220)),
        )
    )

    bbox = timeline.bbox_for_frame(5)

    assert bbox.x == 10
    assert bbox.y == 20
    assert bbox.width == 110
    assert bbox.height == 210


def test_roi_timeline_loads_csv(tmp_path: Path):
    path = tmp_path / "roi.csv"
    path.write_text(
        "frame_index,x,y,width,height\n"
        "10,20,30,120,220\n"
        "0,0,10,100,200\n",
        encoding="utf-8",
    )

    timeline = RoiTimeline.from_csv(path)

    assert timeline.keyframes[0].frame_index == 0
    assert timeline.bbox_for_frame(10).x == 20


def test_pose_extraction_padding_applies_after_keyframe_interpolation():
    timeline = RoiTimeline(
        (
            RoiKeyframe(0, BBox(0, 10, 100, 200)),
            RoiKeyframe(10, BBox(20, 30, 120, 220)),
        )
    )

    bbox = pose_extraction_roi(
        frame_index=5,
        roi_timeline=timeline,
        static_roi=None,
        padding_fraction=0.2,
    )

    assert bbox == BBox(x=-12, y=-22, width=154, height=294)
    assert timeline.bbox_for_frame(5) == BBox(x=10, y=20, width=110, height=210)
