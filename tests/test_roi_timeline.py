from pathlib import Path

from acl_motion.video.roi import BBox, RoiKeyframe, RoiTimeline


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
