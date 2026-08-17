"""ROI propagation helpers for sparse human keyframes."""

from __future__ import annotations

from collections.abc import Iterable

from acl_motion.annotations.models import RoiKeyframeAnnotation
from acl_motion.video.roi import BBox, RoiKeyframe, RoiTimeline


def propagated_bbox(
    keyframes: Iterable[RoiKeyframeAnnotation | RoiKeyframe],
    frame_index: int,
) -> BBox:
    """Return the linearly propagated ROI for a frame.

    Behavior outside the annotated interval intentionally matches the existing
    pose pipeline: the nearest endpoint keyframe is held constant.
    """

    timeline = roi_timeline_from_keyframes(tuple(keyframes))
    return timeline.bbox_for_frame(frame_index)


def roi_timeline_from_keyframes(
    keyframes: Iterable[RoiKeyframeAnnotation | RoiKeyframe],
) -> RoiTimeline:
    """Convert human annotation keyframes to the existing RoiTimeline type."""

    roi_keyframes = tuple(
        sorted(
            (
                RoiKeyframe(frame_index=_frame_index(item), bbox=_bbox(item))
                for item in keyframes
            ),
            key=lambda item: item.frame_index,
        )
    )
    return RoiTimeline(roi_keyframes)


def propagated_roi_records(
    keyframes: Iterable[RoiKeyframeAnnotation | RoiKeyframe],
    frames: Iterable[int],
) -> list[dict]:
    """Return propagated ROI records for a set of source frames."""

    timeline = roi_timeline_from_keyframes(tuple(keyframes))
    records = []
    for frame_index in frames:
        bbox = timeline.bbox_for_frame(int(frame_index))
        records.append(
            {
                "frame_index": int(frame_index),
                "x": bbox.x,
                "y": bbox.y,
                "width": bbox.width,
                "height": bbox.height,
            }
        )
    return records


def _frame_index(keyframe: RoiKeyframeAnnotation | RoiKeyframe) -> int:
    return int(keyframe.frame_index)


def _bbox(keyframe: RoiKeyframeAnnotation | RoiKeyframe) -> BBox:
    return keyframe.bbox
