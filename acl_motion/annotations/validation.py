"""Validation and agreement utilities for human annotations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from acl_motion.annotations.models import HumanAnnotationSession, RoiKeyframeAnnotation
from acl_motion.annotations.propagation import propagated_bbox
from acl_motion.cases.annotations import EventAnnotation
from acl_motion.video.roi import BBox, RoiKeyframe


@dataclass(frozen=True, slots=True)
class AnnotationValidationResult:
    """Validation result shown before save/finalization."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: dict

    @property
    def ok(self) -> bool:
        """Return True when there are no blocking errors."""

        return not self.errors

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class RoiAgreementSummary:
    """Bounding-box agreement between two independently propagated timelines."""

    frames_evaluated: int
    mean_iou: float | None
    median_iou: float | None
    minimum_iou: float | None
    low_agreement_frames: tuple[dict, ...]

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "frames_evaluated": self.frames_evaluated,
            "mean_iou": self.mean_iou,
            "median_iou": self.median_iou,
            "minimum_iou": self.minimum_iou,
            "low_agreement_frames": list(self.low_agreement_frames),
        }


@dataclass(frozen=True, slots=True)
class EventAnchorComparison:
    """Human-vs-development event anchor timing difference."""

    human_frame: int | None
    development_frame: int | None
    frame_difference: int | None
    absolute_frame_difference: int | None
    time_difference_ms: float | None

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "human_frame": self.human_frame,
            "development_frame": self.development_frame,
            "frame_difference": self.frame_difference,
            "absolute_frame_difference": self.absolute_frame_difference,
            "time_difference_ms": self.time_difference_ms,
        }


def validate_annotation_session(
    session: HumanAnnotationSession,
    *,
    frame_count: int | None = None,
) -> AnnotationValidationResult:
    """Validate a human annotation session for save/review."""

    errors: list[str] = []
    warnings: list[str] = []
    if not session.roi_keyframes:
        warnings.append("No target ROI keyframe has been saved yet.")
    if session.movement_window is None:
        warnings.append("No Movement End has been marked yet.")
    for keyframe in session.roi_keyframes:
        if keyframe.bbox.width <= 0 or keyframe.bbox.height <= 0:
            errors.append(f"ROI at frame {keyframe.frame_index} has invalid area.")
        if frame_count is not None and keyframe.frame_index >= frame_count:
            errors.append(f"ROI keyframe {keyframe.frame_index} is outside the video.")
    window = session.movement_window
    if window is not None:
        if frame_count is not None and window.movement_end_frame >= frame_count:
            errors.append(f"Movement End frame {window.movement_end_frame} is outside the video.")
        if window.movement_end_frame < window.movement_start_frame:
            errors.append("Movement End is before Movement Start.")
    summary = {
        "target_roi_keyframes": session.manual_roi_keyframe_count,
        "movement_start": window.movement_start_frame if window is not None else (
            session.roi_keyframes[0].frame_index if session.roi_keyframes else None
        ),
        "last_keyframe": (
            session.roi_keyframes[-1].frame_index if session.roi_keyframes else None
        ),
        "movement_end": window.movement_end_frame if window is not None else None,
        "movement_duration_ms": window.duration_ms if window is not None else None,
        "confidence": (
            session.event_confidence_label.value if session.event_confidence_label else None
        ),
    }
    return AnnotationValidationResult(tuple(errors), tuple(warnings), summary)


def bbox_iou(a: BBox, b: BBox) -> float:
    """Return intersection-over-union for two image-space boxes."""

    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    intersection_width = max(0.0, ix2 - ix1)
    intersection_height = max(0.0, iy2 - iy1)
    intersection = intersection_width * intersection_height
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def compare_roi_timelines(
    human_keyframes: tuple[RoiKeyframeAnnotation | RoiKeyframe, ...],
    development_keyframes: tuple[RoiKeyframeAnnotation | RoiKeyframe, ...],
    frames: tuple[int, ...],
    *,
    low_iou_threshold: float = 0.5,
) -> RoiAgreementSummary:
    """Compare two annotation timelines after independent propagation."""

    if not human_keyframes or not development_keyframes or not frames:
        return RoiAgreementSummary(0, None, None, None, ())
    values: list[float] = []
    low_frames: list[dict] = []
    for frame_index in frames:
        human_bbox = propagated_bbox(human_keyframes, frame_index)
        development_bbox = propagated_bbox(development_keyframes, frame_index)
        value = bbox_iou(human_bbox, development_bbox)
        values.append(value)
        if value < low_iou_threshold:
            low_frames.append({"frame_index": int(frame_index), "iou": value})
    return RoiAgreementSummary(
        frames_evaluated=len(values),
        mean_iou=mean(values),
        median_iou=median(values),
        minimum_iou=min(values),
        low_agreement_frames=tuple(low_frames),
    )


def compare_event_anchors(
    human: EventAnnotation,
    development: EventAnnotation,
    *,
    fps: float,
) -> EventAnchorComparison:
    """Compare human and development event anchors in frames and milliseconds."""

    human_frame = human.t0_frame()
    development_frame = development.t0_frame()
    if human_frame is None or development_frame is None:
        return EventAnchorComparison(human_frame, development_frame, None, None, None)
    frame_difference = int(human_frame - development_frame)
    time_difference_ms = frame_difference / fps * 1000 if fps else None
    return EventAnchorComparison(
        human_frame=human_frame,
        development_frame=development_frame,
        frame_difference=frame_difference,
        absolute_frame_difference=abs(frame_difference),
        time_difference_ms=time_difference_ms,
    )


def _area(bbox: BBox) -> float:
    return bbox.width * bbox.height
