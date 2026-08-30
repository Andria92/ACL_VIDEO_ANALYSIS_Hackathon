"""Validation and agreement utilities for human annotations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from acl_motion.annotations.models import HumanAnnotationSession, RoiKeyframeAnnotation
from acl_motion.annotations.propagation import propagated_bbox
from acl_motion.cases.annotations import EventAnnotation
from acl_motion.cases.models import InjurySide
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


@dataclass(frozen=True, slots=True)
class IndependentSessionAgreement:
    """Agreement summary for two separately created annotation sessions."""

    independent_sessions: bool
    same_case_and_source: bool
    roi_agreement: RoiAgreementSummary
    target_availability_exact_agreement: float | None
    unavailable_frame_jaccard: float | None
    movement_start_difference_frames: int | None
    movement_end_difference_frames: int | None
    movement_start_difference_ms: float | None
    movement_end_difference_ms: float | None
    injured_side_agreement: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "independent_sessions": self.independent_sessions,
            "same_case_and_source": self.same_case_and_source,
            "roi_agreement": self.roi_agreement.to_dict(),
            "target_availability_exact_agreement": self.target_availability_exact_agreement,
            "unavailable_frame_jaccard": self.unavailable_frame_jaccard,
            "movement_start_difference_frames": self.movement_start_difference_frames,
            "movement_end_difference_frames": self.movement_end_difference_frames,
            "movement_start_difference_ms": self.movement_start_difference_ms,
            "movement_end_difference_ms": self.movement_end_difference_ms,
            "injured_side_agreement": self.injured_side_agreement,
            "warnings": list(self.warnings),
            "interpretation": (
                "Agreement measures annotation repeatability. It does not establish that either "
                "annotation is biomechanically or clinically correct."
            ),
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
    if session.injured_side is InjurySide.UNKNOWN:
        warnings.append(
            "The documented injured knee is unknown. It will remain unknown; "
            "injury laterality is not inferred from movement or video appearance."
        )
    for keyframe in session.roi_keyframes:
        if keyframe.bbox.width <= 0 or keyframe.bbox.height <= 0:
            errors.append(f"ROI at frame {keyframe.frame_index} has invalid area.")
        if frame_count is not None and keyframe.frame_index >= frame_count:
            errors.append(f"ROI keyframe {keyframe.frame_index} is outside the video.")
    for interval in session.target_unavailable_intervals:
        if frame_count is not None and interval.end_frame >= frame_count:
            errors.append(
                "Target-unavailable interval "
                f"{interval.start_frame}-{interval.end_frame} is outside the video."
            )
    for interval in session.target_accepted_intervals:
        if frame_count is not None and interval.end_frame >= frame_count:
            errors.append(
                "Target-accepted interval "
                f"{interval.start_frame}-{interval.end_frame} is outside the video."
            )
    window = session.movement_window
    if window is not None:
        if frame_count is not None and window.movement_end_frame >= frame_count:
            errors.append(f"Movement End frame {window.movement_end_frame} is outside the video.")
        if window.movement_end_frame < window.movement_start_frame:
            errors.append("Movement End is before Movement Start.")
        for interval in session.target_accepted_intervals:
            if (
                interval.start_frame < window.movement_start_frame
                or interval.end_frame > window.movement_end_frame
            ):
                errors.append(
                    "Target-accepted interval "
                    f"{interval.start_frame}-{interval.end_frame} is outside the "
                    "generated Movement Window and cannot be pose-reviewed."
                )
    summary = {
        "target_roi_keyframes": session.manual_roi_keyframe_count,
        "target_unavailable_intervals": len(session.target_unavailable_intervals),
        "target_unavailable_frames": session.manual_target_unavailable_frame_count,
        "target_accepted_intervals": len(session.target_accepted_intervals),
        "target_accepted_frames": session.manual_target_accepted_frame_count,
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
        "injured_side": session.injured_side.value,
        "injury_laterality_source": session.injury_laterality_source,
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


def compare_independent_annotation_sessions(
    first: HumanAnnotationSession,
    second: HumanAnnotationSession,
    *,
    frame_count: int,
    fps: float,
    low_iou_threshold: float = 0.5,
) -> IndependentSessionAgreement:
    """Compare repeated sessions while preserving availability disagreements."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    first_provenance = first.provenance
    second_provenance = second.provenance
    same_case_and_source = (
        first_provenance.case_id == second_provenance.case_id
        and first_provenance.source_id == second_provenance.source_id
    )
    independent_sessions = (
        first_provenance.annotation_session_id
        != second_provenance.annotation_session_id
        and first_provenance.annotator_id != second_provenance.annotator_id
    )
    warnings = []
    if not same_case_and_source:
        warnings.append("The sessions do not describe the same registered case and source view.")
    if not independent_sessions:
        warnings.append(
            "Different session and annotator identifiers are required for independent agreement."
        )

    all_frames = tuple(range(frame_count))
    first_unavailable = {
        frame_index
        for frame_index in all_frames
        if first.target_unavailable_interval_at(frame_index) is not None
    }
    second_unavailable = {
        frame_index
        for frame_index in all_frames
        if second.target_unavailable_interval_at(frame_index) is not None
    }
    availability_agreement = sum(
        (frame_index in first_unavailable) == (frame_index in second_unavailable)
        for frame_index in all_frames
    ) / frame_count
    unavailable_union = first_unavailable.union(second_unavailable)
    unavailable_jaccard = (
        len(first_unavailable.intersection(second_unavailable)) / len(unavailable_union)
        if unavailable_union
        else 1.0
    )
    comparable_roi_frames = tuple(
        frame_index
        for frame_index in all_frames
        if frame_index not in first_unavailable
        and frame_index not in second_unavailable
    )
    roi_agreement = compare_roi_timelines(
        first.roi_keyframes,
        second.roi_keyframes,
        comparable_roi_frames,
        low_iou_threshold=low_iou_threshold,
    )

    first_window = first.movement_window
    second_window = second.movement_window
    start_difference = (
        first_window.movement_start_frame - second_window.movement_start_frame
        if first_window is not None and second_window is not None
        else None
    )
    end_difference = (
        first_window.movement_end_frame - second_window.movement_end_frame
        if first_window is not None and second_window is not None
        else None
    )
    return IndependentSessionAgreement(
        independent_sessions=independent_sessions,
        same_case_and_source=same_case_and_source,
        roi_agreement=roi_agreement,
        target_availability_exact_agreement=availability_agreement,
        unavailable_frame_jaccard=unavailable_jaccard,
        movement_start_difference_frames=start_difference,
        movement_end_difference_frames=end_difference,
        movement_start_difference_ms=(
            start_difference / fps * 1000 if start_difference is not None and fps else None
        ),
        movement_end_difference_ms=(
            end_difference / fps * 1000 if end_difference is not None and fps else None
        ),
        injured_side_agreement=first.injured_side is second.injured_side,
        warnings=tuple(warnings),
    )


def _area(bbox: BBox) -> float:
    return bbox.width * bbox.height
