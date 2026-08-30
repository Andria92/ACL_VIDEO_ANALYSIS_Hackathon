"""Typed human annotation models for the M5.5 annotation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

from acl_motion.cases.annotations import AnchorType, EventAnnotation
from acl_motion.cases.models import InjurySide
from acl_motion.video.roi import BBox

ANNOTATION_UI_VERSION = "m5_5_annotation_ui_v1"


class AnnotatorType(StrEnum):
    """Allowed annotation provenance types."""

    HUMAN = "HUMAN"
    DEVELOPMENT = "DEVELOPMENT"


class OperatorFlag(StrEnum):
    """Optional human flags attached to a frame/keyframe."""

    PLAYER_OVERLAP = "PLAYER_OVERLAP"
    TARGET_PARTIALLY_OCCLUDED = "TARGET_PARTIALLY_OCCLUDED"
    TARGET_NOT_VISIBLE = "TARGET_NOT_VISIBLE"
    CAMERA_CUT = "CAMERA_CUT"
    ROI_DIFFICULT = "ROI_DIFFICULT"
    CANNOT_JUDGE = "CANNOT_JUDGE"
    OTHER = "OTHER"


class EventConfidence(StrEnum):
    """User-facing event-anchor confidence levels."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


CONFIDENCE_VALUES = {
    EventConfidence.HIGH: 0.9,
    EventConfidence.MODERATE: 0.6,
    EventConfidence.LOW: 0.3,
}


@dataclass(frozen=True, slots=True)
class AnnotationCase:
    """A local video source that can be annotated."""

    slug: str
    case_id: str
    source_id: str
    player_name: str
    video_path: Path
    view_id: str | None = None
    view_label: str = "Primary view"
    primary_view: bool = True
    perspective: str = "unknown"
    occlusion_level: str = "unknown"
    view_quality: str = "unknown"
    slow_motion: bool = False
    cropped_or_zoomed: bool = False
    real_time_scale: float | None = None
    injured_side: InjurySide = InjurySide.UNKNOWN
    injury_laterality_source: str = ""
    development_roi_path: Path | None = None
    development_event_path: Path | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-ready case representation."""

        return {
            "slug": self.slug,
            "case_id": self.case_id,
            "source_id": self.source_id,
            "view_id": self.view_id or self.source_id,
            "view_label": self.view_label,
            "primary_view": self.primary_view,
            "perspective": self.perspective,
            "occlusion_level": self.occlusion_level,
            "view_quality": self.view_quality,
            "slow_motion": self.slow_motion,
            "cropped_or_zoomed": self.cropped_or_zoomed,
            "real_time_scale": self.real_time_scale,
            "injured_side": InjurySide(self.injured_side).value,
            "injury_laterality_source": self.injury_laterality_source,
            "player_name": self.player_name,
            "video_path": str(self.video_path),
            "development_roi_available": self.development_roi_path is not None
            and self.development_roi_path.exists(),
            "development_event_available": self.development_event_path is not None
            and self.development_event_path.exists(),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class RoiKeyframeAnnotation:
    """A human-authored target-athlete ROI keyframe."""

    frame_index: int
    bbox: BBox
    flags: tuple[OperatorFlag, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("ROI keyframe frame_index must be non-negative.")
        object.__setattr__(self, "flags", tuple(_operator_flag(flag) for flag in self.flags))

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "frame_index": self.frame_index,
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height,
            },
            "flags": [flag.value for flag in self.flags],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RoiKeyframeAnnotation:
        """Create a keyframe from serialized data."""

        bbox = data.get("bbox") or data
        return cls(
            frame_index=int(data["frame_index"]),
            bbox=BBox(
                x=float(bbox["x"]),
                y=float(bbox["y"]),
                width=float(bbox["width"]),
                height=float(bbox["height"]),
            ),
            flags=tuple(_operator_flag(flag) for flag in data.get("flags", ())),
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class TargetUnavailableIntervalAnnotation:
    """Inclusive human-authored interval where target identity is unavailable.

    No ROI is propagated and no pose candidate is accepted inside this interval.
    The interval records absence as evidence instead of encoding it as a fake box.
    """

    start_frame: int
    end_frame: int
    reason: OperatorFlag = OperatorFlag.TARGET_NOT_VISIBLE
    note: str = ""

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < 0:
            raise ValueError("Target-unavailable interval frames must be non-negative.")
        if self.end_frame < self.start_frame:
            raise ValueError("Target-unavailable interval end must be at or after its start.")
        object.__setattr__(self, "reason", _operator_flag(self.reason))

    @property
    def frame_count(self) -> int:
        """Return the inclusive number of unavailable source frames."""

        return self.end_frame - self.start_frame + 1

    def contains(self, frame_index: int) -> bool:
        """Return whether frame_index falls inside this inclusive interval."""

        return self.start_frame <= frame_index <= self.end_frame

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "reason": self.reason.value,
            "note": self.note,
            "annotation_source": "human_ui",
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetUnavailableIntervalAnnotation:
        """Create an interval from serialized data."""

        return cls(
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            reason=_operator_flag(data.get("reason", OperatorFlag.TARGET_NOT_VISIBLE.value)),
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class TargetAcceptedIntervalAnnotation:
    """Inclusive interval where a human confirmed the selected pose is the target.

    Acceptance confirms target identity only. Downstream pose-quality rules may still
    withhold a frame when the skeleton itself is missing or too incomplete to measure.
    """

    start_frame: int
    end_frame: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < 0:
            raise ValueError("Target-accepted interval frames must be non-negative.")
        if self.end_frame < self.start_frame:
            raise ValueError("Target-accepted interval end must be at or after its start.")

    @property
    def frame_count(self) -> int:
        """Return the inclusive number of human-reviewed source frames."""

        return self.end_frame - self.start_frame + 1

    def contains(self, frame_index: int) -> bool:
        """Return whether frame_index falls inside this inclusive interval."""

        return self.start_frame <= frame_index <= self.end_frame

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "decision": "ACCEPTED",
            "note": self.note,
            "annotation_source": "human_ui_pose_review",
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetAcceptedIntervalAnnotation:
        """Create an accepted interval from serialized data."""

        return cls(
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class MovementWindowAnnotation:
    """Human-observable movement analysis window.

    Movement Start is an observability boundary, normally inferred from the first
    manual target ROI keyframe. Movement End is the visible end of the sequence;
    it is not a biological injury instant.
    """

    movement_start_frame: int
    movement_end_frame: int
    movement_start_timestamp_ms: float
    movement_end_timestamp_ms: float
    confidence: EventConfidence | None = None
    rationale: str = ""
    source: str = "human_ui_movement_window"

    def __post_init__(self) -> None:
        if self.movement_start_frame < 0 or self.movement_end_frame < 0:
            raise ValueError("Movement window frames must be non-negative.")
        if self.movement_end_frame < self.movement_start_frame:
            raise ValueError("Movement End must be at or after Movement Start.")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", EventConfidence(self.confidence))

    @property
    def duration_ms(self) -> float:
        """Return movement-window duration in milliseconds."""

        return self.movement_end_timestamp_ms - self.movement_start_timestamp_ms

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "movement_start_frame": self.movement_start_frame,
            "movement_start_timestamp_ms": self.movement_start_timestamp_ms,
            "movement_end_frame": self.movement_end_frame,
            "movement_end_timestamp_ms": self.movement_end_timestamp_ms,
            "movement_duration_ms": self.duration_ms,
            "confidence": self.confidence.value if self.confidence else None,
            "rationale": self.rationale,
            "source": self.source,
            "terminology_note": (
                "Movement End is the visible end of the selected movement sequence, "
                "not an ACL rupture or injury instant."
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MovementWindowAnnotation:
        """Create a MovementWindowAnnotation from serialized data."""

        confidence = data.get("confidence")
        return cls(
            movement_start_frame=int(data["movement_start_frame"]),
            movement_start_timestamp_ms=float(data["movement_start_timestamp_ms"]),
            movement_end_frame=int(data["movement_end_frame"]),
            movement_end_timestamp_ms=float(data["movement_end_timestamp_ms"]),
            confidence=EventConfidence(confidence) if confidence else None,
            rationale=str(data.get("rationale", "")),
            source=str(data.get("source", "human_ui_movement_window")),
        )


@dataclass(frozen=True, slots=True)
class AnnotationProvenance:
    """Provenance for one annotation session."""

    annotation_session_id: str
    annotator_type: AnnotatorType
    annotator_id: str
    created_at: str
    updated_at: str
    case_id: str
    source_id: str
    video_path: str
    view_id: str | None = None
    ui_tool_version: str = ANNOTATION_UI_VERSION

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        source_id: str,
        video_path: str | Path,
        annotator_id: str,
        view_id: str | None = None,
    ) -> AnnotationProvenance:
        """Create human provenance for a user-saved annotation session."""

        now = datetime.now(UTC).isoformat()
        return cls(
            annotation_session_id=str(uuid4()),
            annotator_type=AnnotatorType.HUMAN,
            annotator_id=annotator_id,
            created_at=now,
            updated_at=now,
            case_id=case_id,
            source_id=source_id,
            video_path=str(video_path),
            view_id=view_id or source_id,
        )

    def touch(self) -> AnnotationProvenance:
        """Return a provenance copy with an updated timestamp."""

        return AnnotationProvenance(
            annotation_session_id=self.annotation_session_id,
            annotator_type=self.annotator_type,
            annotator_id=self.annotator_id,
            created_at=self.created_at,
            updated_at=datetime.now(UTC).isoformat(),
            case_id=self.case_id,
            source_id=self.source_id,
            video_path=self.video_path,
            view_id=self.view_id or self.source_id,
            ui_tool_version=self.ui_tool_version,
        )

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "annotation_session_id": self.annotation_session_id,
            "annotator_type": self.annotator_type.value,
            "annotator_id": self.annotator_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "case_id": self.case_id,
            "source_id": self.source_id,
            "view_id": self.view_id or self.source_id,
            "video_path": self.video_path,
            "ui_tool_version": self.ui_tool_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AnnotationProvenance:
        """Load provenance from serialized data."""

        return cls(
            annotation_session_id=str(data["annotation_session_id"]),
            annotator_type=AnnotatorType(data.get("annotator_type", AnnotatorType.HUMAN.value)),
            annotator_id=str(data.get("annotator_id", "")),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            case_id=str(data["case_id"]),
            source_id=str(data["source_id"]),
            video_path=str(data["video_path"]),
            view_id=str(data.get("view_id") or data["source_id"]),
            ui_tool_version=str(data.get("ui_tool_version", ANNOTATION_UI_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class HumanAnnotationSession:
    """Complete human annotation session state."""

    provenance: AnnotationProvenance
    roi_keyframes: tuple[RoiKeyframeAnnotation, ...] = ()
    target_unavailable_intervals: tuple[TargetUnavailableIntervalAnnotation, ...] = ()
    target_accepted_intervals: tuple[TargetAcceptedIntervalAnnotation, ...] = ()
    movement_window: MovementWindowAnnotation | None = None
    event_annotation: EventAnnotation | None = None
    event_confidence_label: EventConfidence | None = None
    injured_side: InjurySide = InjurySide.UNKNOWN
    injury_laterality_source: str = ""
    operator_flags: tuple[OperatorFlag, ...] = ()
    notes: str = ""
    finalized: bool = False

    def __post_init__(self) -> None:
        sorted_keyframes = tuple(sorted(self.roi_keyframes, key=lambda item: item.frame_index))
        frames = [item.frame_index for item in sorted_keyframes]
        if len(frames) != len(set(frames)):
            raise ValueError("Human ROI keyframes must not contain duplicate frame indices.")
        object.__setattr__(self, "roi_keyframes", sorted_keyframes)
        sorted_intervals = tuple(
            sorted(
                self.target_unavailable_intervals,
                key=lambda item: (item.start_frame, item.end_frame),
            )
        )
        for previous, current in pairwise(sorted_intervals):
            if current.start_frame <= previous.end_frame:
                raise ValueError("Human target-unavailable intervals must not overlap.")
        interval_keyframes = [
            keyframe.frame_index
            for keyframe in sorted_keyframes
            if any(interval.contains(keyframe.frame_index) for interval in sorted_intervals)
        ]
        if interval_keyframes:
            raise ValueError(
                "Human ROI keyframes cannot exist inside target-unavailable intervals: "
                f"{interval_keyframes}."
            )
        object.__setattr__(self, "target_unavailable_intervals", sorted_intervals)
        sorted_accepted = tuple(
            sorted(
                self.target_accepted_intervals,
                key=lambda item: (item.start_frame, item.end_frame),
            )
        )
        for previous, current in pairwise(sorted_accepted):
            if current.start_frame <= previous.end_frame:
                raise ValueError("Human target-accepted intervals must not overlap.")
        conflicting_reviews = [
            (accepted.start_frame, accepted.end_frame)
            for accepted in sorted_accepted
            if any(
                accepted.start_frame <= unavailable.end_frame
                and accepted.end_frame >= unavailable.start_frame
                for unavailable in sorted_intervals
            )
        ]
        if conflicting_reviews:
            raise ValueError(
                "Human target-accepted intervals cannot overlap target-unavailable "
                f"intervals: {conflicting_reviews}."
            )
        object.__setattr__(self, "target_accepted_intervals", sorted_accepted)
        object.__setattr__(self, "injured_side", InjurySide(self.injured_side))
        object.__setattr__(
            self,
            "operator_flags",
            tuple(_operator_flag(flag) for flag in self.operator_flags),
        )

    @property
    def manual_roi_keyframe_count(self) -> int:
        """Return the number of human ROI keyframes."""

        return len(self.roi_keyframes)

    @property
    def manual_target_unavailable_frame_count(self) -> int:
        """Return the number of source frames explicitly excluded by the operator."""

        return sum(interval.frame_count for interval in self.target_unavailable_intervals)

    @property
    def manual_target_accepted_frame_count(self) -> int:
        """Return the number of source frames explicitly accepted after pose review."""

        return sum(interval.frame_count for interval in self.target_accepted_intervals)

    def target_unavailable_interval_at(
        self,
        frame_index: int,
    ) -> TargetUnavailableIntervalAnnotation | None:
        """Return the human exclusion interval containing frame_index, if any."""

        return next(
            (
                interval
                for interval in self.target_unavailable_intervals
                if interval.contains(frame_index)
            ),
            None,
        )

    def target_accepted_interval_at(
        self,
        frame_index: int,
    ) -> TargetAcceptedIntervalAnnotation | None:
        """Return the human acceptance interval containing frame_index, if any."""

        return next(
            (
                interval
                for interval in self.target_accepted_intervals
                if interval.contains(frame_index)
            ),
            None,
        )

    def replace_keyframe(self, keyframe: RoiKeyframeAnnotation) -> HumanAnnotationSession:
        """Return a copy with the keyframe at the same frame replaced."""

        keyframes = [item for item in self.roi_keyframes if item.frame_index != keyframe.frame_index]
        keyframes.append(keyframe)
        return self.with_changes(roi_keyframes=tuple(keyframes))

    def delete_keyframe(self, frame_index: int) -> HumanAnnotationSession:
        """Return a copy without a keyframe at frame_index."""

        return self.with_changes(
            roi_keyframes=tuple(
                item for item in self.roi_keyframes if item.frame_index != frame_index
            )
        )

    def with_changes(self, **changes) -> HumanAnnotationSession:
        """Return a copy with selected fields changed and provenance updated."""

        data = {
            "provenance": self.provenance.touch(),
            "roi_keyframes": self.roi_keyframes,
            "target_unavailable_intervals": self.target_unavailable_intervals,
            "target_accepted_intervals": self.target_accepted_intervals,
            "movement_window": self.movement_window,
            "event_annotation": self.event_annotation,
            "event_confidence_label": self.event_confidence_label,
            "injured_side": self.injured_side,
            "injury_laterality_source": self.injury_laterality_source,
            "operator_flags": self.operator_flags,
            "notes": self.notes,
            "finalized": self.finalized,
        }
        data.update(changes)
        return HumanAnnotationSession(**data)

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""

        return {
            "provenance": self.provenance.to_dict(),
            "manual_roi_keyframe_count": self.manual_roi_keyframe_count,
            "roi_keyframes": [keyframe.to_dict() for keyframe in self.roi_keyframes],
            "manual_target_unavailable_frame_count": self.manual_target_unavailable_frame_count,
            "target_unavailable_intervals": [
                interval.to_dict() for interval in self.target_unavailable_intervals
            ],
            "manual_target_accepted_frame_count": self.manual_target_accepted_frame_count,
            "target_accepted_intervals": [
                interval.to_dict() for interval in self.target_accepted_intervals
            ],
            "movement_window": (
                self.movement_window.to_dict() if self.movement_window is not None else None
            ),
            "event_annotation": (
                self.event_annotation.to_dict() if self.event_annotation is not None else None
            ),
            "event_confidence_label": (
                self.event_confidence_label.value if self.event_confidence_label else None
            ),
            "injured_side": self.injured_side.value,
            "injury_laterality_source": self.injury_laterality_source,
            "operator_flags": [flag.value for flag in self.operator_flags],
            "notes": self.notes,
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HumanAnnotationSession:
        """Load a human annotation session from serialized data."""

        event_data = data.get("event_annotation")
        event_annotation = None
        if event_data is not None:
            event_data = dict(event_data)
            event_data["event_anchor_type"] = AnchorType(event_data["event_anchor_type"])
            event_annotation = EventAnnotation(**event_data)
        confidence = data.get("event_confidence_label")
        return cls(
            provenance=AnnotationProvenance.from_dict(data["provenance"]),
            roi_keyframes=tuple(
                RoiKeyframeAnnotation.from_dict(item)
                for item in data.get("roi_keyframes", ())
            ),
            target_unavailable_intervals=tuple(
                TargetUnavailableIntervalAnnotation.from_dict(item)
                for item in data.get("target_unavailable_intervals", ())
            ),
            target_accepted_intervals=tuple(
                TargetAcceptedIntervalAnnotation.from_dict(item)
                for item in data.get("target_accepted_intervals", ())
            ),
            movement_window=(
                MovementWindowAnnotation.from_dict(data["movement_window"])
                if data.get("movement_window")
                else None
            ),
            event_annotation=event_annotation,
            event_confidence_label=EventConfidence(confidence) if confidence else None,
            injured_side=InjurySide(data.get("injured_side", InjurySide.UNKNOWN.value)),
            injury_laterality_source=str(data.get("injury_laterality_source", "")),
            operator_flags=tuple(
                _operator_flag(flag) for flag in data.get("operator_flags", ())
            ),
            notes=str(data.get("notes", "")),
            finalized=bool(data.get("finalized", False)),
        )


def confidence_value(confidence: EventConfidence | str | float | None) -> float | None:
    """Return a numeric confidence value compatible with EventAnnotation."""

    if confidence is None or confidence == "":
        return None
    if isinstance(confidence, int | float):
        return float(confidence)
    return CONFIDENCE_VALUES[EventConfidence(str(confidence).lower())]


def _operator_flag(flag: OperatorFlag | str) -> OperatorFlag:
    if isinstance(flag, OperatorFlag):
        return flag
    return OperatorFlag(str(flag))
