"""Canonical pose output models.

Raw landmark values are immutable records. Later cleaning/smoothing stages should
write separate outputs rather than overwriting these rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from acl_motion.video.roi import BBox


class QualityFlagCode(StrEnum):
    """Structured pose-quality and availability flags."""

    LOW_LANDMARK_CONFIDENCE = "LOW_LANDMARK_CONFIDENCE"
    LOW_LANDMARK_VISIBILITY = "LOW_LANDMARK_VISIBILITY"
    ATHLETE_TOO_SMALL = "ATHLETE_TOO_SMALL"
    TARGET_IDENTITY_UNCERTAIN = "TARGET_IDENTITY_UNCERTAIN"
    LANDMARK_TEMPORAL_JUMP = "LANDMARK_TEMPORAL_JUMP"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    HUMAN_TARGET_UNAVAILABLE = "HUMAN_TARGET_UNAVAILABLE"
    INSUFFICIENT_LOWER_LIMB_COVERAGE = "INSUFFICIENT_LOWER_LIMB_COVERAGE"
    INSUFFICIENT_CORE_COVERAGE = "INSUFFICIENT_CORE_COVERAGE"
    LANDMARK_OUTSIDE_IMAGE = "LANDMARK_OUTSIDE_IMAGE"
    ROI_OUTSIDE_IMAGE = "ROI_OUTSIDE_IMAGE"


@dataclass(frozen=True, slots=True)
class QualityFlag:
    """A traceable quality flag attached to a frame or landmark."""

    code: QualityFlagCode
    message: str = ""
    frame_index: int | None = None
    landmark_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_code": self.code.value,
            "quality_message": self.message,
            "quality_frame_index": self.frame_index,
            "quality_landmark_name": self.landmark_name,
        }


@dataclass(frozen=True, slots=True)
class Landmark:
    """One canonical 2D landmark observation for one frame."""

    name: str
    x_px: float | None
    y_px: float | None
    x_norm: float | None
    y_norm: float | None
    confidence: float | None = None
    visibility: float | None = None
    presence: float | None = None
    observed: bool = False
    backend_specific_metadata: dict[str, Any] = field(default_factory=dict)
    quality_flags: tuple[QualityFlag, ...] = field(default_factory=tuple)

    def to_row(
        self,
        *,
        case_id: str | None,
        source_id: str,
        frame_index: int,
        timestamp_ms: float,
        backend: str,
        target_bbox: BBox | None,
        frame_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a flat long-format row for persistence."""

        backend_metadata = dict(frame_metadata or {})
        backend_metadata.update(self.backend_specific_metadata)
        row: dict[str, Any] = {
            "case_id": case_id,
            "source_id": source_id,
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "landmark_name": self.name,
            "x_px": self.x_px,
            "y_px": self.y_px,
            "x_norm": self.x_norm,
            "y_norm": self.y_norm,
            "confidence": self.confidence,
            "visibility": self.visibility,
            "presence": self.presence,
            "observed": self.observed,
            "backend": backend,
            "backend_metadata": backend_metadata,
            "quality_flags": [flag.to_dict() for flag in self.quality_flags],
        }
        if target_bbox is not None:
            row.update(target_bbox.to_dict(prefix="target_bbox"))
        else:
            row.update(
                {
                    "target_bbox_x": None,
                    "target_bbox_y": None,
                    "target_bbox_width": None,
                    "target_bbox_height": None,
                }
            )
        return row


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """Canonical pose output for one video frame."""

    frame_index: int
    timestamp_ms: float
    source_id: str
    backend: str
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    target_bbox: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_flags: tuple[QualityFlag, ...] = field(default_factory=tuple)

    def iter_rows(self, case_id: str | None = None) -> Iterable[dict[str, Any]]:
        """Yield long-format landmark rows."""

        for landmark in self.landmarks.values():
            yield landmark.to_row(
                case_id=case_id,
                source_id=self.source_id,
                frame_index=self.frame_index,
                timestamp_ms=self.timestamp_ms,
                backend=self.backend,
                target_bbox=self.target_bbox,
                frame_metadata=self.metadata,
            )


@dataclass(frozen=True, slots=True)
class PoseSequence:
    """Pose frames for one source and one backend analysis run."""

    case_id: str | None
    source_id: str
    backend: str
    frames: tuple[PoseFrame, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_landmark_rows(self) -> Iterable[dict[str, Any]]:
        """Yield all long-format landmark rows in frame order."""

        for frame in self.frames:
            yield from frame.iter_rows(case_id=self.case_id)

    def to_dataframe(self):
        """Create a pandas DataFrame without making pandas a model import dependency."""

        import pandas as pd

        return pd.DataFrame(list(self.iter_landmark_rows()))
