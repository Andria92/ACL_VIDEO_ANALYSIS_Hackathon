"""Typed case and video-source models.

ACL diagnosis, injury laterality, and clinical context are provenance fields. They are
not inferred from the video pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path


class InjurySide(StrEnum):
    """Documented ACL laterality from external provenance."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class CameraView(StrEnum):
    """Approximate broadcast camera view labels.

    These are intentionally "ish" labels because broadcast footage is not an
    anatomical measurement plane.
    """

    SAGITTAL_ISH = "sagittal-ish"
    FRONTAL_ISH = "frontal-ish"
    OBLIQUE = "oblique"
    HIGH_WIDE = "high-wide"
    UNKNOWN = "unknown"


class SourceType(StrEnum):
    """Video source type for one view of one documented event."""

    ORIGINAL_BROADCAST = "original_broadcast"
    CLOSE_REPLAY = "close_replay"
    SLOW_MOTION_REPLAY = "slow_motion_replay"
    SOCIAL_MEDIA_CROP = "social_media_crop"
    SPECTATOR_RECORDING = "spectator_recording"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class VideoSource:
    """A single video view/source for one documented ACL case."""

    source_id: str
    case_id: str
    file_path: Path
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    source_type: SourceType = SourceType.UNKNOWN
    replay_type: str | None = None
    camera_view: CameraView = CameraView.UNKNOWN
    camera_view_confidence: float | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ACLCase:
    """Documented ACL injury case with one or more video sources."""

    case_id: str
    player_name: str
    match_date: date | None = None
    team: str | None = None
    opponent: str | None = None
    acl_status: str = "documented_acl_injury"
    injured_side: InjurySide = InjurySide.UNKNOWN
    clinical_metadata_status: str = "unverified"
    provenance_notes: str = ""
    video_sources: tuple[VideoSource, ...] = field(default_factory=tuple)

    def with_video_source(self, source: VideoSource) -> ACLCase:
        """Return a copy with an additional video source for the same case."""

        if source.case_id != self.case_id:
            raise ValueError(
                f"VideoSource case_id {source.case_id!r} does not match {self.case_id!r}."
            )
        return ACLCase(
            case_id=self.case_id,
            player_name=self.player_name,
            match_date=self.match_date,
            team=self.team,
            opponent=self.opponent,
            acl_status=self.acl_status,
            injured_side=self.injured_side,
            clinical_metadata_status=self.clinical_metadata_status,
            provenance_notes=self.provenance_notes,
            video_sources=(*self.video_sources, source),
        )
