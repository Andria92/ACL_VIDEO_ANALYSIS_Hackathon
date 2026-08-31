"""Human-linked contextual video clips that are excluded from measurement pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from acl_motion.persistence import atomic_write_json, path_lock

CONTEXT_CLIP_ROLE = "REAL_TIME_CONTEXT"
CONTEXT_CLIPS_FILENAME = "context_video_clips_human.json"


@dataclass(frozen=True, slots=True)
class ContextVideoClip:
    """A human-linked real-time clip used only for visual event context."""

    clip_id: str
    case_id: str
    video_path: Path
    source_video_path: Path
    start_seconds: float
    end_seconds: float
    created_at: str
    created_by: str = "researcher_01"
    label: str = "Real-time injury sequence"
    role: str = CONTEXT_CLIP_ROLE

    def __post_init__(self) -> None:
        if not self.clip_id.strip():
            raise ValueError("Context clip id cannot be empty.")
        if not self.case_id.strip():
            raise ValueError("Context clip case id cannot be empty.")
        if self.role != CONTEXT_CLIP_ROLE:
            raise ValueError(f"Unsupported context clip role: {self.role}")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("Context clip end must be after its non-negative start.")

    @property
    def duration_seconds(self) -> float:
        """Return the selected context duration."""

        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict:
        """Return a JSON-ready provenance record."""

        return {
            "clip_id": self.clip_id,
            "case_id": self.case_id,
            "role": self.role,
            "label": self.label,
            "video_path": str(self.video_path),
            "source_video_path": str(self.source_video_path),
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "use_for_measurements": False,
            "use_for_movement_narrative": False,
            "automated_contact_interpretation": False,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ContextVideoClip:
        """Create a context clip from a stored registry record."""

        return cls(
            clip_id=str(payload["clip_id"]),
            case_id=str(payload["case_id"]),
            role=str(payload.get("role", CONTEXT_CLIP_ROLE)),
            label=str(payload.get("label", "Real-time injury sequence")),
            video_path=Path(str(payload["video_path"])),
            source_video_path=Path(str(payload["source_video_path"])),
            start_seconds=float(payload["start_seconds"]),
            end_seconds=float(payload["end_seconds"]),
            created_at=str(payload["created_at"]),
            created_by=str(payload.get("created_by", "researcher_01")),
        )


def new_context_video_clip(
    *,
    case_id: str,
    video_path: str | Path,
    source_video_path: str | Path,
    start_seconds: float,
    end_seconds: float,
    created_by: str = "researcher_01",
) -> ContextVideoClip:
    """Create a new context-only clip record with human provenance."""

    return ContextVideoClip(
        clip_id=f"context_{uuid4().hex}",
        case_id=case_id,
        video_path=Path(video_path).resolve(),
        source_video_path=Path(source_video_path).resolve(),
        start_seconds=float(start_seconds),
        end_seconds=float(end_seconds),
        created_at=datetime.now(UTC).isoformat(),
        created_by=created_by.strip() or "researcher_01",
    )


def context_clip_registry_path(data_root: str | Path = "data") -> Path:
    """Return the canonical human context-clip registry path."""

    return Path(data_root) / "annotations" / "human" / CONTEXT_CLIPS_FILENAME


def load_context_video_clips(path: str | Path) -> tuple[ContextVideoClip, ...]:
    """Load valid context clips while preserving unavailable-file provenance."""

    registry_path = Path(path)
    if not registry_path.exists():
        return ()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    clips = []
    for record in payload.get("clips", ()):
        try:
            clips.append(ContextVideoClip.from_dict(record))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(clips)


def save_context_video_clip(clip: ContextVideoClip, path: str | Path) -> Path:
    """Append or replace one context clip in the human registry."""

    registry_path = Path(path)
    with path_lock(registry_path):
        clips = [
            item
            for item in load_context_video_clips(registry_path)
            if item.clip_id != clip.clip_id
        ]
        clips.append(clip)
        atomic_write_json(
            registry_path,
            {"clips": [item.to_dict() for item in clips]},
        )
    return registry_path


def context_video_clips_for_case(
    case_id: str,
    path: str | Path,
) -> tuple[ContextVideoClip, ...]:
    """Return context clips linked to one documented injury case."""

    return tuple(
        clip for clip in load_context_video_clips(path) if clip.case_id == case_id
    )


def context_video_clip_by_id(
    clip_id: str,
    *,
    case_id: str,
    path: str | Path,
) -> ContextVideoClip:
    """Return one clip only when its id and associated injury case both match."""

    for clip in context_video_clips_for_case(case_id, path):
        if clip.clip_id == clip_id:
            return clip
    raise KeyError("Unknown real-time context clip for this injury case.")
