"""Manual multi-view alignment anchors for one ACL event."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from acl_motion.persistence import atomic_write_json

ALIGNMENT_VERSION = "optional_multiview_alignment_v1"


@dataclass(frozen=True, slots=True)
class ViewAlignmentAnchor:
    """A human-marked event that corresponds across two or more views."""

    anchor_id: str
    label: str
    case_id: str
    view_frames: dict[str, int]
    notes: str = ""
    created_by: str = "researcher_01"
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.anchor_id:
            raise ValueError("Alignment anchor_id is required.")
        if not self.label:
            raise ValueError("Alignment label is required.")
        if len(self.view_frames) < 2:
            raise ValueError("An alignment anchor requires at least two view frames.")
        cleaned = {str(view): int(frame) for view, frame in self.view_frames.items()}
        if any(frame < 0 for frame in cleaned.values()):
            raise ValueError("Alignment frame indices must be non-negative.")
        object.__setattr__(self, "view_frames", cleaned)
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Return a JSON-ready anchor representation."""

        return {
            "anchor_id": self.anchor_id,
            "label": self.label,
            "case_id": self.case_id,
            "view_frames": self.view_frames,
            "notes": self.notes,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ViewAlignmentAnchor:
        """Load one anchor from JSON data."""

        return cls(
            anchor_id=str(data["anchor_id"]),
            label=str(data["label"]),
            case_id=str(data["case_id"]),
            view_frames={str(key): int(value) for key, value in data["view_frames"].items()},
            notes=str(data.get("notes", "")),
            created_by=str(data.get("created_by", "researcher_01")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True, slots=True)
class ViewAlignmentSet:
    """Manual alignment anchors for one ACL case."""

    case_id: str
    anchors: tuple[ViewAlignmentAnchor, ...] = ()
    alignment_version: str = ALIGNMENT_VERSION

    def to_dict(self) -> dict:
        """Return a JSON-ready alignment payload."""

        return {
            "case_id": self.case_id,
            "alignment_version": self.alignment_version,
            "anchor_count": len(self.anchors),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "note": (
                "Alignment anchors relate local source frames across views; they do not "
                "overwrite any original source frame numbers."
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ViewAlignmentSet:
        """Load an alignment set from JSON data."""

        return cls(
            case_id=str(data["case_id"]),
            anchors=tuple(
                ViewAlignmentAnchor.from_dict(item) for item in data.get("anchors", ())
            ),
            alignment_version=str(data.get("alignment_version", ALIGNMENT_VERSION)),
        )

    def upsert(self, anchor: ViewAlignmentAnchor) -> ViewAlignmentSet:
        """Return a copy with ``anchor`` added or replacing the same anchor_id."""

        if anchor.case_id != self.case_id:
            raise ValueError("Alignment anchor case_id does not match this alignment set.")
        anchors = [item for item in self.anchors if item.anchor_id != anchor.anchor_id]
        anchors.append(anchor)
        anchors.sort(key=lambda item: item.anchor_id)
        return ViewAlignmentSet(case_id=self.case_id, anchors=tuple(anchors))


def view_alignment_path(output_dir: str | Path, case_id: str) -> Path:
    """Return the human alignment path for a case without using raw frame numbers."""

    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("_")
    return Path(output_dir) / f"{safe_case_id}_view_alignment_human.json"


def load_view_alignment(output_dir: str | Path, case_id: str) -> ViewAlignmentSet:
    """Load human view alignment anchors, returning an empty set if none exist."""

    path = view_alignment_path(output_dir, case_id)
    if not path.exists():
        return ViewAlignmentSet(case_id=case_id)
    return ViewAlignmentSet.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_view_alignment(alignment: ViewAlignmentSet, output_dir: str | Path) -> Path:
    """Persist human view alignment anchors."""

    path = view_alignment_path(output_dir, alignment.case_id)
    if "_human" not in path.stem:
        raise ValueError(f"View alignment filename must include '_human': {path}")
    return atomic_write_json(path, alignment.to_dict())
