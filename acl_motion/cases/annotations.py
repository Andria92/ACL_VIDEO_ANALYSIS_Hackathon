"""Manual event annotations for event-relative analysis.

Event timing is a human-authored analytical reference. The video pipeline never
infers the physiological moment of ACL rupture.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from acl_motion.persistence import atomic_write_json


class AnchorType(StrEnum):
    """Supported manual event anchors."""

    INITIAL_CONTACT = "initial_contact"
    CRITICAL_PLANT = "critical_plant"
    ESTIMATED_EVENT_START = "estimated_event_start"
    ESTIMATED_EVENT_END = "estimated_event_end"
    ESTIMATED_EVENT_ANCHOR = "estimated_event_anchor"
    COLLAPSE = "collapse"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EventAnnotation:
    """Human-authored frame annotations for a source video."""

    case_id: str
    source_id: str
    view_id: str | None = None
    analysis_start_frame: int | None = None
    analysis_end_frame: int | None = None
    initial_contact_frame: int | None = None
    critical_plant_frame: int | None = None
    estimated_event_start_frame: int | None = None
    estimated_event_end_frame: int | None = None
    event_anchor_frame: int | None = None
    event_anchor_type: AnchorType = AnchorType.UNKNOWN
    collapse_frame: int | None = None
    annotation_confidence: float | None = None
    annotation_method: str = "manual"
    annotator: str = ""
    notes: str = ""

    def t0_frame(self) -> int | None:
        """Return the preferred manual t0 frame, if one has been supplied."""

        return (
            self.event_anchor_frame
            if self.event_anchor_frame is not None
            else self.critical_plant_frame
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        data = asdict(self)
        data["event_anchor_type"] = self.event_anchor_type.value
        return data


def load_event_annotation(path: str | Path) -> EventAnnotation:
    """Load a manual event annotation JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "event_anchor_type" in data:
        data["event_anchor_type"] = AnchorType(data["event_anchor_type"])
    return EventAnnotation(**data)


def write_event_annotation(annotation: EventAnnotation, path: str | Path) -> Path:
    """Write a manual event annotation JSON file."""

    output = Path(path)
    return atomic_write_json(output, annotation.to_dict())
