"""Structured semantic movement observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ObservationEvidenceStatus(StrEnum):
    """User-facing evidence state for semantic observations."""

    SUPPORTED = "SUPPORTED"
    LIMITED = "LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class MovementObservation:
    """A deterministic, evidence-backed movement interpretation."""

    observation_id: str
    case_id: str
    category: str
    title: str
    plain_language_summary: str
    technical_feature_names: tuple[str, ...] = ()
    value: float | None = None
    unit: str = ""
    start_value: float | None = None
    end_value: float | None = None
    change: float | None = None
    peak_value: float | None = None
    peak_source_frame: int | None = None
    peak_movement_relative_ms: float | None = None
    evidence_status: ObservationEvidenceStatus | str = ObservationEvidenceStatus.UNAVAILABLE
    evidence_completeness: float | None = None
    quality_reasons: tuple[str, ...] = ()
    source_frames: tuple[int, ...] = ()
    technical_explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_status", ObservationEvidenceStatus(self.evidence_status))
        forbidden = ("abnormal", "pathological", "dangerous", "risky")
        text = " ".join(
            [
                self.title,
                self.plain_language_summary,
                self.technical_explanation,
                " ".join(self.quality_reasons),
            ]
        ).lower()
        if any(term in text for term in forbidden):
            raise ValueError("MovementObservation text must remain task-neutral and non-clinical.")

    def to_dict(self) -> dict:
        """Return a JSON-ready observation record."""

        payload = asdict(self)
        payload["evidence_status"] = self.evidence_status.value
        return payload
