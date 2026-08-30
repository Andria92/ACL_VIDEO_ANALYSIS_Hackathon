"""Human review of whether phase evidence covers the intended visible event."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from acl_motion.annotations.models import AnnotationCase

EVENT_INTERVAL_REVIEW_VERSION = "human_event_interval_review_v1"
EVENT_INTERVAL_REVIEW_QUESTION = (
    "Does the supported phase interval include the visible event you intended to study?"
)
SUPPORTED_PHASE_STATUSES = {
    "SUPPORTED",
    "SUPPORTED_PARTIAL_WINDOW",
    "SUPPORTED_EVIDENCE_INTERVAL",
}


def event_interval_review_path(data_root: str | Path, slug: str) -> Path:
    """Return the per-view human event-coverage decision path."""

    return (
        Path(data_root)
        / "annotations"
        / "human"
        / f"{slug}_event_interval_review_human.json"
    )


def load_event_interval_review(
    case: AnnotationCase,
    *,
    data_root: str | Path = "data",
) -> dict:
    """Load one current Yes/No review, defaulting new supported views to Yes."""

    root = Path(data_root)
    path = event_interval_review_path(root, case.slug)
    current_phase_hash = _phase_artifact_hash(root, case.slug)
    default = _default_yes_review(case, path)
    if not path.exists():
        return default

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {
            **default,
            "decision": None,
            "visible_event_in_supported_phase_interval": None,
            "eligible_for_injury_event_comparison": False,
            "review_status": "REVIEW_FILE_INVALID",
        }

    decision = str(payload.get("decision", "")).lower()
    if decision not in {"yes", "no"}:
        return {
            **default,
            "decision": None,
            "visible_event_in_supported_phase_interval": None,
            "eligible_for_injury_event_comparison": False,
            "review_status": "REVIEW_FILE_INVALID",
        }
    if not current_phase_hash or payload.get("movement_phases_sha256") != current_phase_hash:
        return {
            **default,
            "decision": None,
            "visible_event_in_supported_phase_interval": None,
            "eligible_for_injury_event_comparison": False,
            "review_status": "REVIEW_REQUIRED_AFTER_REGENERATION",
            "previous_decision": decision,
        }

    return {
        **payload,
        "question": EVENT_INTERVAL_REVIEW_QUESTION,
        "decision": decision,
        "visible_event_in_supported_phase_interval": decision == "yes",
        "eligible_for_injury_event_comparison": decision == "yes",
        "review_status": "REVIEWED",
        "decision_source": "human_operator",
        "path": path.name,
    }


def save_event_interval_review(
    case: AnnotationCase,
    *,
    decision: str,
    reviewer_id: str = "researcher_01",
    data_root: str | Path = "data",
) -> dict:
    """Persist an operator-supplied Yes/No event-coverage decision for one view."""

    answer = str(decision).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError("The visible-event review decision must be yes or no.")

    root = Path(data_root)
    phase_path = _phase_artifact_path(root, case.slug)
    if not phase_path.exists():
        raise ValueError("Generate the movement phase analysis before reviewing event coverage.")
    movement_story = json.loads(phase_path.read_text(encoding="utf-8"))
    phase_status = str(movement_story.get("status", ""))
    phases = list(movement_story.get("phases") or [])
    if phase_status not in SUPPORTED_PHASE_STATUSES or not phases:
        raise ValueError("No supported phase interval is available for this review.")

    path = event_interval_review_path(root, case.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EVENT_INTERVAL_REVIEW_VERSION,
        "question": EVENT_INTERVAL_REVIEW_QUESTION,
        "case_id": case.case_id,
        "source_id": case.source_id,
        "view_id": case.view_id or case.source_id,
        "case_slug": case.slug,
        "decision": answer,
        "visible_event_in_supported_phase_interval": answer == "yes",
        "eligible_for_injury_event_comparison": answer == "yes",
        "review_status": "REVIEWED",
        "decision_source": "human_operator",
        "reviewer_id": str(reviewer_id).strip() or "researcher_01",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "phase_status": phase_status,
        "phase_count": len(phases),
        "supported_interval": _supported_interval(movement_story, phases),
        "movement_phases_sha256": _sha256(phase_path),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return {**payload, "path": path.name}


def _default_yes_review(case: AnnotationCase, path: Path) -> dict:
    return {
        "schema_version": EVENT_INTERVAL_REVIEW_VERSION,
        "question": EVENT_INTERVAL_REVIEW_QUESTION,
        "case_id": case.case_id,
        "source_id": case.source_id,
        "view_id": case.view_id or case.source_id,
        "case_slug": case.slug,
        "decision": "yes",
        "visible_event_in_supported_phase_interval": True,
        "eligible_for_injury_event_comparison": True,
        "review_status": "DEFAULT_YES",
        "decision_source": "default_policy",
        "reviewer_id": None,
        "reviewed_at": None,
        "path": path.name,
    }


def _supported_interval(movement_story: dict, phases: list[dict]) -> dict:
    scope = movement_story.get("metadata", {}).get("analysis_scope", {}) or {}
    starts = [int(phase["start_frame"]) for phase in phases if "start_frame" in phase]
    ends = [int(phase["end_frame"]) for phase in phases if "end_frame" in phase]
    return {
        "start_frame": scope.get("start_frame", min(starts) if starts else None),
        "end_frame": scope.get("end_frame", max(ends) if ends else None),
    }


def _phase_artifact_path(root: Path, slug: str) -> Path:
    return root / "phases" / "human" / f"{slug}_movement_phases.json"


def _phase_artifact_hash(root: Path, slug: str) -> str | None:
    path = _phase_artifact_path(root, slug)
    return _sha256(path) if path.exists() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
