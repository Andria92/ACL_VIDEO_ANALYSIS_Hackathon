"""Case-level human review of contact in the visible injury event."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from acl_motion.annotations.models import AnnotationCase
from acl_motion.persistence import atomic_write_json, path_lock

INJURY_MECHANISM_REVIEW_VERSION = "human_injury_mechanism_review_v1"
INJURY_MECHANISM_REVIEW_QUESTION = (
    "Was this injury event contact, non-contact, or indirect contact?"
)
INJURY_MECHANISM_OPTIONS = {
    "direct_contact": {
        "label": "Contact",
        "definition": "Force was applied directly to the injured knee.",
    },
    "non_contact": {
        "label": "Non-contact",
        "definition": "No external contact was mechanically linked to the injury action.",
    },
    "indirect_contact": {
        "label": "Indirect contact",
        "definition": (
            "Contact away from the injured knee remained mechanically linked to the "
            "injury action."
        ),
    },
}


def injury_mechanism_review_path(data_root: str | Path, case_id: str) -> Path:
    """Return the shared case-level injury-mechanism review path."""

    safe_case_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(case_id)
    ).strip("_")
    if not safe_case_id:
        raise ValueError("A case identifier is required for injury-mechanism review.")
    return (
        Path(data_root)
        / "annotations"
        / "human"
        / f"{safe_case_id}_injury_mechanism_review_human.json"
    )


def load_injury_mechanism_review(
    case: AnnotationCase,
    *,
    data_root: str | Path = "data",
) -> dict:
    """Load the case-level decision, using existing research only as a prompt."""

    root = Path(data_root)
    path = injury_mechanism_review_path(root, case.case_id)
    base = _base_payload(case, path)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {**base, "review_status": "REVIEW_FILE_INVALID"}
        decision = _normalize_decision(payload.get("decision"))
        if decision is None:
            return {**base, "review_status": "REVIEW_FILE_INVALID"}
        return {
            **base,
            **payload,
            "question": INJURY_MECHANISM_REVIEW_QUESTION,
            "options": INJURY_MECHANISM_OPTIONS,
            "decision": decision,
            "decision_label": INJURY_MECHANISM_OPTIONS[decision]["label"],
            "review_status": "REVIEWED",
            "decision_source": "human_operator",
            "review_required": False,
            "path": path.name,
        }

    decision, source = _existing_research_decision(root, case.case_id)
    if decision is None:
        return base
    return {
        **base,
        "decision": decision,
        "decision_label": INJURY_MECHANISM_OPTIONS[decision]["label"],
        "review_status": "EXISTING_RESEARCH_LABEL",
        "decision_source": source,
        "review_required": True,
    }


def save_injury_mechanism_review(
    case: AnnotationCase,
    *,
    decision: str,
    reviewer_id: str = "researcher_01",
    data_root: str | Path = "data",
) -> dict:
    """Persist one of the three supported contact-mechanism decisions."""

    normalized = _normalize_decision(decision)
    if normalized is None:
        raise ValueError(
            "The injury mechanism must be contact, non-contact, or indirect contact."
        )

    path = injury_mechanism_review_path(data_root, case.case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": INJURY_MECHANISM_REVIEW_VERSION,
        "question": INJURY_MECHANISM_REVIEW_QUESTION,
        "case_id": case.case_id,
        "decision": normalized,
        "decision_label": INJURY_MECHANISM_OPTIONS[normalized]["label"],
        "review_status": "REVIEWED",
        "decision_source": "human_operator",
        "review_required": False,
        "reviewer_id": str(reviewer_id).strip() or "researcher_01",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "reviewed_from_source_id": case.source_id,
        "reviewed_from_view_id": case.view_id or case.source_id,
        "shared_across_case_views": True,
        "options": INJURY_MECHANISM_OPTIONS,
    }
    with path_lock(path):
        atomic_write_json(path, payload, trailing_newline=True)
    return {**payload, "path": path.name}


def _base_payload(case: AnnotationCase, path: Path) -> dict:
    return {
        "schema_version": INJURY_MECHANISM_REVIEW_VERSION,
        "question": INJURY_MECHANISM_REVIEW_QUESTION,
        "case_id": case.case_id,
        "decision": None,
        "decision_label": None,
        "review_status": "NOT_REVIEWED",
        "decision_source": None,
        "review_required": True,
        "reviewer_id": None,
        "reviewed_at": None,
        "shared_across_case_views": True,
        "options": INJURY_MECHANISM_OPTIONS,
        "path": path.name,
    }


def _existing_research_decision(root: Path, case_id: str) -> tuple[str | None, str | None]:
    sources_path = root / "annotations" / "human" / "injury_report_sources.json"
    sources = _read_json(sources_path).get("cases", {})
    if isinstance(sources, dict):
        record = sources.get(str(case_id), {})
        if isinstance(record, dict):
            decision = _normalize_decision(record.get("classification"))
            if decision is not None:
                return decision, "existing_injury_report_review"

    metadata_path = (
        root / "annotations" / "human" / "case_research_metadata_human.json"
    )
    metadata = _read_json(metadata_path).get("cases", {})
    if isinstance(metadata, dict):
        record = metadata.get(str(case_id), {})
        if isinstance(record, dict):
            decision = _normalize_decision(record.get("contact_mechanism"))
            if decision is not None:
                return decision, "existing_case_research_metadata"
    return None, None


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_decision(value: object) -> str | None:
    aliases = {
        "contact": "direct_contact",
        "direct_contact": "direct_contact",
        "direct-contact": "direct_contact",
        "non_contact": "non_contact",
        "non-contact": "non_contact",
        "indirect_contact": "indirect_contact",
        "indirect-contact": "indirect_contact",
    }
    return aliases.get(str(value or "").strip().lower())
