"""Human-supplied research metadata for registered ACL injury cases."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from acl_motion.persistence import atomic_write_json, path_lock

RESEARCH_METADATA_FILENAME = "case_research_metadata_human.json"
RESEARCH_METADATA_VERSION = "case_research_metadata_v3_harmonized_taxonomy"

POSITION_GROUPS = (
    "unknown",
    "goalkeeper",
    "defender",
    "midfielder",
    "forward",
)

CASE_DETAIL_FIELDS = (
    "player_name",
    "injury_date",
    "league",
    "competition",
    "team",
    "opponent",
    "position_group",
    "match_minute",
    "date_of_birth",
)


def research_metadata_path(output_dir: str | Path) -> Path:
    """Return the human research-metadata path for an annotation directory."""

    return Path(output_dir) / RESEARCH_METADATA_FILENAME


def load_research_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load case-keyed human metadata, tolerating absent or malformed files."""

    metadata_path = Path(path)
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cases = payload.get("cases", {})
    if isinstance(cases, list):
        return {
            str(item.get("case_id")): item
            for item in cases
            if isinstance(item, dict) and item.get("case_id")
        }
    if isinstance(cases, dict):
        return {
            str(case_id): value
            for case_id, value in cases.items()
            if isinstance(value, dict)
        }
    return {}


def case_details(
    path: str | Path,
    case_id: str,
    *,
    fallback_player_name: str = "",
) -> dict[str, str]:
    """Return normalized operator-editable details for one registered case."""

    record = load_research_metadata(path).get(str(case_id), {})
    return {
        "player_name": str(record.get("player_name") or fallback_player_name).strip(),
        "injury_date": str(record.get("injury_date", "")).strip(),
        "league": str(record.get("league", "")).strip(),
        "competition": str(record.get("competition", "")).strip(),
        "team": str(record.get("team", "")).strip(),
        "opponent": str(record.get("opponent", "")).strip(),
        "position_group": _normalize_position(record.get("position_group", "unknown")),
        "match_minute": str(record.get("match_minute", "")).strip(),
        "date_of_birth": str(record.get("date_of_birth", "")).strip(),
    }


def save_case_details(
    path: str | Path,
    case_id: str,
    details: Mapping[str, Any],
    *,
    annotator_id: str,
    fallback_player_name: str = "",
) -> dict[str, str]:
    """Merge operator-supplied case details while preserving other research fields."""

    metadata_path = Path(path)
    normalized = _normalize_case_details(
        details,
        fallback_player_name=fallback_player_name,
    )
    with path_lock(metadata_path):
        cases = load_research_metadata(metadata_path)
        existing = dict(cases.get(str(case_id), {}))
        source = f"human_operator_annotation_ui:{annotator_id.strip() or 'researcher_01'}"
        existing.update(normalized)
        existing["metadata_source"] = source
        existing["updated_at"] = datetime.now(UTC).isoformat()
        provenance = dict(existing.get("field_provenance", {}))
        for field in CASE_DETAIL_FIELDS:
            if normalized[field] and normalized[field] != "unknown":
                provenance[field] = source
        existing["field_provenance"] = provenance
        cases[str(case_id)] = existing
        atomic_write_json(
            metadata_path,
            {
                "metadata_version": RESEARCH_METADATA_VERSION,
                "cases": cases,
            },
        )
    return normalized


def delete_case_details(path: str | Path, case_id: str) -> None:
    """Atomically remove one case while preserving concurrently stored metadata."""

    metadata_path = Path(path)
    with path_lock(metadata_path):
        if not metadata_path.exists():
            return
        cases = load_research_metadata(metadata_path)
        if str(case_id) not in cases:
            return
        cases.pop(str(case_id), None)
        atomic_write_json(
            metadata_path,
            {
                "metadata_version": RESEARCH_METADATA_VERSION,
                "cases": cases,
            },
            trailing_newline=True,
        )


def _normalize_case_details(
    details: Mapping[str, Any],
    *,
    fallback_player_name: str,
) -> dict[str, str]:
    player_name = str(details.get("player_name") or fallback_player_name).strip()
    injury_date = _normalize_date(details.get("injury_date", ""), "Injury date")
    date_of_birth = _normalize_date(details.get("date_of_birth", ""), "Date of birth")
    match_minute = str(details.get("match_minute", "")).strip()
    if match_minute and not re.fullmatch(r"\d{1,3}(?:\+\d{1,2})?", match_minute):
        raise ValueError("Match minute must look like 67 or 45+2.")
    return {
        "player_name": player_name,
        "injury_date": injury_date,
        "league": str(details.get("league", "")).strip(),
        "competition": str(details.get("competition", "")).strip(),
        "team": str(details.get("team", "")).strip(),
        "opponent": str(details.get("opponent", "")).strip(),
        "position_group": _normalize_position(details.get("position_group", "unknown")),
        "match_minute": match_minute,
        "date_of_birth": date_of_birth,
    }


def _normalize_date(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format.") from exc
    return text


def _normalize_position(value: Any) -> str:
    position = str(value or "unknown").strip().lower()
    if position not in POSITION_GROUPS:
        raise ValueError(f"Position must be one of: {', '.join(POSITION_GROUPS)}.")
    return position
