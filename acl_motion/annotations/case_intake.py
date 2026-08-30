"""Persistent intake for grouping multiple video views under one injury case."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from acl_motion.annotations.models import AnnotationCase
from acl_motion.annotations.research_metadata import case_details, save_case_details
from acl_motion.cases.models import InjurySide


def injury_case_options(
    cases: Iterable[AnnotationCase],
    *,
    research_metadata_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Return one UI option per injury event, including shared human metadata."""

    grouped: dict[str, list[AnnotationCase]] = {}
    for case in cases:
        grouped.setdefault(case.case_id, []).append(case)

    options = []
    for case_id, views in grouped.items():
        ordered = sorted(
            views,
            key=lambda item: (0 if item.primary_view else 1, item.view_label.casefold(), item.slug),
        )
        representative = ordered[0]
        details = case_details(
            research_metadata_path,
            case_id,
            fallback_player_name=representative.player_name,
        )
        options.append(
            {
                "case_id": case_id,
                "player_name": details["player_name"] or representative.player_name,
                "injury_date": details["injury_date"],
                "team": details["team"],
                "opponent": details["opponent"],
                "competition": details["competition"],
                "position_group": details["position_group"],
                "match_minute": details["match_minute"],
                "injured_side": InjurySide(representative.injured_side).value,
                "view_count": len(views),
            }
        )
    return tuple(
        sorted(
            options,
            key=lambda item: (
                str(item["player_name"]).casefold(),
                str(item["injury_date"]),
                str(item["case_id"]),
            ),
        )
    )


def register_analysis_clip(
    payload: Mapping[str, Any],
    *,
    video_path: str | Path,
    cases: Iterable[AnnotationCase],
    imported_cases_path: str | Path,
    research_metadata_path: str | Path,
) -> tuple[AnnotationCase, dict[str, str]]:
    """Register a cut as a new view of a new or existing injury case."""

    clip_path = Path(video_path).resolve()
    if not clip_path.exists() or not clip_path.is_file():
        raise ValueError("The cut video is unavailable and cannot be assigned.")

    registered_cases = tuple(cases)
    if any(item.video_path.resolve() == clip_path for item in registered_cases):
        raise ValueError("This cut is already assigned to an injury case.")

    mode = str(payload.get("assignment_mode", "existing")).strip().lower()
    if mode not in {"new", "existing"}:
        raise ValueError("Assignment mode must be new or existing.")

    if mode == "existing":
        case_id = str(payload.get("case_id", "")).strip()
        siblings = tuple(item for item in registered_cases if item.case_id == case_id)
        if not siblings:
            raise ValueError("Choose a registered injury case for this video view.")
        representative = min(
            siblings,
            key=lambda item: (0 if item.primary_view else 1, item.slug),
        )
        player_name = representative.player_name
        injured_side = InjurySide(representative.injured_side)
        details = case_details(
            research_metadata_path,
            case_id,
            fallback_player_name=player_name,
        )
        player_name = details["player_name"] or player_name
    else:
        player_name = str(payload.get("player_name", "")).strip()
        if not player_name:
            raise ValueError("Player name is required when creating an injury case.")
        injury_date = _required_iso_date(payload.get("injury_date"))
        duplicate_case_id = _matching_injury_case_id(
            player_name,
            injury_date,
            registered_cases,
            research_metadata_path=research_metadata_path,
        )
        if duplicate_case_id is not None:
            raise ValueError(
                "An injury case already exists for this player and date. "
                f"Add this video to the existing case ({duplicate_case_id})."
            )
        case_id = _unique_case_id(
            f"imported_{_safe_identifier(player_name)}_{injury_date.replace('-', '_')}_acl_candidate",
            registered_cases,
        )
        try:
            injured_side = InjurySide(str(payload.get("injured_side", "unknown")).lower())
        except ValueError as exc:
            raise ValueError("Injured knee must be left, right, or unknown.") from exc
        details = save_case_details(
            research_metadata_path,
            case_id,
            {
                "player_name": player_name,
                "injury_date": injury_date,
                "team": payload.get("team", ""),
                "opponent": payload.get("opponent", ""),
                "competition": payload.get("competition", ""),
                "position_group": payload.get("position_group", "unknown"),
                "match_minute": payload.get("match_minute", ""),
                "date_of_birth": "",
            },
            annotator_id=str(payload.get("created_by", "researcher_01")),
            fallback_player_name=player_name,
        )
        siblings = ()

    view_number = _next_view_number(case_id, registered_cases)
    slug_base = (
        f"imported_{_safe_identifier(player_name)}_"
        f"{str(details.get('injury_date', '')).replace('-', '_') or 'undated'}_view_{view_number:02d}"
    )
    slug = _unique_slug(slug_base, registered_cases)
    source_id = slug
    case = AnnotationCase(
        slug=slug,
        case_id=case_id,
        source_id=source_id,
        view_id=source_id,
        view_label=str(payload.get("view_label", "")).strip()
        or ("Primary video view" if not siblings else f"Additional video view {view_number}"),
        primary_view=not siblings,
        perspective=str(payload.get("perspective", "unknown")).strip() or "unknown",
        occlusion_level="unknown",
        view_quality="human_annotation_pending",
        slow_motion=_as_bool(payload.get("slow_motion", False)),
        cropped_or_zoomed=_as_bool(payload.get("cropped_or_zoomed", False)),
        injured_side=injured_side,
        injury_laterality_source=(
            f"human_operator_case_intake:{str(payload.get('created_by', 'researcher_01')).strip()}"
            if injured_side is not InjurySide.UNKNOWN
            else ""
        ),
        player_name=player_name,
        video_path=clip_path,
        notes="Cut in Video Cutter and assigned to an injury case before annotation.",
    )
    _append_case_record(
        imported_cases_path,
        case,
        source_video_path=str(payload.get("source_video_path", "")).strip(),
        clip_start_seconds=payload.get("clip_start_seconds"),
        clip_end_seconds=payload.get("clip_end_seconds"),
    )
    return case, details


def _append_case_record(
    path: str | Path,
    case: AnnotationCase,
    *,
    source_video_path: str,
    clip_start_seconds: Any,
    clip_end_seconds: Any,
) -> Path:
    registry_path = Path(path)
    records: list[dict[str, Any]] = []
    if registry_path.exists():
        try:
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            records = [dict(item) for item in loaded.get("cases", ()) if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            records = []
    if any(str(item.get("slug")) == case.slug for item in records):
        raise ValueError("A video view with this identifier is already registered.")
    record = {
        "slug": case.slug,
        "case_id": case.case_id,
        "source_id": case.source_id,
        "view_id": case.view_id or case.source_id,
        "view_label": case.view_label,
        "primary_view": case.primary_view,
        "perspective": case.perspective,
        "occlusion_level": case.occlusion_level,
        "view_quality": case.view_quality,
        "slow_motion": case.slow_motion,
        "cropped_or_zoomed": case.cropped_or_zoomed,
        "real_time_scale": case.real_time_scale,
        "injured_side": InjurySide(case.injured_side).value,
        "injury_laterality_source": case.injury_laterality_source,
        "player_name": case.player_name,
        "video_path": str(case.video_path),
        "notes": case.notes,
        "source_video_path": source_video_path,
        "clip_start_seconds": _optional_float(clip_start_seconds),
        "clip_end_seconds": _optional_float(clip_end_seconds),
    }
    records.append(record)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"cases": records}, indent=2), encoding="utf-8")
    return registry_path


def _required_iso_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Match / injury date is required when creating an injury case.")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError("Match / injury date must use YYYY-MM-DD format.") from exc


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")[:64] or "case"


def _unique_case_id(base: str, cases: Iterable[AnnotationCase]) -> str:
    existing = {case.case_id for case in cases}
    if base not in existing:
        return base
    for suffix in range(2, 1000):
        candidate = f"{base}_{suffix}"
        if candidate not in existing:
            return candidate
    raise ValueError("Could not create a unique injury case identifier.")


def _matching_injury_case_id(
    player_name: str,
    injury_date: str,
    cases: Iterable[AnnotationCase],
    *,
    research_metadata_path: str | Path,
) -> str | None:
    """Return an existing player-and-date case instead of creating a duplicate event."""

    expected_player = _safe_identifier(player_name)
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        details = case_details(
            research_metadata_path,
            case.case_id,
            fallback_player_name=case.player_name,
        )
        existing_player = _safe_identifier(details.get("player_name") or case.player_name)
        existing_date = str(details.get("injury_date", "")).strip()
        if existing_player == expected_player and existing_date == injury_date:
            return case.case_id
    return None


def _unique_slug(base: str, cases: Iterable[AnnotationCase]) -> str:
    existing = {case.slug for case in cases}
    if base not in existing:
        return base
    for suffix in range(2, 1000):
        candidate = f"{base}_{suffix}"
        if candidate not in existing:
            return candidate
    raise ValueError("Could not create a unique video view identifier.")


def _next_view_number(case_id: str, cases: Iterable[AnnotationCase]) -> int:
    return sum(1 for case in cases if case.case_id == case_id) + 1


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)
