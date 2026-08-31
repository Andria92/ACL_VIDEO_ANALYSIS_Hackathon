"""Evidence-gated cross-case exploratory data preparation.

The exploration layer treats one registered ACL event as one statistical unit.
Multiple camera views are never counted as independent observations. Instead, the
strongest available evidence view is selected separately for each feature.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from acl_motion.analytics.similarity import similarity_readiness
from acl_motion.annotations.event_interval_review import (
    SUPPORTED_PHASE_STATUSES,
    load_event_interval_review,
)
from acl_motion.annotations.models import AnnotationCase
from acl_motion.annotations.research_metadata import (
    RESEARCH_METADATA_FILENAME,
    load_research_metadata,
)

SUMMARY_SUFFIX = "_case_feature_summary.parquet"
EXPLORATION_VERSION = "cross_case_descriptive_eda_v3_harmonized_case_metadata"
DEFAULT_SUMMARY_DIR = Path("data/analytics/human")
DEFAULT_RESEARCH_METADATA_PATH = Path("data/annotations/human") / RESEARCH_METADATA_FILENAME
DEFAULT_INJURY_REPORTS_PATH = Path("data/annotations/human/injury_report_sources.json")
DEFAULT_SIGNATURE_DIR = Path("data/signatures/human")
DEFAULT_SEMANTICS_DIR = Path("data/semantics/human")
DEFAULT_DATA_ROOT = Path("data")
HOME_SUMMARY_CACHE_VERSION = "exploration_home_summary_cache_v1"
_HOME_SUMMARY_CACHE_LOCK = Lock()

CONTACT_MECHANISMS = (
    "non_contact",
    "indirect_contact",
    "direct_contact",
    "unclear",
    "uncertain",
    "unknown",
)

UNRESOLVED_CONTACT_MECHANISMS = frozenset({"unclear", "uncertain", "unknown"})

SUMMARY_STATISTICS = (
    "mean",
    "range",
    "pre_late_change",
    "geometry_completeness",
    "dynamic_completeness",
)

FEATURE_PLAIN_LANGUAGE = {
    "contralateral_elbow_angle_2d_deg": (
        "The angle at the elbow on the same side as the knee that was not injured. "
        "A larger angle means the arm looks straighter in the video."
    ),
    "contralateral_hka_angle_2d_deg": (
        "The angle made by the hip, knee, and ankle on the leg that was not injured. "
        "A larger angle means that three-point chain looks straighter in the video."
    ),
    "elbow_projected_bilateral_absolute_difference_deg": (
        "The size of the difference between the two elbow angles. It shows how far "
        "apart the arms look, without saying which arm has the larger angle."
    ),
    "elbow_projected_bilateral_difference_deg": (
        "The elbow angle on the injured-knee side minus the elbow angle on the other "
        "side. A positive value means the injured-knee side has the larger angle."
    ),
    "hka_projected_bilateral_absolute_difference_deg": (
        "The size of the difference between the injured and opposite hip-knee-ankle "
        "angles. It ignores which leg has the larger angle."
    ),
    "hka_projected_bilateral_difference_deg": (
        "The injured-leg hip-knee-ankle angle minus the opposite-leg angle. A positive "
        "value means the injured leg has the larger angle."
    ),
    "injured_elbow_angle_2d_deg": (
        "The angle at the elbow on the same side as the injured knee. A larger angle "
        "means the arm looks straighter in the video."
    ),
    "injured_hka_angle_2d_deg": (
        "The angle made by the hip, knee, and ankle on the injured leg. A larger angle "
        "means that three-point chain looks straighter in the video."
    ),
    "knee_line_deviation_bilateral_absolute_difference": (
        "The size of the difference between the two knees' distances from their "
        "hip-to-ankle lines, measured in image pixels."
    ),
    "knee_line_deviation_bilateral_difference": (
        "The injured knee's line deviation minus the opposite knee's deviation, in "
        "image pixels. The sign depends on the two-dimensional image direction."
    ),
    "knee_line_deviation_normalized_bilateral_absolute_difference": (
        "The size of the difference between the two knee-line deviations after "
        "adjusting for the player's visible body size."
    ),
    "knee_line_deviation_normalized_bilateral_difference": (
        "The injured knee's line deviation minus the opposite knee's deviation after "
        "adjusting for visible body size."
    ),
    "left_elbow_angle_2d_deg": (
        "The angle made by the left shoulder, elbow, and wrist. A larger angle means "
        "the left arm looks straighter in the video."
    ),
    "left_hka_angle_2d_deg": (
        "The angle made by the left hip, knee, and ankle. A larger angle means the "
        "left-leg chain looks straighter in the video."
    ),
    "left_knee_ankle_distance_normalized": (
        "The visible distance between the left knee and ankle, adjusted for the "
        "player's body size. Camera angle can change this value."
    ),
    "left_knee_ankle_x_offset_normalized": (
        "How far left or right the left knee appears from the left ankle, adjusted for "
        "the player's visible body size."
    ),
    "left_knee_line_deviation_2d": (
        "How far the left knee appears from the straight line joining the left hip and "
        "ankle, measured in image pixels."
    ),
    "left_knee_line_deviation_normalized": (
        "How far the left knee appears from the left hip-to-ankle line, adjusted for "
        "the player's visible body size."
    ),
    "left_upper_arm_orientation_2d_deg": (
        "The direction of the line from the left shoulder to the left elbow in the "
        "video image."
    ),
    "left_wrist_pelvis_distance_normalized": (
        "The visible straight-line distance from the left wrist to the centre of the "
        "hips, adjusted for body size."
    ),
    "left_wrist_pelvis_x_offset_normalized": (
        "How far left or right the left wrist appears from the centre of the hips, "
        "adjusted for body size."
    ),
    "projected_hip_line_angle_deg": (
        "The tilt of the line joining the left and right hips in the video image. It is "
        "not a three-dimensional measure of pelvic rotation."
    ),
    "projected_shoulder_line_angle_deg": (
        "The tilt of the line joining the left and right shoulders in the video image. "
        "It is not a three-dimensional measure of torso rotation."
    ),
    "projected_shoulder_pelvis_orientation_difference_deg": (
        "The difference between the shoulder-line tilt and the hip-line tilt. It shows "
        "how differently those two lines are oriented in the image."
    ),
    "projected_shoulder_pelvis_x_offset_normalized": (
        "How far left or right the centre of the shoulders appears from the centre of "
        "the hips, adjusted for body size."
    ),
    "projected_shoulder_pelvis_x_offset_px": (
        "How far left or right the centre of the shoulders appears from the centre of "
        "the hips, measured in image pixels."
    ),
    "projected_trunk_axis_angle_deg": (
        "The direction of the line from the centre of the hips to the centre of the "
        "shoulders in the video image."
    ),
    "right_elbow_angle_2d_deg": (
        "The angle made by the right shoulder, elbow, and wrist. A larger angle means "
        "the right arm looks straighter in the video."
    ),
    "right_hka_angle_2d_deg": (
        "The angle made by the right hip, knee, and ankle. A larger angle means the "
        "right-leg chain looks straighter in the video."
    ),
    "right_knee_ankle_distance_normalized": (
        "The visible distance between the right knee and ankle, adjusted for the "
        "player's body size. Camera angle can change this value."
    ),
    "right_knee_ankle_x_offset_normalized": (
        "How far left or right the right knee appears from the right ankle, adjusted "
        "for the player's visible body size."
    ),
    "right_knee_line_deviation_2d": (
        "How far the right knee appears from the straight line joining the right hip "
        "and ankle, measured in image pixels."
    ),
    "right_knee_line_deviation_normalized": (
        "How far the right knee appears from the right hip-to-ankle line, adjusted for "
        "the player's visible body size."
    ),
    "right_upper_arm_orientation_2d_deg": (
        "The direction of the line from the right shoulder to the right elbow in the "
        "video image."
    ),
    "right_wrist_pelvis_distance_normalized": (
        "The visible straight-line distance from the right wrist to the centre of the "
        "hips, adjusted for body size."
    ),
    "right_wrist_pelvis_x_offset_normalized": (
        "How far left or right the right wrist appears from the centre of the hips, "
        "adjusted for body size."
    ),
}


def load_exploration_payload(
    cases: Iterable[AnnotationCase],
    *,
    summary_dir: str | Path = DEFAULT_SUMMARY_DIR,
    research_metadata_path: str | Path = DEFAULT_RESEARCH_METADATA_PATH,
    injury_reports_path: str | Path = DEFAULT_INJURY_REPORTS_PATH,
    signature_dir: str | Path = DEFAULT_SIGNATURE_DIR,
    semantics_dir: str | Path = DEFAULT_SEMANTICS_DIR,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Build a JSON-safe descriptive EDA payload from completed case summaries."""

    case_list = tuple(cases)
    metadata = load_research_metadata(Path(research_metadata_path))
    injury_reports = _load_injury_reports(Path(injury_reports_path))
    all_rows = _load_summary_rows(
        Path(summary_dir),
        case_list,
        metadata,
        injury_reports.get("cases", {}),
    )
    all_rows = _attach_phase_evidence(all_rows, Path(semantics_dir))
    all_rows = _attach_event_interval_reviews(all_rows, case_list, Path(data_root))
    selected = _select_feature_evidence_views(all_rows)
    similarity_view_rows = all_rows.loc[
        all_rows["event_comparison_eligible"]
        & all_rows["phase_status"].isin(SUPPORTED_PHASE_STATUSES)
    ].copy()
    similarity_selected = _select_feature_evidence_views(similarity_view_rows)
    records = [_json_safe_record(row) for row in selected.to_dict(orient="records")]
    similarity_records = [
        _json_safe_record(row) for row in similarity_selected.to_dict(orient="records")
    ]
    similarity_view_records = [
        _json_safe_record(row) for row in similarity_view_rows.to_dict(orient="records")
    ]
    events = _event_rows(all_rows, selected)
    features = _feature_rows(selected)

    return {
        "exploration_version": EXPLORATION_VERSION,
        "analysis_unit": "registered_injury_case",
        "analysis_unit_note": (
            "One registered injury event is one statistical unit. Frames and multiple "
            "camera views are not counted as independent cases."
        ),
        "summary": _summary_counts(all_rows, selected, events),
        "events": events,
        "features": features,
        "records": records,
        "similarity_records": similarity_records,
        "similarity_view_records": similarity_view_records,
        "statistics": list(SUMMARY_STATISTICS),
        "contact_mechanisms": list(CONTACT_MECHANISMS),
        "mechanism_methodology": injury_reports.get("methodology", {}),
        "mechanism_review": {
            "reviewed_at": injury_reports.get("reviewed_at", ""),
            "review_standard": injury_reports.get("review_standard", ""),
        },
        "readiness": _analysis_readiness(events, selected),
        "similarity": _similarity_readiness(
            Path(signature_dir),
            similarity_records,
            events,
            similarity_view_records,
        ),
        "test_families": _test_families(),
        "provenance": {
            "summary_directory": _public_reference(Path(summary_dir)),
            "research_metadata_path": _public_reference(Path(research_metadata_path)),
            "injury_reports_path": _public_reference(Path(injury_reports_path)),
            "phase_evidence_directory": _public_reference(Path(semantics_dir)),
            "preferred_view_rule": (
                "Feature by feature: geometry eligibility, geometry coverage, dynamic "
                "eligibility, dynamic coverage, then primary-view status."
            ),
            "projected_angles_averaged_across_views": False,
            "similarity_view_rule": (
                "Only completed event-covered views are eligible. Each injury-to-injury "
                "comparison retains the eligible view pair with the highest primary-lens "
                "similarity; views are never averaged or counted as separate cases."
            ),
        },
    }


def load_exploration_summary_payload(
    cases: Iterable[AnnotationCase],
    *,
    summary_dir: str | Path = DEFAULT_SUMMARY_DIR,
    research_metadata_path: str | Path = DEFAULT_RESEARCH_METADATA_PATH,
    injury_reports_path: str | Path = DEFAULT_INJURY_REPORTS_PATH,
    signature_dir: str | Path = DEFAULT_SIGNATURE_DIR,
    semantics_dir: str | Path = DEFAULT_SEMANTICS_DIR,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Build only the home-page counts and evidence-gated similarity readiness."""

    case_list = tuple(cases)
    metadata = load_research_metadata(Path(research_metadata_path))
    injury_reports = _load_injury_reports(Path(injury_reports_path))
    all_rows = _load_summary_rows(
        Path(summary_dir),
        case_list,
        metadata,
        injury_reports.get("cases", {}),
    )
    all_rows = _attach_phase_evidence(all_rows, Path(semantics_dir))
    all_rows = _attach_event_interval_reviews(all_rows, case_list, Path(data_root))
    selected = _select_feature_evidence_views(all_rows)
    similarity_view_rows = all_rows.loc[
        all_rows["event_comparison_eligible"]
        & all_rows["phase_status"].isin(SUPPORTED_PHASE_STATUSES)
    ].copy()
    similarity_selected = _select_feature_evidence_views(similarity_view_rows)
    events = _event_rows(all_rows, selected)
    similarity_records = [
        _json_safe_record(row) for row in similarity_selected.to_dict(orient="records")
    ]
    similarity_view_records = [
        _json_safe_record(row) for row in similarity_view_rows.to_dict(orient="records")
    ]
    return {
        "summary": _summary_counts(all_rows, selected, events),
        "similarity": _similarity_readiness(
            Path(signature_dir),
            similarity_records,
            events,
            similarity_view_records,
        ),
    }


def load_cached_exploration_summary_payload(
    cases: Iterable[AnnotationCase],
    *,
    cache_path: str | Path,
    summary_dir: str | Path = DEFAULT_SUMMARY_DIR,
    research_metadata_path: str | Path = DEFAULT_RESEARCH_METADATA_PATH,
    injury_reports_path: str | Path = DEFAULT_INJURY_REPORTS_PATH,
    signature_dir: str | Path = DEFAULT_SIGNATURE_DIR,
    semantics_dir: str | Path = DEFAULT_SEMANTICS_DIR,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Return a fingerprinted compact cache, rebuilding only when evidence changes."""

    case_list = tuple(cases)
    destination = Path(cache_path)
    fingerprint = _home_summary_fingerprint(
        case_list,
        summary_dir=Path(summary_dir),
        research_metadata_path=Path(research_metadata_path),
        injury_reports_path=Path(injury_reports_path),
        signature_dir=Path(signature_dir),
        semantics_dir=Path(semantics_dir),
        data_root=Path(data_root),
    )
    with _HOME_SUMMARY_CACHE_LOCK:
        cached = _read_home_summary_cache(destination, fingerprint)
        if cached is not None:
            return cached
        payload = load_exploration_summary_payload(
            case_list,
            summary_dir=summary_dir,
            research_metadata_path=research_metadata_path,
            injury_reports_path=injury_reports_path,
            signature_dir=signature_dir,
            semantics_dir=semantics_dir,
            data_root=data_root,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "cache_version": HOME_SUMMARY_CACHE_VERSION,
                    "fingerprint": fingerprint,
                    "payload": payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return payload


def _read_home_summary_cache(path: Path, fingerprint: str) -> dict[str, Any] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        not isinstance(cached, dict)
        or cached.get("cache_version") != HOME_SUMMARY_CACHE_VERSION
        or cached.get("fingerprint") != fingerprint
        or not isinstance(cached.get("payload"), dict)
    ):
        return None
    payload = cached["payload"]
    if not isinstance(payload.get("summary"), dict) or not isinstance(
        payload.get("similarity"), dict
    ):
        return None
    return payload


def _home_summary_fingerprint(
    cases: tuple[AnnotationCase, ...],
    *,
    summary_dir: Path,
    research_metadata_path: Path,
    injury_reports_path: Path,
    signature_dir: Path,
    semantics_dir: Path,
    data_root: Path,
) -> str:
    """Fingerprint compact-summary inputs using identities and filesystem metadata."""

    parts = [HOME_SUMMARY_CACHE_VERSION]
    parts.extend(
        f"case:{case.slug}:{case.case_id}:{case.source_id}:{int(case.primary_view)}"
        for case in sorted(cases, key=lambda item: item.slug)
    )
    paths = {
        research_metadata_path,
        injury_reports_path,
        *summary_dir.glob(f"*{SUMMARY_SUFFIX}"),
        *signature_dir.glob("*_case_movement_signature_long.csv"),
        *semantics_dir.glob("*_observable_movement_descriptions.json"),
        *(data_root / "annotations" / "human").glob(
            "*_event_interval_review_human.json"
        ),
        *(data_root / "phases" / "human").glob("*_movement_phases.json"),
    }
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            stat = path.stat()
        except OSError:
            parts.append(f"missing:{path}")
            continue
        parts.append(f"file:{path}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _summary_counts(
    all_rows: pd.DataFrame,
    selected: pd.DataFrame,
    events: list[dict[str, Any]],
) -> dict[str, int]:
    """Return shared summary counts without materialising the full explorer response."""

    return {
        "analysed_case_count": (
            int(selected["statistical_unit_id"].nunique()) if not selected.empty else 0
        ),
        "analysed_view_count": (
            int(all_rows["source_id"].nunique()) if not all_rows.empty else 0
        ),
        "feature_count": (
            int(selected["feature_name"].nunique()) if not selected.empty else 0
        ),
        "known_injury_side_count": sum(
            event["injured_side"] in {"left", "right"} for event in events
        ),
        "known_contact_mechanism_count": sum(
            event["contact_mechanism"] not in UNRESOLVED_CONTACT_MECHANISMS
            for event in events
        ),
        "unclear_contact_mechanism_count": sum(
            event["contact_mechanism"] in UNRESOLVED_CONTACT_MECHANISMS
            for event in events
        ),
        "mechanism_source_count": sum(
            bool(event["mechanism_sources"]) for event in events
        ),
        "phase_supported_case_count": sum(
            bool(event["reference_pool_eligible"]) for event in events
        ),
    }


def _load_injury_reports(path: Path) -> dict[str, Any]:
    """Load the curated mechanism evidence register without failing the explorer."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"cases": {}}
    if not isinstance(payload, dict):
        return {"cases": {}}
    cases = payload.get("cases")
    payload["cases"] = cases if isinstance(cases, dict) else {}
    return payload


def _attach_phase_evidence(rows: pd.DataFrame, semantics_dir: Path) -> pd.DataFrame:
    """Attach source-level phase support without treating absent evidence as support."""

    if rows.empty:
        output = rows.copy()
        output["phase_status"] = pd.Series(dtype="object")
        return output
    phase_by_source: dict[str, str] = {}
    for path in sorted(semantics_dir.glob("*_observable_movement_descriptions.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        metadata = payload.get("metadata", {})
        source_id = str(metadata.get("source_id") or "")
        if not source_id:
            continue
        phase_by_source[source_id] = str(metadata.get("phase_status") or "UNKNOWN").upper()
    output = rows.copy()
    output["phase_status"] = output["source_id"].map(phase_by_source).fillna("UNKNOWN")
    return output


def _attach_event_interval_reviews(
    rows: pd.DataFrame,
    cases: tuple[AnnotationCase, ...],
    data_root: Path,
) -> pd.DataFrame:
    """Attach per-view event coverage without turning views into cases."""

    output = rows.copy()
    if output.empty:
        output["event_interval_review_decision"] = pd.Series(dtype="object")
        output["event_interval_review_status"] = pd.Series(dtype="object")
        output["event_comparison_eligible"] = pd.Series(dtype="bool")
        return output
    review_by_source = {
        case.source_id: load_event_interval_review(case, data_root=data_root)
        for case in cases
    }
    output["event_interval_review_decision"] = output["source_id"].map(
        {
            source_id: review.get("decision")
            for source_id, review in review_by_source.items()
        }
    )
    output["event_interval_review_status"] = output["source_id"].map(
        {
            source_id: review.get("review_status")
            for source_id, review in review_by_source.items()
        }
    ).fillna("VIEW_NOT_REGISTERED")
    output["event_comparison_eligible"] = output["source_id"].map(
        {
            source_id: bool(review.get("eligible_for_injury_event_comparison"))
            for source_id, review in review_by_source.items()
        }
    ).fillna(False).astype(bool)
    return output


def assess_group_test_eligibility(
    group_sizes: dict[str, int],
    *,
    minimum_per_group: int = 5,
) -> dict[str, Any]:
    """Return a conservative test recommendation from independent-case group counts."""

    usable = {str(name): int(size) for name, size in group_sizes.items() if int(size) > 0}
    if len(usable) < 2:
        return {
            "status": "INSUFFICIENT_GROUPS",
            "eligible": False,
            "reason": "At least two populated groups are required.",
            "recommended_tests": [],
        }
    smallest = min(usable.values())
    if smallest < minimum_per_group:
        return {
            "status": "INSUFFICIENT_INDEPENDENT_CASES",
            "eligible": False,
            "reason": (
                f"Each group needs at least {minimum_per_group} independent cases for this "
                f"exploratory gate; the smallest group contains {smallest}."
            ),
            "recommended_tests": [],
        }
    if len(usable) == 2:
        tests = ["Welch t-test", "Mann-Whitney U", "permutation test"]
    else:
        tests = ["Welch ANOVA", "Kruskal-Wallis", "permutation ANOVA"]
    return {
        "status": "EXPLORATORY_TEST_AVAILABLE",
        "eligible": True,
        "reason": (
            "The minimum independent-case count is met. Distribution, outlier, and "
            "missingness checks are still required before selecting a test."
        ),
        "recommended_tests": tests,
    }


def _load_summary_rows(
    summary_dir: Path,
    cases: tuple[AnnotationCase, ...],
    metadata: dict[str, dict[str, Any]],
    injury_reports: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    source_lookup = {case.source_id: case for case in cases}
    case_lookup: dict[str, AnnotationCase] = {}
    for case in cases:
        case_lookup.setdefault(case.case_id, case)
    frames: list[pd.DataFrame] = []
    for path in sorted(summary_dir.glob(f"*{SUMMARY_SUFFIX}")):
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        required = {"case_id", "source_id", "feature_name"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["summary_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    case_details = []
    for row in combined.itertuples(index=False):
        case = source_lookup.get(str(row.source_id)) or case_lookup.get(str(row.case_id))
        research = metadata.get(str(row.case_id), {})
        registered_injured_side = (
            str(case.injured_side.value) if case is not None else "unknown"
        )
        researched_injured_side = str(research.get("injured_side", "")).lower()
        injured_side = (
            researched_injured_side
            if researched_injured_side in {"left", "right"}
            else registered_injured_side
        )
        mechanism_evidence = injury_reports.get(str(row.case_id), {})
        mechanism_sources = mechanism_evidence.get("sources", [])
        if not isinstance(mechanism_sources, list):
            mechanism_sources = []
        primary_mechanism_source = (
            str(mechanism_sources[0].get("url", ""))
            if mechanism_sources and isinstance(mechanism_sources[0], dict)
            else ""
        )
        case_details.append(
            {
                "player_name": str(
                    mechanism_evidence.get(
                        "canonical_player_name",
                        research.get(
                            "player_name",
                            case.player_name if case else _label_from_identifier(row.case_id),
                        ),
                    )
                ),
                "view_label": case.view_label if case else _label_from_identifier(row.source_id),
                "case_slug": case.slug if case else "",
                "perspective": case.perspective if case else "unknown",
                "injured_side": injured_side,
                "primary_view": bool(case.primary_view) if case else False,
                "contact_mechanism": str(
                    mechanism_evidence.get(
                        "classification", research.get("contact_mechanism", "unknown")
                    )
                ),
                "contact_mechanism_source": str(
                    primary_mechanism_source
                    or research.get("contact_mechanism_source", "")
                ),
                "previous_contact_mechanism": mechanism_evidence.get(
                    "previous_classification"
                ),
                "mechanism_confidence": str(
                    mechanism_evidence.get("confidence", "not_reviewed")
                ),
                "mechanism_verification_status": str(
                    mechanism_evidence.get("verification_status", "not_reviewed")
                ),
                "mechanism_evidence_basis": str(
                    mechanism_evidence.get("evidence_basis", "")
                ),
                "mechanism_change_status": str(
                    mechanism_evidence.get("change_status", "not_reviewed")
                ),
                "mechanism_rationale": str(mechanism_evidence.get("rationale", "")),
                "mechanism_investigation_status": str(
                    mechanism_evidence.get("investigation_status", "complete")
                ),
                "mechanism_investigation_note": str(
                    mechanism_evidence.get("investigation_note", "")
                ),
                "mechanism_sources": mechanism_sources,
                "injury_date": str(research.get("injury_date", "")),
                "league": str(research.get("league", "")),
                "competition": str(research.get("competition", "")),
                "team": str(research.get("team", "")),
                "position_group": str(research.get("position_group", "unknown")),
                "match_minute": str(research.get("match_minute", "")),
                "date_of_birth": str(research.get("date_of_birth", "")),
                "age_at_injury": _age_at_injury(
                    research.get("date_of_birth"), research.get("injury_date")
                ),
                "age_group": _age_group(
                    _age_at_injury(
                        research.get("date_of_birth"), research.get("injury_date")
                    )
                ),
                "preferred_foot": str(research.get("preferred_foot", "unknown")),
                "preferred_foot_source": str(
                    research.get("preferred_foot_source", "")
                ),
                "preferred_foot_source_url": str(
                    research.get("preferred_foot_source_url", "")
                ),
                "preferred_foot_knee_injured": research.get(
                    "preferred_foot_knee_injured"
                ),
                "ea_fc_audit_status": str(
                    research.get("ea_fc_audit_status", "not_reviewed")
                ),
                "metadata_source": str(research.get("metadata_source", "")),
                "registered_case_id": str(row.case_id),
                "statistical_unit_id": str(
                    mechanism_evidence.get(
                        "statistical_unit_id",
                        research.get("statistical_unit_id", str(row.case_id)),
                    )
                ),
            }
        )
    details = pd.DataFrame(case_details, index=combined.index)
    combined = pd.concat([combined, details], axis=1)
    for column in SUMMARY_STATISTICS:
        if column not in combined:
            combined[column] = np.nan
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    for column in (
        "analytics_eligible",
        "geometry_analytics_eligible",
        "dynamic_analytics_eligible",
    ):
        if column not in combined:
            combined[column] = False
        combined[column] = combined[column].fillna(False).astype(bool)
    return combined


def _select_feature_evidence_views(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    ranked = rows.copy()
    ranked["_geometry_eligible"] = ranked["geometry_analytics_eligible"].astype(int)
    ranked["_dynamic_eligible"] = ranked["dynamic_analytics_eligible"].astype(int)
    ranked["_primary_view"] = ranked["primary_view"].astype(int)
    ranked = ranked.sort_values(
        [
            "statistical_unit_id",
            "feature_name",
            "_geometry_eligible",
            "geometry_completeness",
            "_dynamic_eligible",
            "dynamic_completeness",
            "_primary_view",
            "source_id",
        ],
        ascending=[True, True, False, False, False, False, False, True],
        na_position="last",
    )
    selected = ranked.drop_duplicates(
        ["statistical_unit_id", "feature_name"], keep="first"
    ).copy()
    view_counts = rows.groupby(["statistical_unit_id", "feature_name"])[
        "source_id"
    ].nunique().rename("view_count")
    selected = selected.join(view_counts, on=["statistical_unit_id", "feature_name"])
    return selected.drop(
        columns=["_geometry_eligible", "_dynamic_eligible", "_primary_view"],
        errors="ignore",
    ).reset_index(drop=True)


def _event_rows(rows: pd.DataFrame, selected: pd.DataFrame) -> list[dict[str, Any]]:
    if selected.empty:
        return []
    events = []
    for case_id, group in selected.groupby("statistical_unit_id", sort=True):
        first = group.iloc[0]
        source_rows = rows.loc[rows["statistical_unit_id"].eq(case_id)]
        phase_statuses = sorted(
            {str(value or "UNKNOWN").upper() for value in source_rows["phase_status"]}
        )
        supported_phase_views = int(
            source_rows.loc[
                source_rows["phase_status"].isin(SUPPORTED_PHASE_STATUSES),
                "source_id",
            ].nunique()
        )
        event_covered_views = int(
            source_rows.loc[
                source_rows["phase_status"].isin(SUPPORTED_PHASE_STATUSES)
                & source_rows["event_comparison_eligible"],
                "source_id",
            ].nunique()
        )
        reference_pool_eligible = event_covered_views > 0
        events.append(
            {
                "case_id": str(case_id),
                "player_name": str(first["player_name"]),
                "injured_side": str(first["injured_side"]),
                "contact_mechanism": str(first["contact_mechanism"]),
                "contact_mechanism_source": str(first["contact_mechanism_source"]),
                "previous_contact_mechanism": _optional_string(
                    first.get("previous_contact_mechanism")
                ),
                "mechanism_confidence": str(first["mechanism_confidence"]),
                "mechanism_verification_status": str(
                    first["mechanism_verification_status"]
                ),
                "mechanism_evidence_basis": str(first["mechanism_evidence_basis"]),
                "mechanism_change_status": str(first["mechanism_change_status"]),
                "mechanism_rationale": str(first["mechanism_rationale"]),
                "mechanism_investigation_status": str(
                    first["mechanism_investigation_status"]
                ),
                "mechanism_investigation_note": str(
                    first["mechanism_investigation_note"]
                ),
                "mechanism_sources": (
                    first["mechanism_sources"]
                    if isinstance(first["mechanism_sources"], list)
                    else []
                ),
                "injury_date": str(first["injury_date"]),
                "league": str(first["league"]),
                "competition": str(first["competition"]),
                "team": str(first["team"]),
                "position_group": str(first["position_group"]),
                "match_minute": str(first["match_minute"]),
                "date_of_birth": str(first["date_of_birth"]),
                "age_at_injury": _json_safe(first["age_at_injury"]),
                "age_group": str(first["age_group"]),
                "preferred_foot": str(first["preferred_foot"]),
                "preferred_foot_source": str(first["preferred_foot_source"]),
                "preferred_foot_source_url": str(
                    first["preferred_foot_source_url"]
                ),
                "preferred_foot_knee_injured": _json_safe(
                    first["preferred_foot_knee_injured"]
                ),
                "ea_fc_audit_status": str(first["ea_fc_audit_status"]),
                "metadata_source": str(first["metadata_source"]),
                "analysed_view_count": int(source_rows["source_id"].nunique()),
                "registered_case_ids": sorted(
                    {str(value) for value in source_rows["registered_case_id"]}
                ),
                "available_feature_count": int(group["mean"].notna().sum()),
                "geometry_eligible_feature_count": int(
                    group["geometry_analytics_eligible"].sum()
                ),
                "dynamic_eligible_feature_count": int(
                    group["dynamic_analytics_eligible"].sum()
                ),
                "unavailable_feature_count": int(group["mean"].isna().sum()),
                "median_geometry_completeness": _finite_median(
                    group["geometry_completeness"]
                ),
                "median_dynamic_completeness": _finite_median(
                    group["dynamic_completeness"]
                ),
                "feature_count": len(group),
                "phase_statuses": phase_statuses,
                "phase_supported_view_count": supported_phase_views,
                "event_covered_view_count": event_covered_views,
                "event_excluded_view_count": int(
                    source_rows.loc[
                        ~source_rows["event_comparison_eligible"], "source_id"
                    ].nunique()
                ),
                "reference_pool_eligible": reference_pool_eligible,
                "reference_pool_reason": (
                    "At least one completed view has supported phases and covers the visible event."
                    if reference_pool_eligible
                    else "No completed event-covered view is eligible for comparison."
                ),
            }
        )
    return events


def _optional_string(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    text = str(value).strip()
    return text or None


def _age_at_injury(date_of_birth: Any, injury_date: Any) -> int | None:
    """Return completed years at injury, or ``None`` for missing/invalid dates."""

    try:
        born = date.fromisoformat(str(date_of_birth or ""))
        injured = date.fromisoformat(str(injury_date or ""))
    except ValueError:
        return None
    return injured.year - born.year - ((injured.month, injured.day) < (born.month, born.day))


def _age_group(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age < 21:
        return "Under 21"
    if age < 26:
        return "21–25"
    if age < 31:
        return "26–30"
    return "31+"


def _feature_rows(selected: pd.DataFrame) -> list[dict[str, Any]]:
    if selected.empty:
        return []
    case_count = int(selected["statistical_unit_id"].nunique())
    features = []
    for feature_name, group in selected.groupby("feature_name", sort=True):
        first = group.iloc[0]
        valid_mean = group["mean"].notna() & group["geometry_analytics_eligible"]
        valid_dynamic = (
            group["pre_late_change"].notna() & group["dynamic_analytics_eligible"]
        )
        geometry = group["geometry_completeness"].dropna()
        dynamic = group["dynamic_completeness"].dropna()
        features.append(
            {
                "feature_name": str(feature_name),
                "label": _feature_label(str(feature_name)),
                "description": _feature_description(str(feature_name)),
                "body_region": str(first.get("body_region", "unknown")),
                "feature_family": str(first.get("feature_family", "unknown")),
                "supported_case_count": int(valid_mean.sum()),
                "available_case_count": int(group["mean"].notna().sum()),
                "unavailable_case_count": int(group["mean"].isna().sum()),
                "unsupported_case_count": int(
                    (group["mean"].notna() & ~group["geometry_analytics_eligible"]).sum()
                ),
                "dynamic_supported_case_count": int(valid_dynamic.sum()),
                "relevant_case_count": case_count,
                "case_coverage": float(valid_mean.sum() / case_count) if case_count else 0.0,
                "median_geometry_completeness": (
                    float(geometry.median()) if not geometry.empty else None
                ),
                "median_dynamic_completeness": (
                    float(dynamic.median()) if not dynamic.empty else None
                ),
            }
        )
    return features


def _similarity_readiness(
    signature_dir: Path,
    records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    view_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report on-demand similarity readiness and legacy signature coverage."""

    case_ids: set[str] = set()
    signature_files = sorted(signature_dir.glob("*_case_movement_signature_long.csv"))
    for path in signature_files:
        try:
            frame = pd.read_csv(path, usecols=["case_id"])
        except (OSError, ValueError):
            continue
        case_ids.update(str(value) for value in frame["case_id"].dropna().unique())

    readiness = similarity_readiness(records, events, view_records=view_records)
    return {
        **readiness,
        "signature_case_count": len(case_ids),
        "signature_file_count": len(signature_files),
    }


def _finite_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.median()) if not finite.empty else None


def _analysis_readiness(events: list[dict[str, Any]], selected: pd.DataFrame) -> dict[str, Any]:
    case_count = len(events)
    paired_feature_max = 0
    if not selected.empty:
        available = selected.loc[
            selected["geometry_analytics_eligible"] & selected["mean"].notna()
        ]
        supported_counts = available.groupby("feature_name")["statistical_unit_id"].nunique()
        if not supported_counts.empty:
            paired_feature_max = int(supported_counts.max())
    contact_counts: dict[str, int] = {}
    for event in events:
        mechanism = str(event["contact_mechanism"])
        if mechanism not in UNRESOLVED_CONTACT_MECHANISMS:
            contact_counts[mechanism] = contact_counts.get(mechanism, 0) + 1
    group_gate = assess_group_test_eligibility(contact_counts)
    correlation_ready = paired_feature_max >= 5
    return {
        "overall_status": "DESCRIPTIVE_ONLY",
        "descriptive_eda": {
            "eligible": case_count > 0,
            "status": "AVAILABLE" if case_count else "NO_ANALYSED_CASES",
        },
        "correlation": {
            "eligible": correlation_ready,
            "status": "EXPLORATORY_ONLY" if correlation_ready else "INSUFFICIENT_PAIRED_CASES",
            "maximum_supported_case_count": paired_feature_max,
            "reason": (
                "Descriptive rank correlations may be inspected, but they are not evidence of "
                "causation or ACL risk."
                if correlation_ready
                else "At least five mutually supported independent cases are required."
            ),
        },
        "contact_group_comparison": {
            **group_gate,
            "group_sizes": contact_counts,
        },
        "confirmatory_inference": {
            "eligible": False,
            "status": "NOT_AVAILABLE",
            "reason": (
                "The present case library is exploratory and no confirmatory analysis plan "
                "or prospective sample-size justification is registered."
            ),
        },
    }


def _test_families() -> list[dict[str, Any]]:
    return [
        {
            "question": "Two independent groups",
            "candidate_tests": ["Welch t-test", "Mann-Whitney U", "permutation test"],
            "required_output": ["effect size", "95% confidence interval", "p and q values"],
        },
        {
            "question": "Three or more independent groups",
            "candidate_tests": ["Welch ANOVA", "Kruskal-Wallis", "permutation ANOVA"],
            "required_output": ["effect size", "95% confidence interval", "p and q values"],
        },
        {
            "question": "Repeated phases within an injury case",
            "candidate_tests": ["paired analysis", "mixed-effects model"],
            "required_output": ["case-level clustering", "within-case dependence"],
        },
        {
            "question": "Relationship between two numeric descriptors",
            "candidate_tests": ["Spearman correlation", "Pearson correlation when justified"],
            "required_output": ["paired case count", "confidence interval", "scatter plot"],
        },
    ]


def _json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "case_id",
        "registered_case_id",
        "statistical_unit_id",
        "source_id",
        "view_id",
        "player_name",
        "view_label",
        "case_slug",
        "perspective",
        "primary_view",
        "injured_side",
        "contact_mechanism",
        "mechanism_confidence",
        "mechanism_verification_status",
        "mechanism_evidence_basis",
        "mechanism_change_status",
        "mechanism_investigation_status",
        "mechanism_investigation_note",
        "injury_date",
        "league",
        "competition",
        "team",
        "position_group",
        "match_minute",
        "date_of_birth",
        "age_at_injury",
        "age_group",
        "preferred_foot",
        "preferred_foot_source",
        "preferred_foot_source_url",
        "preferred_foot_knee_injured",
        "ea_fc_audit_status",
        "metadata_source",
        "feature_name",
        "body_region",
        "feature_family",
        "unit",
        "mean",
        "range",
        "range_semantics",
        "angular_statistics_version",
        "comparison_statistics_version",
        "comparison_support_scope",
        "profile_version",
        "pre_late_change",
        "geometry_completeness",
        "dynamic_completeness",
        "analytics_eligibility",
        "geometry_analytics_eligible",
        "dynamic_analytics_eligible",
        "quality_category",
        "primary_rejection_reason",
        "eligibility_reason",
        "summary_path",
        "view_count",
        "phase_status",
        "event_interval_review_decision",
        "event_interval_review_status",
        "event_comparison_eligible",
    }
    output = {key: _json_safe(value) for key, value in record.items() if key in keep}
    if output.get("summary_path"):
        output["summary_path"] = Path(str(output["summary_path"])).name
    return output


def _public_reference(path: Path) -> str:
    """Avoid exposing parent directories in browser-facing provenance."""

    return path.name if path.is_absolute() else path.as_posix()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return str(value)


def _feature_label(feature_name: str) -> str:
    tokens = feature_name.split("_")
    is_projected_2d = "2d" in tokens
    tokens = [token for token in tokens if token not in {"2d", "deg"}]
    replacements = {
        "hka": "HKA",
        "injured": "injured-side",
        "contralateral": "opposite-side",
    }
    words = [replacements.get(token, token) for token in tokens]
    label = " ".join(words)
    if is_projected_2d:
        label = f"{label} (2D)"
    return label[:1].upper() + label[1:]


def _feature_description(feature_name: str) -> str:
    return FEATURE_PLAIN_LANGUAGE.get(
        feature_name,
        (
            "A two-dimensional measurement taken from the player's visible movement "
            "in the video. Camera angle and landmark visibility can affect it."
        ),
    )


def _label_from_identifier(value: object) -> str:
    return str(value).replace("_", " ").strip().title() or "Unknown case"
