"""Controlled observable movement vocabulary for user-facing descriptions.

The vocabulary converts supported projected measurements into deterministic,
plain-language movement descriptions. It does not create clustering features,
clinical labels, football-action labels, or archetype names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acl_motion.geometry.angles import wrapped_angle_difference_deg

VOCABULARY_VERSION = "m5_10_controlled_movement_vocabulary_v1"
DESCRIPTION_VERSION = "m5_10_observable_movement_description_v1"

FORBIDDEN_DESCRIPTION_PHRASES = (
    "acl mechanism",
    "acl loading",
    "dangerous movement",
    "risky movement",
    "injury-causing movement",
    "valgus collapse",
    "unstable landing",
    "bad landing",
    "poor technique",
    "abnormal asymmetry",
    "high-risk plant",
    "sudden turn",
    "sudden stop",
    "knee collapse",
    "knee valgus",
    "knee varus",
    "plant",
    "foot strike",
    "ground contact",
    "stance",
    "support foot",
    "lumbar rotation",
    "spinal rotation",
    "trunk collapse",
)

PATH_DESCRIPTOR_IDS = (
    "PROJECTED_MOVEMENT_CONTINUES",
    "PROJECTED_DIRECTION_CHANGE",
    "RAPID_PROJECTED_DIRECTION_CHANGE",
    "PROJECTED_SPEED_INCREASE",
    "PROJECTED_SLOWDOWN",
    "RAPID_PROJECTED_SLOWDOWN",
    "PROJECTED_STOP_OR_NEAR_STOP",
)

ORIENTATION_FEATURES = {
    "projected_trunk_axis_angle_deg",
    "projected_hip_line_angle_deg",
    "projected_shoulder_line_angle_deg",
    "projected_shoulder_pelvis_orientation_difference_deg",
    "right_upper_arm_orientation_2d_deg",
    "left_upper_arm_orientation_2d_deg",
}


class DescriptionEvidenceStatus(StrEnum):
    """Evidence state for observable movement descriptions."""

    SUPPORTED = "SUPPORTED"
    LIMITED = "LIMITED"
    WITHHELD = "WITHHELD"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class MovementDescriptorDefinition:
    """Registry definition for one allowed deterministic movement phrase."""

    descriptor_id: str
    family: str
    user_label: str
    description: str
    required_features: tuple[str, ...]
    required_evidence: str
    temporal_rule: str
    magnitude_rule: str
    view_requirements: tuple[str, ...]
    forbidden_when: tuple[str, ...]
    explanation_template: str
    version: str = VOCABULARY_VERSION
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready registry record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservableMovementDescription:
    """Evidence-backed user-facing movement description."""

    descriptor_id: str
    case_id: str
    scope_type: str
    scope_start: int | None
    scope_end: int | None
    family: str
    user_label: str
    summary: str
    evidence_status: DescriptionEvidenceStatus | str
    evidence_reason: str
    supporting_features: tuple[str, ...]
    supporting_values: dict[str, Any]
    change_magnitude: float | None
    duration_ms: float | None
    robust_rate: float | None
    source_frames: tuple[int, ...]
    view_requirements: tuple[str, ...]
    provenance: str
    visualisation_hint: str
    salience_score: float
    version: str = DESCRIPTION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_status",
            DescriptionEvidenceStatus(self.evidence_status),
        )
        text = (
            f"{self.descriptor_id} {self.user_label} {self.summary} "
            f"{self.evidence_reason} {self.provenance}"
        ).lower()
        if any(phrase in text for phrase in FORBIDDEN_DESCRIPTION_PHRASES):
            raise ValueError("Observable movement descriptions must remain non-clinical.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready description record."""

        payload = asdict(self)
        payload["evidence_status"] = self.evidence_status.value
        return payload


@dataclass(frozen=True, slots=True)
class MovementVocabularyConfig:
    """Configurable descriptive rules for engineering-threshold language."""

    minimum_interval_frames: int = 8
    minimum_supported_samples: int = 8
    minimum_feature_coverage: float = 0.55
    minimum_dynamic_supported_fraction: float = 0.35
    angle_change_deg: float = 10.0
    bilateral_change_deg: float = 6.0
    orientation_shift_deg: float = 15.0
    upper_body_shift_deg: float = 15.0
    rapid_rate_deg_per_s: float = 150.0
    rapid_min_consecutive_samples: int = 3
    max_default_descriptions: int = 4

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready config record."""

        return asdict(self)


def build_controlled_movement_vocabulary() -> tuple[MovementDescriptorDefinition, ...]:
    """Return the M5.10 controlled descriptor registry."""

    common_view = ("generic_projected_2d_view",)
    path_view = ("validated_projected_movement_path",)
    definitions = [
        _definition(
            "PROJECTED_MOVEMENT_CONTINUES",
            "MOVEMENT PATH",
            "Projected movement continued",
            ("path:compensated_x", "path:compensated_y"),
            path_view,
            "Path must pass QA and contain a continuous supported path segment.",
        ),
        _definition(
            "PROJECTED_DIRECTION_CHANGE",
            "MOVEMENT PATH",
            "Projected travel direction changed",
            ("path:projected_heading_deg",),
            path_view,
            "Validated heading evidence must change across a supported interval.",
        ),
        _definition(
            "RAPID_PROJECTED_DIRECTION_CHANGE",
            "MOVEMENT PATH",
            "Rapid projected direction change",
            ("path:projected_heading_deg",),
            path_view,
            "Validated heading change must be sustained and rate-supported.",
        ),
        _definition(
            "PROJECTED_SPEED_INCREASE",
            "MOVEMENT PATH",
            "Projected speed increased",
            ("path:normalized_projected_speed_per_s",),
            path_view,
            "Validated projected speed must increase across a supported interval.",
        ),
        _definition(
            "PROJECTED_SLOWDOWN",
            "MOVEMENT PATH",
            "Projected slowdown",
            ("path:normalized_projected_speed_per_s",),
            path_view,
            "Validated projected speed must decrease across a supported interval.",
        ),
        _definition(
            "RAPID_PROJECTED_SLOWDOWN",
            "MOVEMENT PATH",
            "Rapid projected slowdown",
            ("path:normalized_projected_speed_per_s",),
            path_view,
            "Projected slowdown must be sustained and rate-supported.",
        ),
        _definition(
            "PROJECTED_STOP_OR_NEAR_STOP",
            "MOVEMENT PATH",
            "Projected movement approached a low-speed state",
            ("path:normalized_projected_speed_per_s",),
            path_view,
            "Validated projected speed must remain low for a sufficient interval.",
        ),
        _definition(
            "RELATIVELY_STABLE_BODY_ORIENTATION",
            "WHOLE-BODY ORIENTATION",
            "Body orientation remained relatively stable",
            ("projected_trunk_axis_angle_deg", "projected_hip_line_angle_deg"),
            common_view,
            "Multiple orientation features must remain below the descriptive change rule.",
        ),
        _definition(
            "WHOLE_BODY_REORIENTATION",
            "WHOLE-BODY ORIENTATION",
            "Whole-body orientation shifted",
            ("projected_trunk_axis_angle_deg", "projected_hip_line_angle_deg"),
            common_view,
            "At least two projected body-orientation measures must shift.",
        ),
        _definition(
            "RAPID_WHOLE_BODY_REORIENTATION",
            "WHOLE-BODY ORIENTATION",
            "Rapid whole-body orientation shift",
            ("projected_trunk_axis_angle_deg", "projected_hip_line_angle_deg"),
            common_view,
            "Orientation shift must be sustained and rate-supported.",
        ),
        _definition(
            "INJURED_HKA_INCREASED",
            "HIP-KNEE-ANKLE CHAIN",
            "Injured-side projected HKA increased",
            ("injured_hka_angle_2d_deg",),
            common_view,
            "Supported injured-side HKA change must exceed the descriptive rule.",
        ),
        _definition(
            "INJURED_HKA_DECREASED",
            "HIP-KNEE-ANKLE CHAIN",
            "Injured-side projected HKA decreased",
            ("injured_hka_angle_2d_deg",),
            common_view,
            "Supported injured-side HKA change must exceed the descriptive rule.",
        ),
        _definition(
            "CONTRALATERAL_HKA_INCREASED",
            "HIP-KNEE-ANKLE CHAIN",
            "Contralateral projected HKA increased",
            ("contralateral_hka_angle_2d_deg",),
            common_view,
            "Supported contralateral HKA change must exceed the descriptive rule.",
        ),
        _definition(
            "CONTRALATERAL_HKA_DECREASED",
            "HIP-KNEE-ANKLE CHAIN",
            "Contralateral projected HKA decreased",
            ("contralateral_hka_angle_2d_deg",),
            common_view,
            "Supported contralateral HKA change must exceed the descriptive rule.",
        ),
        _definition(
            "LOWER_LIMB_CONFIGURATION_CHANGED",
            "HIP-KNEE-ANKLE CHAIN",
            "Lower-limb configuration changed",
            ("injured_hka_angle_2d_deg", "contralateral_hka_angle_2d_deg"),
            common_view,
            "At least one supported projected HKA chain must change.",
        ),
        _definition(
            "RAPID_LOWER_LIMB_CONFIGURATION_CHANGE",
            "HIP-KNEE-ANKLE CHAIN",
            "Rapid lower-limb configuration change",
            ("injured_hka_angle_2d_deg", "contralateral_hka_angle_2d_deg"),
            common_view,
            "Projected HKA change must be sustained and rate-supported.",
        ),
        _definition(
            "RELATIVELY_STABLE_LOWER_LIMB_CONFIGURATION",
            "HIP-KNEE-ANKLE CHAIN",
            "Lower-limb configuration remained relatively stable",
            ("injured_hka_angle_2d_deg", "contralateral_hka_angle_2d_deg"),
            common_view,
            "Projected HKA changes must remain below the descriptive change rule.",
        ),
        _definition(
            "BILATERAL_DIFFERENCE_INCREASED",
            "BILATERAL LOWER-LIMB RELATIONSHIP",
            "Bilateral lower-limb difference increased",
            ("hka_projected_bilateral_absolute_difference_deg",),
            common_view,
            "Absolute injured-vs-contralateral HKA relationship must increase.",
        ),
        _definition(
            "BILATERAL_DIFFERENCE_DECREASED",
            "BILATERAL LOWER-LIMB RELATIONSHIP",
            "Bilateral lower-limb difference decreased",
            ("hka_projected_bilateral_absolute_difference_deg",),
            common_view,
            "Absolute injured-vs-contralateral HKA relationship must decrease.",
        ),
        _definition(
            "BILATERAL_RELATIONSHIP_DIVERGING",
            "BILATERAL LOWER-LIMB RELATIONSHIP",
            "Bilateral lower-limb relationship diverged",
            ("hka_projected_bilateral_absolute_difference_deg",),
            common_view,
            "Absolute bilateral relationship must increase beyond the descriptive rule.",
        ),
        _definition(
            "BILATERAL_RELATIONSHIP_CONVERGING",
            "BILATERAL LOWER-LIMB RELATIONSHIP",
            "Bilateral lower-limb relationship converged",
            ("hka_projected_bilateral_absolute_difference_deg",),
            common_view,
            "Absolute bilateral relationship must decrease beyond the descriptive rule.",
        ),
        _definition(
            "BILATERAL_RELATIONSHIP_CROSSING",
            "BILATERAL LOWER-LIMB RELATIONSHIP",
            "Bilateral lower-limb relationship crossed",
            ("hka_projected_bilateral_difference_deg",),
            common_view,
            "Signed injured-minus-contralateral HKA relationship must change sign.",
        ),
        _definition(
            "BILATERAL_RELATIONSHIP_RELATIVELY_STABLE",
            "BILATERAL LOWER-LIMB RELATIONSHIP",
            "Bilateral lower-limb relationship remained relatively stable",
            ("hka_projected_bilateral_absolute_difference_deg",),
            common_view,
            "Absolute bilateral relationship must remain below the descriptive change rule.",
        ),
        _definition(
            "TRUNK_ORIENTATION_SHIFT",
            "TRUNK & PELVIS",
            "Trunk orientation shifted",
            ("projected_trunk_axis_angle_deg",),
            common_view,
            "Projected trunk-axis change must exceed the descriptive rule.",
        ),
        _definition(
            "RAPID_TRUNK_ORIENTATION_SHIFT",
            "TRUNK & PELVIS",
            "Rapid trunk orientation shift",
            ("projected_trunk_axis_angle_deg",),
            common_view,
            "Projected trunk-axis shift must be sustained and rate-supported.",
        ),
        _definition(
            "PELVIS_ORIENTATION_SHIFT",
            "TRUNK & PELVIS",
            "Pelvis orientation shifted",
            ("projected_hip_line_angle_deg",),
            common_view,
            "Projected hip-line orientation change must exceed the descriptive rule.",
        ),
        _definition(
            "SHOULDER_PELVIS_RELATIONSHIP_CHANGED",
            "TRUNK & PELVIS",
            "Shoulder-pelvis relationship changed",
            ("projected_shoulder_pelvis_orientation_difference_deg",),
            common_view,
            "Shoulder-line relative to hip-line orientation change must exceed the rule.",
        ),
        _definition(
            "RELATIVELY_STABLE_TRUNK_PELVIS",
            "TRUNK & PELVIS",
            "Trunk and pelvis remained relatively stable",
            ("projected_trunk_axis_angle_deg", "projected_hip_line_angle_deg"),
            common_view,
            "Trunk/pelvis changes must remain below the descriptive change rule.",
        ),
        _definition(
            "ARM_CONFIGURATION_CHANGED",
            "UPPER BODY",
            "Arm configuration changed",
            ("right_elbow_angle_2d_deg", "left_elbow_angle_2d_deg"),
            common_view,
            "Supported projected elbow change must exceed the descriptive rule.",
        ),
        _definition(
            "BILATERAL_ARM_RELATIONSHIP_CHANGED",
            "UPPER BODY",
            "Bilateral arm relationship changed",
            ("elbow_projected_bilateral_absolute_difference_deg",),
            common_view,
            "Projected bilateral elbow relationship must change.",
        ),
        _definition(
            "UPPER_BODY_REORIENTATION",
            "UPPER BODY",
            "Upper-body orientation shifted",
            ("right_upper_arm_orientation_2d_deg", "left_upper_arm_orientation_2d_deg"),
            common_view,
            "Supported projected upper-arm orientation change must exceed the rule.",
        ),
        _definition(
            "RELATIVELY_STABLE_UPPER_BODY",
            "UPPER BODY",
            "Upper body remained relatively stable",
            ("right_upper_arm_orientation_2d_deg", "left_upper_arm_orientation_2d_deg"),
            common_view,
            "Upper-body changes must remain below the descriptive change rule.",
        ),
        _definition(
            "MOVEMENT_PATH_WITHHELD",
            "EVIDENCE / UNAVAILABLE",
            "Movement path interpretation withheld",
            ("path:compensated_x", "path:compensated_y"),
            path_view,
            "Projected movement path must be withheld when path QA is not supported.",
            enabled=True,
        ),
        _definition(
            "PHASE_SEGMENTATION_WITHHELD",
            "EVIDENCE / UNAVAILABLE",
            "Movement phase segmentation withheld",
            ("movement_phases",),
            common_view,
            "Phase descriptions require supported phase segmentation.",
            enabled=True,
        ),
    ]
    return tuple(definitions)


def build_observable_movement_description_payload(
    *,
    case_id: str,
    source_id: str,
    dynamic_df: pd.DataFrame,
    frame_quality: pd.DataFrame,
    path_summary: dict,
    movement_window: Any,
    phase_status: str | None = None,
    config: MovementVocabularyConfig | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready controlled-vocabulary movement-description payload."""

    cfg = config or MovementVocabularyConfig()
    registry = build_controlled_movement_vocabulary()
    intervals = _supported_intervals(frame_quality, cfg)
    movement_window_metadata = _movement_window_dict(movement_window)
    descriptions: list[ObservableMovementDescription] = []
    withheld = _withheld_descriptions(
        case_id=case_id,
        path_summary=path_summary,
        phase_status=phase_status,
    )
    for interval in intervals:
        descriptions.extend(
            _descriptions_for_interval(
                case_id=case_id,
                dynamic_df=dynamic_df,
                interval=interval,
                cfg=cfg,
            )
        )
    default_story = _default_story_descriptions(descriptions, cfg)
    payload = {
        "metadata": {
            "case_id": case_id,
            "source_id": source_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "vocabulary_version": VOCABULARY_VERSION,
            "description_version": DESCRIPTION_VERSION,
            "movement_window": movement_window_metadata,
            "path_quality_status": path_summary.get("overall_status", "UNAVAILABLE"),
            "phase_status": phase_status or "UNKNOWN",
            "clustering_policy": (
                "Semantic strings are not clustering inputs; future clustering uses "
                "CaseMovementSignature numeric descriptors."
            ),
            "threshold_note": (
                "Magnitude/rate thresholds are transparent engineering rules for "
                "description, not clinical or population-normal thresholds."
            ),
        },
        "movement_families": sorted({item.family for item in registry}),
        "descriptor_registry": [item.to_dict() for item in registry],
        "forbidden_descriptors": list(FORBIDDEN_DESCRIPTION_PHRASES),
        "magnitude_rules": cfg.to_dict(),
        "clip_evidence_coverage": _clip_evidence_coverage(
            frame_quality,
            intervals,
            movement_window_metadata,
        ),
        "supported_intervals": intervals,
        "default_story_descriptions": [item.to_dict() for item in default_story],
        "descriptions": [item.to_dict() for item in descriptions],
        "withheld_descriptions": [item.to_dict() for item in withheld],
    }
    return _json_ready(payload)


def write_observable_descriptions_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write controlled movement descriptions to disk."""

    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return output


def _definition(
    descriptor_id: str,
    family: str,
    user_label: str,
    required_features: tuple[str, ...],
    view_requirements: tuple[str, ...],
    description: str,
    *,
    enabled: bool = True,
) -> MovementDescriptorDefinition:
    return MovementDescriptorDefinition(
        descriptor_id=descriptor_id,
        family=family,
        user_label=user_label,
        description=description,
        required_features=required_features,
        required_evidence="supported feature samples inside one valid target interval",
        temporal_rule="do not bridge identity uncertainty, invalid track segments, or unsupported gaps",
        magnitude_rule="configured engineering threshold; not a clinical or population-normal threshold",
        view_requirements=view_requirements,
        forbidden_when=FORBIDDEN_DESCRIPTION_PHRASES,
        explanation_template=(
            "Show start/end values, change magnitude, supported samples, duration, and QC reason."
        ),
        enabled=enabled,
    )


def _supported_intervals(frame_quality: pd.DataFrame, cfg: MovementVocabularyConfig) -> list[dict]:
    if frame_quality.empty or "frame_status" not in frame_quality.columns:
        return []
    valid = frame_quality[frame_quality["frame_status"].eq("VALID_TARGET")].copy()
    if valid.empty:
        return []
    if "valid_segment_id" not in valid.columns:
        valid["valid_segment_id"] = _contiguous_segment_ids(valid["source_frame_index"])
    intervals = []
    for segment_id, rows in valid.groupby("valid_segment_id", dropna=False, sort=False):
        rows = rows.sort_values("source_frame_index")
        if len(rows) < cfg.minimum_interval_frames:
            continue
        start = int(rows["source_frame_index"].iloc[0])
        end = int(rows["source_frame_index"].iloc[-1])
        start_ms = _optional_float(rows["timestamp_ms"].iloc[0])
        end_ms = _optional_float(rows["timestamp_ms"].iloc[-1])
        intervals.append(
            {
                "scope_type": "SUPPORTED_INTERVAL",
                "scope_id": f"supported_interval_{len(intervals) + 1:03d}",
                "valid_segment_id": None if pd.isna(segment_id) else str(segment_id),
                "start_frame": start,
                "end_frame": end,
                "start_timestamp_ms": start_ms,
                "end_timestamp_ms": end_ms,
                "duration_ms": end_ms - start_ms if start_ms is not None and end_ms is not None else None,
                "frame_count": len(rows),
                "label": f"Supported evidence interval, source frames {start}-{end}",
                "snapshot_frames": _snapshot_frames_from_interval(rows),
            }
        )
    return intervals


def _clip_evidence_coverage(
    frame_quality: pd.DataFrame,
    intervals: list[dict],
    movement_window: dict[str, Any],
) -> dict[str, Any]:
    if frame_quality.empty or "source_frame_index" not in frame_quality.columns:
        return {
            "status": "UNAVAILABLE",
            "note": "Clip frame coverage is unavailable because frame QC was not supplied.",
        }

    rows = frame_quality.sort_values("source_frame_index").copy()
    status_column = "frame_status" if "frame_status" in rows.columns else None
    statuses = (
        rows[status_column].fillna("UNKNOWN").astype(str)
        if status_column
        else pd.Series(["UNKNOWN"] * len(rows), index=rows.index)
    )
    source_frames = rows["source_frame_index"].astype(int)
    clip_start = int(source_frames.min())
    clip_end = int(source_frames.max())
    movement_start = _optional_int(movement_window.get("movement_start_frame"))
    movement_end = _optional_int(movement_window.get("movement_end_frame"))
    supported_frames = int(statuses.eq("VALID_TARGET").sum())
    last_supported = max((int(item["end_frame"]) for item in intervals), default=None)
    post_supported = (
        rows[source_frames > last_supported].copy()
        if last_supported is not None
        else rows.iloc[0:0].copy()
    )
    post_statuses = (
        post_supported[status_column].fillna("UNKNOWN").astype(str)
        if status_column and not post_supported.empty
        else pd.Series(dtype=str)
    )

    coverage = {
        "status": "AVAILABLE",
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "total_clip_frames": len(rows),
        "annotated_movement_start_frame": movement_start,
        "annotated_movement_end_frame": movement_end,
        "valid_target_frames": supported_frames,
        "withheld_or_unsupported_frames": int(len(rows) - supported_frames),
        "frame_status_counts": {str(key): int(value) for key, value in statuses.value_counts().items()},
        "supported_source_ranges": [
            {
                "start_frame": int(item["start_frame"]),
                "end_frame": int(item["end_frame"]),
                "frame_count": int(item["frame_count"]),
            }
            for item in intervals
        ],
        "last_supported_source_frame": last_supported,
        "has_frames_after_supported_interval": bool(last_supported is not None and clip_end > last_supported),
        "supported_interval_reaches_annotated_movement_end": bool(
            last_supported is not None and movement_end is not None and last_supported >= movement_end
        ),
        "note": (
            "Supported intervals are evidence/QC boundaries, not claims about the final video frame."
        ),
    }
    if not post_supported.empty:
        post_frames = post_supported["source_frame_index"].astype(int)
        coverage["post_supported_frame_range"] = {
            "start_frame": int(post_frames.min()),
            "end_frame": int(post_frames.max()),
            "frame_count": len(post_supported),
        }
        coverage["post_supported_status_counts"] = {
            str(key): int(value) for key, value in post_statuses.value_counts().items()
        }
        coverage["post_supported_note"] = (
            "Frames after the supported interval are present in the clip but withheld "
            "from controlled movement descriptions by quality checks."
        )
    return coverage


def _contiguous_segment_ids(frames: pd.Series) -> pd.Series:
    ids = []
    current = 1
    previous = None
    for frame in frames.astype(int):
        if previous is not None and frame != previous + 1:
            current += 1
        ids.append(current)
        previous = frame
    return pd.Series(ids, index=frames.index)


def _snapshot_frames_from_interval(rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows = rows.sort_values("source_frame_index")
    frames = rows["source_frame_index"].astype(int).tolist()
    if not frames:
        return []
    labels = ["Start", "25%", "50%", "75%", "End"] if len(frames) >= 12 else ["Start", "Mid", "End"]
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0] if len(labels) == 5 else [0.0, 0.5, 1.0]
    snapshots = []
    for label, fraction in zip(labels, fractions, strict=True):
        target = frames[0] + fraction * (frames[-1] - frames[0])
        nearest = min(frames, key=lambda item: abs(item - target))
        row = rows[rows["source_frame_index"].astype(int).eq(int(nearest))].iloc[0]
        snapshots.append(
            {
                "label": label,
                "source_frame_index": int(nearest),
                "timestamp_ms": _optional_float(row.get("timestamp_ms")),
            }
        )
    deduped = []
    seen = set()
    for snapshot in snapshots:
        frame = snapshot["source_frame_index"]
        if frame in seen:
            continue
        seen.add(frame)
        deduped.append(snapshot)
    return deduped


def _descriptions_for_interval(
    *,
    case_id: str,
    dynamic_df: pd.DataFrame,
    interval: dict[str, Any],
    cfg: MovementVocabularyConfig,
) -> list[ObservableMovementDescription]:
    results: list[ObservableMovementDescription] = []
    metrics = {
        feature: _feature_evidence(dynamic_df, feature, interval, cfg)
        for feature in (
            "injured_hka_angle_2d_deg",
            "contralateral_hka_angle_2d_deg",
            "hka_projected_bilateral_difference_deg",
            "hka_projected_bilateral_absolute_difference_deg",
            "projected_trunk_axis_angle_deg",
            "projected_hip_line_angle_deg",
            "projected_shoulder_line_angle_deg",
            "projected_shoulder_pelvis_orientation_difference_deg",
            "right_elbow_angle_2d_deg",
            "right_upper_arm_orientation_2d_deg",
            "left_elbow_angle_2d_deg",
            "left_upper_arm_orientation_2d_deg",
        )
    }
    injured = metrics["injured_hka_angle_2d_deg"]
    contralateral = metrics["contralateral_hka_angle_2d_deg"]
    hka_metrics = [metric for metric in (injured, contralateral) if metric["supported"]]
    if hka_metrics:
        dominant = max(hka_metrics, key=lambda item: abs(item["change"]))
        if abs(dominant["change"]) >= cfg.angle_change_deg:
            results.append(
                _description(
                    case_id,
                    interval,
                    "LOWER_LIMB_CONFIGURATION_CHANGED",
                    "HIP-KNEE-ANKLE CHAIN",
                    "Lower-limb configuration changed",
                    (
                        "The projected configuration of the lower-limb chain changed across "
                        "the supported interval."
                    ),
                    hka_metrics,
                    visualisation_hint="pose_start_mid_end_hka_arc",
                    salience_base=0.88,
                )
            )
            hka_descriptor = (
                "INJURED_HKA_INCREASED"
                if dominant["feature_name"] == "injured_hka_angle_2d_deg" and dominant["change"] > 0
                else "INJURED_HKA_DECREASED"
                if dominant["feature_name"] == "injured_hka_angle_2d_deg"
                else "CONTRALATERAL_HKA_INCREASED"
                if dominant["change"] > 0
                else "CONTRALATERAL_HKA_DECREASED"
            )
            results.append(
                _description(
                    case_id,
                    interval,
                    hka_descriptor,
                    "HIP-KNEE-ANKLE CHAIN",
                    _hka_user_label(hka_descriptor),
                    _hka_summary(hka_descriptor),
                    (dominant,),
                    visualisation_hint="hka_start_end_overlay",
                    salience_base=0.62,
                )
            )
            if _rapid_supported(dominant, cfg):
                results.append(
                    _description(
                        case_id,
                        interval,
                        "RAPID_LOWER_LIMB_CONFIGURATION_CHANGE",
                        "HIP-KNEE-ANKLE CHAIN",
                        "Rapid lower-limb configuration change",
                        (
                            "The projected lower-limb configuration changed rapidly under "
                            "the configured robust-rate rule."
                        ),
                        (dominant,),
                        visualisation_hint="pose_start_mid_end_hka_arc",
                        salience_base=0.76,
                    )
                )
    signed = metrics["hka_projected_bilateral_difference_deg"]
    absolute = metrics["hka_projected_bilateral_absolute_difference_deg"]
    if signed["supported"] and absolute["supported"]:
        signed_start = signed["start_value"]
        signed_end = signed["end_value"]
        crossed = (signed_start < 0 < signed_end) or (signed_start > 0 > signed_end)
        abs_change = absolute["end_value"] - absolute["start_value"]
        if crossed:
            results.append(
                _description(
                    case_id,
                    interval,
                    "BILATERAL_RELATIONSHIP_CROSSING",
                    "BILATERAL LOWER-LIMB RELATIONSHIP",
                    "Bilateral lower-limb relationship crossed",
                    (
                        "The signed injured-minus-contralateral projected lower-limb "
                        "relationship changed sign during the supported interval."
                    ),
                    (signed, absolute),
                    visualisation_hint="bilateral_compact_trajectory",
                    salience_base=0.82,
                )
            )
        if abs(abs_change) >= cfg.bilateral_change_deg:
            descriptor_id = (
                "BILATERAL_DIFFERENCE_INCREASED"
                if abs_change > 0
                else "BILATERAL_DIFFERENCE_DECREASED"
            )
            results.append(
                _description(
                    case_id,
                    interval,
                    descriptor_id,
                    "BILATERAL LOWER-LIMB RELATIONSHIP",
                    (
                        "Bilateral lower-limb difference increased"
                        if abs_change > 0
                        else "Bilateral lower-limb difference decreased"
                    ),
                    (
                        "The absolute projected difference between the injured and "
                        "contralateral lower-limb chains changed across the supported interval."
                    ),
                    (absolute,),
                    visualisation_hint="bilateral_compact_trajectory",
                    salience_base=0.78,
                )
            )
    trunk = metrics["projected_trunk_axis_angle_deg"]
    pelvis = metrics["projected_hip_line_angle_deg"]
    shoulder = metrics["projected_shoulder_line_angle_deg"]
    shoulder_pelvis = metrics["projected_shoulder_pelvis_orientation_difference_deg"]
    orientation_changed = [
        metric
        for metric, threshold in (
            (trunk, cfg.orientation_shift_deg),
            (pelvis, cfg.orientation_shift_deg),
            (shoulder, cfg.orientation_shift_deg),
            (shoulder_pelvis, cfg.orientation_shift_deg),
        )
        if metric["supported"] and abs(metric["change"]) >= threshold
    ]
    if trunk["supported"] and abs(trunk["change"]) >= cfg.orientation_shift_deg:
        results.append(
            _description(
                case_id,
                interval,
                "TRUNK_ORIENTATION_SHIFT",
                "TRUNK & PELVIS",
                "Trunk orientation shifted",
                "Projected trunk orientation shifted during the supported interval.",
                (trunk,),
                visualisation_hint="trunk_axis_start_end_overlay",
                salience_base=0.8,
            )
        )
    if pelvis["supported"] and abs(pelvis["change"]) >= cfg.orientation_shift_deg:
        results.append(
            _description(
                case_id,
                interval,
                "PELVIS_ORIENTATION_SHIFT",
                "TRUNK & PELVIS",
                "Pelvis orientation shifted",
                "Projected hip-line orientation shifted during the supported interval.",
                (pelvis,),
                visualisation_hint="hip_line_start_end_overlay",
                salience_base=0.72,
            )
        )
    if shoulder_pelvis["supported"] and abs(shoulder_pelvis["change"]) >= cfg.orientation_shift_deg:
        results.append(
            _description(
                case_id,
                interval,
                "SHOULDER_PELVIS_RELATIONSHIP_CHANGED",
                "TRUNK & PELVIS",
                "Shoulder-pelvis relationship changed",
                (
                    "The projected shoulder-line relative to hip-line relationship changed "
                    "during the supported interval."
                ),
                (shoulder_pelvis,),
                visualisation_hint="shoulder_pelvis_axis_overlay",
                salience_base=0.68,
            )
        )
    upper_candidates = [
        metrics["right_upper_arm_orientation_2d_deg"],
        metrics["left_upper_arm_orientation_2d_deg"],
        metrics["right_elbow_angle_2d_deg"],
        metrics["left_elbow_angle_2d_deg"],
    ]
    upper_changed = [
        metric
        for metric in upper_candidates
        if metric["supported"] and abs(metric["change"]) >= cfg.upper_body_shift_deg
    ]
    if upper_changed:
        dominant_upper = max(upper_changed, key=lambda item: abs(item["change"]))
        descriptor = (
            "UPPER_BODY_REORIENTATION"
            if "upper_arm_orientation" in dominant_upper["feature_name"]
            else "ARM_CONFIGURATION_CHANGED"
        )
        results.append(
            _description(
                case_id,
                interval,
                descriptor,
                "UPPER BODY",
                (
                    "Upper-body orientation shifted"
                    if descriptor == "UPPER_BODY_REORIENTATION"
                    else "Arm configuration changed"
                ),
                (
                    "Supported projected upper-body measurements changed across the "
                    "supported interval."
                ),
                (dominant_upper,),
                visualisation_hint="upper_body_start_end_overlay",
                salience_base=0.58,
            )
        )
    whole_body_metrics = [
        *orientation_changed,
        *([max(hka_metrics, key=lambda item: abs(item["change"]))] if hka_metrics else []),
        *upper_changed[:1],
    ]
    whole_body_metrics = _unique_metric_records(
        [
            metric
            for metric in whole_body_metrics
            if metric["supported"] and abs(metric["change"]) >= cfg.angle_change_deg
        ]
    )
    if len(whole_body_metrics) >= 2:
        results.append(
            _description(
                case_id,
                interval,
                "WHOLE_BODY_REORIENTATION",
                "WHOLE-BODY ORIENTATION",
                "Whole-body orientation shifted",
                (
                    "Multiple projected body-orientation measures shifted across the "
                    "supported interval."
                ),
                tuple(whole_body_metrics[:4]),
                visualisation_hint="ghosted_pose_sequence",
                salience_base=0.9,
            )
        )
    return results


def _feature_evidence(
    dynamic_df: pd.DataFrame,
    feature_name: str,
    interval: dict[str, Any],
    cfg: MovementVocabularyConfig,
) -> dict[str, Any]:
    start = int(interval["start_frame"])
    end = int(interval["end_frame"])
    relevant_frames = int(interval["frame_count"])
    rows = dynamic_df[
        dynamic_df["feature_name"].eq(feature_name)
        & dynamic_df["source_frame_index"].astype(int).between(start, end, inclusive="both")
    ].copy()
    supported = rows[rows["feature_status"].eq("SUPPORTED") & rows["feature_value"].notna()].copy()
    supported = supported.sort_values("source_frame_index")
    if len(supported) < cfg.minimum_supported_samples:
        return {
            "feature_name": feature_name,
            "supported": False,
            "reason": "Fewer than the configured supported samples were available.",
            "supported_samples": len(supported),
            "relevant_frames": relevant_frames,
            "coverage": len(supported) / relevant_frames if relevant_frames else 0.0,
        }
    coverage = len(supported) / relevant_frames if relevant_frames else 0.0
    if coverage < cfg.minimum_feature_coverage:
        return {
            "feature_name": feature_name,
            "supported": False,
            "reason": "Supported-sample coverage is below the descriptor rule.",
            "supported_samples": len(supported),
            "relevant_frames": relevant_frames,
            "coverage": coverage,
        }
    start_value = float(supported.iloc[0]["feature_value"])
    end_value = float(supported.iloc[-1]["feature_value"])
    change = _feature_change(feature_name, start_value, end_value)
    dynamic = rows[
        rows["dynamic_status"].eq("SUPPORTED")
        & rows["robust_dynamic_rate"].notna()
    ].copy()
    robust_rates = dynamic["robust_dynamic_rate"].astype(float) if len(dynamic) else pd.Series(dtype=float)
    rapid_run = _max_consecutive_above(dynamic, cfg.rapid_rate_deg_per_s)
    return {
        "feature_name": feature_name,
        "supported": True,
        "reason": "Supported by feature-level QC inside one valid target interval.",
        "supported_samples": len(supported),
        "relevant_frames": relevant_frames,
        "coverage": float(coverage),
        "start_value": start_value,
        "end_value": end_value,
        "change": float(change),
        "unit": str(supported.iloc[0].get("unit", "")),
        "start_frame": int(supported.iloc[0]["source_frame_index"]),
        "end_frame": int(supported.iloc[-1]["source_frame_index"]),
        "source_frames": tuple(int(item) for item in supported["source_frame_index"]),
        "dynamic_supported_samples": len(dynamic),
        "dynamic_supported_fraction": float(len(dynamic) / relevant_frames if relevant_frames else 0.0),
        "max_abs_robust_rate": (
            float(robust_rates.abs().max()) if len(robust_rates) else None
        ),
        "median_abs_robust_rate": (
            float(robust_rates.abs().median()) if len(robust_rates) else None
        ),
        "rapid_max_consecutive_samples": int(rapid_run),
    }


def _feature_change(feature_name: str, start_value: float, end_value: float) -> float:
    if feature_name in ORIENTATION_FEATURES:
        return wrapped_angle_difference_deg(end_value, start_value)
    return end_value - start_value


def _max_consecutive_above(rows: pd.DataFrame, threshold: float) -> int:
    if rows.empty:
        return 0
    high = rows[rows["robust_dynamic_rate"].astype(float).abs().ge(threshold)]
    max_run = 0
    current = 0
    previous = None
    for frame in high["source_frame_index"].astype(int):
        if previous is None or frame == previous + 1:
            current += 1
        else:
            current = 1
        max_run = max(max_run, current)
        previous = frame
    return int(max_run)


def _description(
    case_id: str,
    interval: dict[str, Any],
    descriptor_id: str,
    family: str,
    user_label: str,
    summary: str,
    metrics: tuple[dict[str, Any], ...],
    *,
    visualisation_hint: str,
    salience_base: float,
) -> ObservableMovementDescription:
    primary = max(metrics, key=lambda item: abs(float(item.get("change", 0.0) or 0.0)))
    change_magnitude = max(abs(float(item.get("change", 0.0) or 0.0)) for item in metrics)
    coverage = min(float(item.get("coverage", 0.0) or 0.0) for item in metrics)
    robust_rate = max(
        [
            float(item["max_abs_robust_rate"])
            for item in metrics
            if item.get("max_abs_robust_rate") is not None
        ],
        default=None,
    )
    evidence_status = "SUPPORTED" if coverage >= 0.75 else "LIMITED"
    source_frames = tuple(
        sorted(
            {
                int(frame)
                for item in metrics
                for frame in (item.get("source_frames") or ())
            }
        )
    )
    salience = salience_base + min(change_magnitude / 90.0, 0.8) * 0.35 + coverage * 0.2
    if robust_rate is not None:
        salience += min(robust_rate / 400.0, 0.5) * 0.1
    return ObservableMovementDescription(
        descriptor_id=descriptor_id,
        case_id=case_id,
        scope_type=str(interval["scope_type"]),
        scope_start=int(interval["start_frame"]),
        scope_end=int(interval["end_frame"]),
        family=family,
        user_label=user_label,
        summary=summary,
        evidence_status=evidence_status,
        evidence_reason=(
            f"{primary['supported_samples']} of {primary['relevant_frames']} interval frames "
            "supported the primary measurement."
        ),
        supporting_features=tuple(item["feature_name"] for item in metrics),
        supporting_values={
            item["feature_name"]: {
                "start_value": item.get("start_value"),
                "end_value": item.get("end_value"),
                "change": item.get("change"),
                "unit": item.get("unit"),
                "supported_samples": item.get("supported_samples"),
                "relevant_frames": item.get("relevant_frames"),
                "coverage": item.get("coverage"),
                "dynamic_supported_samples": item.get("dynamic_supported_samples"),
                "dynamic_supported_fraction": item.get("dynamic_supported_fraction"),
                "max_abs_robust_rate": item.get("max_abs_robust_rate"),
                "rapid_max_consecutive_samples": item.get("rapid_max_consecutive_samples"),
            }
            for item in metrics
        },
        change_magnitude=float(change_magnitude),
        duration_ms=_optional_float(interval.get("duration_ms")),
        robust_rate=robust_rate,
        source_frames=source_frames,
        view_requirements=("generic_projected_2d_view",),
        provenance="human target annotation + processed pose QC + controlled vocabulary rule",
        visualisation_hint=visualisation_hint,
        salience_score=float(salience),
    )


def _hka_user_label(descriptor_id: str) -> str:
    return {
        "INJURED_HKA_INCREASED": "Injured-side projected HKA increased",
        "INJURED_HKA_DECREASED": "Injured-side projected HKA decreased",
        "CONTRALATERAL_HKA_INCREASED": "Contralateral projected HKA increased",
        "CONTRALATERAL_HKA_DECREASED": "Contralateral projected HKA decreased",
    }[descriptor_id]


def _hka_summary(descriptor_id: str) -> str:
    side = "injured-side" if descriptor_id.startswith("INJURED") else "contralateral"
    direction = "increased" if descriptor_id.endswith("INCREASED") else "decreased"
    return (
        f"The {side} projected hip-knee-ankle chain {direction} across the "
        "supported interval."
    )


def _rapid_supported(metric: dict[str, Any], cfg: MovementVocabularyConfig) -> bool:
    return (
        float(metric.get("dynamic_supported_fraction", 0.0) or 0.0)
        >= cfg.minimum_dynamic_supported_fraction
        and int(metric.get("rapid_max_consecutive_samples", 0) or 0)
        >= cfg.rapid_min_consecutive_samples
    )


def _unique_metric_records(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for metric in metrics:
        name = metric["feature_name"]
        if name in seen:
            continue
        seen.add(name)
        output.append(metric)
    return output


def _default_story_descriptions(
    descriptions: list[ObservableMovementDescription],
    cfg: MovementVocabularyConfig,
) -> list[ObservableMovementDescription]:
    supported = [
        item
        for item in descriptions
        if item.evidence_status in {DescriptionEvidenceStatus.SUPPORTED, DescriptionEvidenceStatus.LIMITED}
    ]
    ranked = sorted(supported, key=lambda item: item.salience_score, reverse=True)
    selected = []
    used_families = set()
    for item in ranked:
        if item.family in used_families:
            continue
        selected.append(item)
        used_families.add(item.family)
        if len(selected) >= cfg.max_default_descriptions:
            break
    return selected


def _withheld_descriptions(
    *,
    case_id: str,
    path_summary: dict,
    phase_status: str | None,
) -> list[ObservableMovementDescription]:
    withheld = []
    path_status = path_summary.get("overall_status", "UNAVAILABLE")
    if path_status != "SUPPORTED":
        reason = (
            path_summary.get("reason")
            or "Projected movement path did not pass scientific quality checks."
        )
        for descriptor_id in PATH_DESCRIPTOR_IDS:
            withheld.append(
                _withheld(
                    case_id,
                    descriptor_id,
                    "MOVEMENT PATH",
                    descriptor_id.replace("_", " ").title(),
                    (
                        "Projected movement-path interpretation was withheld because "
                        f"camera-motion/path QA status is {path_status}."
                    ),
                    reason,
                    ("validated_projected_movement_path",),
                )
            )
    if phase_status == "SUPPORTED_EVIDENCE_INTERVAL":
        withheld.append(
            _withheld(
                case_id,
                "PHASE_SEGMENTATION_WITHHELD",
                "EVIDENCE / AVAILABLE INTERVAL",
                "Distinct movement phases not detected",
                (
                    "A supported measurement interval is available, but it is not "
                    "presented as a phase sequence because no supported transition was detected."
                ),
                (
                    "The interval remains useful for observable measurements but does not "
                    "establish a before/after transition or injury timing."
                ),
                ("generic_projected_2d_view",),
            )
        )
    elif phase_status and phase_status not in {"SUPPORTED", "SUPPORTED_PARTIAL_WINDOW"}:
        withheld.append(
            _withheld(
                case_id,
                "PHASE_SEGMENTATION_WITHHELD",
                "EVIDENCE / UNAVAILABLE",
                "Movement phase segmentation withheld",
                "Distinct movement phases were not inferred reliably for this case.",
                "Continuous multidimensional evidence was insufficient after target/path QC.",
                ("generic_projected_2d_view",),
            )
        )
    return withheld


def _withheld(
    case_id: str,
    descriptor_id: str,
    family: str,
    user_label: str,
    summary: str,
    evidence_reason: str,
    view_requirements: tuple[str, ...],
) -> ObservableMovementDescription:
    return ObservableMovementDescription(
        descriptor_id=descriptor_id,
        case_id=case_id,
        scope_type="WHOLE_MOVEMENT",
        scope_start=None,
        scope_end=None,
        family=family,
        user_label=user_label,
        summary=summary,
        evidence_status="WITHHELD",
        evidence_reason=evidence_reason,
        supporting_features=(),
        supporting_values={},
        change_magnitude=None,
        duration_ms=None,
        robust_rate=None,
        source_frames=(),
        view_requirements=view_requirements,
        provenance="controlled vocabulary availability rule",
        visualisation_hint="unavailable_state",
        salience_score=0.0,
    )


def _movement_window_dict(movement_window: Any) -> dict[str, Any]:
    if hasattr(movement_window, "to_dict"):
        return movement_window.to_dict()
    if isinstance(movement_window, dict):
        return dict(movement_window)
    return {}


def _optional_int(value: Any) -> int | None:
    try:
        output = int(value)
    except (TypeError, ValueError):
        return None
    return output


def _optional_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if np.isfinite(output) else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, DescriptionEvidenceStatus):
        return value.value
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
