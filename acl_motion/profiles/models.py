"""Serializable MovementProfile models for Milestone 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class BodyRegion(StrEnum):
    """User-facing body-region grouping."""

    WHOLE_BODY = "whole_body"
    LOWER_LIMB = "lower_limb"
    TRUNK_PELVIS = "trunk_pelvis"
    UPPER_BODY = "upper_body"
    BILATERAL = "bilateral"
    EVIDENCE = "evidence"


class FeatureFamily(StrEnum):
    """Feature-family grouping for future explainable analytics."""

    LOWER_LIMB_GEOMETRY = "lower_limb_geometry"
    KNEE_GEOMETRY = "knee_geometry"
    TRUNK_PELVIS = "trunk_pelvis"
    UPPER_LIMB = "upper_limb"
    WRIST_GEOMETRY = "wrist_geometry"
    BILATERAL = "bilateral"
    DYNAMICS = "dynamics"


class QualityCategory(StrEnum):
    """User-readable evidence category."""

    SUPPORTED = "SUPPORTED"
    LIMITED = "LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


class AnalyticsEligibility(StrEnum):
    """Transparent feature eligibility for later analytics."""

    ANALYTICS_READY = "ANALYTICS_READY"
    GEOMETRY_ONLY = "GEOMETRY_ONLY"
    DYNAMIC_ONLY = "DYNAMIC_ONLY"
    LIMITED_COVERAGE = "LIMITED_COVERAGE"
    LOW_DYNAMIC_RELIABILITY = "LOW_DYNAMIC_RELIABILITY"
    UNSUPPORTED = "UNSUPPORTED"
    EXCLUDE_FROM_DEFAULT_ANALYTICS = "EXCLUDE_FROM_DEFAULT_ANALYTICS"


@dataclass(frozen=True, slots=True)
class ProfileThresholds:
    """Configurable analysis-quality thresholds, not clinical validation thresholds."""

    geometry_ready_completeness: float = 0.45
    geometry_limited_completeness: float = 0.15
    dynamic_ready_completeness: float = 0.50
    dynamic_limited_completeness: float = 0.20
    minimum_supported_geometry_frames: int = 3
    minimum_supported_dynamic_samples: int = 3


@dataclass(frozen=True, slots=True)
class TraceableValue:
    """A summary value with timing and source-frame traceability."""

    value: float | None
    event_relative_ms: float | None
    source_frame_index: int | None
    analysis_frame_index: int | None
    feature_status: str
    dynamic_status: str = ""


@dataclass(frozen=True, slots=True)
class ProfileWindowSummary:
    """Feature summary inside an event-relative window."""

    window_name: str
    window_start_ms: float
    window_end_ms: float
    geometry_status: str
    geometry_completeness: float
    mean: float | None
    range: float | None
    change: float | None
    robust_max_rate: float | None
    time_robust_max_rate_ms: float | None


@dataclass(frozen=True, slots=True)
class FeatureMovementProfile:
    """Per-feature movement and evidence summary."""

    feature_name: str
    body_region: str
    feature_family: str
    unit: str
    geometry_status: str
    geometry_completeness: float
    dynamic_status: str
    dynamic_completeness: float
    quality_category: str
    minimum: TraceableValue
    maximum: TraceableValue
    range: float | None
    mean: float | None
    value_at_t0: TraceableValue
    baseline_value: float | None
    change_baseline_to_t0: float | None
    peak_robust_rate: TraceableValue
    window_summaries: tuple[ProfileWindowSummary, ...]
    primary_rejection_reason: str
    landmarks_used: tuple[str, ...]
    analytics_eligibility: str
    geometry_analytics_eligible: bool
    dynamic_analytics_eligible: bool
    eligibility_reason: str
    notes: str


@dataclass(frozen=True, slots=True)
class EvidenceOverview:
    """Compact evidence overview for future UI display."""

    supported_features: int
    limited_features: int
    unavailable_features: int
    geometry_coverage: float
    dynamic_coverage: float
    event_anchor_confidence: float | None
    primary_limitations: tuple[str, ...]
    human_target_verified: bool
    manual_roi_corrections: int | None
    overall_quality_label: str
    component_breakdown: dict


@dataclass(frozen=True, slots=True)
class EvidenceProfile:
    """Case/view evidence and measurement reliability profile."""

    case_id: str
    source_id: str
    view_id: str
    target_annotation_method: str
    manual_roi_keyframe_count: int | None
    target_tracking_coverage: float | None
    pose_frame_coverage: float | None
    upper_body_landmark_coverage: float | None
    core_landmark_coverage: float | None
    lower_limb_landmark_coverage: float | None
    geometry_feature_coverage: float
    dynamic_feature_coverage: float
    interpolation_fraction: float | None
    rejected_fraction: float | None
    identity_uncertainty_fraction: float | None
    target_loss_fraction: float | None
    supported_feature_count: int
    limited_feature_count: int
    unavailable_feature_count: int
    analytics_ready_feature_count: int
    primary_evidence_limitations: tuple[str, ...]
    evidence_overview: EvidenceOverview


@dataclass(frozen=True, slots=True)
class MovementProfile:
    """Structured case/view movement profile."""

    case: dict
    source: dict
    event_annotation: dict
    body_region_profiles: dict[str, list[FeatureMovementProfile]]
    trajectory_summaries: tuple[FeatureMovementProfile, ...]
    temporal_dynamic_summaries: tuple[FeatureMovementProfile, ...]
    evidence_profile: EvidenceProfile
    feature_availability: dict
    analytics_eligibility: dict
    provenance: dict
    explanatory_metadata: dict
    movement_profile_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-ready dictionary."""

        return _jsonable(asdict(self))


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
