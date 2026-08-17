"""Build MovementProfile objects from M4.1 hardened dynamic evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from acl_motion.cases.annotations import EventAnnotation
from acl_motion.events.dynamic_reliability import DYNAMIC_RELIABILITY_VERSION
from acl_motion.geometry.features import FEATURE_SET_VERSION, GEOMETRY_VERSION
from acl_motion.profiles.models import (
    AnalyticsEligibility,
    BodyRegion,
    EvidenceOverview,
    EvidenceProfile,
    FeatureMovementProfile,
    MovementProfile,
    ProfileThresholds,
    ProfileWindowSummary,
    QualityCategory,
    TraceableValue,
)
from acl_motion.profiles.registry import classify_feature

PROFILE_VERSION = "m5_movement_profile_v1"
DEFAULT_PROFILE_THRESHOLDS = ProfileThresholds()
ROBUST_SUPPORTED_STATUS = "SUPPORTED"
GEOMETRY_SUPPORTED_STATUS = "SUPPORTED"


def build_movement_profile(
    dynamic_df: pd.DataFrame,
    *,
    event_annotation: EventAnnotation,
    pose_reliability_summary: dict | None = None,
    dynamic_quality_summary: dict | None = None,
    geometry_feature_summary: dict | None = None,
    thresholds: ProfileThresholds = DEFAULT_PROFILE_THRESHOLDS,
    manual_roi_keyframe_count: int | None = None,
    provenance: dict | None = None,
) -> MovementProfile:
    """Build a serializable MovementProfile for one case/view."""

    _validate_dynamic_input(dynamic_df)
    feature_profiles = tuple(
        _build_feature_profile(feature_name, rows, thresholds)
        for feature_name, rows in dynamic_df.groupby("feature_name", sort=True)
    )
    evidence_profile = _build_evidence_profile(
        dynamic_df,
        feature_profiles,
        event_annotation,
        pose_reliability_summary or {},
        manual_roi_keyframe_count,
    )
    body_regions = _group_by_region(feature_profiles)
    availability = _feature_availability(feature_profiles)
    eligibility = _analytics_eligibility(feature_profiles, thresholds)
    summary = _movement_summary(feature_profiles, evidence_profile)
    first = dynamic_df.iloc[0]
    profile_provenance = {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "geometry_version": GEOMETRY_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "dynamic_reliability_version": DYNAMIC_RELIABILITY_VERSION,
        "thresholds": asdict(thresholds),
        "dynamic_quality_summary": dynamic_quality_summary or {},
        "geometry_feature_summary": geometry_feature_summary or {},
        **(provenance or {}),
    }
    return MovementProfile(
        case={"case_id": str(first["case_id"])},
        source={
            "source_id": str(first["source_id"]),
            "view_id": str(first["view_id"]),
        },
        event_annotation=event_annotation.to_dict(),
        body_region_profiles=body_regions,
        trajectory_summaries=feature_profiles,
        temporal_dynamic_summaries=feature_profiles,
        evidence_profile=evidence_profile,
        feature_availability=availability,
        analytics_eligibility=eligibility,
        provenance=profile_provenance,
        explanatory_metadata={
            "terminology": (
                "Generic image-plane projected movement; not ACL risk, injury probability, "
                "diagnosis, or true 3D biomechanics."
            ),
            "missingness_policy": "Unavailable values remain null and are not interpolated for M5.",
        },
        movement_profile_summary=summary,
    )


def build_case_feature_summary(profile: MovementProfile) -> pd.DataFrame:
    """Build long case x feature summary table for future M6 feature matrix work."""

    rows: list[dict] = []
    for feature in profile.trajectory_summaries:
        pre_late = _window_by_name(feature, "PRE_LATE")
        bilateral_summary = {}
        if feature.body_region == BodyRegion.BILATERAL.value:
            bilateral_summary = {
                "minimum": feature.minimum.value,
                "maximum": feature.maximum.value,
                "range": feature.range,
            }
        rows.append(
            {
                "case_id": profile.case["case_id"],
                "source_id": profile.source["source_id"],
                "view_id": profile.source["view_id"],
                "feature_name": feature.feature_name,
                "body_region": feature.body_region,
                "feature_family": feature.feature_family,
                "geometry_completeness": feature.geometry_completeness,
                "dynamic_completeness": feature.dynamic_completeness,
                "minimum": feature.minimum.value,
                "minimum_event_relative_ms": feature.minimum.event_relative_ms,
                "minimum_source_frame_index": feature.minimum.source_frame_index,
                "maximum": feature.maximum.value,
                "maximum_event_relative_ms": feature.maximum.event_relative_ms,
                "maximum_source_frame_index": feature.maximum.source_frame_index,
                "range": feature.range,
                "mean": feature.mean,
                "value_at_t0": feature.value_at_t0.value,
                "t0_status": feature.value_at_t0.feature_status,
                "pre_late_mean": pre_late.mean if pre_late else None,
                "pre_late_change": pre_late.change if pre_late else None,
                "peak_robust_dynamic_rate": feature.peak_robust_rate.value,
                "peak_robust_rate_event_relative_ms": feature.peak_robust_rate.event_relative_ms,
                "peak_robust_rate_source_frame_index": feature.peak_robust_rate.source_frame_index,
                "bilateral_summary_if_applicable": json.dumps(
                    _json_ready(bilateral_summary),
                    sort_keys=True,
                ),
                "analytics_eligibility": feature.analytics_eligibility,
                "geometry_analytics_eligible": feature.geometry_analytics_eligible,
                "dynamic_analytics_eligible": feature.dynamic_analytics_eligible,
                "analytics_eligible": (
                    feature.geometry_analytics_eligible or feature.dynamic_analytics_eligible
                ),
                "eligibility_reason": feature.eligibility_reason,
                "quality_category": feature.quality_category,
                "primary_rejection_reason": feature.primary_rejection_reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["case_id", "feature_name"]).reset_index(drop=True)


def write_profile_json(profile: MovementProfile, path: str | Path) -> Path:
    """Write a MovementProfile JSON file."""

    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(profile.to_dict()), indent=2, allow_nan=False), encoding="utf-8")
    return output


def write_evidence_profile_json(evidence_profile: EvidenceProfile, path: str | Path) -> Path:
    """Write an EvidenceProfile JSON file."""

    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(asdict(evidence_profile)), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


def _build_feature_profile(
    feature_name: str,
    rows: pd.DataFrame,
    thresholds: ProfileThresholds,
) -> FeatureMovementProfile:
    rows = rows.sort_values("event_relative_ms")
    body_region, family = classify_feature(feature_name)
    supported = rows[rows["feature_status"].eq(GEOMETRY_SUPPORTED_STATUS)]
    geometry_completeness = len(supported) / len(rows) if len(rows) else 0.0
    dynamic_eligible_rows = rows[
        rows["raw_dynamic_status"].ne("NOT_DYNAMIC_FEATURE")
        & rows["feature_status"].eq(GEOMETRY_SUPPORTED_STATUS)
    ]
    robust_supported = dynamic_eligible_rows[
        dynamic_eligible_rows["dynamic_status"].eq(ROBUST_SUPPORTED_STATUS)
    ]
    dynamic_completeness = (
        len(robust_supported) / len(dynamic_eligible_rows) if len(dynamic_eligible_rows) else 0.0
    )
    geometry_status = _geometry_status(geometry_completeness, supported, thresholds)
    dynamic_status = _dynamic_status(dynamic_completeness, dynamic_eligible_rows, robust_supported, thresholds)
    eligibility, geometry_ready, dynamic_ready, eligibility_reason = _eligibility(
        geometry_completeness,
        len(supported),
        dynamic_completeness,
        len(robust_supported),
        len(dynamic_eligible_rows),
        thresholds,
    )
    quality_category = _quality_category(geometry_status, dynamic_status, eligibility)
    minimum = _extreme_trace(supported, "idxmin")
    maximum = _extreme_trace(supported, "idxmax")
    peak_rate = _peak_robust_rate_trace(robust_supported)
    value_at_t0 = _t0_trace(rows)
    baseline = _baseline_value(rows)
    change_baseline = (
        value_at_t0.value - baseline
        if value_at_t0.value is not None and baseline is not None
        else None
    )
    primary_reason = _primary_reason(rows)
    landmarks = _landmarks_used(rows)
    return FeatureMovementProfile(
        feature_name=feature_name,
        body_region=body_region.value,
        feature_family=family.value,
        unit=str(rows.iloc[0]["unit"]),
        geometry_status=geometry_status,
        geometry_completeness=float(geometry_completeness),
        dynamic_status=dynamic_status,
        dynamic_completeness=float(dynamic_completeness),
        quality_category=quality_category.value,
        minimum=minimum,
        maximum=maximum,
        range=_range(supported["feature_value"]),
        mean=_mean(supported["feature_value"]),
        value_at_t0=value_at_t0,
        baseline_value=baseline,
        change_baseline_to_t0=change_baseline,
        peak_robust_rate=peak_rate,
        window_summaries=tuple(_window_summaries(rows)),
        primary_rejection_reason=primary_reason,
        landmarks_used=landmarks,
        analytics_eligibility=eligibility.value,
        geometry_analytics_eligible=geometry_ready,
        dynamic_analytics_eligible=dynamic_ready,
        eligibility_reason=eligibility_reason,
        notes=_feature_note(feature_name, geometry_status, dynamic_status, primary_reason),
    )


def _window_summaries(rows: pd.DataFrame) -> list[ProfileWindowSummary]:
    windows = (
        ("PRE_EARLY", -500.0, -250.0),
        ("PRE_LATE", -250.0, 0.0),
        ("EARLY_POST", 0.0, 100.0),
        ("POST", 100.0, 200.0),
    )
    summaries: list[ProfileWindowSummary] = []
    for name, start, end in windows:
        window = rows[rows["event_relative_ms"].ge(start) & rows["event_relative_ms"].lt(end)]
        supported = window[window["feature_status"].eq(GEOMETRY_SUPPORTED_STATUS)]
        completeness = len(supported) / len(window) if len(window) else 0.0
        robust = window[window["dynamic_status"].eq(ROBUST_SUPPORTED_STATUS)]
        summaries.append(
            ProfileWindowSummary(
                window_name=name,
                window_start_ms=start,
                window_end_ms=end,
                geometry_status="SUPPORTED" if len(window) and completeness >= 0.5 else "INSUFFICIENT_COMPLETENESS",
                geometry_completeness=float(completeness),
                mean=_mean(supported["feature_value"]),
                range=_range(supported["feature_value"]),
                change=_change(supported["feature_value"]),
                robust_max_rate=_max_abs(robust["robust_dynamic_rate"]),
                time_robust_max_rate_ms=_time_of_abs_max(robust, "robust_dynamic_rate"),
            )
        )
    return summaries


def _build_evidence_profile(
    dynamic_df: pd.DataFrame,
    features: tuple[FeatureMovementProfile, ...],
    annotation: EventAnnotation,
    reliability: dict,
    manual_roi_keyframe_count: int | None,
) -> EvidenceProfile:
    supported = [f for f in features if f.quality_category == QualityCategory.SUPPORTED.value]
    limited = [f for f in features if f.quality_category == QualityCategory.LIMITED.value]
    unavailable = [f for f in features if f.quality_category == QualityCategory.UNAVAILABLE.value]
    ready = [f for f in features if f.analytics_eligibility == AnalyticsEligibility.ANALYTICS_READY.value]
    frame_counts = reliability.get("frame_status_counts", {})
    total_frames = reliability.get("total_frames") or sum(frame_counts.values()) or None
    limitations = _primary_limitations(dynamic_df, reliability, features)
    geometry_coverage = float(np.mean([f.geometry_completeness for f in features])) if features else 0.0
    dynamic_coverage = float(np.mean([f.dynamic_completeness for f in features])) if features else 0.0
    overview = EvidenceOverview(
        supported_features=len(supported),
        limited_features=len(limited),
        unavailable_features=len(unavailable),
        geometry_coverage=geometry_coverage,
        dynamic_coverage=dynamic_coverage,
        event_anchor_confidence=annotation.annotation_confidence,
        primary_limitations=tuple(limitations),
        human_target_verified=annotation.annotation_method.startswith("manual")
        or bool(manual_roi_keyframe_count),
        manual_roi_corrections=manual_roi_keyframe_count,
        overall_quality_label=_overall_quality_label(len(supported), len(limited), len(features), geometry_coverage),
        component_breakdown={
            "supported_fraction": len(supported) / len(features) if features else 0.0,
            "limited_fraction": len(limited) / len(features) if features else 0.0,
            "geometry_coverage": geometry_coverage,
            "dynamic_coverage": dynamic_coverage,
        },
    )
    first = dynamic_df.iloc[0]
    return EvidenceProfile(
        case_id=str(first["case_id"]),
        source_id=str(first["source_id"]),
        view_id=str(first["view_id"]),
        target_annotation_method=annotation.annotation_method,
        manual_roi_keyframe_count=manual_roi_keyframe_count,
        target_tracking_coverage=_optional_float(reliability.get("target_tracking_coverage")),
        pose_frame_coverage=_optional_float(reliability.get("pose_frame_coverage")),
        upper_body_landmark_coverage=_landmark_group_coverage(
            reliability, ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
        ),
        core_landmark_coverage=_landmark_group_coverage(
            reliability, ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        ),
        lower_limb_landmark_coverage=_landmark_group_coverage(
            reliability, ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
        ),
        geometry_feature_coverage=geometry_coverage,
        dynamic_feature_coverage=dynamic_coverage,
        interpolation_fraction=_optional_float(reliability.get("interpolated_landmark_row_fraction")),
        rejected_fraction=_optional_float(reliability.get("rejected_landmark_row_fraction")),
        identity_uncertainty_fraction=_frame_fraction(frame_counts, total_frames, "TARGET_IDENTITY_UNCERTAIN"),
        target_loss_fraction=_frame_fraction(frame_counts, total_frames, "TARGET_NOT_FOUND"),
        supported_feature_count=len(supported),
        limited_feature_count=len(limited),
        unavailable_feature_count=len(unavailable),
        analytics_ready_feature_count=len(ready),
        primary_evidence_limitations=tuple(limitations),
        evidence_overview=overview,
    )


def _geometry_status(
    completeness: float,
    supported: pd.DataFrame,
    thresholds: ProfileThresholds,
) -> str:
    if supported.empty:
        return QualityCategory.UNAVAILABLE.value
    if (
        completeness >= thresholds.geometry_ready_completeness
        and len(supported) >= thresholds.minimum_supported_geometry_frames
    ):
        return QualityCategory.SUPPORTED.value
    if completeness >= thresholds.geometry_limited_completeness:
        return QualityCategory.LIMITED.value
    return QualityCategory.UNAVAILABLE.value


def _dynamic_status(
    completeness: float,
    eligible: pd.DataFrame,
    robust_supported: pd.DataFrame,
    thresholds: ProfileThresholds,
) -> str:
    if eligible.empty:
        return "NOT_DYNAMIC_FEATURE"
    if (
        completeness >= thresholds.dynamic_ready_completeness
        and len(robust_supported) >= thresholds.minimum_supported_dynamic_samples
    ):
        return QualityCategory.SUPPORTED.value
    if completeness >= thresholds.dynamic_limited_completeness or len(robust_supported) > 0:
        return QualityCategory.LIMITED.value
    return QualityCategory.UNAVAILABLE.value


def _eligibility(
    geometry_completeness: float,
    supported_geometry_frames: int,
    dynamic_completeness: float,
    supported_dynamic_samples: int,
    dynamic_eligible_samples: int,
    thresholds: ProfileThresholds,
) -> tuple[AnalyticsEligibility, bool, bool, str]:
    geometry_ready = (
        geometry_completeness >= thresholds.geometry_ready_completeness
        and supported_geometry_frames >= thresholds.minimum_supported_geometry_frames
    )
    dynamic_ready = (
        dynamic_eligible_samples > 0
        and dynamic_completeness >= thresholds.dynamic_ready_completeness
        and supported_dynamic_samples >= thresholds.minimum_supported_dynamic_samples
    )
    if not supported_geometry_frames:
        return AnalyticsEligibility.UNSUPPORTED, False, False, "No supported geometry frames."
    if geometry_ready and dynamic_ready:
        return AnalyticsEligibility.ANALYTICS_READY, True, True, "Geometry and robust dynamics meet analysis-quality thresholds."
    if geometry_ready and dynamic_eligible_samples == 0:
        return AnalyticsEligibility.GEOMETRY_ONLY, True, False, "Geometry is supported; feature is not configured for dynamics."
    if geometry_ready and not dynamic_ready:
        return AnalyticsEligibility.LOW_DYNAMIC_RELIABILITY, True, False, "Geometry is supported but robust dynamic evidence is below threshold."
    if geometry_completeness >= thresholds.geometry_limited_completeness:
        return AnalyticsEligibility.LIMITED_COVERAGE, False, False, "Geometry coverage is limited below default analytics threshold."
    return AnalyticsEligibility.EXCLUDE_FROM_DEFAULT_ANALYTICS, False, False, "Insufficient evidence for default analytics."


def _quality_category(
    geometry_status: str,
    dynamic_status: str,
    eligibility: AnalyticsEligibility,
) -> QualityCategory:
    if geometry_status == QualityCategory.UNAVAILABLE.value:
        return QualityCategory.UNAVAILABLE
    if eligibility == AnalyticsEligibility.ANALYTICS_READY:
        return QualityCategory.SUPPORTED
    if geometry_status == QualityCategory.SUPPORTED.value and dynamic_status in {
        QualityCategory.SUPPORTED.value,
        "NOT_DYNAMIC_FEATURE",
    }:
        return QualityCategory.SUPPORTED
    return QualityCategory.LIMITED


def _extreme_trace(supported: pd.DataFrame, method: str) -> TraceableValue:
    values = supported["feature_value"].dropna()
    if values.empty:
        return TraceableValue(None, None, None, None, "UNAVAILABLE")
    index = getattr(values, method)()
    row = supported.loc[index]
    return _trace(row, "feature_value")


def _peak_robust_rate_trace(robust: pd.DataFrame) -> TraceableValue:
    values = robust["robust_dynamic_rate"].dropna().abs()
    if values.empty:
        return TraceableValue(None, None, None, None, "UNAVAILABLE", "UNAVAILABLE")
    max_value = values.max()
    candidates = robust.loc[values[values.eq(max_value)].index]
    row = candidates.loc[candidates["event_relative_ms"].abs().idxmin()]
    return TraceableValue(
        value=_optional_float(row["robust_dynamic_rate"]),
        event_relative_ms=_optional_float(row["event_relative_ms"]),
        source_frame_index=_optional_int(row["source_frame_index"]),
        analysis_frame_index=_optional_int(row["analysis_frame_index"]),
        feature_status=str(row["feature_status"]),
        dynamic_status=str(row["dynamic_status"]),
    )


def _t0_trace(rows: pd.DataFrame) -> TraceableValue:
    t0 = rows[np.isclose(rows["event_relative_ms"].astype(float), 0.0)]
    if t0.empty:
        return TraceableValue(None, None, None, None, "NO_T0_FRAME")
    row = t0.iloc[0]
    if row["feature_status"] != GEOMETRY_SUPPORTED_STATUS:
        return TraceableValue(
            None,
            _optional_float(row["event_relative_ms"]),
            _optional_int(row["source_frame_index"]),
            _optional_int(row["analysis_frame_index"]),
            str(row["feature_status"]),
            str(row["dynamic_status"]),
        )
    return _trace(row, "feature_value")


def _trace(row: pd.Series, value_column: str) -> TraceableValue:
    return TraceableValue(
        value=_optional_float(row[value_column]),
        event_relative_ms=_optional_float(row["event_relative_ms"]),
        source_frame_index=_optional_int(row["source_frame_index"]),
        analysis_frame_index=_optional_int(row["analysis_frame_index"]),
        feature_status=str(row["feature_status"]),
        dynamic_status=str(row.get("dynamic_status", "")),
    )


def _baseline_value(rows: pd.DataFrame) -> float | None:
    baseline = rows[
        rows["event_relative_ms"].ge(-500.0)
        & rows["event_relative_ms"].lt(-250.0)
        & rows["feature_status"].eq(GEOMETRY_SUPPORTED_STATUS)
    ]["feature_value"]
    return _mean(baseline)


def _primary_reason(rows: pd.DataFrame) -> str:
    geometry_reasons = rows["rejection_reason"].replace("", np.nan).dropna()
    if not geometry_reasons.empty:
        return str(geometry_reasons.mode().iloc[0])
    dynamic_reasons = rows["dynamic_rejection_reason"].replace("", np.nan).dropna()
    if dynamic_reasons.empty:
        return ""
    return str(dynamic_reasons.mode().iloc[0])


def _landmarks_used(rows: pd.DataFrame) -> tuple[str, ...]:
    for value in rows["landmarks_used"]:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, list | tuple) and value:
            return tuple(str(item) for item in value)
    return ()


def _feature_note(
    feature_name: str,
    geometry_status: str,
    dynamic_status: str,
    primary_reason: str,
) -> str:
    readable = feature_name.replace("_", " ")
    if geometry_status == QualityCategory.SUPPORTED.value and dynamic_status == QualityCategory.SUPPORTED.value:
        return f"{readable} geometry and robust rate-of-change evidence are supported by current analysis-quality rules."
    if geometry_status == QualityCategory.SUPPORTED.value:
        return f"{readable} geometry is measurable, but rate-of-change evidence is limited or unavailable."
    if geometry_status == QualityCategory.LIMITED.value:
        return f"{readable} has limited coverage; interpret as partial evidence only."
    reason = primary_reason or "the required supported feature evidence was unavailable."
    return f"{readable} is unavailable or excluded because {reason}"


def _group_by_region(features: tuple[FeatureMovementProfile, ...]) -> dict[str, list[FeatureMovementProfile]]:
    grouped = {region.value: [] for region in BodyRegion}
    for feature in features:
        grouped.setdefault(feature.body_region, []).append(feature)
    return {region: items for region, items in grouped.items() if items}


def _feature_availability(features: tuple[FeatureMovementProfile, ...]) -> dict:
    return {
        "by_quality_category": _count_by([feature.quality_category for feature in features]),
        "by_body_region": {
            region: _count_by([feature.quality_category for feature in group])
            for region, group in _group_by_region(features).items()
        },
    }


def _analytics_eligibility(
    features: tuple[FeatureMovementProfile, ...],
    thresholds: ProfileThresholds,
) -> dict:
    return {
        "thresholds": asdict(thresholds),
        "by_state": _count_by([feature.analytics_eligibility for feature in features]),
        "geometry_analytics_ready_features": [
            feature.feature_name for feature in features if feature.geometry_analytics_eligible
        ],
        "dynamic_analytics_ready_features": [
            feature.feature_name for feature in features if feature.dynamic_analytics_eligible
        ],
        "excluded_from_default_dynamic_analytics": [
            feature.feature_name for feature in features if not feature.dynamic_analytics_eligible
        ],
    }


def _movement_summary(
    features: tuple[FeatureMovementProfile, ...],
    evidence: EvidenceProfile,
) -> dict:
    strongest = sorted(
        features,
        key=lambda item: (item.quality_category == QualityCategory.SUPPORTED.value, item.geometry_completeness, item.dynamic_completeness),
        reverse=True,
    )[:6]
    excluded = [
        feature.feature_name
        for feature in features
        if feature.analytics_eligibility
        in {
            AnalyticsEligibility.UNSUPPORTED.value,
            AnalyticsEligibility.EXCLUDE_FROM_DEFAULT_ANALYTICS.value,
            AnalyticsEligibility.LIMITED_COVERAGE.value,
        }
    ]
    return {
        "movement_features_considered": len(features),
        "supported": evidence.supported_feature_count,
        "limited": evidence.limited_feature_count,
        "unavailable": evidence.unavailable_feature_count,
        "analytics_ready": evidence.analytics_ready_feature_count,
        "strongest_evidence_features": [feature.feature_name for feature in strongest],
        "primary_limitations": list(evidence.primary_evidence_limitations),
        "excluded_from_default_analytics": excluded,
    }


def _primary_limitations(
    dynamic_df: pd.DataFrame,
    reliability: dict,
    features: tuple[FeatureMovementProfile, ...],
) -> list[str]:
    limitations: list[str] = []
    frame_counts = reliability.get("frame_status_counts", {})
    if frame_counts.get("TARGET_NOT_FOUND", 0):
        limitations.append("target loss intervals")
    if frame_counts.get("TARGET_IDENTITY_UNCERTAIN", 0):
        limitations.append("target identity uncertainty")
    if frame_counts.get("LOW_POSE_CONFIDENCE", 0):
        limitations.append("low pose confidence intervals")
    dynamic_reasons = (
        dynamic_df["dynamic_rejection_reason"].replace("", np.nan).dropna().value_counts().head(2)
    )
    limitations.extend(str(reason) for reason in dynamic_reasons.index)
    unavailable = [feature.primary_rejection_reason for feature in features if feature.primary_rejection_reason]
    if unavailable:
        limitations.append(pd.Series(unavailable).mode().iloc[0])
    return list(dict.fromkeys(limitations))[:6]


def _landmark_group_coverage(reliability: dict, landmarks: tuple[str, ...]) -> float | None:
    values = [
        reliability.get(f"{landmark}_coverage")
        for landmark in landmarks
        if reliability.get(f"{landmark}_coverage") is not None
    ]
    return float(np.mean(values)) if values else None


def _frame_fraction(frame_counts: dict, total_frames: int | None, status: str) -> float | None:
    if not total_frames:
        return None
    return float(frame_counts.get(status, 0) / total_frames)


def _overall_quality_label(
    supported_count: int,
    limited_count: int,
    total_count: int,
    geometry_coverage: float,
) -> str:
    supported_fraction = supported_count / total_count if total_count else 0.0
    if supported_fraction >= 0.75 and geometry_coverage >= 0.65:
        return "HIGH EVIDENCE"
    if supported_fraction >= 0.5 and geometry_coverage >= 0.45:
        return "MODERATE EVIDENCE"
    if supported_count or limited_count:
        return "LIMITED EVIDENCE"
    return "UNAVAILABLE EVIDENCE"


def _window_by_name(feature: FeatureMovementProfile, name: str) -> ProfileWindowSummary | None:
    for window in feature.window_summaries:
        if window.window_name == name:
            return window
    return None


def _mean(values: pd.Series) -> float | None:
    clean = values.dropna()
    return float(clean.mean()) if not clean.empty else None


def _range(values: pd.Series) -> float | None:
    clean = values.dropna()
    return float(clean.max() - clean.min()) if not clean.empty else None


def _change(values: pd.Series) -> float | None:
    clean = values.dropna()
    return float(clean.iloc[-1] - clean.iloc[0]) if len(clean) >= 2 else None


def _max_abs(values: pd.Series) -> float | None:
    clean = values.dropna().abs()
    return float(clean.max()) if not clean.empty else None


def _time_of_abs_max(rows: pd.DataFrame, column: str) -> float | None:
    values = rows[column].dropna().abs()
    if values.empty:
        return None
    return _optional_float(rows.loc[values.idxmax(), "event_relative_ms"])


def _count_by(values: list[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in pd.Series(values).value_counts().sort_index().items()}


def _optional_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_dynamic_input(dynamic_df: pd.DataFrame) -> None:
    required = {
        "case_id",
        "source_id",
        "view_id",
        "feature_name",
        "feature_value",
        "feature_status",
        "dynamic_status",
        "raw_dynamic_status",
        "robust_dynamic_rate",
        "event_relative_ms",
        "source_frame_index",
        "analysis_frame_index",
        "landmarks_used",
    }
    missing = sorted(required.difference(dynamic_df.columns))
    if missing:
        raise ValueError(f"Dynamic feature table is missing required columns: {missing}")


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
