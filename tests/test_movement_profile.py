from __future__ import annotations

import pandas as pd

from acl_motion.cases.annotations import AnchorType, EventAnnotation
from acl_motion.profiles.builder import build_case_feature_summary, build_movement_profile
from acl_motion.profiles.models import (
    AnalyticsEligibility,
    BodyRegion,
    FeatureFamily,
    QualityCategory,
)
from acl_motion.profiles.registry import classify_feature


def test_profile_construction_populates_supported_summary_fields() -> None:
    profile = build_movement_profile(_dynamic_df("left_hka_angle_2d_deg"), event_annotation=_annotation())
    feature = profile.trajectory_summaries[0]

    assert feature.geometry_status == QualityCategory.SUPPORTED.value
    assert feature.dynamic_status == QualityCategory.SUPPORTED.value
    assert feature.maximum.value == 50.0
    assert feature.maximum.source_frame_index == 5
    assert feature.peak_robust_rate.value == 100.0


def test_geometry_only_feature_keeps_geometry_eligible() -> None:
    df = _dynamic_df("left_knee_line_deviation_2d")
    df["raw_dynamic_status"] = "NOT_DYNAMIC_FEATURE"
    df["dynamic_status"] = "NOT_DYNAMIC_FEATURE"
    df["robust_dynamic_rate"] = pd.NA
    profile = build_movement_profile(df, event_annotation=_annotation())
    feature = profile.trajectory_summaries[0]

    assert feature.geometry_analytics_eligible is True
    assert feature.dynamic_analytics_eligible is False
    assert feature.analytics_eligibility == AnalyticsEligibility.GEOMETRY_ONLY.value


def test_unsupported_feature_is_unavailable_with_reason() -> None:
    df = _dynamic_df("left_hka_angle_2d_deg", statuses=["LOW_CONFIDENCE"] * 5)
    profile = build_movement_profile(df, event_annotation=_annotation())
    feature = profile.trajectory_summaries[0]

    assert feature.quality_category == QualityCategory.UNAVAILABLE.value
    assert feature.value_at_t0.value is None
    assert feature.primary_rejection_reason == "low confidence"


def test_t0_unavailable_is_preserved() -> None:
    df = _dynamic_df("left_hka_angle_2d_deg")
    df.loc[df["event_relative_ms"].eq(0), "feature_status"] = "INVALID_TARGET_FRAME"
    df.loc[df["event_relative_ms"].eq(0), "feature_value"] = pd.NA
    profile = build_movement_profile(df, event_annotation=_annotation())
    feature = profile.trajectory_summaries[0]

    assert feature.value_at_t0.value is None
    assert feature.value_at_t0.feature_status == "INVALID_TARGET_FRAME"


def test_feature_completeness_uses_supported_frame_fraction() -> None:
    statuses = ["SUPPORTED", "SUPPORTED", "LOW_CONFIDENCE", "SUPPORTED", "LOW_CONFIDENCE"]
    profile = build_movement_profile(_dynamic_df("left_hka_angle_2d_deg", statuses=statuses), event_annotation=_annotation())

    assert profile.trajectory_summaries[0].geometry_completeness == 0.6


def test_evidence_categories_map_transparently() -> None:
    supported = build_movement_profile(_dynamic_df("left_hka_angle_2d_deg"), event_annotation=_annotation())
    limited = build_movement_profile(
        _dynamic_df("left_hka_angle_2d_deg", statuses=["SUPPORTED", "LOW_CONFIDENCE", "LOW_CONFIDENCE", "LOW_CONFIDENCE", "SUPPORTED"]),
        event_annotation=_annotation(),
    )

    assert supported.trajectory_summaries[0].quality_category == QualityCategory.SUPPORTED.value
    assert limited.trajectory_summaries[0].quality_category == QualityCategory.LIMITED.value


def test_rejection_reason_aggregation() -> None:
    df = _dynamic_df("left_hka_angle_2d_deg", statuses=["LOW_CONFIDENCE"] * 5)
    df["rejection_reason"] = ["target loss", "target loss", "low confidence", "target loss", "low confidence"]
    profile = build_movement_profile(df, event_annotation=_annotation())

    assert profile.trajectory_summaries[0].primary_rejection_reason == "target loss"


def test_peak_traceability_preserves_source_frame_and_event_time() -> None:
    profile = build_movement_profile(_dynamic_df("left_hka_angle_2d_deg"), event_annotation=_annotation())
    feature = profile.trajectory_summaries[0]

    assert feature.peak_robust_rate.source_frame_index == 3
    assert feature.peak_robust_rate.event_relative_ms == 0.0


def test_carpenter_style_low_coverage_still_builds_profile() -> None:
    statuses = ["INVALID_TARGET_FRAME", "SUPPORTED", "SUPPORTED", "INVALID_TARGET_FRAME", "TARGET_NOT_FOUND"]
    profile = build_movement_profile(_dynamic_df("left_hka_angle_2d_deg", statuses=statuses), event_annotation=_annotation())
    feature = profile.trajectory_summaries[0]

    assert profile.case["case_id"] == "case"
    assert feature.quality_category in {QualityCategory.LIMITED.value, QualityCategory.UNAVAILABLE.value}


def test_low_dynamic_confidence_is_not_dynamic_analytics_ready() -> None:
    df = _dynamic_df("left_hka_angle_2d_deg")
    df["dynamic_status"] = "LOW_DYNAMIC_CONFIDENCE"
    df["robust_dynamic_rate"] = 10.0
    profile = build_movement_profile(df, event_annotation=_annotation())
    feature = profile.trajectory_summaries[0]

    assert feature.geometry_analytics_eligible is True
    assert feature.dynamic_analytics_eligible is False


def test_case_feature_summary_has_eligibility_columns() -> None:
    profile = build_movement_profile(_dynamic_df("left_hka_angle_2d_deg"), event_annotation=_annotation())
    summary = build_case_feature_summary(profile)

    assert {"geometry_analytics_eligible", "dynamic_analytics_eligible", "analytics_eligibility"}.issubset(summary.columns)
    assert summary["bilateral_summary_if_applicable"].iloc[0] == "{}"


def test_human_roi_keyframes_mark_target_as_verified() -> None:
    annotation = EventAnnotation(
        case_id="case",
        source_id="source",
        view_id="source",
        event_anchor_frame=3,
        event_anchor_type=AnchorType.ESTIMATED_EVENT_END,
        annotation_confidence=0.6,
        annotation_method="human_movement_window_compatibility",
    )

    profile = build_movement_profile(
        _dynamic_df("left_hka_angle_2d_deg"),
        event_annotation=annotation,
        manual_roi_keyframe_count=4,
    )

    assert profile.evidence_profile.evidence_overview.human_target_verified is True


def test_wrist_relative_to_pelvis_groups_with_upper_body() -> None:
    body_region, feature_family = classify_feature("left_wrist_pelvis_x_offset_normalized")

    assert body_region == BodyRegion.UPPER_BODY
    assert feature_family == FeatureFamily.WRIST_GEOMETRY


def _annotation() -> EventAnnotation:
    return EventAnnotation(
        case_id="case",
        source_id="source",
        view_id="source",
        event_anchor_frame=3,
        event_anchor_type=AnchorType.CRITICAL_PLANT,
        annotation_confidence=0.8,
        annotation_method="manual_frame_review",
    )


def _dynamic_df(feature_name: str, statuses: list[str] | None = None) -> pd.DataFrame:
    statuses = statuses or ["SUPPORTED"] * 5
    rows = []
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for i, status in enumerate(statuses, start=1):
        supported = status == "SUPPORTED"
        rows.append(
            {
                "case_id": "case",
                "source_id": "source",
                "view_id": "source",
                "frame_index": i,
                "source_frame_index": i,
                "analysis_frame_index": i - 1,
                "timestamp_ms": (i - 1) * 100.0,
                "event_relative_ms": (i - 3) * 100.0,
                "event_anchor_frame": 3,
                "event_anchor_type": "critical_plant",
                "feature_name": feature_name,
                "feature_value": values[i - 1] if supported else pd.NA,
                "unit": "deg",
                "feature_status": status,
                "quality_status": "frame:VALID_TARGET",
                "completeness": 1.0 if supported else 0.0,
                "landmarks_used": ["hip", "knee", "ankle"],
                "input_interpolated": False,
                "input_smoothed": True,
                "rejection_reason": "" if supported else "low confidence",
                "frame_status": "VALID_TARGET",
                "metadata": {},
                "feature_segment_id": f"{feature_name}_segment_001" if supported else "",
                "dynamic_value": 100.0 if i > 1 and supported else pd.NA,
                "dynamic_unit": "deg/s",
                "dynamic_status": "SUPPORTED" if supported else "MISSING_FEATURE",
                "dynamic_rejection_reason": "" if supported else "Feature value is unavailable.",
                "raw_first_difference_rate": 100.0 if i > 1 and supported else pd.NA,
                "raw_dynamic_status": "SUPPORTED" if i > 1 and supported else "INSUFFICIENT_PREVIOUS_POINT",
                "raw_dynamic_rejection_reason": "",
                "robust_dynamic_rate": 100.0 if supported else pd.NA,
                "robust_dynamic_unit": "deg/s",
                "dynamic_quality": "HIGH" if supported else "UNAVAILABLE",
                "dynamic_method": "local_linear_regression_slope",
                "dynamic_parameters": {},
                "dynamic_window_start_frame": 1,
                "dynamic_window_end_frame": 5,
                "dynamic_window_start_ms": 0.0,
                "dynamic_window_end_ms": 400.0,
                "dynamic_valid_samples": 5 if supported else 0,
                "local_residual": 0.0,
                "local_jitter_metric": 0.0,
                "temporal_stability_score": 0.0,
            }
        )
    return pd.DataFrame(rows)
