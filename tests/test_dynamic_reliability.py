from __future__ import annotations

import pandas as pd

from acl_motion.cases.annotations import AnchorType, EventAnnotation
from acl_motion.events.dynamic_reliability import (
    RobustDynamicStatus,
    build_dynamic_spike_audit,
    harden_dynamic_reliability,
)
from acl_motion.events.temporal import build_event_relative_features


def test_stable_linear_trajectory_raw_and_robust_agree() -> None:
    dynamic = _hardened("left_hka_angle_2d_deg", [0, 10, 20, 30])
    supported = dynamic[dynamic["dynamic_status"].eq("SUPPORTED")]

    assert not supported.empty
    assert supported["robust_dynamic_rate"].dropna().sub(100).abs().max() < 1e-6
    raw = dynamic[dynamic["raw_dynamic_status"].eq("SUPPORTED")]["raw_first_difference_rate"]
    assert raw.sub(100).abs().max() < 1e-6


def test_single_frame_spike_preserves_raw_and_flags_dynamic_qc() -> None:
    dynamic = _hardened("left_hka_angle_2d_deg", [20, 21, 80, 22, 23])
    spike_rows = dynamic[dynamic["raw_first_difference_rate"].abs().ge(500)]

    assert not spike_rows.empty
    assert RobustDynamicStatus.TEMPORAL_OUTLIER.value in set(spike_rows["dynamic_status"])
    assert spike_rows["raw_first_difference_rate"].abs().max() == 590.0


def test_genuine_rapid_smooth_movement_remains_supported() -> None:
    dynamic = _hardened("left_hka_angle_2d_deg", [20, 30, 45, 65, 90])
    supported = dynamic[dynamic["dynamic_status"].eq("SUPPORTED")]

    assert len(supported) >= 3
    assert dynamic["robust_dynamic_rate"].dropna().max() > 100
    assert "TEMPORAL_OUTLIER" not in set(dynamic["dynamic_status"])


def test_robust_derivative_does_not_span_feature_segments() -> None:
    event = _event_df(
        "left_hka_angle_2d_deg",
        frames=[1, 2, 5, 6],
        timestamps=[0, 100, 400, 500],
        values=[0, 10, 50, 60],
        statuses=["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"],
    )
    event.loc[event["source_frame_index"].isin([1, 2]), "feature_segment_id"] = (
        "left_hka_angle_2d_deg_segment_001"
    )
    event.loc[event["source_frame_index"].isin([5, 6]), "feature_segment_id"] = (
        "left_hka_angle_2d_deg_segment_002"
    )
    dynamic = harden_dynamic_reliability(event)

    assert dynamic["dynamic_status"].eq("INSUFFICIENT_NEIGHBORHOOD").all()
    assert dynamic["dynamic_window_start_frame"].max() <= 6


def test_missing_neighborhood_returns_insufficient_neighborhood() -> None:
    dynamic = _hardened("left_hka_angle_2d_deg", [0, 10])

    assert dynamic["dynamic_status"].eq("INSUFFICIENT_NEIGHBORHOOD").all()


def test_irregular_timestamps_affect_robust_slope() -> None:
    event = _event_df(
        "left_hka_angle_2d_deg",
        frames=[1, 2, 3],
        timestamps=[0, 50, 200],
        values=[0, 5, 20],
    )
    dynamic = harden_dynamic_reliability(event)

    assert dynamic["robust_dynamic_rate"].dropna().iloc[1] == 100.0


def test_orientation_wrapping_remains_stable() -> None:
    dynamic = _hardened("projected_hip_line_angle_deg", [178, 179, -179, -178])

    assert dynamic["robust_dynamic_rate"].dropna().abs().max() < 25
    assert dynamic["raw_first_difference_rate"].dropna().abs().max() < 25


def test_metadata_and_spike_audit_preserve_traceability() -> None:
    dynamic = _hardened("left_hka_angle_2d_deg", [20, 21, 80, 22, 23])
    audit = build_dynamic_spike_audit(dynamic, features=("left_hka_angle_2d_deg",), top_n=2)

    assert {
        "source_frame_prev",
        "source_frame_current",
        "event_relative_ms",
        "dynamic_status",
        "local_jitter_metric",
        "landmarks_used",
    }.issubset(audit.columns)
    assert not audit.empty


def _hardened(feature_name: str, values: list[float]) -> pd.DataFrame:
    frames = list(range(1, len(values) + 1))
    timestamps = [i * 100 for i in range(len(values))]
    return harden_dynamic_reliability(
        _event_df(feature_name, frames=frames, timestamps=timestamps, values=values)
    )


def _event_df(
    feature_name: str,
    *,
    frames: list[int],
    timestamps: list[float],
    values: list[float],
    statuses: list[str] | None = None,
) -> pd.DataFrame:
    base = _feature_df(feature_name, frames, timestamps, values, statuses=statuses)
    return build_event_relative_features(base, _annotation(frames[0]))


def _annotation(anchor_frame: int) -> EventAnnotation:
    return EventAnnotation(
        case_id="case",
        source_id="source",
        view_id="source",
        event_anchor_frame=anchor_frame,
        event_anchor_type=AnchorType.CRITICAL_PLANT,
        annotation_confidence=1.0,
    )


def _feature_df(
    feature_name: str,
    frames: list[int],
    timestamps: list[float],
    values: list[float],
    *,
    statuses: list[str] | None = None,
) -> pd.DataFrame:
    statuses = statuses or ["SUPPORTED"] * len(frames)
    rows = []
    for frame, timestamp, value, status in zip(frames, timestamps, values, statuses, strict=True):
        rows.append(
            {
                "case_id": "case",
                "source_id": "source",
                "frame_index": frame,
                "source_frame_index": frame,
                "analysis_frame_index": frame - frames[0],
                "timestamp_ms": timestamp,
                "feature_name": feature_name,
                "feature_value": value,
                "unit": "deg" if feature_name.endswith("_deg") else "body_scale",
                "status": status,
                "quality_status": "frame:VALID_TARGET",
                "landmarks_used": ["a", "b"],
                "completeness": 1.0 if status == "SUPPORTED" else 0.5,
                "observed": status == "SUPPORTED",
                "input_interpolated": False,
                "input_smoothed": True,
                "view_suitability": "not_assessed",
                "rejection_reason": "" if status == "SUPPORTED" else "test unavailable",
                "frame_status": "VALID_TARGET",
                "metadata": {},
            }
        )
    return pd.DataFrame(rows)
