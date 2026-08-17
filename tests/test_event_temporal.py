from __future__ import annotations

import math

import numpy as np
import pandas as pd

from acl_motion.cases.annotations import AnchorType, EventAnnotation
from acl_motion.events.temporal import (
    EventWindow,
    build_event_relative_features,
    build_event_summary,
    build_window_summaries,
)


def test_event_relative_time_uses_anchor_timestamp() -> None:
    df = _feature_df("left_hka_angle_2d_deg", [10, 11, 12], [100.0, 133.0, 166.0], [1, 2, 3])
    event = build_event_relative_features(df, _annotation(11))

    times = event.drop_duplicates("source_frame_index")["event_relative_ms"].tolist()

    assert times == [-33.0, 0.0, 33.0]


def test_angular_velocity_uses_elapsed_seconds() -> None:
    df = _feature_df("left_hka_angle_2d_deg", [1, 2, 3], [0.0, 100.0, 200.0], [10, 20, 35])
    event = build_event_relative_features(df, _annotation(1))

    values = event["dynamic_value"].tolist()

    assert math.isnan(values[0])
    assert values[1] == 100.0
    assert values[2] == 150.0


def test_irregular_time_intervals_use_actual_time() -> None:
    df = _feature_df("left_hka_angle_2d_deg", [1, 2, 3], [0.0, 50.0, 200.0], [0, 5, 20])
    event = build_event_relative_features(df, _annotation(1))

    assert event.iloc[1]["dynamic_value"] == 100.0
    assert event.iloc[2]["dynamic_value"] == 100.0


def test_no_derivative_across_segment_boundary() -> None:
    df = _feature_df(
        "left_hka_angle_2d_deg",
        [1, 2, 3, 4],
        [0.0, 100.0, 200.0, 300.0],
        [10, 20, np.nan, 40],
        statuses=["SUPPORTED", "SUPPORTED", "LOW_CONFIDENCE", "SUPPORTED"],
    )
    event = build_event_relative_features(df, _annotation(1))

    assert event.iloc[1]["dynamic_status"] == "SUPPORTED"
    assert event.iloc[3]["dynamic_status"] == "INSUFFICIENT_PREVIOUS_POINT"
    assert math.isnan(event.iloc[3]["dynamic_value"])


def test_missing_geometry_produces_missing_dynamics() -> None:
    df = _feature_df(
        "left_hka_angle_2d_deg",
        [1, 2],
        [0.0, 100.0],
        [10, np.nan],
        statuses=["SUPPORTED", "INSUFFICIENT_LANDMARKS"],
    )
    event = build_event_relative_features(df, _annotation(1))

    assert event.iloc[1]["dynamic_status"] == "UNSUPPORTED_FEATURE"
    assert math.isnan(event.iloc[1]["dynamic_value"])


def test_orientation_wrap_does_not_create_large_velocity_jump() -> None:
    df = _feature_df(
        "projected_hip_line_angle_deg",
        [1, 2, 3, 4],
        [0.0, 100.0, 200.0, 300.0],
        [178.0, 179.0, -179.0, -178.0],
    )
    event = build_event_relative_features(df, _annotation(1))

    assert event.iloc[1]["dynamic_value"] == 10.0
    assert event.iloc[2]["dynamic_value"] == 20.0
    assert event.iloc[3]["dynamic_value"] == 10.0


def test_window_summaries_statuses() -> None:
    df = _feature_df(
        "left_hka_angle_2d_deg",
        [1, 2, 3, 4],
        [-100.0, -50.0, 0.0, 50.0],
        [10.0, 20.0, np.nan, 40.0],
        statuses=["SUPPORTED", "SUPPORTED", "LOW_CONFIDENCE", "SUPPORTED"],
    )
    event = build_event_relative_features(df, _annotation(3))
    summaries = build_window_summaries(
        event,
        windows=(
            EventWindow("FULL", -100.0, 0.0),
            EventWindow("PARTIAL_OK", 0.0, 100.0),
            EventWindow("INSUFFICIENT", -100.0, 100.0),
            EventWindow("EMPTY", 500.0, 600.0),
        ),
        minimum_completeness=0.75,
    ).set_index("window_name")

    assert summaries.loc["FULL", "window_status"] == "SUPPORTED"
    assert summaries.loc["FULL", "mean"] == 15.0
    assert summaries.loc["PARTIAL_OK", "window_status"] == "INSUFFICIENT_COMPLETENESS"
    assert math.isnan(summaries.loc["PARTIAL_OK", "mean"])
    assert summaries.loc["INSUFFICIENT", "window_status"] == "SUPPORTED"
    assert summaries.loc["EMPTY", "window_status"] == "EMPTY_WINDOW"


def test_t0_unavailable_remains_unavailable() -> None:
    df = _feature_df(
        "left_hka_angle_2d_deg",
        [1, 2],
        [0.0, 100.0],
        [10.0, np.nan],
        statuses=["SUPPORTED", "LOW_CONFIDENCE"],
    )
    event = build_event_relative_features(df, _annotation(2))
    windows = build_window_summaries(event)
    summary = build_event_summary(
        event,
        windows,
        _annotation(2),
        event_annotation_file="annotation.json",
        feature_input_file="features.parquet",
    )

    feature_summary = summary["feature_summaries"]["left_hka_angle_2d_deg"]
    assert math.isnan(feature_summary["value_at_t0"])
    assert feature_summary["value_at_t0_status"] == "UNAVAILABLE_AT_T0"


def test_bilateral_dynamics_summary() -> None:
    df = _feature_df(
        "hka_projected_bilateral_difference_deg",
        [1, 2, 3],
        [0.0, 100.0, 200.0],
        [5.0, -10.0, 20.0],
    )
    event = build_event_relative_features(df, _annotation(2))
    windows = build_window_summaries(event, windows=(EventWindow("PRE_EARLY", -150.0, -50.0),))
    summary = build_event_summary(
        event,
        windows,
        _annotation(2),
        event_annotation_file="annotation.json",
        feature_input_file="features.parquet",
    )

    bilateral = summary["bilateral_summaries"]["hka_projected_bilateral_difference_deg"]
    assert bilateral["mean_signed_difference"] == 5.0
    assert bilateral["peak_absolute_difference"] == 20.0
    assert bilateral["time_peak_absolute_difference_ms"] == 100.0
    assert bilateral["difference_at_t0"] == -10.0
    assert bilateral["maximum_rate_of_change"] == 300.0


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
