"""Event-relative trajectories and first-derivative summaries for Milestone 4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from acl_motion.cases.annotations import EventAnnotation
from acl_motion.geometry.angles import wrapped_angle_difference_deg
from acl_motion.geometry.features import FEATURE_SET_VERSION, GEOMETRY_VERSION

TEMPORAL_ENGINE_VERSION = "m4_event_relative_temporal_v1"
DYNAMIC_CONFIGURATION_VERSION = "first_derivative_supported_segments_v1"
WINDOW_CONFIGURATION_VERSION = "m4_default_event_windows_v1"
SUPPORTED_STATUS = "SUPPORTED"


@dataclass(frozen=True, slots=True)
class EventWindow:
    """A half-open event-relative window in milliseconds."""

    name: str
    start_ms: float
    end_ms: float

    def contains(self, values: pd.Series) -> pd.Series:
        """Return rows inside the half-open interval [start_ms, end_ms)."""

        return values.ge(self.start_ms) & values.lt(self.end_ms)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_EVENT_WINDOWS: tuple[EventWindow, ...] = (
    EventWindow("PRE_EARLY", -500.0, -250.0),
    EventWindow("PRE_LATE", -250.0, 0.0),
    EventWindow("EARLY_POST", 0.0, 100.0),
    EventWindow("POST", 100.0, 200.0),
)

ORIENTATION_FEATURES = {
    "projected_trunk_axis_angle_deg",
    "projected_hip_line_angle_deg",
    "projected_shoulder_line_angle_deg",
    "projected_shoulder_pelvis_orientation_difference_deg",
    "left_upper_arm_orientation_2d_deg",
    "right_upper_arm_orientation_2d_deg",
}

ANGLE_RATE_FEATURES = {
    "left_hka_angle_2d_deg",
    "right_hka_angle_2d_deg",
    "injured_hka_angle_2d_deg",
    "contralateral_hka_angle_2d_deg",
    "hka_projected_bilateral_difference_deg",
    "hka_projected_bilateral_absolute_difference_deg",
    "left_elbow_angle_2d_deg",
    "right_elbow_angle_2d_deg",
    "injured_elbow_angle_2d_deg",
    "contralateral_elbow_angle_2d_deg",
    "elbow_projected_bilateral_difference_deg",
    "elbow_projected_bilateral_absolute_difference_deg",
    *ORIENTATION_FEATURES,
}

NORMALIZED_RATE_SUFFIXES = (
    "_normalized",
    "_normalized_bilateral_difference",
    "_normalized_bilateral_absolute_difference",
)

BILATERAL_FEATURES = {
    "hka_projected_bilateral_difference_deg",
    "elbow_projected_bilateral_difference_deg",
    "knee_line_deviation_bilateral_difference",
    "knee_line_deviation_normalized_bilateral_difference",
}


def build_event_relative_features(
    feature_df: pd.DataFrame,
    annotation: EventAnnotation,
) -> pd.DataFrame:
    """Add event-relative timing, valid feature segments, and first derivatives."""

    _validate_geometry_input(feature_df)
    t0_frame = annotation.t0_frame()
    if t0_frame is None:
        raise ValueError("Manual event annotation must include event_anchor_frame or critical_plant_frame.")
    t0_timestamp = _anchor_timestamp_ms(feature_df, t0_frame)
    output = _base_event_table(feature_df, annotation, t0_frame, t0_timestamp)
    output = _assign_feature_segments(output)
    output = _add_dynamic_values(output)
    return output


def build_window_summaries(
    event_df: pd.DataFrame,
    *,
    windows: tuple[EventWindow, ...] = DEFAULT_EVENT_WINDOWS,
    minimum_completeness: float = 0.5,
) -> pd.DataFrame:
    """Summarize each feature inside configurable event-relative windows."""

    rows: list[dict] = []
    for feature_name, feature_rows in event_df.groupby("feature_name", sort=True):
        for window in windows:
            in_window = feature_rows[window.contains(feature_rows["event_relative_ms"])]
            expected = len(in_window)
            supported = in_window[in_window["feature_status"].eq(SUPPORTED_STATUS)]
            valid = len(supported)
            completeness = valid / expected if expected else 0.0
            status = _window_status(expected, completeness, minimum_completeness)
            values = supported["feature_value"] if status == "SUPPORTED" else pd.Series(dtype=float)
            dynamic_values = (
                supported["dynamic_value"].dropna()
                if status == "SUPPORTED"
                else pd.Series(dtype=float)
            )
            rows.append(
                {
                    "case_id": _first_or_empty(event_df, "case_id"),
                    "source_id": _first_or_empty(event_df, "source_id"),
                    "view_id": _first_or_empty(event_df, "view_id"),
                    "feature_name": feature_name,
                    "window_name": window.name,
                    "window_start_ms": window.start_ms,
                    "window_end_ms": window.end_ms,
                    "expected_frame_count": int(expected),
                    "valid_frame_count": int(valid),
                    "window_completeness": float(completeness),
                    "window_status": status,
                    "mean": _stat(values, "mean"),
                    "minimum": _stat(values, "min"),
                    "maximum": _stat(values, "max"),
                    "range": _range(values),
                    "first_valid_value": _first_value(values),
                    "last_valid_value": _last_value(values),
                    "change_over_window": _change(values),
                    "maximum_absolute_rate_of_change": _max_abs(dynamic_values),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_name", "window_start_ms"]).reset_index(drop=True)


def build_event_summary(
    event_df: pd.DataFrame,
    window_summaries: pd.DataFrame,
    annotation: EventAnnotation,
    *,
    event_annotation_file: str,
    feature_input_file: str,
    minimum_window_completeness: float = 0.5,
    windows: tuple[EventWindow, ...] = DEFAULT_EVENT_WINDOWS,
) -> dict:
    """Build a structured per-case MovementEventSummary-like object."""

    generated_at = datetime.now(UTC).isoformat()
    feature_summaries = _feature_summaries(event_df, window_summaries)
    bilateral_summaries = {
        name: summary
        for name, summary in feature_summaries.items()
        if name in BILATERAL_FEATURES
    }
    run_metadata = {
        "run_id": (
            f"{annotation.case_id}_m4_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        ),
        "generated_at": generated_at,
        "case_id": annotation.case_id,
        "source_id": annotation.source_id,
        "view_id": annotation.view_id or annotation.source_id,
        "event_annotation_file": event_annotation_file,
        "event_anchor_frame": annotation.t0_frame(),
        "event_anchor_type": annotation.event_anchor_type.value,
        "event_anchor_confidence": annotation.annotation_confidence,
        "feature_input_file": feature_input_file,
        "geometry_version": GEOMETRY_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "temporal_engine_version": TEMPORAL_ENGINE_VERSION,
        "window_configuration": [window.to_dict() for window in windows],
        "window_configuration_version": WINDOW_CONFIGURATION_VERSION,
        "minimum_window_completeness": minimum_window_completeness,
        "dynamic_configuration": {
            "version": DYNAMIC_CONFIGURATION_VERSION,
            "derivative": "first_difference_within_supported_feature_segment",
            "timebase": "timestamp_ms",
            "orientation_wrap": "wrapped_angle_difference_deg_for_orientation_features",
        },
    }
    return {
        "run_metadata": run_metadata,
        "case_id": annotation.case_id,
        "source_id": annotation.source_id,
        "view_id": annotation.view_id or annotation.source_id,
        "event_anchor": annotation.to_dict(),
        "analysis_window_start_ms": float(event_df["event_relative_ms"].min()),
        "analysis_window_end_ms": float(event_df["event_relative_ms"].max()),
        "event_relative_feature_rows": len(event_df),
        "feature_count": int(event_df["feature_name"].nunique()),
        "supported_feature_rows": int(event_df["feature_status"].eq(SUPPORTED_STATUS).sum()),
        "supported_dynamic_rows": int(event_df["dynamic_status"].eq("SUPPORTED").sum()),
        "dynamic_completeness": _dynamic_completeness(event_df),
        "feature_summaries": feature_summaries,
        "window_summaries": _window_summary_records(window_summaries),
        "bilateral_summaries": bilateral_summaries,
        "feature_completeness": _event_feature_completeness(event_df),
        "quality_metadata": {
            "feature_status_counts": _value_counts(event_df["feature_status"]),
            "dynamic_status_counts": _value_counts(event_df["dynamic_status"]),
            "primary_rejection_reasons": _value_counts(
                event_df["rejection_reason"].replace("", np.nan).dropna()
            ),
        },
    }


def _validate_geometry_input(feature_df: pd.DataFrame) -> None:
    required = {
        "case_id",
        "source_id",
        "frame_index",
        "source_frame_index",
        "analysis_frame_index",
        "timestamp_ms",
        "feature_name",
        "feature_value",
        "unit",
        "status",
        "quality_status",
        "landmarks_used",
        "completeness",
        "input_interpolated",
        "input_smoothed",
        "rejection_reason",
        "metadata",
    }
    missing = sorted(required.difference(feature_df.columns))
    if missing:
        raise ValueError(f"Geometry feature table is missing required columns: {missing}")


def _anchor_timestamp_ms(feature_df: pd.DataFrame, t0_frame: int) -> float:
    frame_rows = feature_df[feature_df["source_frame_index"].eq(t0_frame)]
    if frame_rows.empty:
        frame_rows = feature_df[feature_df["frame_index"].eq(t0_frame)]
    if frame_rows.empty:
        raise ValueError(f"Event anchor frame {t0_frame} is not present in geometry features.")
    return float(frame_rows.iloc[0]["timestamp_ms"])


def _base_event_table(
    feature_df: pd.DataFrame,
    annotation: EventAnnotation,
    t0_frame: int,
    t0_timestamp: float,
) -> pd.DataFrame:
    output = feature_df.copy()
    output["view_id"] = annotation.view_id or annotation.source_id
    output["event_relative_ms"] = output["timestamp_ms"].astype(float) - t0_timestamp
    output["event_anchor_frame"] = int(t0_frame)
    output["event_anchor_type"] = annotation.event_anchor_type.value
    output = output.rename(columns={"status": "feature_status"})
    return output[
        [
            "case_id",
            "source_id",
            "view_id",
            "frame_index",
            "source_frame_index",
            "analysis_frame_index",
            "timestamp_ms",
            "event_relative_ms",
            "event_anchor_frame",
            "event_anchor_type",
            "feature_name",
            "feature_value",
            "unit",
            "feature_status",
            "quality_status",
            "completeness",
            "landmarks_used",
            "input_interpolated",
            "input_smoothed",
            "rejection_reason",
            "frame_status",
            "metadata",
        ]
    ]


def _assign_feature_segments(event_df: pd.DataFrame) -> pd.DataFrame:
    output = event_df.sort_values(["feature_name", "timestamp_ms"]).copy()
    output["feature_segment_id"] = ""
    for feature_name, index in output.groupby("feature_name", sort=False).groups.items():
        rows = output.loc[index].sort_values("timestamp_ms")
        segment_number = 0
        previous_supported = False
        previous_frame = None
        previous_timestamp = None
        median_dt = _median_frame_dt(rows)
        for row_index, row in rows.iterrows():
            supported = row["feature_status"] == SUPPORTED_STATUS
            contiguous = (
                previous_supported
                and previous_frame is not None
                and int(row["source_frame_index"]) == previous_frame + 1
                and previous_timestamp is not None
                and float(row["timestamp_ms"]) - previous_timestamp <= median_dt * 1.5
            )
            if supported:
                if not contiguous:
                    segment_number += 1
                output.at[row_index, "feature_segment_id"] = (
                    f"{feature_name}_segment_{segment_number:03d}"
                )
            previous_supported = supported
            previous_frame = int(row["source_frame_index"])
            previous_timestamp = float(row["timestamp_ms"])
    return output.sort_values(["feature_name", "timestamp_ms"]).reset_index(drop=True)


def _add_dynamic_values(event_df: pd.DataFrame) -> pd.DataFrame:
    output = event_df.copy()
    output["dynamic_value"] = np.nan
    output["dynamic_unit"] = ""
    output["dynamic_status"] = "NOT_DYNAMIC_FEATURE"
    output["dynamic_rejection_reason"] = ""
    for feature_name, rows in output.groupby("feature_name", sort=False):
        if not _is_dynamic_feature(feature_name):
            continue
        dynamic_unit = _dynamic_unit(rows.iloc[0]["unit"])
        output.loc[rows.index, "dynamic_unit"] = dynamic_unit
        output.loc[rows.index, "dynamic_status"] = "UNSUPPORTED_FEATURE"
        output.loc[rows.index, "dynamic_rejection_reason"] = (
            "Feature is unsupported at this frame."
        )
        for _, segment in rows[rows["feature_segment_id"].ne("")].groupby("feature_segment_id"):
            previous_index = None
            for row_index, row in segment.sort_values("timestamp_ms").iterrows():
                if previous_index is None:
                    output.at[row_index, "dynamic_status"] = "INSUFFICIENT_PREVIOUS_POINT"
                    output.at[row_index, "dynamic_rejection_reason"] = (
                        "First supported point in feature segment."
                    )
                    previous_index = row_index
                    continue
                previous = output.loc[previous_index]
                dt_s = (float(row["timestamp_ms"]) - float(previous["timestamp_ms"])) / 1000.0
                if dt_s <= 0:
                    output.at[row_index, "dynamic_status"] = "INVALID_TIME_DELTA"
                    output.at[row_index, "dynamic_rejection_reason"] = (
                        "Timestamp did not increase within feature segment."
                    )
                    previous_index = row_index
                    continue
                delta = _feature_delta(
                    feature_name,
                    float(previous["feature_value"]),
                    float(row["feature_value"]),
                )
                output.at[row_index, "dynamic_value"] = delta / dt_s
                output.at[row_index, "dynamic_status"] = "SUPPORTED"
                output.at[row_index, "dynamic_rejection_reason"] = ""
                previous_index = row_index
    return output


def _feature_summaries(event_df: pd.DataFrame, window_summaries: pd.DataFrame) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    baseline_rows = window_summaries[window_summaries["window_name"].eq("PRE_EARLY")]
    baseline_by_feature = baseline_rows.set_index("feature_name") if not baseline_rows.empty else None
    for feature_name, rows in event_df.groupby("feature_name", sort=True):
        supported = rows[rows["feature_status"].eq(SUPPORTED_STATUS)].sort_values("event_relative_ms")
        dynamics = rows[rows["dynamic_status"].eq("SUPPORTED")].sort_values("event_relative_ms")
        value_at_t0, t0_status = _value_at_t0(rows)
        baseline_mean = np.nan
        change_baseline_to_t0 = np.nan
        if baseline_by_feature is not None and feature_name in baseline_by_feature.index:
            baseline = baseline_by_feature.loc[feature_name]
            if baseline["window_status"] == "SUPPORTED":
                baseline_mean = float(baseline["mean"])
        if np.isfinite(value_at_t0) and np.isfinite(baseline_mean):
            change_baseline_to_t0 = value_at_t0 - baseline_mean
        summary = {
            "status": "SUPPORTED" if not supported.empty else "NO_SUPPORTED_VALUES",
            "supported_frames": len(supported),
            "relevant_frames": len(rows),
            "completeness": float(len(supported) / len(rows)) if len(rows) else 0.0,
            "minimum": _stat(supported["feature_value"], "min"),
            "maximum": _stat(supported["feature_value"], "max"),
            "range": _range(supported["feature_value"]),
            "mean": _stat(supported["feature_value"], "mean"),
            "standard_deviation": _stat(supported["feature_value"], "std"),
            "time_of_minimum_ms": _time_of_extreme(supported, "idxmin"),
            "time_of_maximum_ms": _time_of_extreme(supported, "idxmax"),
            "value_at_t0": value_at_t0,
            "value_at_t0_status": t0_status,
            "pre_event_baseline_window": "PRE_EARLY",
            "pre_event_baseline_mean": baseline_mean,
            "change_baseline_to_t0": change_baseline_to_t0,
            "pre_event_change": change_baseline_to_t0,
            "post_event_change": _post_event_change(rows, value_at_t0),
            "maximum_absolute_rate_of_change": _max_abs(dynamics["dynamic_value"]),
            "time_of_maximum_absolute_rate_of_change_ms": _time_of_max_abs_rate(dynamics),
        }
        if feature_name in BILATERAL_FEATURES:
            summary.update(_bilateral_fields(rows, dynamics, value_at_t0, change_baseline_to_t0))
        summaries[feature_name] = summary
    return summaries


def _event_feature_completeness(event_df: pd.DataFrame) -> dict[str, float]:
    return {
        feature_name: float(rows["feature_status"].eq(SUPPORTED_STATUS).mean())
        for feature_name, rows in event_df.groupby("feature_name", sort=True)
    }


def _window_summary_records(window_summaries: pd.DataFrame) -> list[dict]:
    return [
        {
            key: _json_scalar(value)
            for key, value in row.items()
        }
        for row in window_summaries.to_dict(orient="records")
    ]


def _bilateral_fields(
    rows: pd.DataFrame,
    dynamics: pd.DataFrame,
    value_at_t0: float,
    change_pre_event: float,
) -> dict:
    supported = rows[rows["feature_status"].eq(SUPPORTED_STATUS)]
    values = supported["feature_value"].dropna()
    abs_values = values.abs()
    peak_time = float("nan")
    if not abs_values.empty:
        peak_time = float(supported.loc[abs_values.idxmax(), "event_relative_ms"])
    return {
        "mean_signed_difference": _stat(values, "mean"),
        "mean_absolute_difference": _stat(abs_values, "mean"),
        "maximum_signed_difference": _stat(values, "max"),
        "minimum_signed_difference": _stat(values, "min"),
        "peak_absolute_difference": _stat(abs_values, "max"),
        "time_peak_absolute_difference_ms": peak_time,
        "difference_at_t0": value_at_t0,
        "change_pre_event": change_pre_event,
        "maximum_rate_of_change": _max_abs(dynamics["dynamic_value"]),
        "time_maximum_rate_of_change_ms": _time_of_max_abs_rate(dynamics),
    }


def _window_status(expected: int, completeness: float, threshold: float) -> str:
    if expected == 0:
        return "EMPTY_WINDOW"
    if completeness < threshold:
        return "INSUFFICIENT_COMPLETENESS"
    return "SUPPORTED"


def _is_dynamic_feature(feature_name: str) -> bool:
    return (
        feature_name in ANGLE_RATE_FEATURES
        or feature_name in BILATERAL_FEATURES
        or feature_name.endswith(NORMALIZED_RATE_SUFFIXES)
    )


def _dynamic_unit(unit: str) -> str:
    if unit == "deg":
        return "deg/s"
    if unit == "body_scale":
        return "body_scale/s"
    return f"{unit}/s" if unit else ""


def _feature_delta(feature_name: str, previous: float, current: float) -> float:
    if feature_name in ORIENTATION_FEATURES:
        return wrapped_angle_difference_deg(current, previous)
    return current - previous


def _median_frame_dt(rows: pd.DataFrame) -> float:
    unique_times = rows[["source_frame_index", "timestamp_ms"]].drop_duplicates().sort_values(
        "source_frame_index"
    )
    deltas = unique_times["timestamp_ms"].diff().dropna()
    if deltas.empty:
        return float("inf")
    return float(deltas.median())


def _value_at_t0(rows: pd.DataFrame) -> tuple[float, str]:
    t0_rows = rows[np.isclose(rows["event_relative_ms"], 0.0)]
    if t0_rows.empty:
        return float("nan"), "NO_T0_FRAME"
    row = t0_rows.iloc[0]
    if row["feature_status"] != SUPPORTED_STATUS:
        return float("nan"), "UNAVAILABLE_AT_T0"
    return float(row["feature_value"]), "SUPPORTED"


def _post_event_change(rows: pd.DataFrame, value_at_t0: float) -> float:
    if not np.isfinite(value_at_t0):
        return float("nan")
    post_rows = rows[
        rows["event_relative_ms"].ge(0.0)
        & rows["event_relative_ms"].le(200.0)
        & rows["feature_status"].eq(SUPPORTED_STATUS)
    ].sort_values("event_relative_ms")
    if post_rows.empty:
        return float("nan")
    return float(post_rows.iloc[-1]["feature_value"] - value_at_t0)


def _dynamic_completeness(event_df: pd.DataFrame) -> float:
    candidate = event_df[event_df["dynamic_status"].ne("NOT_DYNAMIC_FEATURE")]
    if candidate.empty:
        return 0.0
    return float(candidate["dynamic_status"].eq("SUPPORTED").mean())


def _time_of_extreme(rows: pd.DataFrame, method: str) -> float:
    values = rows["feature_value"].dropna()
    if values.empty:
        return float("nan")
    index = getattr(values, method)()
    return float(rows.loc[index, "event_relative_ms"])


def _time_of_max_abs_rate(rows: pd.DataFrame) -> float:
    values = rows["dynamic_value"].dropna().abs()
    if values.empty:
        return float("nan")
    index = values.idxmax()
    return float(rows.loc[index, "event_relative_ms"])


def _stat(values: pd.Series, method: str) -> float:
    clean = values.dropna()
    if clean.empty:
        return float("nan")
    return float(getattr(clean, method)())


def _range(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.max() - clean.min())


def _first_value(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.iloc[0]) if not clean.empty else float("nan")


def _last_value(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.iloc[-1]) if not clean.empty else float("nan")


def _change(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.iloc[-1] - clean.iloc[0])


def _max_abs(values: pd.Series) -> float:
    clean = values.dropna().abs()
    if clean.empty:
        return float("nan")
    return float(clean.max())


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _first_or_empty(df: pd.DataFrame, column: str) -> str:
    return str(df.iloc[0][column]) if not df.empty and column in df.columns else ""


def _json_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
