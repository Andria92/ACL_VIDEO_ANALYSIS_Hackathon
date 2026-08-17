"""Dynamic reliability hardening for Milestone 4.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
import pandas as pd

from acl_motion.events.temporal import DEFAULT_EVENT_WINDOWS, ORIENTATION_FEATURES
from acl_motion.geometry.angles import wrapped_angle_difference_deg

DYNAMIC_RELIABILITY_VERSION = "m4_1_dynamic_reliability_v1"
ROBUST_DYNAMIC_METHOD = "local_linear_regression_slope"


class RobustDynamicStatus(StrEnum):
    """Reliability state for the robust local dynamic estimate."""

    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_NEIGHBORHOOD = "INSUFFICIENT_NEIGHBORHOOD"
    SEGMENT_BOUNDARY = "SEGMENT_BOUNDARY"
    TEMPORAL_OUTLIER = "TEMPORAL_OUTLIER"
    EXCESSIVE_LOCAL_JITTER = "EXCESSIVE_LOCAL_JITTER"
    LOW_DYNAMIC_CONFIDENCE = "LOW_DYNAMIC_CONFIDENCE"
    MISSING_FEATURE = "MISSING_FEATURE"
    INVALID_TIME_INTERVAL = "INVALID_TIME_INTERVAL"
    NOT_DYNAMIC_FEATURE = "NOT_DYNAMIC_FEATURE"


@dataclass(frozen=True, slots=True)
class RobustDynamicConfig:
    """Parameters for local derivative estimation and dynamic QC."""

    half_window_samples: int = 2
    minimum_samples: int = 3
    temporal_outlier_residual_ratio: float = 1.25
    low_confidence_residual_ratio: float = 0.75
    reversal_rate_multiplier: float = 2.5
    epsilon: float = 1e-9

    def to_dict(self) -> dict:
        return asdict(self)


AUDIT_FEATURES = (
    "left_hka_angle_2d_deg",
    "right_hka_angle_2d_deg",
    "hka_projected_bilateral_difference_deg",
    "projected_trunk_axis_angle_deg",
    "projected_hip_line_angle_deg",
    "projected_shoulder_line_angle_deg",
    "left_elbow_angle_2d_deg",
    "right_elbow_angle_2d_deg",
)


def harden_dynamic_reliability(
    event_df: pd.DataFrame,
    *,
    config: RobustDynamicConfig | None = None,
) -> pd.DataFrame:
    """Preserve raw first differences and add robust dynamic estimates/QC."""

    cfg = config or RobustDynamicConfig()
    output = event_df.copy()
    _validate_event_table(output)
    output = _preserve_raw_first_difference(output)
    output = _initialize_robust_columns(output, cfg)

    dynamic_rows = output["raw_dynamic_status"].ne("NOT_DYNAMIC_FEATURE")
    for _, segment in output[dynamic_rows & output["feature_segment_id"].ne("")].groupby(
        "feature_segment_id", sort=False
    ):
        output = _estimate_segment_robust_rates(output, segment, cfg)
    return output.sort_values(["feature_name", "timestamp_ms"]).reset_index(drop=True)


def build_dynamic_quality_summary(
    dynamic_df: pd.DataFrame,
    *,
    config: RobustDynamicConfig | None = None,
    event_annotation_file: str = "",
    event_features_file: str = "",
) -> dict:
    """Summarize dynamic reliability for one case/view."""

    cfg = config or RobustDynamicConfig()
    eligible = _eligible_dynamic_rows(dynamic_df)
    raw_supported = eligible["raw_dynamic_status"].eq("SUPPORTED")
    robust_supported = eligible["dynamic_status"].eq(RobustDynamicStatus.SUPPORTED.value)
    feature_coverage = _feature_dynamic_coverage(eligible)
    return {
        "run_metadata": {
            "run_id": (
                f"{_first(dynamic_df, 'case_id')}_m4_1_"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "case_id": _first(dynamic_df, "case_id"),
            "source_id": _first(dynamic_df, "source_id"),
            "view_id": _first(dynamic_df, "view_id"),
            "event_annotation_file": event_annotation_file,
            "event_features_file": event_features_file,
            "dynamic_reliability_version": DYNAMIC_RELIABILITY_VERSION,
            "robust_dynamic_method": ROBUST_DYNAMIC_METHOD,
            "robust_dynamic_parameters": cfg.to_dict(),
            "dynamic_qc_basis": (
                "local trajectory residuals and simple reversal checks; no biological rate ceiling"
            ),
        },
        "total_eligible_dynamic_samples": len(eligible),
        "raw_supported_dynamic_samples": int(raw_supported.sum()),
        "supported_robust_dynamic_samples": int(robust_supported.sum()),
        "raw_dynamic_completeness": _mean_bool(raw_supported),
        "robust_dynamic_completeness": _mean_bool(robust_supported),
        "dynamic_status_counts": _value_counts(dynamic_df["dynamic_status"]),
        "raw_dynamic_status_counts": _value_counts(dynamic_df["raw_dynamic_status"]),
        "dynamic_quality_counts": _value_counts(dynamic_df["dynamic_quality"]),
        "top_raw_rate": _max_abs_value(eligible, "raw_first_difference_rate"),
        "top_robust_supported_rate": _max_abs_value(
            eligible[eligible["dynamic_status"].eq(RobustDynamicStatus.SUPPORTED.value)],
            "robust_dynamic_rate",
        ),
        "feature_dynamic_coverage": feature_coverage,
        "features_with_strongest_dynamic_coverage": _coverage_extremes(feature_coverage, ascending=False),
        "features_with_weakest_dynamic_coverage": _coverage_extremes(feature_coverage, ascending=True),
        "main_dynamic_rejection_reasons": _value_counts(
            dynamic_df["dynamic_rejection_reason"].replace("", np.nan).dropna()
        ),
    }


def build_dynamic_window_summaries(
    dynamic_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize raw and robust rates inside the default M4 event windows."""

    rows: list[dict] = []
    for feature_name, feature_rows in dynamic_df.groupby("feature_name", sort=True):
        for window in DEFAULT_EVENT_WINDOWS:
            in_window = feature_rows[window.contains(feature_rows["event_relative_ms"])]
            eligible = _eligible_dynamic_rows(in_window)
            raw_supported = eligible[eligible["raw_dynamic_status"].eq("SUPPORTED")]
            robust_supported = eligible[
                eligible["dynamic_status"].eq(RobustDynamicStatus.SUPPORTED.value)
            ]
            rows.append(
                {
                    "case_id": _first(dynamic_df, "case_id"),
                    "source_id": _first(dynamic_df, "source_id"),
                    "view_id": _first(dynamic_df, "view_id"),
                    "feature_name": feature_name,
                    "window_name": window.name,
                    "window_start_ms": window.start_ms,
                    "window_end_ms": window.end_ms,
                    "eligible_dynamic_samples": len(eligible),
                    "raw_supported_dynamic_samples": len(raw_supported),
                    "robust_supported_dynamic_samples": len(robust_supported),
                    "raw_dynamic_completeness": _fraction(len(raw_supported), len(eligible)),
                    "robust_dynamic_completeness": _fraction(len(robust_supported), len(eligible)),
                    "raw_maximum_absolute_rate": _max_abs_value(
                        raw_supported, "raw_first_difference_rate"
                    ),
                    "robust_maximum_absolute_rate": _max_abs_value(
                        robust_supported, "robust_dynamic_rate"
                    ),
                    "time_raw_maximum_absolute_rate_ms": _time_of_max_abs(
                        raw_supported, "raw_first_difference_rate"
                    ),
                    "time_robust_maximum_absolute_rate_ms": _time_of_max_abs(
                        robust_supported, "robust_dynamic_rate"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_name", "window_start_ms"]).reset_index(
        drop=True
    )


def build_dynamic_spike_audit(
    dynamic_df: pd.DataFrame,
    *,
    features: tuple[str, ...] = AUDIT_FEATURES,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build a top-spike audit table from absolute raw first-difference rates."""

    rows: list[pd.DataFrame] = []
    for feature_name in features:
        feature_rows = dynamic_df[
            dynamic_df["feature_name"].eq(feature_name)
            & dynamic_df["raw_dynamic_status"].eq("SUPPORTED")
            & dynamic_df["raw_first_difference_rate"].notna()
        ].copy()
        if feature_rows.empty:
            continue
        feature_rows["absolute_raw_rate"] = feature_rows["raw_first_difference_rate"].abs()
        rows.append(feature_rows.nlargest(top_n, "absolute_raw_rate"))
    if not rows:
        return pd.DataFrame()
    audit = pd.concat(rows, ignore_index=True)
    audit = audit.sort_values(["feature_name", "absolute_raw_rate"], ascending=[True, False])
    return audit[
        [
            "case_id",
            "feature_name",
            "previous_source_frame_index",
            "current_source_frame_index",
            "event_relative_ms",
            "previous_value",
            "current_value",
            "delta_value",
            "delta_time_ms",
            "raw_first_difference_rate",
            "robust_dynamic_rate",
            "dynamic_status",
            "dynamic_quality",
            "local_jitter_metric",
            "local_residual",
            "input_quality_prev",
            "input_quality_current",
            "landmarks_used",
        ]
    ].rename(
        columns={
            "previous_source_frame_index": "source_frame_prev",
            "current_source_frame_index": "source_frame_current",
            "raw_first_difference_rate": "raw_rate",
            "robust_dynamic_rate": "robust_rate",
        }
    )


def _validate_event_table(df: pd.DataFrame) -> None:
    required = {
        "feature_name",
        "feature_value",
        "feature_status",
        "feature_segment_id",
        "source_frame_index",
        "timestamp_ms",
        "dynamic_value",
        "dynamic_status",
        "dynamic_rejection_reason",
        "quality_status",
        "landmarks_used",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Event-relative feature table is missing required columns: {missing}")


def _preserve_raw_first_difference(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["raw_first_difference_rate"] = output["dynamic_value"]
    output["raw_dynamic_status"] = output["dynamic_status"]
    output["raw_dynamic_rejection_reason"] = output["dynamic_rejection_reason"]
    output["previous_source_frame_index"] = np.nan
    output["current_source_frame_index"] = output["source_frame_index"]
    output["previous_timestamp_ms"] = np.nan
    output["current_timestamp_ms"] = output["timestamp_ms"]
    output["previous_value"] = np.nan
    output["current_value"] = output["feature_value"]
    output["delta_value"] = np.nan
    output["delta_time_ms"] = np.nan
    output["input_quality_prev"] = ""
    output["input_quality_current"] = output["quality_status"]

    for _, segment in output[output["feature_segment_id"].ne("")].groupby(
        "feature_segment_id", sort=False
    ):
        previous_index = None
        for row_index, row in segment.sort_values("timestamp_ms").iterrows():
            if previous_index is not None and row["raw_dynamic_status"] == "SUPPORTED":
                previous = output.loc[previous_index]
                output.at[row_index, "previous_source_frame_index"] = previous[
                    "source_frame_index"
                ]
                output.at[row_index, "previous_timestamp_ms"] = previous["timestamp_ms"]
                output.at[row_index, "previous_value"] = previous["feature_value"]
                output.at[row_index, "delta_time_ms"] = (
                    float(row["timestamp_ms"]) - float(previous["timestamp_ms"])
                )
                output.at[row_index, "delta_value"] = (
                    float(row["feature_value"]) - float(previous["feature_value"])
                )
                output.at[row_index, "input_quality_prev"] = previous["quality_status"]
            previous_index = row_index
    return output


def _initialize_robust_columns(df: pd.DataFrame, cfg: RobustDynamicConfig) -> pd.DataFrame:
    output = df.copy()
    output["robust_dynamic_rate"] = np.nan
    output["robust_dynamic_unit"] = output["dynamic_unit"]
    output["dynamic_status"] = np.where(
        output["raw_dynamic_status"].eq("NOT_DYNAMIC_FEATURE"),
        RobustDynamicStatus.NOT_DYNAMIC_FEATURE.value,
        RobustDynamicStatus.MISSING_FEATURE.value,
    )
    output["dynamic_quality"] = np.where(
        output["raw_dynamic_status"].eq("NOT_DYNAMIC_FEATURE"), "NOT_APPLICABLE", "UNAVAILABLE"
    )
    output["dynamic_rejection_reason"] = np.where(
        output["raw_dynamic_status"].eq("NOT_DYNAMIC_FEATURE"),
        "Feature is not configured for dynamic analysis.",
        "Feature value is unavailable.",
    )
    output["dynamic_method"] = ROBUST_DYNAMIC_METHOD
    output["dynamic_parameters"] = [cfg.to_dict()] * len(output)
    output["dynamic_window_start_frame"] = np.nan
    output["dynamic_window_end_frame"] = np.nan
    output["dynamic_window_start_ms"] = np.nan
    output["dynamic_window_end_ms"] = np.nan
    output["dynamic_valid_samples"] = 0
    output["local_residual"] = np.nan
    output["local_jitter_metric"] = np.nan
    output["temporal_stability_score"] = np.nan
    output["spike_reversal_detected"] = False
    return output


def _estimate_segment_robust_rates(
    output: pd.DataFrame,
    segment: pd.DataFrame,
    cfg: RobustDynamicConfig,
) -> pd.DataFrame:
    supported = segment[segment["feature_status"].eq("SUPPORTED")].sort_values("timestamp_ms")
    if supported.empty:
        return output
    feature_name = str(supported.iloc[0]["feature_name"])
    supported_indices = list(supported.index)
    continuous_values = _continuous_values(
        supported["feature_value"].astype(float).to_numpy(),
        feature_name,
    )
    supported = supported.assign(_continuous_value=continuous_values)
    raw_abs_median = _median_abs_supported_rate(supported)

    for position, row_index in enumerate(supported_indices):
        row = supported.loc[row_index]
        start = max(0, position - cfg.half_window_samples)
        end = min(len(supported_indices), position + cfg.half_window_samples + 1)
        window = supported.iloc[start:end]
        _write_window_metadata(output, row_index, window)
        if len(window) < cfg.minimum_samples:
            _set_dynamic_status(
                output,
                row_index,
                RobustDynamicStatus.INSUFFICIENT_NEIGHBORHOOD,
                "Fewer than the configured minimum supported neighboring samples.",
            )
            continue
        times_s = (window["timestamp_ms"].astype(float).to_numpy() - float(row["timestamp_ms"])) / 1000.0
        if len(np.unique(times_s)) < 2:
            _set_dynamic_status(
                output,
                row_index,
                RobustDynamicStatus.INVALID_TIME_INTERVAL,
                "Local dynamic window does not contain increasing timestamps.",
            )
            continue
        values = window["_continuous_value"].astype(float).to_numpy()
        slope, intercept = np.polyfit(times_s, values, 1)
        predicted = slope * times_s + intercept
        residuals = values - predicted
        current_offset = position - start
        local_residual = abs(float(residuals[current_offset]))
        local_jitter = float(np.median(np.abs(residuals)))
        local_successive = _median_abs_successive_difference(values)
        stability_score = local_jitter / (local_successive + cfg.epsilon)
        output.at[row_index, "robust_dynamic_rate"] = float(slope)
        output.at[row_index, "local_residual"] = local_residual
        output.at[row_index, "local_jitter_metric"] = local_jitter
        output.at[row_index, "temporal_stability_score"] = stability_score

        residual_ratio = local_residual / (local_successive + cfg.epsilon)
        jitter_ratio = local_jitter / (local_successive + cfg.epsilon)
        reversal = _has_rate_reversal(output, supported_indices, position, raw_abs_median, cfg)
        output.at[row_index, "spike_reversal_detected"] = reversal
        if reversal and residual_ratio >= cfg.temporal_outlier_residual_ratio:
            _set_dynamic_status(
                output,
                row_index,
                RobustDynamicStatus.TEMPORAL_OUTLIER,
                "Large local residual plus opposite-direction one-step reversal.",
                quality="LIMITED",
            )
        elif jitter_ratio >= cfg.low_confidence_residual_ratio:
            _set_dynamic_status(
                output,
                row_index,
                RobustDynamicStatus.LOW_DYNAMIC_CONFIDENCE,
                "Local trajectory residuals are high relative to local movement.",
                quality="LIMITED",
            )
        else:
            _set_dynamic_status(
                output,
                row_index,
                RobustDynamicStatus.SUPPORTED,
                "",
                quality="HIGH",
            )
    return output


def _write_window_metadata(output: pd.DataFrame, row_index: int, window: pd.DataFrame) -> None:
    output.at[row_index, "dynamic_window_start_frame"] = int(window["source_frame_index"].min())
    output.at[row_index, "dynamic_window_end_frame"] = int(window["source_frame_index"].max())
    output.at[row_index, "dynamic_window_start_ms"] = float(window["timestamp_ms"].min())
    output.at[row_index, "dynamic_window_end_ms"] = float(window["timestamp_ms"].max())
    output.at[row_index, "dynamic_valid_samples"] = len(window)


def _set_dynamic_status(
    output: pd.DataFrame,
    row_index: int,
    status: RobustDynamicStatus,
    reason: str,
    *,
    quality: str = "UNAVAILABLE",
) -> None:
    output.at[row_index, "dynamic_status"] = status.value
    output.at[row_index, "dynamic_rejection_reason"] = reason
    output.at[row_index, "dynamic_quality"] = quality


def _continuous_values(values: np.ndarray, feature_name: str) -> np.ndarray:
    if feature_name not in ORIENTATION_FEATURES:
        return values
    continuous = [float(values[0])]
    for value in values[1:]:
        continuous.append(continuous[-1] + wrapped_angle_difference_deg(float(value), continuous[-1]))
    return np.array(continuous, dtype=float)


def _median_abs_supported_rate(supported: pd.DataFrame) -> float:
    rates = supported["raw_first_difference_rate"].dropna().abs().sort_values()
    if rates.empty:
        return 0.0
    lower_half_count = max(1, len(rates) // 2)
    return float(rates.iloc[:lower_half_count].median())


def _median_abs_successive_difference(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(values))))


def _has_rate_reversal(
    output: pd.DataFrame,
    supported_indices: list[int],
    position: int,
    raw_abs_median: float,
    cfg: RobustDynamicConfig,
) -> bool:
    row_index = supported_indices[position]
    current_rate = output.at[row_index, "raw_first_difference_rate"]
    if not np.isfinite(current_rate):
        return False
    threshold = max(raw_abs_median * cfg.reversal_rate_multiplier, cfg.epsilon)
    if abs(float(current_rate)) < threshold:
        return False
    neighbor_rates = []
    if position + 1 < len(supported_indices):
        next_rate = output.at[supported_indices[position + 1], "raw_first_difference_rate"]
        if np.isfinite(next_rate):
            neighbor_rates.append(float(next_rate))
    if position > 0:
        previous_rate = output.at[supported_indices[position - 1], "raw_first_difference_rate"]
        if np.isfinite(previous_rate):
            neighbor_rates.append(float(previous_rate))
    return any(
        np.sign(rate) == -np.sign(float(current_rate)) and abs(rate) >= threshold
        for rate in neighbor_rates
    )


def _eligible_dynamic_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["raw_dynamic_status"].ne("NOT_DYNAMIC_FEATURE")
        & df["feature_status"].eq("SUPPORTED")
    ]


def _feature_dynamic_coverage(eligible: pd.DataFrame) -> dict[str, dict]:
    coverage: dict[str, dict] = {}
    for feature_name, rows in eligible.groupby("feature_name", sort=True):
        supported = rows["dynamic_status"].eq(RobustDynamicStatus.SUPPORTED.value)
        coverage[feature_name] = {
            "eligible_dynamic_samples": len(rows),
            "supported_robust_dynamic_samples": int(supported.sum()),
            "robust_dynamic_completeness": _mean_bool(supported),
            "raw_dynamic_completeness": _mean_bool(rows["raw_dynamic_status"].eq("SUPPORTED")),
        }
    return coverage


def _coverage_extremes(coverage: dict[str, dict], *, ascending: bool) -> list[dict]:
    rows = [
        {"feature_name": name, **values}
        for name, values in coverage.items()
        if values["eligible_dynamic_samples"] > 0
    ]
    rows.sort(key=lambda row: row["robust_dynamic_completeness"], reverse=not ascending)
    return rows[:5]


def _max_abs_value(df: pd.DataFrame, column: str) -> float | None:
    values = df[column].dropna().abs()
    if values.empty:
        return None
    return float(values.max())


def _time_of_max_abs(df: pd.DataFrame, column: str) -> float | None:
    values = df[column].dropna().abs()
    if values.empty:
        return None
    return float(df.loc[values.idxmax(), "event_relative_ms"])


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _mean_bool(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _first(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return ""
    return str(df.iloc[0][column])
