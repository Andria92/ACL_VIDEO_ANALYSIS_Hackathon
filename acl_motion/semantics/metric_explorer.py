"""Metric registry and multiscale statistics for the Results metric explorer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from acl_motion.geometry.angles import wrapped_angle_difference_deg
from acl_motion.geometry.angular_semantics import (
    ANGULAR_STATISTICS_VERSION,
    AngleType,
    angle_type_for_metric,
    angular_difference,
    angular_mean,
    angular_standard_deviation,
    measurement_range_for_metric,
    range_semantics_for_metric,
)


class SelectionMode(StrEnum):
    """Shared UI/data selection modes for multiscale movement exploration."""

    WHOLE_MOVEMENT = "WHOLE_MOVEMENT"
    PHASE = "PHASE"
    FIVE_FRAME_WINDOW = "FIVE_FRAME_WINDOW"
    SINGLE_FRAME = "SINGLE_FRAME"


@dataclass(frozen=True, slots=True)
class MetricVisualisationSpec:
    """Visualisation registry record for one numeric movement metric."""

    metric_name: str
    display_label: str
    metric_family: str
    temporal: bool
    signed: bool
    categorical: bool
    bilateral: bool
    distributional: bool
    angular: bool
    angle_type: str | None
    unit: str
    preferred_visualisation: str
    no_visualisation_reason: str = ""
    paired_metric_name: str | None = None
    evidence_note: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-ready visualisation spec."""

        return asdict(self)


METRIC_CATEGORY_LABELS = {
    "movement_path": "Movement Path",
    "hip_knee_ankle_chain": "Hip-Knee-Ankle Chain",
    "knee_relative_to_ankle": "Knee / Ankle Diagnostics",
    "hip_thigh": "Hip / Thigh",
    "trunk_pelvis": "Trunk & Pelvis",
    "upper_body": "Upper Body",
    "bilateral_limb_relationship": "Bilateral Limb Relationship",
    "dynamics": "Dynamics",
    "evidence_quality": "Evidence / Quality",
}


def build_metric_explorer_payload(
    *,
    dynamic_df: pd.DataFrame,
    path_df: pd.DataFrame,
    movement_story: dict,
) -> dict:
    """Build the reusable multiscale metric explorer payload."""

    series = _all_metric_series(dynamic_df, path_df)
    specs = {
        metric_name: _visualisation_spec(metric_name, rows)
        for metric_name, rows in series.items()
    }
    phases = movement_story.get("phases", [])
    phase_stats = {
        metric_name: [
            {
                "phase_id": phase["phase_id"],
                "phase_index": phase["phase_index"],
                "phase_title": phase["title"],
                **metric_statistics(
                    rows,
                    start_frame=int(phase["start_frame"]),
                    end_frame=int(phase["end_frame"]),
                ),
            }
            for phase in phases
        ]
        for metric_name, rows in series.items()
    }
    whole_stats = {
        metric_name: metric_statistics(rows)
        for metric_name, rows in series.items()
    }
    angular_metric_names = [
        name
        for name, spec in specs.items()
        if spec.angular
    ]
    return {
        "selection_modes": [mode.value for mode in SelectionMode],
        "window_convention": (
            "Five-frame selections begin at the selected anchor frame and include up to "
            "the next four frames, clamped to the selected phase when a phase is active."
        ),
        "boundary_behavior": (
            "+1/-1 and +5/-5 remain within the selected phase by default; graph or scrub "
            "selection may update the selected phase when the chosen frame belongs elsewhere."
        ),
        "metric_category_labels": METRIC_CATEGORY_LABELS,
        "metrics": {name: spec.to_dict() for name, spec in specs.items()},
        "categories": _categories(specs),
        "series": {name: _records_json(rows) for name, rows in series.items()},
        "angular_metric_names": angular_metric_names,
        "whole_movement_statistics": whole_stats,
        "phase_statistics": phase_stats,
        "angular_heatmap": _angular_heatmap(angular_metric_names, phase_stats),
        "visualisation_contract": {
            "future_cohort_reference": "ACL case-library reference only; not a normal population.",
            "future_alignment_note": (
                "Future cohort trajectories must use normalized Movement Window time or "
                "validated comparable phase alignment, never raw frame indices alone."
            ),
            "future_similarity_note": (
                "Future top-k similar cases and clusters should use standardized analytical "
                "feature space with mutually supported descriptors, report features used and "
                "missing, and not use UMAP display distance. Similarity would describe only "
                "the measured movement representation, not injury mechanism, biological cause, "
                "tissue loading, or clinical condition."
            ),
        },
    }


def metric_statistics(
    rows: pd.DataFrame,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict:
    """Summarize supported metric values inside an optional frame interval."""

    subset = rows.copy()
    metric_name = str(rows["metric_name"].iloc[0]) if not rows.empty else ""
    if start_frame is not None:
        subset = subset[subset["source_frame_index"].astype(int).ge(int(start_frame))]
    if end_frame is not None:
        subset = subset[subset["source_frame_index"].astype(int).le(int(end_frame))]
    relevant_n = len(subset)
    supported = subset[subset["evidence_status"].eq("SUPPORTED") & subset["value"].notna()].copy()
    values = pd.to_numeric(supported["value"], errors="coerce").dropna()
    if values.empty:
        return {
            "supported_n": 0,
            "relevant_n": relevant_n,
            "completeness": 0.0 if relevant_n else None,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "range": None,
            "raw_range": None,
            "range_semantics": range_semantics_for_metric(metric_name),
            "angular_statistics_version": ANGULAR_STATISTICS_VERSION,
            "q1": None,
            "q3": None,
            "iqr": None,
            "summary_semantics": (
                "circular"
                if angle_type_for_metric(metric_name) in {AngleType.DIRECTED, AngleType.AXIS}
                else "linear"
            ),
            "start_value": None,
            "end_value": None,
            "change": None,
            "absolute_change": None,
            "total_absolute_change": None,
            "start_frame": None,
            "end_frame": None,
            "minimum_frame": None,
            "maximum_frame": None,
            "peak_frame_to_frame_change": None,
            "peak_frame_to_frame_change_frame": None,
            "raw_start_angle": None,
            "raw_end_angle": None,
            "angle_type": _angle_type_value(metric_name),
            "canonical_signed_change": None,
            "canonical_absolute_change": None,
        }
    first = supported.iloc[0]
    last = supported.iloc[-1]
    angle_type = angle_type_for_metric(metric_name)
    circular_summary = angle_type in {AngleType.DIRECTED, AngleType.AXIS}
    q1 = None if circular_summary else float(values.quantile(0.25))
    q3 = None if circular_summary else float(values.quantile(0.75))
    min_row = supported.loc[pd.to_numeric(supported["value"], errors="coerce").idxmin()]
    max_row = supported.loc[pd.to_numeric(supported["value"], errors="coerce").idxmax()]
    frame_changes = _frame_to_frame_changes(metric_name, supported)
    peak_change = frame_changes["change"].abs().max() if not frame_changes.empty else np.nan
    peak_change_row = (
        frame_changes.loc[frame_changes["change"].abs().idxmax()]
        if not frame_changes.empty and np.isfinite(peak_change)
        else None
    )
    has_change_support = len(values) > 1
    change = (
        _metric_change(metric_name, float(first["value"]), float(last["value"]))
        if has_change_support
        else None
    )
    display_fields = (
        _canonical_change_fields(
            metric_name,
            float(first["value"]),
            float(last["value"]),
            change,
        )
        if has_change_support
        else {
            "raw_start_angle": float(first["value"]) if _angle_type_value(metric_name) else None,
            "raw_end_angle": float(last["value"]) if _angle_type_value(metric_name) else None,
            "angle_type": _angle_type_value(metric_name),
            "canonical_signed_change": None,
            "canonical_absolute_change": None,
        }
    )
    return {
        "supported_n": len(values),
        "relevant_n": relevant_n,
        "completeness": float(len(values) / relevant_n) if relevant_n else None,
        "mean": (
            _finite_or_none(angular_mean(values, angle_type))
            if circular_summary
            else float(values.mean())
        ),
        "median": None if circular_summary else float(values.median()),
        "standard_deviation": (
            _finite_or_none(angular_standard_deviation(values, angle_type))
            if circular_summary and has_change_support
            else float(values.std(ddof=1)) if has_change_support else None
        ),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "range": measurement_range_for_metric(metric_name, values),
        "raw_range": float(values.max() - values.min()),
        "range_semantics": range_semantics_for_metric(metric_name),
        "angular_statistics_version": ANGULAR_STATISTICS_VERSION,
        "q1": q1,
        "q3": q3,
        "iqr": None if circular_summary else float(q3 - q1),
        "summary_semantics": "circular" if circular_summary else "linear",
        "start_value": float(first["value"]),
        "end_value": float(last["value"]),
        "change": change,
        "absolute_change": abs(change) if change is not None else None,
        "total_absolute_change": _total_absolute_change(metric_name, values),
        "start_frame": int(first["source_frame_index"]),
        "end_frame": int(last["source_frame_index"]),
        "minimum_frame": int(min_row["source_frame_index"]),
        "maximum_frame": int(max_row["source_frame_index"]),
        "peak_frame_to_frame_change": (
            float(peak_change_row["change"]) if peak_change_row is not None else None
        ),
        "peak_frame_to_frame_change_frame": (
            int(peak_change_row["source_frame_index"]) if peak_change_row is not None else None
        ),
        **display_fields,
    }


def selection_statistics(
    rows: pd.DataFrame,
    *,
    mode: SelectionMode | str,
    frame: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict:
    """Return statistics for one explicit selection mode."""

    selection_mode = SelectionMode(mode)
    if selection_mode is SelectionMode.SINGLE_FRAME:
        if frame is None:
            raise ValueError("Single-frame statistics require frame.")
        subset = rows[rows["source_frame_index"].astype(int).eq(int(frame))]
        if subset.empty:
            return {"mode": selection_mode.value, "frame": int(frame), "current_value": None, "evidence_state": "UNAVAILABLE"}
        row = subset.iloc[0]
        return {
            "mode": selection_mode.value,
            "frame": int(frame),
            "movement_end_relative_ms": _optional_float(row.get("movement_end_relative_ms")),
            "current_value": _optional_float(row.get("value")),
            "evidence_state": str(row.get("evidence_status", "UNAVAILABLE")),
        }
    stats = metric_statistics(rows, start_frame=start_frame, end_frame=end_frame)
    stats["mode"] = selection_mode.value
    if selection_mode is SelectionMode.FIVE_FRAME_WINDOW:
        subset = rows[
            rows["source_frame_index"].astype(int).between(int(start_frame), int(end_frame), inclusive="both")
        ]
        stats["frames"] = _records_json(subset)
    return stats


def _all_metric_series(dynamic_df: pd.DataFrame, path_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for feature_name, rows in dynamic_df.groupby("feature_name", sort=True):
        metric_name = str(feature_name)
        output[metric_name] = _geometry_metric_series(metric_name, rows)
        dynamic_metric = f"dynamic_rate:{feature_name}"
        output[dynamic_metric] = _dynamic_metric_series(dynamic_metric, rows)
    output.update(_path_metric_series(path_df))
    return output


def _geometry_metric_series(metric_name: str, rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.sort_values("source_frame_index")[
        [
            "source_frame_index",
            "analysis_frame_index",
            "timestamp_ms",
            "movement_elapsed_ms",
            "movement_end_relative_ms",
            "feature_value",
            "feature_status",
            "unit",
            "rejection_reason",
        ]
    ].copy()
    output["metric_name"] = metric_name
    output["value"] = pd.to_numeric(output["feature_value"], errors="coerce").where(
        output["feature_status"].eq("SUPPORTED")
    )
    output["evidence_status"] = output["feature_status"].where(
        output["feature_status"].eq("SUPPORTED"),
        "UNAVAILABLE",
    )
    output["quality_reason"] = output["rejection_reason"].fillna("")
    return _metric_columns(output)


def _dynamic_metric_series(metric_name: str, rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.sort_values("source_frame_index")[
        [
            "source_frame_index",
            "analysis_frame_index",
            "timestamp_ms",
            "movement_elapsed_ms",
            "movement_end_relative_ms",
            "robust_dynamic_rate",
            "dynamic_status",
            "robust_dynamic_unit",
            "dynamic_rejection_reason",
        ]
    ].copy()
    output["metric_name"] = metric_name
    output["value"] = pd.to_numeric(output["robust_dynamic_rate"], errors="coerce").where(
        output["dynamic_status"].eq("SUPPORTED")
    )
    output["evidence_status"] = output["dynamic_status"].where(
        output["dynamic_status"].eq("SUPPORTED"),
        "UNAVAILABLE",
    )
    output["unit"] = output["robust_dynamic_unit"].fillna("feature units/s")
    output["quality_reason"] = output["dynamic_rejection_reason"].fillna("")
    return _metric_columns(output)


def _path_metric_series(path_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    path = path_df.sort_values("source_frame_index").copy()
    path["analysis_frame_index"] = path["source_frame_index"]
    specs = {
        "path:compensated_x": ("compensated_x", "body-center x"),
        "path:compensated_y": ("compensated_y", "body-center y"),
        "path:projected_heading_deg": ("projected_heading_deg", "deg"),
        "path:normalized_projected_speed_per_s": (
            "normalized_projected_speed_per_s",
            "body-scale units/s",
        ),
    }
    output = {}
    for metric_name, (column, unit) in specs.items():
        path_columns = [
            "source_frame_index",
            "analysis_frame_index",
            "timestamp_ms",
            "movement_elapsed_ms",
            "movement_end_relative_ms",
            column,
            "path_status",
            "path_rejection_reason",
        ]
        if "path_segment_id" in path.columns:
            path_columns.append("path_segment_id")
        rows = path[path_columns].copy()
        rows["metric_name"] = metric_name
        rows["value"] = pd.to_numeric(rows[column], errors="coerce").where(
            rows["path_status"].eq("SUPPORTED")
        )
        rows["evidence_status"] = rows["path_status"].where(
            rows["path_status"].eq("SUPPORTED"),
            "UNAVAILABLE",
        )
        rows["unit"] = unit
        rows["quality_reason"] = rows["path_rejection_reason"].fillna("")
        output[metric_name] = _metric_columns(rows)
    heading = output["path:projected_heading_deg"].copy()
    heading["metric_name"] = "path:projected_heading_change_deg"
    heading["value"] = _heading_change_values(output["path:projected_heading_deg"])
    heading["unit"] = "deg"
    heading["evidence_status"] = heading["evidence_status"].where(heading["value"].notna(), "UNAVAILABLE")
    output["path:projected_heading_change_deg"] = heading
    return output


def _heading_change_values(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(index=rows.index, dtype=float)
    values = rows["value"].to_numpy(dtype=float)
    changes = [np.nan]
    for previous, current in pairwise(values):
        if np.isfinite(previous) and np.isfinite(current):
            changes.append(wrapped_angle_difference_deg(float(current), float(previous)))
        else:
            changes.append(np.nan)
    return pd.Series(changes, index=rows.index)


def _metric_columns(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric_name",
        "source_frame_index",
        "analysis_frame_index",
        "timestamp_ms",
        "movement_elapsed_ms",
        "movement_end_relative_ms",
        "value",
        "unit",
        "evidence_status",
        "quality_reason",
    ]
    if "path_segment_id" in rows.columns:
        columns.append("path_segment_id")
    return rows[columns].reset_index(drop=True)


def _visualisation_spec(metric_name: str, rows: pd.DataFrame) -> MetricVisualisationSpec:
    family = _metric_family(metric_name)
    unit = str(rows["unit"].dropna().iloc[0]) if rows["unit"].notna().any() else ""
    paired = _paired_metric(metric_name)
    preferred = "paired_line" if paired else "line"
    if metric_name == "path:compensated_x":
        preferred = "path_2d"
    elif family == "evidence_quality":
        preferred = "horizontal_bar"
    elif family == "bilateral_limb_relationship" and "absolute_difference" in metric_name:
        preferred = "line_with_phase_bars"
    return MetricVisualisationSpec(
        metric_name=metric_name,
        display_label=_display_label(metric_name),
        metric_family=family,
        temporal=True,
        signed=_is_signed(metric_name),
        categorical=False,
        bilateral=family == "bilateral_limb_relationship" or paired is not None,
        distributional=True,
        angular=_is_angular_metric(metric_name, unit),
        angle_type=_angle_type_value(metric_name),
        unit=unit,
        preferred_visualisation=preferred,
        paired_metric_name=paired,
        evidence_note="Unsupported/rejected samples are omitted and shown as gaps.",
    )


def _metric_family(metric_name: str) -> str:
    base = metric_name.removeprefix("dynamic_rate:")
    if metric_name.startswith("path:"):
        return "movement_path"
    if metric_name.startswith("dynamic_rate:"):
        return "dynamics"
    if any(token in base for token in ("bilateral", "injured_", "contralateral_")):
        return "bilateral_limb_relationship"
    if "knee_ankle" in base:
        return "knee_relative_to_ankle"
    if any(token in base for token in ("hka", "knee_line")):
        return "hip_knee_ankle_chain"
    if any(token in base for token in ("elbow", "upper_arm", "wrist")):
        return "upper_body"
    if any(token in base for token in ("trunk", "hip_line", "shoulder_line", "pelvis")):
        return "trunk_pelvis"
    return "hip_thigh"


def _paired_metric(metric_name: str) -> str | None:
    pairs = {
        "injured_hka_angle_2d_deg": "contralateral_hka_angle_2d_deg",
        "contralateral_hka_angle_2d_deg": "injured_hka_angle_2d_deg",
        "left_hka_angle_2d_deg": "right_hka_angle_2d_deg",
        "right_hka_angle_2d_deg": "left_hka_angle_2d_deg",
        "left_elbow_angle_2d_deg": "right_elbow_angle_2d_deg",
        "right_elbow_angle_2d_deg": "left_elbow_angle_2d_deg",
        "left_upper_arm_orientation_2d_deg": "right_upper_arm_orientation_2d_deg",
        "right_upper_arm_orientation_2d_deg": "left_upper_arm_orientation_2d_deg",
        "left_knee_ankle_x_offset_normalized": "right_knee_ankle_x_offset_normalized",
        "right_knee_ankle_x_offset_normalized": "left_knee_ankle_x_offset_normalized",
        "left_knee_ankle_distance_normalized": "right_knee_ankle_distance_normalized",
        "right_knee_ankle_distance_normalized": "left_knee_ankle_distance_normalized",
    }
    return pairs.get(metric_name)


def _is_signed(metric_name: str) -> bool:
    return any(token in metric_name for token in ("difference", "offset", "orientation", "heading", "line"))


def _is_angular_metric(metric_name: str, unit: str) -> bool:
    return unit == "deg" and not metric_name.startswith("dynamic_rate:")


def _frame_to_frame_changes(metric_name: str, supported: pd.DataFrame) -> pd.DataFrame:
    rows = supported.sort_values("source_frame_index")
    changes = []
    previous_value: float | None = None
    for _, row in rows.iterrows():
        value = float(row["value"])
        if previous_value is not None:
            changes.append(
                {
                    "source_frame_index": int(row["source_frame_index"]),
                    "change": _metric_change(metric_name, previous_value, value),
                }
            )
        previous_value = value
    return pd.DataFrame(changes)


def _total_absolute_change(metric_name: str, values: pd.Series) -> float | None:
    numeric = values.to_numpy(dtype=float)
    if len(numeric) < 2:
        return None
    total = 0.0
    for previous, current in pairwise(numeric):
        if np.isfinite(previous) and np.isfinite(current):
            total += abs(_metric_change(metric_name, float(previous), float(current)))
    return float(total)


def _angular_heatmap(
    angular_metric_names: list[str],
    phase_stats: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    aggregates = ("mean", "range", "change", "absolute_change", "total_absolute_change")
    return {
        aggregate: [
            {
                "metric_name": metric_name,
                "phase_id": stats["phase_id"],
                "phase_index": stats["phase_index"],
                "value": stats.get(aggregate),
                "supported_n": stats.get("supported_n"),
                "completeness": stats.get("completeness"),
            }
            for metric_name in angular_metric_names
            for stats in phase_stats.get(metric_name, [])
        ]
        for aggregate in aggregates
    }


def _categories(specs: dict[str, MetricVisualisationSpec]) -> dict[str, list[dict]]:
    categories: dict[str, list[dict]] = {key: [] for key in METRIC_CATEGORY_LABELS}
    for spec in sorted(specs.values(), key=_metric_sort_key):
        categories.setdefault(spec.metric_family, []).append(
            {
                "metric_name": spec.metric_name,
                "display_label": spec.display_label,
                "preferred_visualisation": spec.preferred_visualisation,
            }
        )
    return {key: value for key, value in categories.items() if value}


def _metric_sort_key(spec: MetricVisualisationSpec) -> tuple[str, int, str]:
    bilateral_priority = {
        "injured_hka_angle_2d_deg": 0,
        "contralateral_hka_angle_2d_deg": 1,
        "hka_projected_bilateral_difference_deg": 2,
        "hka_projected_bilateral_absolute_difference_deg": 3,
    }
    return (
        spec.metric_family,
        bilateral_priority.get(spec.metric_name, 100),
        spec.display_label,
    )


def _display_label(metric_name: str) -> str:
    if metric_name.startswith("dynamic_rate:"):
        return f"Robust rate: {_display_label(metric_name.removeprefix('dynamic_rate:'))}"
    labels = {
        "path:compensated_x": "Camera-compensated projected path",
        "path:compensated_y": "Camera-compensated projected path y",
        "path:projected_heading_deg": "Projected heading",
        "path:projected_heading_change_deg": "Projected heading change",
        "path:normalized_projected_speed_per_s": "Projected speed",
        "left_knee_ankle_distance_normalized": (
            "Left projected segment-length / foreshortening diagnostic"
        ),
        "right_knee_ankle_distance_normalized": (
            "Right projected segment-length / foreshortening diagnostic"
        ),
        "left_hka_angle_2d_deg": "Left HKA angle",
        "right_hka_angle_2d_deg": "Right HKA angle",
        "injured_hka_angle_2d_deg": "Injured HKA angle",
        "contralateral_hka_angle_2d_deg": "Contralateral HKA angle",
        "hka_projected_bilateral_difference_deg": (
            "Injured-contralateral signed HKA difference"
        ),
        "hka_projected_bilateral_absolute_difference_deg": (
            "Injured-contralateral absolute HKA difference"
        ),
        "elbow_projected_bilateral_difference_deg": (
            "Injured-contralateral signed elbow difference"
        ),
        "elbow_projected_bilateral_absolute_difference_deg": (
            "Injured-contralateral absolute elbow difference"
        ),
        "knee_line_deviation_bilateral_difference": (
            "Injured-contralateral knee-line deviation difference"
        ),
        "knee_line_deviation_normalized_bilateral_difference": (
            "Injured-contralateral normalized knee-line deviation difference"
        ),
    }
    return labels.get(metric_name, metric_name.replace("_", " ").replace("2d", "2D").title())


def _metric_change(metric_name: str, start_value: float, end_value: float) -> float:
    if metric_name.endswith("_deg") and any(
        token in metric_name for token in ("orientation", "heading", "line", "upper_arm")
    ):
        return wrapped_angle_difference_deg(end_value, start_value)
    return end_value - start_value


def _angle_type_value(metric_name: str) -> str | None:
    angle_type = angle_type_for_metric(metric_name)
    return angle_type.value if angle_type is not None else None


def _canonical_change_fields(
    metric_name: str,
    start_value: float,
    end_value: float,
    existing_change: float,
) -> dict[str, float | str | None]:
    angle_type = angle_type_for_metric(metric_name)
    canonical_change = (
        angular_difference(start_value, end_value, angle_type)
        if angle_type is not None
        else existing_change
    )
    return {
        "raw_start_angle": start_value if angle_type is not None else None,
        "raw_end_angle": end_value if angle_type is not None else None,
        "angle_type": angle_type.value if angle_type is not None else None,
        "canonical_signed_change": float(canonical_change),
        "canonical_absolute_change": abs(float(canonical_change)),
    }


def _records_json(rows: pd.DataFrame) -> list[dict]:
    return [{str(key): _json_ready(value) for key, value in row.items()} for row in rows.to_dict(orient="records")]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value) if not isinstance(value, list | tuple | dict) else False:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if np.isnan(number) or np.isinf(number) else number
    return value


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
