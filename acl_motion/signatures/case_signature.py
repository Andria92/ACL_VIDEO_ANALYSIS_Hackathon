"""Clustering-ready case-level movement-signature prototype.

The signature is a compact, auditable representation of one human-validated ACL
event. It prepares future matrix construction without running clustering,
nearest-neighbor search, UMAP, association rules, or archetype assignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

SIGNATURE_VERSION = "m5_9_1_case_movement_signature_v1"
REGISTRY_VERSION = "m5_9_1_clustering_feature_registry_v1"

STANDARD_WINDOWS = {
    "whole_window": None,
    "final_1000ms": (-1000.0, 0.0),
    "final_500ms": (-500.0, 0.0),
    "final_250ms": (-250.0, 0.0),
    "progress_0_25": (0.0, 0.25),
    "progress_25_50": (0.25, 0.50),
    "progress_50_75": (0.50, 0.75),
    "progress_75_100": (0.75, 1.0),
}

DIAGNOSTIC_ONLY_PATTERNS = {
    "knee_ankle_distance_normalized": (
        "Projected same-segment length is retained as foreshortening/landmark-quality diagnostic only."
    )
}


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One explicitly approved candidate descriptor for future matrix construction."""

    feature_name: str
    family: str
    source_metric: str
    aggregation: str
    time_scope: str
    required_view: str
    minimum_coverage: float
    core_or_optional: str
    scientific_reason: str
    enabled_for_clustering: bool
    version: str = REGISTRY_VERSION


@dataclass(frozen=True, slots=True)
class CaseMovementSignature:
    """Structured case-level signature with long and matrix-preview tables."""

    case_id: str
    source_id: str
    run_metadata: dict[str, Any]
    registry: pd.DataFrame
    long_table: pd.DataFrame
    matrix_preview: pd.DataFrame
    summary: dict[str, Any]


def build_clustering_feature_registry(*, path_enabled: bool = False) -> pd.DataFrame:
    """Return the explicit future-clustering descriptor registry."""

    entries = [
        RegistryEntry(
            "injured_hka_median_whole",
            "HKA",
            "injured_hka_angle_2d_deg",
            "median",
            "whole_window",
            "generic_projected_view",
            0.55,
            "core_candidate",
            "Compact central tendency of injured-side projected HKA chain.",
            True,
        ),
        RegistryEntry(
            "injured_hka_iqr_whole",
            "HKA",
            "injured_hka_angle_2d_deg",
            "iqr",
            "whole_window",
            "generic_projected_view",
            0.55,
            "optional_characterization",
            "Robust variability of injured-side projected HKA chain.",
            True,
        ),
        RegistryEntry(
            "injured_hka_start_end_change_whole",
            "HKA",
            "injured_hka_angle_2d_deg",
            "start_to_end_change",
            "whole_window",
            "generic_projected_view",
            0.55,
            "core_candidate",
            "Directional change of supported injured-side projected HKA evidence.",
            True,
        ),
        RegistryEntry(
            "contralateral_hka_start_end_change_whole",
            "HKA",
            "contralateral_hka_angle_2d_deg",
            "start_to_end_change",
            "whole_window",
            "generic_projected_view",
            0.55,
            "optional_characterization",
            "Companion projected HKA change without deleting left/right variables.",
            True,
        ),
        RegistryEntry(
            "hka_bilateral_abs_median_whole",
            "Bilateral",
            "hka_projected_bilateral_absolute_difference_deg",
            "median",
            "whole_window",
            "generic_projected_view",
            0.55,
            "core_candidate",
            "Compact projected bilateral HKA relationship magnitude.",
            True,
        ),
        RegistryEntry(
            "hka_bilateral_abs_max_whole",
            "Bilateral",
            "hka_projected_bilateral_absolute_difference_deg",
            "maximum",
            "whole_window",
            "generic_projected_view",
            0.55,
            "optional_characterization",
            "Peak supported projected bilateral HKA relationship.",
            True,
        ),
        RegistryEntry(
            "hka_bilateral_signed_change_whole",
            "Bilateral",
            "hka_projected_bilateral_difference_deg",
            "start_to_end_change",
            "whole_window",
            "generic_projected_view",
            0.55,
            "core_candidate",
            "Signed projected bilateral HKA relationship trajectory change.",
            True,
        ),
        RegistryEntry(
            "hka_bilateral_abs_median_final_500ms",
            "Bilateral",
            "hka_projected_bilateral_absolute_difference_deg",
            "median",
            "final_500ms",
            "generic_projected_view",
            0.45,
            "core_candidate",
            "Final-window projected bilateral HKA relationship magnitude.",
            True,
        ),
        RegistryEntry(
            "trunk_axis_range_whole",
            "Trunk/Pelvis",
            "projected_trunk_axis_angle_deg",
            "range",
            "whole_window",
            "generic_projected_view",
            0.55,
            "core_candidate",
            "Supported projected trunk-axis orientation range.",
            True,
        ),
        RegistryEntry(
            "trunk_axis_change_whole",
            "Trunk/Pelvis",
            "projected_trunk_axis_angle_deg",
            "start_to_end_change",
            "whole_window",
            "generic_projected_view",
            0.55,
            "core_candidate",
            "Supported projected trunk-axis orientation change.",
            True,
        ),
        RegistryEntry(
            "shoulder_pelvis_orientation_range_whole",
            "Trunk/Pelvis",
            "projected_shoulder_pelvis_orientation_difference_deg",
            "range",
            "whole_window",
            "generic_projected_view",
            0.55,
            "optional_characterization",
            "Projected shoulder-line relative to hip-line variation.",
            True,
        ),
        RegistryEntry(
            "upper_arm_orientation_change_whole",
            "Upper Body",
            "right_upper_arm_orientation_2d_deg",
            "start_to_end_change",
            "whole_window",
            "generic_projected_view",
            0.45,
            "optional_characterization",
            "Compact upper-arm projected orientation change; redundancy controlled later.",
            True,
        ),
        RegistryEntry(
            "elbow_bilateral_abs_median_whole",
            "Upper Body",
            "elbow_projected_bilateral_absolute_difference_deg",
            "median",
            "whole_window",
            "generic_projected_view",
            0.45,
            "optional_characterization",
            "Compact bilateral projected elbow relationship.",
            True,
        ),
        RegistryEntry(
            "path_heading_change_whole",
            "Movement Path",
            "path:projected_heading_deg",
            "start_to_end_change",
            "whole_window",
            "path_validated_projected_view",
            0.55,
            "optional_characterization",
            "Only enabled after movement path passes scientific QA.",
            path_enabled,
        ),
        RegistryEntry(
            "path_speed_median_final_500ms",
            "Movement Path",
            "path:normalized_projected_speed_per_s",
            "median",
            "final_500ms",
            "path_validated_projected_view",
            0.45,
            "optional_characterization",
            "Final-window projected speed descriptor only if path passes QA.",
            path_enabled,
        ),
        RegistryEntry(
            "phase_count_supported",
            "Phase Structure",
            "movement_phases",
            "count",
            "whole_window",
            "generic_projected_view",
            0.0,
            "optional_characterization",
            "Within-case number of supported phases; not used to align phase numbers across cases.",
            True,
        ),
        RegistryEntry(
            "strongest_transition_timing_normalized",
            "Phase Structure",
            "movement_phases",
            "normalized_timing_of_maximum",
            "whole_window",
            "generic_projected_view",
            0.0,
            "optional_characterization",
            "Normalized timing of strongest within-case transition.",
            True,
        ),
        RegistryEntry(
            "strongest_transition_magnitude",
            "Phase Structure",
            "movement_phases",
            "maximum",
            "whole_window",
            "generic_projected_view",
            0.0,
            "optional_characterization",
            "Magnitude of strongest supported multivariate transition.",
            True,
        ),
    ]
    return pd.DataFrame([asdict(entry) for entry in entries])


def build_case_movement_signature(
    *,
    case_id: str,
    source_id: str,
    dynamic_df: pd.DataFrame,
    path_df: pd.DataFrame,
    movement_story: dict,
    movement_window: dict,
    path_quality_summary: dict | None = None,
) -> CaseMovementSignature:
    """Build a Press-only prototype fixed-length signature preview."""

    path_enabled = bool((path_quality_summary or {}).get("overall_status") == "SUPPORTED")
    registry = build_clustering_feature_registry(path_enabled=path_enabled)
    long_rows: list[dict[str, Any]] = []
    for _, entry in registry.iterrows():
        row = _descriptor_row(
            case_id=case_id,
            source_id=source_id,
            entry=entry,
            dynamic_df=dynamic_df,
            path_df=path_df,
            movement_story=movement_story,
            movement_window=movement_window,
            path_enabled=path_enabled,
        )
        long_rows.append(row)
    long_table = pd.DataFrame(long_rows)
    matrix = _matrix_preview(case_id, long_table)
    summary = _signature_summary(long_table, registry)
    run_metadata = {
        "run_id": f"{case_id}_signature_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": datetime.now(UTC).isoformat(),
        "case_id": case_id,
        "source_id": source_id,
        "signature_version": SIGNATURE_VERSION,
        "registry_version": REGISTRY_VERSION,
        "movement_window": movement_window,
        "path_quality_status": (path_quality_summary or {}).get("overall_status", "UNAVAILABLE"),
        "missingness_policy": "unsupported descriptors remain NA; no zero filling or mean imputation",
        "phase_alignment_policy": "phase numbers are within-case labels and are not cross-case alignment keys",
    }
    return CaseMovementSignature(
        case_id=case_id,
        source_id=source_id,
        run_metadata=run_metadata,
        registry=registry,
        long_table=long_table,
        matrix_preview=matrix,
        summary=summary,
    )


def _descriptor_row(
    *,
    case_id: str,
    source_id: str,
    entry: pd.Series,
    dynamic_df: pd.DataFrame,
    path_df: pd.DataFrame,
    movement_story: dict,
    movement_window: dict,
    path_enabled: bool,
) -> dict[str, Any]:
    descriptor = str(entry["feature_name"])
    source_metric = str(entry["source_metric"])
    family = str(entry["family"])
    reason = ""
    excluded_by = ""
    value = np.nan
    coverage = np.nan
    supported_frames = 0
    relevant_frames = 0
    evidence = "UNAVAILABLE"
    timing_ms = np.nan
    timing_progress = np.nan
    view_suitability = _view_suitability(entry, path_enabled=path_enabled)
    if _diagnostic_only(source_metric):
        excluded_by = "scientific_rule"
        reason = _diagnostic_only(source_metric)
    elif source_metric.startswith("path:") and not path_enabled:
        excluded_by = "evidence"
        reason = "Movement path remains QA_REQUIRED/UNAVAILABLE and is not eligible."
    elif source_metric == "movement_phases":
        value, evidence, reason, timing_ms, timing_progress = _phase_descriptor(
            entry,
            movement_story,
            movement_window,
        )
        supported_frames = len(movement_story.get("phases", []))
        relevant_frames = supported_frames
        coverage = 1.0 if supported_frames else 0.0
    else:
        rows = _metric_rows(source_metric, dynamic_df, path_df)
        scoped = _scope_rows(rows, str(entry["time_scope"]), movement_window)
        relevant_frames = len(scoped)
        supported = scoped[scoped["evidence_status"].eq("SUPPORTED") & scoped["value"].notna()]
        supported_frames = len(supported)
        coverage = supported_frames / relevant_frames if relevant_frames else 0.0
        if coverage < float(entry["minimum_coverage"]):
            excluded_by = "evidence"
            reason = "Coverage below descriptor minimum."
        else:
            value, timing_ms, timing_progress = _aggregate(
                supported,
                str(entry["aggregation"]),
                movement_window,
            )
            if np.isfinite(value):
                evidence = "SUPPORTED"
            else:
                excluded_by = "evidence"
                reason = "Aggregation unavailable from supported samples."
    eligible = (
        bool(entry["enabled_for_clustering"])
        and evidence == "SUPPORTED"
        and not excluded_by
    )
    if not bool(entry["enabled_for_clustering"]) and not excluded_by:
        excluded_by = "scientific_rule"
        reason = "Registry entry is disabled for future clustering eligibility."
    return {
        "case_id": case_id,
        "source_id": source_id,
        "feature_name": descriptor,
        "family": family,
        "source_metric": source_metric,
        "aggregation": str(entry["aggregation"]),
        "time_scope": str(entry["time_scope"]),
        "value": float(value) if np.isfinite(value) else np.nan,
        "evidence": evidence,
        "coverage": float(coverage) if np.isfinite(coverage) else np.nan,
        "supported_frames": int(supported_frames),
        "relevant_frames": int(relevant_frames),
        "missing_frames": int(max(relevant_frames - supported_frames, 0)),
        "missingness_fraction": (
            float(1.0 - coverage) if np.isfinite(coverage) else np.nan
        ),
        "eligible_for_future_clustering": bool(eligible),
        "core_or_optional": str(entry["core_or_optional"]),
        "excluded_by": excluded_by,
        "exclusion_reason": reason,
        "required_view": str(entry["required_view"]),
        "view_suitability": view_suitability,
        "source_feature_names": source_metric,
        "provenance": "human_target_annotation + processed_pose + robust_dynamic_features",
        "movement_end_relative_ms": float(timing_ms) if np.isfinite(timing_ms) else np.nan,
        "normalized_movement_progress": (
            float(timing_progress) if np.isfinite(timing_progress) else np.nan
        ),
        "signature_version": SIGNATURE_VERSION,
        "registry_version": REGISTRY_VERSION,
    }


def _view_suitability(entry: pd.Series, *, path_enabled: bool) -> str:
    required_view = str(entry["required_view"])
    if required_view == "path_validated_projected_view" and not path_enabled:
        return "PATH_QA_REQUIRED"
    return "SUPPORTED_GENERIC_PROJECTED_VIEW"


def _metric_rows(metric: str, dynamic_df: pd.DataFrame, path_df: pd.DataFrame) -> pd.DataFrame:
    if metric.startswith("path:"):
        column = metric.removeprefix("path:")
        if path_df.empty or column not in path_df.columns:
            return pd.DataFrame()
        rows = path_df.copy()
        rows["value"] = pd.to_numeric(rows[column], errors="coerce").where(
            rows["path_status"].eq("SUPPORTED")
        )
        rows["evidence_status"] = rows["path_status"].where(
            rows["path_status"].eq("SUPPORTED"),
            "UNAVAILABLE",
        )
        return rows
    rows = dynamic_df[dynamic_df["feature_name"].eq(metric)].copy()
    if rows.empty:
        return rows
    rows["value"] = pd.to_numeric(rows["feature_value"], errors="coerce").where(
        rows["feature_status"].eq("SUPPORTED")
    )
    rows["evidence_status"] = rows["feature_status"].where(
        rows["feature_status"].eq("SUPPORTED"),
        "UNAVAILABLE",
    )
    return rows


def _scope_rows(rows: pd.DataFrame, scope: str, movement_window: dict) -> pd.DataFrame:
    if rows.empty:
        return rows
    scoped = rows.copy()
    duration = _duration_ms(movement_window)
    if "normalized_movement_progress" not in scoped.columns:
        elapsed = (
            pd.to_numeric(scoped["movement_elapsed_ms"], errors="coerce")
            if "movement_elapsed_ms" in scoped.columns
            else pd.Series(np.nan, index=scoped.index)
        )
        if elapsed.notna().any() and duration > 0:
            scoped["normalized_movement_progress"] = elapsed / duration
        else:
            end_relative = (
                pd.to_numeric(scoped["movement_end_relative_ms"], errors="coerce")
                if "movement_end_relative_ms" in scoped.columns
                else pd.Series(np.nan, index=scoped.index)
            )
            scoped["normalized_movement_progress"] = 1.0 + end_relative / duration if duration > 0 else np.nan
    bounds = STANDARD_WINDOWS.get(scope)
    if bounds is None:
        return scoped
    start, end = bounds
    if scope.startswith("progress"):
        return scoped[
            scoped["normalized_movement_progress"].ge(start)
            & scoped["normalized_movement_progress"].lt(end if end < 1.0 else end + 1e-9)
        ].copy()
    return scoped[
        scoped["movement_end_relative_ms"].astype(float).between(start, end, inclusive="both")
    ].copy()


def _aggregate(
    supported: pd.DataFrame,
    aggregation: str,
    movement_window: dict,
) -> tuple[float, float, float]:
    values = pd.to_numeric(supported["value"], errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, np.nan
    ordered = supported.loc[values.index].sort_values("movement_end_relative_ms")
    if aggregation == "median":
        return float(values.median()), np.nan, np.nan
    if aggregation == "iqr":
        return float(values.quantile(0.75) - values.quantile(0.25)), np.nan, np.nan
    if aggregation == "range":
        return float(values.max() - values.min()), np.nan, np.nan
    if aggregation == "maximum":
        row = ordered.loc[pd.to_numeric(ordered["value"], errors="coerce").idxmax()]
        return _value_with_timing(row, movement_window)
    if aggregation == "start_to_end_change":
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        value = float(last["value"] - first["value"])
        timing_ms = float(last.get("movement_end_relative_ms", np.nan))
        return value, timing_ms, _progress(last, movement_window)
    return np.nan, np.nan, np.nan


def _phase_descriptor(
    entry: pd.Series,
    movement_story: dict,
    movement_window: dict,
) -> tuple[float, str, str, float, float]:
    phases = movement_story.get("phases", [])
    transitions = movement_story.get("transitions", [])
    aggregation = str(entry["aggregation"])
    if aggregation == "count":
        if phases:
            return float(len(phases)), "SUPPORTED", "", np.nan, np.nan
        return (
            0.0,
            "UNAVAILABLE",
            "Phase segmentation unavailable from current supported evidence.",
            np.nan,
            np.nan,
        )
    if not transitions:
        return np.nan, "UNAVAILABLE", "No supported phase transitions.", np.nan, np.nan
    strongest = max(transitions, key=lambda item: float(item.get("change_score", 0.0) or 0.0))
    timing_ms = float(strongest.get("movement_end_relative_ms", np.nan))
    progress = _progress_from_end_relative(timing_ms, movement_window)
    if aggregation == "maximum":
        return float(strongest.get("change_score", np.nan)), "SUPPORTED", "", timing_ms, progress
    if aggregation == "normalized_timing_of_maximum":
        return progress, "SUPPORTED", "", timing_ms, progress
    return np.nan, "UNAVAILABLE", "Unsupported phase-structure aggregation.", np.nan, np.nan


def _value_with_timing(row: pd.Series, movement_window: dict) -> tuple[float, float, float]:
    timing_ms = float(row.get("movement_end_relative_ms", np.nan))
    return float(row["value"]), timing_ms, _progress(row, movement_window)


def _progress(row: pd.Series, movement_window: dict) -> float:
    value = row.get("normalized_movement_progress")
    if value is not None and pd.notna(value):
        return float(value)
    timing_ms = float(row.get("movement_end_relative_ms", np.nan))
    return _progress_from_end_relative(timing_ms, movement_window)


def _progress_from_end_relative(timing_ms: float, movement_window: dict) -> float:
    duration = _duration_ms(movement_window)
    if not np.isfinite(timing_ms) or duration <= 0:
        return np.nan
    return float(1.0 + timing_ms / duration)


def _duration_ms(movement_window: dict) -> float:
    duration = movement_window.get("duration_ms", movement_window.get("movement_duration_ms"))
    if duration is None:
        start = movement_window.get("movement_start_frame")
        end = movement_window.get("movement_end_frame")
        return float(end - start) if start is not None and end is not None else np.nan
    return float(duration)


def _diagnostic_only(metric: str) -> str:
    for pattern, reason in DIAGNOSTIC_ONLY_PATTERNS.items():
        if pattern in metric:
            return reason
    return ""


def _matrix_preview(case_id: str, long_table: pd.DataFrame) -> pd.DataFrame:
    values = {
        row["feature_name"]: row["value"]
        for _, row in long_table.iterrows()
        if bool(row["eligible_for_future_clustering"])
    }
    return pd.DataFrame([{"case_id": case_id, **values}])


def _signature_summary(long_table: pd.DataFrame, registry: pd.DataFrame) -> dict[str, Any]:
    return {
        "candidate_descriptor_count": len(registry),
        "eligible_descriptor_count": int(long_table["eligible_for_future_clustering"].sum()),
        "unavailable_descriptor_count": int(long_table["evidence"].ne("SUPPORTED").sum()),
        "excluded_descriptor_count": int(long_table["excluded_by"].ne("").sum()),
        "exclusion_counts": long_table["excluded_by"].replace("", np.nan).dropna().value_counts().to_dict(),
        "descriptor_counts_by_family": long_table["family"].value_counts().to_dict(),
    }
