"""Feature-aware geometry coverage for human-reviewed movement evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TARGET_PRESENT_FRAME_STATUSES = {
    "VALID_TARGET",
    "PARTIAL_POSE",
    "LOW_POSE_CONFIDENCE",
}


def build_geometry_coverage_evidence(
    dynamic_df: pd.DataFrame,
    frame_quality: pd.DataFrame,
) -> dict[str, Any]:
    """Return complementary, non-substitutable geometry coverage measures.

    ``movement_window_coverage`` keeps every source frame in the denominator.
    ``reviewed_frame_yield`` asks how often a feature was supported specifically
    on frames whose target identity a human accepted. ``target_present_yield``
    asks the same question on frames with defensible target-region pose evidence.
    None of the conditional yields replaces complete-window coverage.
    """

    required_dynamic = {"source_frame_index", "feature_name", "feature_status"}
    missing_dynamic = required_dynamic.difference(dynamic_df.columns)
    if missing_dynamic:
        raise ValueError(
            "Dynamic feature evidence is missing required columns: "
            f"{sorted(missing_dynamic)}"
        )
    required_quality = {"source_frame_index", "frame_status"}
    missing_quality = required_quality.difference(frame_quality.columns)
    if missing_quality:
        raise ValueError(
            "Frame quality evidence is missing required columns: "
            f"{sorted(missing_quality)}"
        )

    timing = _frame_timing(dynamic_df)
    all_frames = set(timing["source_frame_index"].astype(int))
    quality = (
        frame_quality.sort_values("source_frame_index")
        .drop_duplicates("source_frame_index", keep="last")
        .copy()
    )
    quality["source_frame_index"] = quality["source_frame_index"].astype(int)
    quality = quality[quality["source_frame_index"].isin(all_frames)]

    accepted = _bool_series(quality, "human_target_accepted")
    unavailable = _bool_series(quality, "human_target_unavailable")
    observed = pd.to_numeric(
        quality.get("observed_landmark_count", pd.Series(0, index=quality.index)),
        errors="coerce",
    ).fillna(0).gt(0)
    valid_target = _bool_series(quality, "valid_target_frame")
    target_present = (
        ~unavailable
        & observed
        & (
            accepted
            | valid_target
            | quality["frame_status"].astype(str).isin(TARGET_PRESENT_FRAME_STATUSES)
        )
    )
    accepted_frames = set(quality.loc[accepted, "source_frame_index"].astype(int))
    target_present_frames = set(
        quality.loc[target_present, "source_frame_index"].astype(int)
    )

    per_feature: list[dict[str, Any]] = []
    for feature_name, rows in dynamic_df.groupby("feature_name", sort=True):
        feature_rows = (
            rows.sort_values("source_frame_index")
            .drop_duplicates("source_frame_index", keep="last")
            .copy()
        )
        supported_mask = feature_rows["feature_status"].astype(str).eq("SUPPORTED")
        if "feature_value" in feature_rows:
            supported_mask &= pd.to_numeric(
                feature_rows["feature_value"], errors="coerce"
            ).notna()
        supported_frames = set(
            feature_rows.loc[supported_mask, "source_frame_index"].astype(int)
        )
        longest = _longest_supported_run(supported_frames, timing)
        per_feature.append(
            {
                "feature_name": str(feature_name),
                "movement_window_supported_frames": len(supported_frames),
                "movement_window_total_frames": len(all_frames),
                "movement_window_coverage": _fraction(
                    len(supported_frames), len(all_frames)
                ),
                "reviewed_supported_frames": len(supported_frames & accepted_frames),
                "reviewed_total_frames": len(accepted_frames),
                "reviewed_frame_yield": _optional_fraction(
                    len(supported_frames & accepted_frames), len(accepted_frames)
                ),
                "target_present_supported_frames": len(
                    supported_frames & target_present_frames
                ),
                "target_present_total_frames": len(target_present_frames),
                "target_present_yield": _optional_fraction(
                    len(supported_frames & target_present_frames),
                    len(target_present_frames),
                ),
                "longest_continuous_supported": longest,
            }
        )

    feature_count = len(per_feature)
    reviewed_opportunities = len(accepted_frames) * feature_count
    target_present_opportunities = len(target_present_frames) * feature_count
    reviewed_supported = sum(item["reviewed_supported_frames"] for item in per_feature)
    target_present_supported = sum(
        item["target_present_supported_frames"] for item in per_feature
    )
    window_supported = sum(
        item["movement_window_supported_frames"] for item in per_feature
    )
    window_opportunities = len(all_frames) * feature_count
    best_window = _best_feature(per_feature, "movement_window_coverage")
    best_reviewed = _best_feature(per_feature, "reviewed_frame_yield")
    best_target_present = _best_feature(per_feature, "target_present_yield")
    best_continuous = max(
        per_feature,
        key=lambda item: (
            item["longest_continuous_supported"]["frame_count"],
            item["movement_window_coverage"],
        ),
        default=None,
    )
    return {
        "definition_version": "feature_aware_geometry_coverage_v1",
        "movement_window_total_frames": len(all_frames),
        "accepted_target_identity_frames": len(accepted_frames),
        "target_present_frames": len(target_present_frames),
        "feature_count": feature_count,
        "movement_window_coverage": _optional_fraction(
            window_supported, window_opportunities
        ),
        "reviewed_frame_geometry_yield": _optional_fraction(
            reviewed_supported, reviewed_opportunities
        ),
        "target_present_geometry_yield": _optional_fraction(
            target_present_supported, target_present_opportunities
        ),
        "best_movement_window_feature": best_window,
        "best_reviewed_feature": best_reviewed,
        "best_target_present_feature": best_target_present,
        "best_continuous_feature": best_continuous,
        "per_feature": per_feature,
        "interpretation": {
            "movement_window_coverage": (
                "Supported feature-frame pairs divided by all feature-frame opportunities "
                "in the complete human Movement Window."
            ),
            "reviewed_frame_geometry_yield": (
                "Supported feature-frame pairs on human-accepted target-identity frames "
                "divided by all feature opportunities on those reviewed frames."
            ),
            "target_present_geometry_yield": (
                "Supported feature-frame pairs divided by feature opportunities on frames "
                "with defensible target-region pose evidence."
            ),
        },
    }


def _frame_timing(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["source_frame_index"]
    if "timestamp_ms" in dynamic_df:
        columns.append("timestamp_ms")
    timing = (
        dynamic_df[columns]
        .drop_duplicates("source_frame_index")
        .sort_values("source_frame_index")
        .reset_index(drop=True)
    )
    timing["source_frame_index"] = timing["source_frame_index"].astype(int)
    if "timestamp_ms" not in timing:
        timing["timestamp_ms"] = timing["source_frame_index"].astype(float)
    return timing


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == object:
        return values.fillna(False).map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes"}
        )
    return values.fillna(False).astype(bool)


def _longest_supported_run(
    supported_frames: set[int],
    timing: pd.DataFrame,
) -> dict[str, Any]:
    if not supported_frames:
        return {
            "start_frame": None,
            "end_frame": None,
            "frame_count": 0,
            "duration_ms": 0.0,
        }
    timestamps = {
        int(row.source_frame_index): float(row.timestamp_ms)
        for row in timing.itertuples(index=False)
    }
    ordered = sorted(supported_frames)
    runs: list[list[int]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if frame == runs[-1][-1] + 1:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    best = max(runs, key=lambda run: (len(run), -run[0]))
    deltas = np.diff(sorted(timestamps.values()))
    positive = deltas[deltas > 0]
    frame_duration = float(np.median(positive)) if len(positive) else 0.0
    return {
        "start_frame": best[0],
        "end_frame": best[-1],
        "frame_count": len(best),
        "duration_ms": max(
            0.0,
            timestamps.get(best[-1], 0.0)
            - timestamps.get(best[0], 0.0)
            + frame_duration,
        ),
    }


def _best_feature(items: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [item for item in items if item.get(metric) is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item[metric]),
            item["longest_continuous_supported"]["frame_count"],
        ),
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _optional_fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None
