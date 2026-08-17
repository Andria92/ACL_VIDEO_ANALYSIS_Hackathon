"""Milestone 2 frame and landmark quality classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from acl_motion.quality.models import FrameQualityStatus, LandmarkQualityStatus

CORE_LANDMARKS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
LOWER_LIMB_LANDMARKS = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
DIAGNOSTIC_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Configurable Milestone 2 quality thresholds."""

    landmark_confidence_threshold: float = 0.25
    frame_median_confidence_threshold: float = 0.35
    core_confidence_threshold: float = 0.35
    min_observed_landmarks: int = 8
    min_core_observed: int = 4
    min_lower_limb_observed: int = 4
    temporal_jump_body_scale_multiplier: float = 0.55
    temporal_jump_min_px: float = 70.0
    scale_jump_fraction: float = 0.55
    suspicious_padding_frames: int = 1
    min_valid_segment_frames: int = 5
    roi_centroid_padding_fraction: float = 0.15
    minimum_pose_roi_overlap_fraction: float = 0.45
    overlap_jump_body_scale_multiplier: float = 0.35


def classify_pose_quality(
    raw_pose: pd.DataFrame,
    config: QualityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify frame-level and landmark-level quality without editing raw pose."""

    cfg = config or QualityConfig()
    frame_quality = classify_frame_quality(raw_pose, cfg)
    landmark_quality = classify_landmark_quality(raw_pose, frame_quality, cfg)
    return frame_quality, landmark_quality


def build_target_identity_diagnostics(frame_quality: pd.DataFrame) -> pd.DataFrame:
    """Return framewise target-pose identity evidence for audit/UI explanation."""

    columns = [
        "frame_index",
        "source_frame_index",
        "analysis_frame_index",
        "timestamp_ms",
        "frame_status",
        "frame_rejection_reason",
        "valid_target_frame",
        "valid_segment_id",
        "backend_pose_count",
        "selected_pose_index",
        "multiple_pose_candidates",
        "target_overlap_ambiguous_flag",
        "identity_jump_flag",
        "identity_uncertain_flag",
        "pelvis_x",
        "pelvis_y",
        "shoulder_mid_x",
        "shoulder_mid_y",
        "pelvis_jump_px",
        "shoulder_jump_px",
        "body_scale_px",
        "body_scale_relative_change",
        "pose_bbox_x",
        "pose_bbox_y",
        "pose_bbox_width",
        "pose_bbox_height",
        "pose_centroid_x",
        "pose_centroid_y",
        "pose_scale_px",
        "pose_scale_relative_change",
        "roi_center_x",
        "roi_center_y",
        "roi_pose_centroid_distance_px",
        "roi_pose_centroid_distance_fraction",
        "pose_roi_overlap_fraction",
        "pose_centroid_consistent_with_roi",
    ]
    existing = [column for column in columns if column in frame_quality.columns]
    return frame_quality[existing].copy()


def classify_frame_quality(raw_pose: pd.DataFrame, config: QualityConfig | None = None) -> pd.DataFrame:
    """Create one frame-level quality row per video frame."""

    cfg = config or QualityConfig()
    frames: list[dict] = []
    for frame_index, frame in raw_pose.groupby("frame_index", sort=True):
        observed = frame[frame["observed"].astype(bool)]
        row: dict = {
            "frame_index": int(frame_index),
            "source_frame_index": _frame_int_value(frame, "source_frame_index", int(frame_index)),
            "analysis_frame_index": _frame_int_value(frame, "analysis_frame_index", int(frame_index)),
            "timestamp_ms": float(frame["timestamp_ms"].iloc[0]),
            "observed_landmark_count": len(observed),
            "observed_fraction": float(len(observed) / len(frame)) if len(frame) else 0.0,
            "median_confidence": _safe_median(frame["confidence"]),
            "core_observed_count": _observed_count(frame, CORE_LANDMARKS),
            "lower_limb_observed_count": _observed_count(frame, LOWER_LIMB_LANDMARKS),
            "core_median_confidence": _safe_median(_landmark_values(frame, CORE_LANDMARKS, "confidence")),
            "lower_limb_median_confidence": _safe_median(
                _landmark_values(frame, LOWER_LIMB_LANDMARKS, "confidence")
            ),
        }
        pelvis = _midpoint(frame, "left_hip", "right_hip")
        shoulders = _midpoint(frame, "left_shoulder", "right_shoulder")
        row["pelvis_x"] = pelvis[0]
        row["pelvis_y"] = pelvis[1]
        row["shoulder_mid_x"] = shoulders[0]
        row["shoulder_mid_y"] = shoulders[1]
        row["body_scale_px"] = _distance(pelvis, shoulders)
        row.update(_roi_pose_diagnostics(frame, observed))
        frames.append(row)

    quality = pd.DataFrame(frames).sort_values("frame_index").reset_index(drop=True)
    quality["pelvis_jump_px"] = _point_jump(quality, "pelvis_x", "pelvis_y")
    quality["shoulder_jump_px"] = _point_jump(quality, "shoulder_mid_x", "shoulder_mid_y")
    body_scales = pd.to_numeric(quality["body_scale_px"], errors="coerce").dropna()
    body_scale_ref = float(np.nanmedian(body_scales)) if not body_scales.empty else np.nan
    if not np.isfinite(body_scale_ref) or body_scale_ref <= 0:
        body_scale_ref = 100.0
    quality["body_scale_reference_px"] = body_scale_ref
    quality["body_scale_relative_change"] = (
        (quality["body_scale_px"] - quality["body_scale_px"].shift(1)).abs() / body_scale_ref
    )
    quality["pose_scale_relative_change"] = (
        (quality["pose_scale_px"] - quality["pose_scale_px"].shift(1)).abs() / body_scale_ref
    )

    jump_threshold = max(cfg.temporal_jump_min_px, body_scale_ref * cfg.temporal_jump_body_scale_multiplier)
    quality["identity_jump_flag"] = (
        (quality["pelvis_jump_px"] > jump_threshold)
        | (quality["shoulder_jump_px"] > jump_threshold)
        | (quality["body_scale_relative_change"] > cfg.scale_jump_fraction)
    )
    quality["identity_jump_flag"] = quality["identity_jump_flag"].fillna(False)
    overlap_jump_threshold = max(45.0, body_scale_ref * cfg.overlap_jump_body_scale_multiplier)
    quality["target_overlap_ambiguous_flag"] = (
        quality["multiple_pose_candidates"].fillna(False)
        & (
            (quality["pelvis_jump_px"] > overlap_jump_threshold)
            | (quality["shoulder_jump_px"] > overlap_jump_threshold)
            | (quality["pose_scale_relative_change"] > cfg.scale_jump_fraction)
        )
    )
    quality["target_overlap_ambiguous_flag"] = quality["target_overlap_ambiguous_flag"].fillna(False)
    if cfg.suspicious_padding_frames > 0 and quality["identity_jump_flag"].any():
        padded = quality["identity_jump_flag"].copy()
        for offset in range(1, cfg.suspicious_padding_frames + 1):
            padded = padded | quality["identity_jump_flag"].shift(offset, fill_value=False)
            padded = padded | quality["identity_jump_flag"].shift(-offset, fill_value=False)
        quality["identity_uncertain_flag"] = padded
    else:
        quality["identity_uncertain_flag"] = quality["identity_jump_flag"]

    statuses: list[str] = []
    reasons: list[str] = []
    for row in quality.to_dict(orient="records"):
        status, reason = _frame_status(row, cfg)
        statuses.append(status.value)
        reasons.append(reason)
    quality["frame_status"] = statuses
    quality["frame_rejection_reason"] = reasons
    quality["valid_target_frame"] = quality["frame_status"].eq(FrameQualityStatus.VALID_TARGET.value)
    quality["valid_segment_id"] = _valid_segments(quality["valid_target_frame"])
    _reject_short_valid_segments(quality, cfg)
    return quality


def classify_landmark_quality(
    raw_pose: pd.DataFrame,
    frame_quality: pd.DataFrame,
    config: QualityConfig | None = None,
) -> pd.DataFrame:
    """Create one landmark-level quality row per raw pose row."""

    cfg = config or QualityConfig()
    merged = raw_pose.merge(
        frame_quality[
            [
                "frame_index",
                "frame_status",
                "valid_target_frame",
                "identity_uncertain_flag",
                "valid_segment_id",
            ]
        ],
        on="frame_index",
        how="left",
    )
    merged = merged.sort_values(["landmark_name", "frame_index"]).reset_index(drop=True)
    merged["landmark_jump_px"] = np.nan
    for _, group in merged.groupby("landmark_name", sort=False):
        jumps = _point_jump(group, "x_px", "y_px")
        merged.loc[group.index, "landmark_jump_px"] = jumps.to_numpy()
    jump_threshold = max(
        cfg.temporal_jump_min_px,
        float(frame_quality["body_scale_reference_px"].iloc[0]) * cfg.temporal_jump_body_scale_multiplier,
    )
    merged["landmark_temporal_outlier"] = merged["landmark_jump_px"] > jump_threshold

    statuses: list[str] = []
    reasons: list[str] = []
    for row in merged.to_dict(orient="records"):
        status, reason = _landmark_status(row, cfg)
        statuses.append(status.value)
        reasons.append(reason)
    merged["landmark_status"] = statuses
    merged["landmark_rejection_reason"] = reasons
    if "source_frame_index" not in merged.columns:
        merged["source_frame_index"] = merged["frame_index"]
    if "analysis_frame_index" not in merged.columns:
        merged["analysis_frame_index"] = merged["frame_index"]
    return merged[
        [
            "case_id",
            "source_id",
            "frame_index",
            "source_frame_index",
            "analysis_frame_index",
            "timestamp_ms",
            "landmark_name",
            "frame_status",
            "valid_target_frame",
            "valid_segment_id",
            "landmark_status",
            "landmark_rejection_reason",
            "landmark_jump_px",
            "landmark_temporal_outlier",
        ]
    ]


def _frame_status(row: dict, cfg: QualityConfig) -> tuple[FrameQualityStatus, str]:
    if row["observed_landmark_count"] == 0:
        return FrameQualityStatus.TARGET_NOT_FOUND, "No observed landmarks returned by pose backend."
    if row["observed_landmark_count"] < cfg.min_observed_landmarks:
        return FrameQualityStatus.PARTIAL_POSE, "Too few observed landmarks."
    if row["core_observed_count"] < cfg.min_core_observed:
        return FrameQualityStatus.PARTIAL_POSE, "Insufficient core landmark coverage."
    if row["lower_limb_observed_count"] < cfg.min_lower_limb_observed:
        return FrameQualityStatus.PARTIAL_POSE, "Insufficient lower-limb landmark coverage."
    if not bool(row.get("pose_centroid_consistent_with_roi", True)):
        return (
            FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN,
            "Detected pose centroid is not defensibly inside the human target ROI.",
        )
    if _finite_lt(
        row.get("pose_roi_overlap_fraction", np.nan),
        cfg.minimum_pose_roi_overlap_fraction,
    ):
        return (
            FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN,
            "Detected skeleton has insufficient overlap with the human target ROI.",
        )
    if bool(row.get("target_overlap_ambiguous_flag", False)):
        return (
            FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN,
            "TARGET_OVERLAP_AMBIGUOUS: multiple pose candidates plus a target-continuity jump.",
        )
    if bool(row["identity_uncertain_flag"]):
        return FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN, "Temporal/body-scale discontinuity suggests target identity uncertainty."
    if (
        _finite_lt(row["median_confidence"], cfg.frame_median_confidence_threshold)
        or _finite_lt(row["core_median_confidence"], cfg.core_confidence_threshold)
    ):
        return FrameQualityStatus.LOW_POSE_CONFIDENCE, "Frame/core landmark confidence below threshold."
    return FrameQualityStatus.VALID_TARGET, ""


def _landmark_status(row: dict, cfg: QualityConfig) -> tuple[LandmarkQualityStatus, str]:
    if not bool(row["observed"]):
        return LandmarkQualityStatus.MISSING, "Landmark is unavailable in raw pose."
    if row["frame_status"] == FrameQualityStatus.TARGET_IDENTITY_UNCERTAIN.value:
        return LandmarkQualityStatus.IDENTITY_UNCERTAIN, "Frame-level target identity is uncertain."
    if row["frame_status"] in {
        FrameQualityStatus.TARGET_NOT_FOUND.value,
        FrameQualityStatus.PARTIAL_POSE.value,
        FrameQualityStatus.INVALID_TRACK_SEGMENT.value,
    }:
        return LandmarkQualityStatus.REJECTED, f"Frame status is {row['frame_status']}."
    if _finite_lt(row["confidence"], cfg.landmark_confidence_threshold):
        return LandmarkQualityStatus.LOW_CONFIDENCE, "Landmark confidence below threshold."
    if bool(row["landmark_temporal_outlier"]):
        return LandmarkQualityStatus.TEMPORAL_OUTLIER, "Landmark displacement exceeds temporal continuity threshold."
    return LandmarkQualityStatus.OBSERVED_VALID, ""


def _observed_count(frame: pd.DataFrame, landmarks: tuple[str, ...]) -> int:
    subset = frame[frame["landmark_name"].isin(landmarks)]
    return int(subset["observed"].astype(bool).sum())


def _landmark_values(frame: pd.DataFrame, landmarks: tuple[str, ...], column: str) -> pd.Series:
    return frame[frame["landmark_name"].isin(landmarks)][column]


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    with np.errstate(all="ignore"):
        value = np.nanmedian(numeric)
    return float(value) if np.isfinite(value) else np.nan


def _frame_int_value(frame: pd.DataFrame, column: str, default: int) -> int:
    if column not in frame.columns:
        return default
    value = frame[column].iloc[0]
    if pd.isna(value):
        return default
    return int(value)


def _midpoint(frame: pd.DataFrame, left_name: str, right_name: str) -> tuple[float, float]:
    points = []
    for name in (left_name, right_name):
        row = frame[frame["landmark_name"].eq(name)]
        if not row.empty and bool(row["observed"].iloc[0]):
            x_value = row["x_px"].iloc[0]
            y_value = row["y_px"].iloc[0]
            if np.isfinite(x_value) and np.isfinite(y_value):
                points.append((float(x_value), float(y_value)))
    if len(points) != 2:
        return np.nan, np.nan
    return (points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2


def _roi_pose_diagnostics(frame: pd.DataFrame, observed: pd.DataFrame) -> dict:
    """Return explainable selected-pose relationship diagnostics for a human ROI."""

    metadata = _metadata_dict(frame.iloc[0].get("backend_metadata"))
    pose_count = int(metadata.get("pose_count", 1) or 1)
    selected_pose_index = int(metadata.get("selected_pose_index", 0) or 0)
    diagnostics: dict[str, Any] = {
        "backend_pose_count": pose_count,
        "selected_pose_index": selected_pose_index,
        "multiple_pose_candidates": pose_count > 1,
        "pose_bbox_x": np.nan,
        "pose_bbox_y": np.nan,
        "pose_bbox_width": np.nan,
        "pose_bbox_height": np.nan,
        "pose_centroid_x": np.nan,
        "pose_centroid_y": np.nan,
        "roi_center_x": np.nan,
        "roi_center_y": np.nan,
        "roi_pose_centroid_distance_px": np.nan,
        "roi_pose_centroid_distance_fraction": np.nan,
        "pose_roi_overlap_fraction": np.nan,
        "pose_scale_px": np.nan,
        "pose_scale_relative_change": np.nan,
        "pose_centroid_consistent_with_roi": True,
    }
    if observed.empty or not _has_roi_columns(frame):
        return diagnostics

    roi_x = float(frame["target_bbox_x"].iloc[0])
    roi_y = float(frame["target_bbox_y"].iloc[0])
    roi_w = float(frame["target_bbox_width"].iloc[0])
    roi_h = float(frame["target_bbox_height"].iloc[0])
    roi_center = (roi_x + roi_w / 2.0, roi_y + roi_h / 2.0)
    x1 = float(observed["x_px"].min())
    y1 = float(observed["y_px"].min())
    x2 = float(observed["x_px"].max())
    y2 = float(observed["y_px"].max())
    pose_w = max(x2 - x1, 0.0)
    pose_h = max(y2 - y1, 0.0)
    pose_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    roi_diag = float(np.hypot(roi_w, roi_h))
    centroid_distance = float(np.hypot(pose_center[0] - roi_center[0], pose_center[1] - roi_center[1]))
    ix1 = max(roi_x, x1)
    iy1 = max(roi_y, y1)
    ix2 = min(roi_x + roi_w, x2)
    iy2 = min(roi_y + roi_h, y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pose_area = pose_w * pose_h
    overlap_fraction = intersection / pose_area if pose_area > 0 else np.nan
    pad_x = roi_w * 0.15
    pad_y = roi_h * 0.15
    centroid_consistent = (
        roi_x - pad_x <= pose_center[0] <= roi_x + roi_w + pad_x
        and roi_y - pad_y <= pose_center[1] <= roi_y + roi_h + pad_y
    )
    diagnostics.update(
        {
            "pose_bbox_x": x1,
            "pose_bbox_y": y1,
            "pose_bbox_width": pose_w,
            "pose_bbox_height": pose_h,
            "pose_centroid_x": pose_center[0],
            "pose_centroid_y": pose_center[1],
            "roi_center_x": roi_center[0],
            "roi_center_y": roi_center[1],
            "roi_pose_centroid_distance_px": centroid_distance,
            "roi_pose_centroid_distance_fraction": centroid_distance / roi_diag if roi_diag > 0 else np.nan,
            "pose_roi_overlap_fraction": overlap_fraction,
            "pose_scale_px": float(np.hypot(pose_w, pose_h)),
            "pose_centroid_consistent_with_roi": bool(centroid_consistent),
        }
    )
    return diagnostics


def _metadata_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return {}


def _has_roi_columns(frame: pd.DataFrame) -> bool:
    return {
        "target_bbox_x",
        "target_bbox_y",
        "target_bbox_width",
        "target_bbox_height",
    }.issubset(frame.columns)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    if not all(np.isfinite(value) for value in (*a, *b)):
        return np.nan
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _point_jump(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    dx = df[x_col].diff()
    dy = df[y_col].diff()
    return np.sqrt(dx**2 + dy**2)


def _finite_lt(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value < threshold)


def _valid_segments(valid: pd.Series) -> pd.Series:
    segment_ids: list[int | None] = []
    current = 0
    in_segment = False
    for is_valid in valid.astype(bool):
        if is_valid:
            if not in_segment:
                current += 1
                in_segment = True
            segment_ids.append(current)
        else:
            in_segment = False
            segment_ids.append(None)
    return pd.Series(segment_ids, index=valid.index, dtype="Int64")


def _reject_short_valid_segments(quality: pd.DataFrame, cfg: QualityConfig) -> None:
    if cfg.min_valid_segment_frames <= 1:
        return
    valid_rows = quality.dropna(subset=["valid_segment_id"])
    if valid_rows.empty:
        return
    short_ids = [
        segment_id
        for segment_id, group in valid_rows.groupby("valid_segment_id")
        if len(group) < cfg.min_valid_segment_frames
    ]
    if not short_ids:
        return
    mask = quality["valid_segment_id"].isin(short_ids)
    quality.loc[mask, "frame_status"] = FrameQualityStatus.INVALID_TRACK_SEGMENT.value
    quality.loc[mask, "frame_rejection_reason"] = (
        "Valid-looking pose segment is too short to establish target identity continuity."
    )
    quality.loc[mask, "valid_target_frame"] = False
    quality["valid_segment_id"] = _valid_segments(quality["valid_target_frame"])
