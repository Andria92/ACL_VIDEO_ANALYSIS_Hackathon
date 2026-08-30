"""Camera-compensated projected movement-path analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from acl_motion.video.roi import BBox


class CameraMotionStatus(StrEnum):
    """QC state for estimated global/background motion."""

    SUPPORTED = "SUPPORTED"
    LOW_BACKGROUND_FEATURE_COUNT = "LOW_BACKGROUND_FEATURE_COUNT"
    UNSTABLE_TRANSFORM = "UNSTABLE_TRANSFORM"
    EXCESSIVE_TRANSFORM_RESIDUAL = "EXCESSIVE_TRANSFORM_RESIDUAL"
    CAMERA_CUT = "CAMERA_CUT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class CameraMotionConfig:
    """Sparse optical-flow settings for projected camera compensation."""

    max_corners: int = 350
    quality_level: float = 0.01
    min_distance_px: int = 8
    minimum_background_features: int = 25
    maximum_translation_mad_px: float = 8.0
    maximum_affine_residual_px: float = 6.0
    maximum_camera_step_px: float = 80.0
    maximum_affine_scale_change_fraction: float = 0.08
    target_roi_padding_fraction: float = 0.2


@dataclass(frozen=True, slots=True)
class PathAnalysisConfig:
    """Projected path descriptor settings."""

    minimum_path_samples: int = 5
    maximum_frame_gap: int = 1
    maximum_time_gap_multiplier: float = 1.5
    direction_window_fraction: float = 0.25
    minimum_direction_window_samples: int = 3
    slowdown_window_ms: float = 500.0
    maximum_step_to_median_ratio: float = 6.0
    minimum_supported_fraction_for_validation: float = 0.55


BODY_CENTER_COLUMNS = (
    "source_frame_index",
    "timestamp_ms",
    "movement_elapsed_ms",
    "movement_end_relative_ms",
    "center_x",
    "center_y",
    "center_source",
    "center_status",
)

PATH_COLUMNS = (
    "source_frame_index",
    "timestamp_ms",
    "movement_elapsed_ms",
    "movement_end_relative_ms",
    "center_x",
    "center_y",
    "center_source",
    "raw_dx",
    "raw_dy",
    "compensated_dx",
    "compensated_dy",
    "compensated_x",
    "compensated_y",
    "camera_dx",
    "camera_dy",
    "camera_compensation_method",
    "background_feature_count",
    "camera_motion_residual_px",
    "projected_heading_deg",
    "normalized_projected_speed_per_s",
    "path_segment_id",
    "path_status",
    "path_rejection_reason",
)


def estimate_background_camera_motion(
    video_path: str | Path,
    source_frames: list[int],
    *,
    roi_for_frame: Callable[[int], BBox],
    config: CameraMotionConfig | None = None,
) -> pd.DataFrame:
    """Estimate translation-only background motion between adjacent source frames.

    The target ROI is masked out before sparse optical flow so target-athlete motion
    does not dominate the global camera-motion estimate. The method intentionally
    does not attempt pitch calibration or true camera geometry.
    """

    cfg = config or CameraMotionConfig()
    import cv2

    frames = sorted(int(frame) for frame in source_frames)
    if not frames:
        return pd.DataFrame()
    capture = cv2.VideoCapture(str(video_path))
    rows = [
        _camera_motion_row(
            frames[0],
            0.0,
            0.0,
            0,
            None,
            CameraMotionStatus.INSUFFICIENT_EVIDENCE,
            method="translation_median_sparse_flow",
        )
    ]
    try:
        previous_frame = _read_gray(capture, frames[0])
        for previous_index, current_index in pairwise(frames):
            current_frame = _read_gray(capture, current_index)
            mask = np.full(previous_frame.shape, 255, dtype=np.uint8)
            try:
                roi = roi_for_frame(previous_index)
                x1, y1, x2, y2 = roi.pad(cfg.target_roi_padding_fraction).clamp(
                    previous_frame.shape[1],
                    previous_frame.shape[0],
                ).as_int_xyxy()
                mask[y1:y2, x1:x2] = 0
            except ValueError:
                pass
            corners = cv2.goodFeaturesToTrack(
                previous_frame,
                mask=mask,
                maxCorners=cfg.max_corners,
                qualityLevel=cfg.quality_level,
                minDistance=cfg.min_distance_px,
            )
            if corners is None or len(corners) < cfg.minimum_background_features:
                rows.append(
                    _camera_motion_row(
                        current_index,
                        0.0,
                        0.0,
                        0,
                        None,
                        CameraMotionStatus.LOW_BACKGROUND_FEATURE_COUNT,
                        method="translation_median_sparse_flow",
                    )
                )
                previous_frame = current_frame
                continue
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, corners, None)
            if next_points is None or status is None:
                rows.append(
                    _camera_motion_row(
                        current_index,
                        0.0,
                        0.0,
                        0,
                        None,
                        CameraMotionStatus.INSUFFICIENT_EVIDENCE,
                        method="translation_median_sparse_flow",
                    )
                )
                previous_frame = current_frame
                continue
            valid = status.reshape(-1).astype(bool)
            start = corners.reshape(-1, 2)[valid]
            end = next_points.reshape(-1, 2)[valid]
            if len(start) < cfg.minimum_background_features:
                rows.append(
                    _camera_motion_row(
                        current_index,
                        0.0,
                        0.0,
                        len(start),
                        None,
                        CameraMotionStatus.LOW_BACKGROUND_FEATURE_COUNT,
                        method="translation_median_sparse_flow",
                    )
                )
                previous_frame = current_frame
                continue
            displacement = end - start
            median_dx, median_dy = np.median(displacement, axis=0)
            residual = np.median(np.linalg.norm(displacement - [median_dx, median_dy], axis=1))
            if np.hypot(median_dx, median_dy) > cfg.maximum_camera_step_px:
                status_code = CameraMotionStatus.CAMERA_CUT
            elif residual > cfg.maximum_translation_mad_px:
                status_code = CameraMotionStatus.UNSTABLE_TRANSFORM
            else:
                status_code = CameraMotionStatus.SUPPORTED
            rows.append(
                _camera_motion_row(
                    current_index,
                    float(median_dx),
                    float(median_dy),
                    len(start),
                    float(residual),
                    status_code,
                    method="translation_median_sparse_flow",
                )
            )
            previous_frame = current_frame
    finally:
        capture.release()
    return pd.DataFrame(rows)


def estimate_background_camera_motion_affine(
    video_path: str | Path,
    source_frames: list[int],
    *,
    roi_for_frame: Callable[[int], BBox],
    config: CameraMotionConfig | None = None,
) -> pd.DataFrame:
    """Estimate partial affine background motion with sparse optical flow and RANSAC."""

    cfg = config or CameraMotionConfig()
    import cv2

    frames = sorted(int(frame) for frame in source_frames)
    if not frames:
        return pd.DataFrame()
    capture = cv2.VideoCapture(str(video_path))
    rows = [
        _camera_motion_row(
            frames[0],
            0.0,
            0.0,
            0,
            None,
            CameraMotionStatus.INSUFFICIENT_EVIDENCE,
            method="partial_affine_ransac_sparse_flow",
            affine_matrix=_identity_affine(),
        )
    ]
    try:
        previous_frame = _read_gray(capture, frames[0])
        for previous_index, current_index in pairwise(frames):
            current_frame = _read_gray(capture, current_index)
            mask = np.full(previous_frame.shape, 255, dtype=np.uint8)
            try:
                roi = roi_for_frame(previous_index)
                x1, y1, x2, y2 = roi.pad(cfg.target_roi_padding_fraction).clamp(
                    previous_frame.shape[1],
                    previous_frame.shape[0],
                ).as_int_xyxy()
                mask[y1:y2, x1:x2] = 0
            except ValueError:
                pass
            corners = cv2.goodFeaturesToTrack(
                previous_frame,
                mask=mask,
                maxCorners=cfg.max_corners,
                qualityLevel=cfg.quality_level,
                minDistance=cfg.min_distance_px,
            )
            if corners is None or len(corners) < cfg.minimum_background_features:
                rows.append(
                    _camera_motion_row(
                        current_index,
                        0.0,
                        0.0,
                        0,
                        None,
                        CameraMotionStatus.LOW_BACKGROUND_FEATURE_COUNT,
                        method="partial_affine_ransac_sparse_flow",
                    )
                )
                previous_frame = current_frame
                continue
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_frame,
                current_frame,
                corners,
                None,
            )
            if next_points is None or status is None:
                rows.append(
                    _camera_motion_row(
                        current_index,
                        0.0,
                        0.0,
                        0,
                        None,
                        CameraMotionStatus.INSUFFICIENT_EVIDENCE,
                        method="partial_affine_ransac_sparse_flow",
                    )
                )
                previous_frame = current_frame
                continue
            valid = status.reshape(-1).astype(bool)
            start = corners.reshape(-1, 2)[valid]
            end = next_points.reshape(-1, 2)[valid]
            if len(start) < cfg.minimum_background_features:
                rows.append(
                    _camera_motion_row(
                        current_index,
                        0.0,
                        0.0,
                        len(start),
                        None,
                        CameraMotionStatus.LOW_BACKGROUND_FEATURE_COUNT,
                        method="partial_affine_ransac_sparse_flow",
                    )
                )
                previous_frame = current_frame
                continue
            matrix, inliers = cv2.estimateAffinePartial2D(
                start,
                end,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99,
            )
            if matrix is None or inliers is None:
                rows.append(
                    _camera_motion_row(
                        current_index,
                        0.0,
                        0.0,
                        len(start),
                        None,
                        CameraMotionStatus.UNSTABLE_TRANSFORM,
                        method="partial_affine_ransac_sparse_flow",
                    )
                )
                previous_frame = current_frame
                continue
            inlier_mask = inliers.reshape(-1).astype(bool)
            support = int(inlier_mask.sum())
            if support < cfg.minimum_background_features:
                rows.append(
                    _camera_motion_row(
                        current_index,
                        0.0,
                        0.0,
                        support,
                        None,
                        CameraMotionStatus.LOW_BACKGROUND_FEATURE_COUNT,
                        method="partial_affine_ransac_sparse_flow",
                        affine_matrix=matrix,
                    )
                )
                previous_frame = current_frame
                continue
            predicted = _apply_affine(matrix, start[inlier_mask])
            residuals = np.linalg.norm(predicted - end[inlier_mask], axis=1)
            residual = float(np.median(residuals))
            tx = float(matrix[0, 2])
            ty = float(matrix[1, 2])
            scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
            if np.hypot(tx, ty) > cfg.maximum_camera_step_px:
                status_code = CameraMotionStatus.CAMERA_CUT
            elif abs(scale - 1.0) > cfg.maximum_affine_scale_change_fraction:
                status_code = CameraMotionStatus.UNSTABLE_TRANSFORM
            elif residual > cfg.maximum_affine_residual_px:
                status_code = CameraMotionStatus.EXCESSIVE_TRANSFORM_RESIDUAL
            else:
                status_code = CameraMotionStatus.SUPPORTED
            rows.append(
                _camera_motion_row(
                    current_index,
                    tx,
                    ty,
                    support,
                    residual,
                    status_code,
                    method="partial_affine_ransac_sparse_flow",
                    affine_matrix=matrix,
                    affine_scale=scale,
                    affine_inlier_fraction=float(support / len(start)),
                )
            )
            previous_frame = current_frame
    finally:
        capture.release()
    return pd.DataFrame(rows)


def body_center_from_processed_pose(processed_pose: pd.DataFrame) -> pd.DataFrame:
    """Build a pelvis-midpoint projected body-center proxy from valid hips."""

    hip_rows = processed_pose[
        processed_pose["landmark_name"].isin(["left_hip", "right_hip"])
        & processed_pose["rejected"].eq(False)
        & processed_pose["smoothed_x"].notna()
        & processed_pose["smoothed_y"].notna()
    ]
    records = []
    for source_frame, rows in hip_rows.groupby("source_frame_index", sort=True):
        names = set(rows["landmark_name"])
        if {"left_hip", "right_hip"}.issubset(names):
            records.append(
                {
                    "source_frame_index": int(source_frame),
                    "timestamp_ms": float(rows["timestamp_ms"].iloc[0]),
                    "movement_elapsed_ms": None,
                    "movement_end_relative_ms": None,
                    "center_x": float(rows["smoothed_x"].mean()),
                    "center_y": float(rows["smoothed_y"].mean()),
                    "center_source": "pelvis_midpoint",
                    "center_status": "SUPPORTED",
                }
            )
    return pd.DataFrame.from_records(records, columns=BODY_CENTER_COLUMNS)


def compensate_projected_path(
    center_df: pd.DataFrame,
    camera_motion_df: pd.DataFrame,
    *,
    scale_reference_px: float,
    config: PathAnalysisConfig | None = None,
) -> pd.DataFrame:
    """Subtract estimated camera motion from a projected body-center trajectory.

    Path runs are never connected across missing target frames, camera transform
    failures, or non-contiguous source frames. Each defensible continuous run gets
    an explicit ``path_segment_id``.
    """

    cfg = config or PathAnalysisConfig()
    if center_df.empty:
        return pd.DataFrame(columns=PATH_COLUMNS)
    centers = center_df.sort_values("source_frame_index").reset_index(drop=True).copy()
    camera = camera_motion_df.set_index("source_frame_index")
    rows = []
    compensated_x = 0.0
    compensated_y = 0.0
    previous: pd.Series | None = None
    current_segment = 0
    median_dt = _median_timestamp_delta(centers)
    for _, current in centers.iterrows():
        source_frame = int(current["source_frame_index"])
        timestamp = float(current["timestamp_ms"])
        camera_row = camera.loc[source_frame] if source_frame in camera.index else None
        contiguous = _path_contiguous(previous, current, median_dt, cfg)
        if previous is None or not contiguous:
            current_segment += 1
            compensated_x = 0.0
            compensated_y = 0.0
            rows.append(
                _path_row(
                    current,
                    compensated_x,
                    compensated_y,
                    None,
                    None,
                    "SUPPORTED",
                    "",
                    path_segment_id=_segment_id(current_segment),
                    compensation_method=_camera_method(camera_row),
                )
            )
            previous = current
            continue
        if camera_row is None:
            rows.append(
                _path_row(
                    current,
                    None,
                    None,
                    None,
                    None,
                    "INSUFFICIENT_EVIDENCE",
                    "No camera-motion estimate for this frame.",
                    path_segment_id="",
                )
            )
            previous = None
            continue
        camera_status = str(camera_row["camera_motion_status"])
        if camera_status != CameraMotionStatus.SUPPORTED.value:
            rows.append(
                _path_row(
                    current,
                    None,
                    None,
                    None,
                    None,
                    camera_status,
                    "Camera compensation was not supported.",
                    path_segment_id="",
                    compensation_method=_camera_method(camera_row),
                    background_feature_count=_optional_int(camera_row.get("background_feature_count")),
                    camera_motion_residual_px=_optional_float(
                        camera_row.get("camera_motion_residual_px")
                    ),
                )
            )
            previous = None
            continue
        raw_dx = float(current["center_x"] - previous["center_x"])
        raw_dy = float(current["center_y"] - previous["center_y"])
        camera_dx, camera_dy = _camera_displacement_at_point(camera_row, previous)
        compensated_dx = raw_dx - camera_dx
        compensated_dy = raw_dy - camera_dy
        compensated_x += compensated_dx
        compensated_y += compensated_dy
        dt = (timestamp - float(previous["timestamp_ms"])) / 1000.0
        speed = (
            np.hypot(compensated_dx, compensated_dy) / scale_reference_px / dt
            if dt > 0 and scale_reference_px > 0
            else None
        )
        heading = projected_heading_deg(compensated_dx, compensated_dy)
        rows.append(
            _path_row(
                current,
                compensated_x,
                compensated_y,
                speed,
                heading,
                "SUPPORTED",
                "",
                path_segment_id=_segment_id(current_segment),
                compensation_method=_camera_method(camera_row),
                raw_dx=raw_dx,
                raw_dy=raw_dy,
                compensated_dx=compensated_dx,
                compensated_dy=compensated_dy,
                camera_dx=camera_dx,
                camera_dy=camera_dy,
                background_feature_count=_optional_int(camera_row.get("background_feature_count")),
                camera_motion_residual_px=(
                    float(camera_row["camera_motion_residual_px"])
                    if pd.notna(camera_row["camera_motion_residual_px"])
                    else None
                ),
            )
        )
        previous = current
    output = pd.DataFrame(rows)
    _reject_short_path_segments(output, cfg)
    return output


def projected_heading_deg(dx: float, dy: float) -> float | None:
    """Return projected video-plane heading using mathematical y-up coordinates."""

    if dx == 0 and dy == 0:
        return None
    return float(np.degrees(np.arctan2(-dy, dx)))


def direction_change_summary(
    path_df: pd.DataFrame,
    *,
    config: PathAnalysisConfig | None = None,
) -> dict:
    """Estimate robust projected direction change from early and late path windows."""

    cfg = config or PathAnalysisConfig()
    supported = path_df[
        path_df["path_status"].eq("SUPPORTED")
        & path_df["compensated_x"].notna()
        & path_df["compensated_y"].notna()
    ].sort_values("movement_end_relative_ms")
    if len(supported) < cfg.minimum_path_samples:
        return {
            "evidence_status": "UNAVAILABLE",
            "reason": "Insufficient supported camera-compensated path samples.",
        }
    window_size = max(
        cfg.minimum_direction_window_samples,
        round(len(supported) * cfg.direction_window_fraction),
    )
    early = supported.head(window_size)
    late = supported.tail(window_size)
    if len(early) < cfg.minimum_direction_window_samples or len(late) < cfg.minimum_direction_window_samples:
        return {
            "evidence_status": "UNAVAILABLE",
            "reason": "Insufficient samples in robust heading windows.",
        }
    early_heading = _heading_from_window(early)
    late_heading = _heading_from_window(late)
    if early_heading is None or late_heading is None:
        return {"evidence_status": "UNAVAILABLE", "reason": "Heading vector length was zero."}
    change = _wrapped_difference(late_heading, early_heading)
    return {
        "evidence_status": "SUPPORTED",
        "incoming_heading_deg": early_heading,
        "late_heading_deg": late_heading,
        "projected_change_of_direction_angle_deg": change,
        "absolute_projected_change_of_direction_angle_deg": abs(change),
        "source_frame_start": int(supported.iloc[0]["source_frame_index"]),
        "source_frame_end": int(supported.iloc[-1]["source_frame_index"]),
        "technical_explanation": (
            "Heading is calculated from early and late camera-compensated body-center "
            "windows in the video plane; it is not a calibrated pitch heading."
        ),
    }


def path_quality_summary(path_df: pd.DataFrame) -> dict:
    """Summarize projected movement-path support and speed descriptors."""

    if path_df.empty:
        return {"overall_status": "UNAVAILABLE", "reason": "No projected body-center samples."}
    status_counts = path_df["path_status"].value_counts().to_dict()
    supported = path_df[path_df["path_status"].eq("SUPPORTED")].copy()
    if supported.empty:
        validation_status = _mode_or_none(path_df.get("path_validation_status"))
        validation_reason = _mode_or_none(path_df.get("path_validation_reason"))
        if validation_status:
            return {
                "overall_status": validation_status,
                "status_counts": status_counts,
                "reason": validation_reason or "Projected movement path did not pass validation.",
                "camera_compensation_method": _path_method_label(path_df),
                "validation": {
                    "validation_status": validation_status,
                    "reason": validation_reason,
                },
            }
        return {
            "overall_status": "UNAVAILABLE",
            "status_counts": status_counts,
            "reason": "No supported camera-compensated path samples.",
        }
    validation = validate_projected_path(path_df)
    if validation["validation_status"] != "SUPPORTED":
        return {
            "overall_status": validation["validation_status"],
            "status_counts": status_counts,
            "supported_samples": len(supported),
            "center_source": str(supported["center_source"].mode().iloc[0]),
            "camera_compensation_method": _path_method_label(path_df),
            "validation": validation,
            "reason": validation["reason"],
        }
    speeds = supported["normalized_projected_speed_per_s"].dropna()
    direction = direction_change_summary(path_df)
    slowdown = _speed_change(supported, -500.0)
    return {
        "overall_status": "SUPPORTED",
        "status_counts": status_counts,
        "supported_samples": len(supported),
        "center_source": str(supported["center_source"].mode().iloc[0]),
        "camera_compensation_method": _path_method_label(path_df),
        "heading_method": "mathematical y-up heading from camera-compensated body-center deltas",
        "speed_method": "body-scale-normalized projected speed; not m/s",
        "mean_normalized_projected_speed_per_s": float(speeds.mean()) if len(speeds) else None,
        "minimum_normalized_projected_speed_per_s": float(speeds.min()) if len(speeds) else None,
        "time_minimum_projected_speed_ms": _time_of_min(supported, "normalized_projected_speed_per_s"),
        "maximum_normalized_projected_speed_per_s": float(speeds.max()) if len(speeds) else None,
        "direction_change": direction,
        "projected_speed_change_final_500ms": slowdown,
        "validation": validation,
    }


def validate_projected_path(
    path_df: pd.DataFrame,
    *,
    config: PathAnalysisConfig | None = None,
) -> dict:
    """Apply conservative scientific QA to a projected movement path."""

    cfg = config or PathAnalysisConfig()
    if path_df.empty:
        return {"validation_status": "UNAVAILABLE", "reason": "No path rows were produced."}
    supported = path_df[path_df["path_status"].eq("SUPPORTED")].copy()
    supported_fraction = len(supported) / len(path_df) if len(path_df) else 0.0
    if len(supported) < cfg.minimum_path_samples:
        return {
            "validation_status": "UNAVAILABLE",
            "reason": "Fewer than the configured supported projected-path samples were available.",
            "supported_fraction": supported_fraction,
        }
    segment_counts = supported.groupby("path_segment_id", dropna=True).size().to_dict()
    max_segment = max(segment_counts.values()) if segment_counts else 0
    if max_segment < cfg.minimum_path_samples:
        return {
            "validation_status": "UNAVAILABLE",
            "reason": "No continuous path segment met the minimum supported sample count.",
            "supported_fraction": supported_fraction,
            "path_segment_counts": segment_counts,
        }
    if supported_fraction < cfg.minimum_supported_fraction_for_validation:
        return {
            "validation_status": "QA_REQUIRED",
            "reason": "Supported path fraction is too low for movement-path semantics.",
            "supported_fraction": supported_fraction,
            "path_segment_counts": segment_counts,
        }
    steps = supported["normalized_projected_speed_per_s"].dropna().abs()
    if len(steps) >= 4:
        median_step = float(steps.median())
        max_step = float(steps.max())
        if median_step > 0 and max_step / median_step > cfg.maximum_step_to_median_ratio:
            return {
                "validation_status": "QA_REQUIRED",
                "reason": "Projected path contains a large step relative to the median supported step.",
                "supported_fraction": supported_fraction,
                "maximum_to_median_step_ratio": max_step / median_step,
                "path_segment_counts": segment_counts,
            }
    return {
        "validation_status": "SUPPORTED",
        "reason": "",
        "supported_fraction": supported_fraction,
        "path_segment_counts": segment_counts,
    }


def enforce_path_validation(
    path_df: pd.DataFrame,
    validation: dict,
) -> pd.DataFrame:
    """Mark path rows unavailable when validation does not support semantics."""

    output = path_df.copy()
    output["path_validation_status"] = validation.get("validation_status", "UNAVAILABLE")
    output["path_validation_reason"] = validation.get("reason", "")
    if validation.get("validation_status") == "SUPPORTED":
        return output
    supported = output["path_status"].eq("SUPPORTED")
    output.loc[supported, "path_status"] = validation.get("validation_status", "QA_REQUIRED")
    output.loc[supported, "path_rejection_reason"] = validation.get(
        "reason",
        "Projected movement path requires scientific QA before use.",
    )
    output.loc[supported, ["projected_heading_deg", "normalized_projected_speed_per_s"]] = np.nan
    return output


def build_path_frame_diagnostics(
    *,
    frame_quality: pd.DataFrame,
    center_df: pd.DataFrame,
    roi_records: pd.DataFrame,
    translation_camera: pd.DataFrame,
    affine_camera: pd.DataFrame,
    translation_path: pd.DataFrame,
    affine_path: pd.DataFrame,
    final_path: pd.DataFrame,
) -> pd.DataFrame:
    """Build aligned framewise diagnostics for movement-path scientific QA."""

    base = frame_quality[
        [
            "source_frame_index",
            "analysis_frame_index",
            "timestamp_ms",
            "frame_status",
            "frame_rejection_reason",
            "valid_segment_id",
            "pelvis_x",
            "pelvis_y",
        ]
    ].drop_duplicates("source_frame_index")
    output = base.rename(columns={"pelvis_x": "raw_pelvis_mid_x", "pelvis_y": "raw_pelvis_mid_y"})
    if not center_df.empty:
        output = output.merge(
            center_df[
                [
                    "source_frame_index",
                    "center_x",
                    "center_y",
                    "center_status",
                ]
            ].rename(
                columns={
                    "center_x": "path_input_center_x",
                    "center_y": "path_input_center_y",
                }
            ),
            on="source_frame_index",
            how="left",
        )
    if not roi_records.empty:
        output = output.merge(roi_records, on="source_frame_index", how="left")
    output = output.merge(
        _camera_diagnostic_columns(translation_camera, "translation"),
        on="source_frame_index",
        how="left",
    )
    output = output.merge(
        _camera_diagnostic_columns(affine_camera, "affine"),
        on="source_frame_index",
        how="left",
    )
    output = output.merge(
        _path_diagnostic_columns(translation_path, "translation"),
        on="source_frame_index",
        how="left",
    )
    output = output.merge(
        _path_diagnostic_columns(affine_path, "affine"),
        on="source_frame_index",
        how="left",
    )
    output = output.merge(
        _path_diagnostic_columns(final_path, "chosen"),
        on="source_frame_index",
        how="left",
    )
    return output.sort_values("source_frame_index").reset_index(drop=True)


def _path_row(
    current: pd.Series,
    compensated_x,
    compensated_y,
    speed,
    heading,
    path_status: str,
    reason: str,
    *,
    path_segment_id: str = "",
    compensation_method: str = "",
    raw_dx=None,
    raw_dy=None,
    compensated_dx=None,
    compensated_dy=None,
    camera_dx=None,
    camera_dy=None,
    background_feature_count=None,
    camera_motion_residual_px=None,
) -> dict:
    return {
        "source_frame_index": int(current["source_frame_index"]),
        "timestamp_ms": float(current["timestamp_ms"]),
        "movement_elapsed_ms": current.get("movement_elapsed_ms"),
        "movement_end_relative_ms": current.get("movement_end_relative_ms"),
        "center_x": float(current["center_x"]),
        "center_y": float(current["center_y"]),
        "center_source": str(current.get("center_source", "pelvis_midpoint")),
        "raw_dx": raw_dx,
        "raw_dy": raw_dy,
        "compensated_dx": compensated_dx,
        "compensated_dy": compensated_dy,
        "compensated_x": compensated_x,
        "compensated_y": compensated_y,
        "camera_dx": camera_dx,
        "camera_dy": camera_dy,
        "camera_compensation_method": compensation_method,
        "background_feature_count": background_feature_count,
        "camera_motion_residual_px": camera_motion_residual_px,
        "projected_heading_deg": heading,
        "normalized_projected_speed_per_s": speed,
        "path_segment_id": path_segment_id,
        "path_status": path_status,
        "path_rejection_reason": reason,
    }


def _camera_motion_row(
    frame_index,
    dx,
    dy,
    count,
    residual,
    status,
    *,
    method: str,
    affine_matrix=None,
    affine_scale=None,
    affine_inlier_fraction=None,
) -> dict:
    matrix = np.asarray(affine_matrix if affine_matrix is not None else _identity_affine(), dtype=float)
    return {
        "source_frame_index": int(frame_index),
        "background_dx_px": float(dx),
        "background_dy_px": float(dy),
        "background_feature_count": int(count),
        "camera_motion_status": status.value if isinstance(status, CameraMotionStatus) else str(status),
        "camera_motion_residual_px": residual,
        "camera_motion_method": method,
        "affine_a": float(matrix[0, 0]),
        "affine_b": float(matrix[0, 1]),
        "affine_tx": float(matrix[0, 2]),
        "affine_c": float(matrix[1, 0]),
        "affine_d": float(matrix[1, 1]),
        "affine_ty": float(matrix[1, 2]),
        "affine_scale": affine_scale,
        "affine_inlier_fraction": affine_inlier_fraction,
    }


def _identity_affine() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return homogeneous @ matrix.T


def _camera_method(camera_row) -> str:
    if camera_row is None:
        return ""
    try:
        return str(camera_row.get("camera_motion_method", "translation_median_sparse_flow"))
    except AttributeError:
        return "translation_median_sparse_flow"


def _camera_displacement_at_point(camera_row: pd.Series, previous: pd.Series) -> tuple[float, float]:
    method = _camera_method(camera_row)
    if method == "partial_affine_ransac_sparse_flow":
        matrix = np.array(
            [
                [
                    float(camera_row.get("affine_a", 1.0)),
                    float(camera_row.get("affine_b", 0.0)),
                    float(camera_row.get("affine_tx", 0.0)),
                ],
                [
                    float(camera_row.get("affine_c", 0.0)),
                    float(camera_row.get("affine_d", 1.0)),
                    float(camera_row.get("affine_ty", 0.0)),
                ],
            ],
            dtype=float,
        )
        point = np.array([[float(previous["center_x"]), float(previous["center_y"])]])
        predicted = _apply_affine(matrix, point)[0]
        return (
            float(predicted[0] - float(previous["center_x"])),
            float(predicted[1] - float(previous["center_y"])),
        )
    return float(camera_row["background_dx_px"]), float(camera_row["background_dy_px"])


def _median_timestamp_delta(centers: pd.DataFrame) -> float:
    deltas = centers["timestamp_ms"].astype(float).diff().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 33.333
    return float(deltas.median())


def _path_contiguous(
    previous: pd.Series | None,
    current: pd.Series,
    median_dt: float,
    cfg: PathAnalysisConfig,
) -> bool:
    if previous is None:
        return False
    frame_gap = int(current["source_frame_index"]) - int(previous["source_frame_index"])
    time_gap = float(current["timestamp_ms"]) - float(previous["timestamp_ms"])
    return (
        0 < frame_gap <= cfg.maximum_frame_gap
        and time_gap > 0
        and time_gap <= median_dt * cfg.maximum_time_gap_multiplier
    )


def _segment_id(segment_number: int) -> str:
    return f"path_segment_{segment_number:03d}"


def _reject_short_path_segments(output: pd.DataFrame, cfg: PathAnalysisConfig) -> None:
    if output.empty or "path_segment_id" not in output.columns:
        return
    supported = output[output["path_status"].eq("SUPPORTED") & output["path_segment_id"].ne("")]
    if supported.empty:
        return
    short_ids = [
        segment_id
        for segment_id, rows in supported.groupby("path_segment_id", sort=False)
        if len(rows) < cfg.minimum_path_samples
    ]
    if not short_ids:
        return
    mask = output["path_segment_id"].isin(short_ids) & output["path_status"].eq("SUPPORTED")
    output.loc[mask, "path_status"] = "INSUFFICIENT_EVIDENCE"
    output.loc[mask, "path_rejection_reason"] = (
        "Continuous path run is shorter than the configured minimum sample count."
    )
    output.loc[mask, ["projected_heading_deg", "normalized_projected_speed_per_s"]] = np.nan


def _path_method_label(path_df: pd.DataFrame) -> str:
    if "camera_compensation_method" not in path_df.columns:
        return "translation-only sparse optical flow with target ROI masked"
    methods = path_df["camera_compensation_method"].replace("", np.nan).dropna()
    if methods.empty:
        return "camera compensation method unavailable"
    method = str(methods.mode().iloc[0])
    if method == "partial_affine_ransac_sparse_flow":
        return "partial affine/RANSAC sparse optical flow with human ROI masked"
    if method == "translation_median_sparse_flow":
        return "translation-only sparse optical flow with human ROI masked"
    return method


def _camera_diagnostic_columns(camera_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if camera_df.empty:
        return pd.DataFrame({"source_frame_index": pd.Series(dtype=int)})
    columns = [
        "source_frame_index",
        "camera_motion_status",
        "camera_motion_method",
        "background_dx_px",
        "background_dy_px",
        "background_feature_count",
        "camera_motion_residual_px",
        "affine_scale",
        "affine_inlier_fraction",
    ]
    existing = [column for column in columns if column in camera_df.columns]
    output = camera_df[existing].copy()
    rename = {
        column: f"{prefix}_{column}"
        for column in existing
        if column != "source_frame_index"
    }
    return output.rename(columns=rename)


def _path_diagnostic_columns(path_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if path_df.empty:
        return pd.DataFrame({"source_frame_index": pd.Series(dtype=int)})
    columns = [
        "source_frame_index",
        "raw_dx",
        "raw_dy",
        "compensated_dx",
        "compensated_dy",
        "compensated_x",
        "compensated_y",
        "projected_heading_deg",
        "normalized_projected_speed_per_s",
        "path_segment_id",
        "path_status",
        "path_rejection_reason",
        "path_validation_status",
        "path_validation_reason",
    ]
    existing = [column for column in columns if column in path_df.columns]
    output = path_df[existing].copy()
    rename = {
        column: f"{prefix}_{column}"
        for column in existing
        if column != "source_frame_index"
    }
    return output.rename(columns=rename)


def _optional_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _mode_or_none(series) -> str | None:
    if series is None:
        return None
    values = series.replace("", np.nan).dropna()
    if values.empty:
        return None
    return str(values.mode().iloc[0])


def _read_gray(capture, frame_index: int):
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not read video frame {frame_index}.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _heading_from_window(window: pd.DataFrame) -> float | None:
    first = window.iloc[0]
    last = window.iloc[-1]
    return projected_heading_deg(
        float(last["compensated_x"] - first["compensated_x"]),
        float(last["compensated_y"] - first["compensated_y"]),
    )


def _wrapped_difference(a: float, b: float) -> float:
    return float((a - b + 180.0) % 360.0 - 180.0)


def _time_of_min(df: pd.DataFrame, column: str) -> float | None:
    values = df[df[column].notna()]
    if values.empty:
        return None
    row = values.loc[values[column].idxmin()]
    return float(row["movement_end_relative_ms"])


def _speed_change(supported: pd.DataFrame, window_start_ms: float) -> dict:
    speeds = supported[supported["normalized_projected_speed_per_s"].notna()]
    final = speeds[speeds["movement_end_relative_ms"].between(window_start_ms, 0.0, inclusive="both")]
    prior = speeds[speeds["movement_end_relative_ms"] < window_start_ms]
    if len(final) < 3 or len(prior) < 3:
        return {
            "evidence_status": "UNAVAILABLE",
            "reason": "Insufficient supported speed samples in comparison windows.",
        }
    change = float(final["normalized_projected_speed_per_s"].mean() - prior["normalized_projected_speed_per_s"].mean())
    if change <= -0.5:
        label = "PROJECTED_SLOWDOWN"
    elif change >= 0.5:
        label = "PROJECTED_ACCELERATION"
    else:
        label = "RELATIVELY_STABLE_PROJECTED_SPEED"
    return {
        "evidence_status": "SUPPORTED",
        "window_start_ms": window_start_ms,
        "mean_prior_projected_speed": float(prior["normalized_projected_speed_per_s"].mean()),
        "mean_final_projected_speed": float(final["normalized_projected_speed_per_s"].mean()),
        "projected_speed_change": change,
        "descriptor": label,
    }
