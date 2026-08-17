"""Conservative coordinate processing for Milestone 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from acl_motion.quality.models import LandmarkQualityStatus, ProcessingStatus


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Configurable coordinate-processing thresholds."""

    max_interpolation_gap_frames: int = 2
    smoothing_window_frames: int = 5
    min_smoothing_segment_frames: int = 5


def process_pose_coordinates(
    raw_pose: pd.DataFrame,
    landmark_quality: pd.DataFrame,
    config: ProcessingConfig | None = None,
) -> pd.DataFrame:
    """Create clean/smoothed coordinates while preserving raw columns."""

    cfg = config or ProcessingConfig()
    processed = raw_pose.merge(
        landmark_quality[
            [
                "frame_index",
                "landmark_name",
                "frame_status",
                "valid_target_frame",
                "valid_segment_id",
                "landmark_status",
                "landmark_rejection_reason",
                "landmark_jump_px",
            ]
        ],
        on=["frame_index", "landmark_name"],
        how="left",
    ).copy()
    processed = processed.rename(
        columns={
            "x_px": "raw_x",
            "y_px": "raw_y",
            "x_norm": "raw_x_norm",
            "y_norm": "raw_y_norm",
        }
    )
    processed["clean_x"] = np.nan
    processed["clean_y"] = np.nan
    processed["smoothed_x"] = np.nan
    processed["smoothed_y"] = np.nan
    processed["interpolated"] = False
    processed["smoothed"] = False
    processed["rejected"] = True
    processed["processing_status"] = ProcessingStatus.MISSING.value

    valid_mask = processed["landmark_status"].eq(LandmarkQualityStatus.OBSERVED_VALID.value)
    processed.loc[valid_mask, "clean_x"] = processed.loc[valid_mask, "raw_x"]
    processed.loc[valid_mask, "clean_y"] = processed.loc[valid_mask, "raw_y"]
    processed.loc[valid_mask, "rejected"] = False
    processed.loc[valid_mask, "processing_status"] = ProcessingStatus.RAW_VALID.value

    _apply_rejection_statuses(processed)
    _interpolate_short_gaps(processed, cfg)
    _smooth_valid_segments(processed, cfg)
    return processed.sort_values(["frame_index", "landmark_name"]).reset_index(drop=True)


def _apply_rejection_statuses(processed: pd.DataFrame) -> None:
    low = processed["landmark_status"].eq(LandmarkQualityStatus.LOW_CONFIDENCE.value)
    outlier = processed["landmark_status"].eq(LandmarkQualityStatus.TEMPORAL_OUTLIER.value)
    identity = processed["landmark_status"].eq(LandmarkQualityStatus.IDENTITY_UNCERTAIN.value)
    rejected = processed["landmark_status"].eq(LandmarkQualityStatus.REJECTED.value)
    missing = processed["landmark_status"].eq(LandmarkQualityStatus.MISSING.value)
    processed.loc[low, "processing_status"] = ProcessingStatus.REJECTED_LOW_CONFIDENCE.value
    processed.loc[outlier, "processing_status"] = ProcessingStatus.REJECTED_TEMPORAL_OUTLIER.value
    processed.loc[identity, "processing_status"] = ProcessingStatus.REJECTED_IDENTITY_UNCERTAIN.value
    processed.loc[rejected, "processing_status"] = ProcessingStatus.REJECTED_INVALID_TRACK_SEGMENT.value
    processed.loc[missing, "processing_status"] = ProcessingStatus.MISSING.value


def _interpolate_short_gaps(processed: pd.DataFrame, cfg: ProcessingConfig) -> None:
    if cfg.max_interpolation_gap_frames <= 0:
        return
    for (landmark_name, segment_id), group in processed.groupby(
        ["landmark_name", "valid_segment_id"], dropna=True, sort=False
    ):
        if pd.isna(segment_id):
            continue
        group = group.sort_values("frame_index")
        valid = group["clean_x"].notna() & group["clean_y"].notna()
        if valid.sum() < 2:
            continue
        interpolated = group[["clean_x", "clean_y"]].interpolate(
            method="linear",
            limit=cfg.max_interpolation_gap_frames,
            limit_area="inside",
        )
        newly_filled = (~valid) & interpolated["clean_x"].notna() & interpolated["clean_y"].notna()
        if not newly_filled.any():
            continue
        indices = group.index[newly_filled]
        processed.loc[indices, "clean_x"] = interpolated.loc[indices, "clean_x"]
        processed.loc[indices, "clean_y"] = interpolated.loc[indices, "clean_y"]
        processed.loc[indices, "interpolated"] = True
        processed.loc[indices, "rejected"] = False
        processed.loc[indices, "processing_status"] = ProcessingStatus.INTERPOLATED.value


def _smooth_valid_segments(processed: pd.DataFrame, cfg: ProcessingConfig) -> None:
    window = cfg.smoothing_window_frames
    if window <= 1:
        processed["smoothed_x"] = processed["clean_x"]
        processed["smoothed_y"] = processed["clean_y"]
        return
    if window % 2 == 0:
        window += 1
    for (_, segment_id), segment in processed.groupby(["landmark_name", "valid_segment_id"], dropna=True):
        if pd.isna(segment_id):
            continue
        segment = segment.sort_values("frame_index")
        usable = segment["clean_x"].notna() & segment["clean_y"].notna()
        if int(usable.sum()) < cfg.min_smoothing_segment_frames:
            indices = segment.index[usable]
            processed.loc[indices, "smoothed_x"] = processed.loc[indices, "clean_x"]
            processed.loc[indices, "smoothed_y"] = processed.loc[indices, "clean_y"]
            continue
        local_window = min(window, int(usable.sum()) if int(usable.sum()) % 2 == 1 else int(usable.sum()) - 1)
        local_window = max(local_window, 3)
        valid_segment = segment.loc[usable]
        smoothed_x = valid_segment["clean_x"].rolling(
            window=local_window,
            center=True,
            min_periods=1,
        ).median()
        smoothed_y = valid_segment["clean_y"].rolling(
            window=local_window,
            center=True,
            min_periods=1,
        ).median()
        processed.loc[valid_segment.index, "smoothed_x"] = smoothed_x
        processed.loc[valid_segment.index, "smoothed_y"] = smoothed_y
        processed.loc[valid_segment.index, "smoothed"] = True
        raw_valid = processed.loc[valid_segment.index, "processing_status"].eq(ProcessingStatus.RAW_VALID.value)
        processed.loc[valid_segment.index[raw_valid], "processing_status"] = ProcessingStatus.SMOOTHED.value
