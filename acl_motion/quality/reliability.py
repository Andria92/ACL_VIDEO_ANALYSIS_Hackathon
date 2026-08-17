"""Pose reliability summaries for Milestone 2."""

from __future__ import annotations

from typing import Any

import pandas as pd

from acl_motion.quality.pose_quality import CORE_LANDMARKS, LOWER_LIMB_LANDMARKS


def build_reliability_summary(
    processed_pose: pd.DataFrame,
    frame_quality: pd.DataFrame,
) -> dict[str, Any]:
    """Build interpretable coverage and processing-quality metrics."""

    total_frames = int(frame_quality["frame_index"].nunique())
    valid_frames = int(frame_quality["valid_target_frame"].sum())
    summary: dict[str, Any] = {
        "total_frames": total_frames,
        "target_tracking_coverage": _fraction(valid_frames, total_frames),
        "pose_frame_coverage": _fraction(
            int((frame_quality["observed_landmark_count"] > 0).sum()),
            total_frames,
        ),
        "frame_status_counts": frame_quality["frame_status"].value_counts().sort_index().to_dict(),
        "low_confidence_intervals": _intervals(
            frame_quality[frame_quality["frame_status"].eq("LOW_POSE_CONFIDENCE")]["frame_index"].tolist()
        ),
        "target_loss_intervals": _intervals(
            frame_quality[frame_quality["frame_status"].eq("TARGET_NOT_FOUND")]["frame_index"].tolist()
        ),
        "identity_uncertainty_intervals": _intervals(
            frame_quality[frame_quality["frame_status"].eq("TARGET_IDENTITY_UNCERTAIN")][
                "frame_index"
            ].tolist()
        ),
        "valid_target_segments": _segments(frame_quality),
        "interpolated_frame_fraction": _fraction(
            int(processed_pose[processed_pose["interpolated"]].frame_index.nunique()),
            total_frames,
        ),
        "rejected_frame_fraction": _fraction(
            int(processed_pose[processed_pose["rejected"]].frame_index.nunique()),
            total_frames,
        ),
        "interpolated_landmark_row_fraction": _fraction(
            int(processed_pose["interpolated"].sum()),
            len(processed_pose),
        ),
        "rejected_landmark_row_fraction": _fraction(int(processed_pose["rejected"].sum()), len(processed_pose)),
        "processing_status_counts": processed_pose["processing_status"].value_counts().sort_index().to_dict(),
        "landmark_status_counts": processed_pose["landmark_status"].value_counts().sort_index().to_dict(),
    }
    for landmark_name, group in processed_pose.groupby("landmark_name"):
        summary[f"{landmark_name}_coverage"] = _fraction(int(group["clean_x"].notna().sum()), total_frames)
    summary["core_landmark_coverage"] = _coverage_for_landmarks(processed_pose, CORE_LANDMARKS, total_frames)
    summary["lower_limb_landmark_coverage"] = _coverage_for_landmarks(
        processed_pose,
        LOWER_LIMB_LANDMARKS,
        total_frames,
    )
    return summary


def _coverage_for_landmarks(processed_pose: pd.DataFrame, landmarks: tuple[str, ...], total_frames: int) -> float:
    subset = processed_pose[processed_pose["landmark_name"].isin(landmarks)]
    expected = total_frames * len(landmarks)
    return _fraction(int(subset["clean_x"].notna().sum()), expected)


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _intervals(frames: list[int]) -> list[dict[str, int]]:
    if not frames:
        return []
    frames = sorted(frames)
    intervals: list[dict[str, int]] = []
    start = frames[0]
    previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        intervals.append({"start_frame": start, "end_frame": previous})
        start = previous = frame
    intervals.append({"start_frame": start, "end_frame": previous})
    return intervals


def _segments(frame_quality: pd.DataFrame) -> list[dict[str, int]]:
    rows = frame_quality.dropna(subset=["valid_segment_id"])
    segments: list[dict[str, int]] = []
    for segment_id, group in rows.groupby("valid_segment_id"):
        segments.append(
            {
                "segment_id": int(segment_id),
                "start_frame": int(group["frame_index"].min()),
                "end_frame": int(group["frame_index"].max()),
                "frame_count": len(group),
            }
        )
    return segments
