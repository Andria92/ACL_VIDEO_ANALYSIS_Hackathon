"""QC-state skeleton overlay rendering."""

from __future__ import annotations

import math
from typing import Any

from acl_motion.video.roi import BBox
from acl_motion.visualisation.overlay import MEDIAPIPE_SKELETON

FRAME_COLORS = {
    "VALID_TARGET": (70, 220, 70),
    "LOW_POSE_CONFIDENCE": (40, 190, 255),
    "PARTIAL_POSE": (40, 190, 255),
    "TARGET_IDENTITY_UNCERTAIN": (220, 90, 220),
    "TARGET_NOT_FOUND": (60, 60, 240),
    "INVALID_TRACK_SEGMENT": (60, 60, 240),
}

LANDMARK_COLORS = {
    "OBSERVED_VALID": (70, 220, 70),
    "LOW_CONFIDENCE": (40, 190, 255),
    "TEMPORAL_OUTLIER": (60, 60, 240),
    "IDENTITY_UNCERTAIN": (220, 90, 220),
    "MISSING": (160, 160, 160),
    "REJECTED": (60, 60, 240),
}


def draw_qc_pose_overlay(frame: Any, pose_rows, frame_status: str, frame_label: str) -> Any:
    """Draw a pose overlay colored by Milestone 2 QC state."""

    import cv2

    output = frame.copy()
    frame_color = FRAME_COLORS.get(frame_status, (200, 200, 200))
    bbox = _bbox_from_rows(pose_rows)
    if bbox is not None:
        x1, y1, x2, y2 = bbox.as_int_xyxy()
        cv2.rectangle(output, (x1, y1), (x2, y2), frame_color, 2)

    points = {}
    for row in pose_rows:
        point = _point_from_row(row, prefer_processed=frame_status == "VALID_TARGET")
        if point is not None:
            points[row["landmark_name"]] = point

    for start, end in MEDIAPIPE_SKELETON:
        if start in points and end in points:
            cv2.line(output, points[start], points[end], frame_color, 2, lineType=cv2.LINE_AA)

    for row in pose_rows:
        point = points.get(row["landmark_name"])
        if point is None:
            continue
        color = LANDMARK_COLORS.get(row.get("landmark_status"), frame_color)
        cv2.circle(output, point, 4, color, -1, lineType=cv2.LINE_AA)

    label = f"{frame_label} | {frame_status}"
    cv2.putText(
        output,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return output


def _point_from_row(row: dict, *, prefer_processed: bool) -> tuple[int, int] | None:
    if prefer_processed:
        x_value = row.get("smoothed_x")
        y_value = row.get("smoothed_y")
        if _is_missing_number(x_value) or _is_missing_number(y_value):
            x_value = row.get("clean_x")
            y_value = row.get("clean_y")
    else:
        x_value = row.get("raw_x")
        y_value = row.get("raw_y")
    if _is_missing_number(x_value) or _is_missing_number(y_value):
        return None
    return round(float(x_value)), round(float(y_value))


def _bbox_from_rows(rows: list[dict]) -> BBox | None:
    if not rows:
        return None
    first = rows[0]
    values = [
        first.get("target_bbox_x"),
        first.get("target_bbox_y"),
        first.get("target_bbox_width"),
        first.get("target_bbox_height"),
    ]
    if any(_is_missing_number(value) for value in values):
        return None
    return BBox(float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _is_missing_number(value: Any) -> bool:
    if value is None:
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True
